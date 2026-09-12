"""DOCX 正文提取器（计划 7.4 节）。

使用标准库 ``zipfile`` 读取内存中的 DOCX 包、``defusedxml`` 流式解析
正文 XML 的第一方窄范围提取器。不引入通用文档转换框架，不依赖桌面
Word，不把压缩包解压到文件系统。

首轮只承诺"段落级正文"：标题与列表只保留文字，表格按出现顺序降级为
文字行（行间 LF、单元格间 TAB）；超链接保留显示文字，字段只取文档内
已保存的显示结果。页眉页脚、脚注尾注、批注、图片等未覆盖内容输出稳定
warning record；会使正文读取顺序不明确的结构（未接受修订、嵌套表格、
altChunk、正文内容控件、AlternateContent）整体归入受控失败。
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from collections.abc import Callable
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree as DefusedElementTree

from hivememory.system.config.attachments import AttachmentParserConfig
from hivememory.system.services.attachments.errors import (
    CONTENT_UNREADABLE,
    RESOURCE_LIMIT,
    AttachmentParseError,
)
from hivememory.system.services.attachments.limits import ParseBudget
from hivememory.system.services.attachments.models import (
    LOCATOR_KIND_PARAGRAPH,
    AttachmentContentBuilder,
    ParsedAttachmentContent,
)

# ---------------------------------------------------------------------------
# Office Open XML 命名空间、内容类型与关系类型（Transitional / Strict 显式识别）
# ---------------------------------------------------------------------------

_W_NS_TRANSITIONAL = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W_NS_STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main"
_W_NS = frozenset({_W_NS_TRANSITIONAL, _W_NS_STRICT})
_NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"

_CT_MAIN_DOCUMENT = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)
_CT_MACRO_MAIN_DOCUMENT = "application/vnd.ms-word.document.macroEnabled.main+xml"
_CT_VBA_PROJECT = "application/vnd.ms-office.vbaProject"

_REL_NS_TRANSITIONAL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_REL_NS_STRICT = "http://purl.oclc.org/ooxml/officeDocument/relationships"
_REL_NS = frozenset({_REL_NS_TRANSITIONAL, _REL_NS_STRICT})
_REL_OFFICE_DOCUMENT = "officeDocument"
_REL_HEADER = "header"
_REL_FOOTER = "footer"
_REL_FOOTNOTES = "footnotes"
_REL_ENDNOTES = "endnotes"
_REL_COMMENTS = "comments"

#: OLE 复合文档魔数：OOXML 加密文档是 CFB 容器而非 ZIP。
_CFB_MAGIC = b"\xd0\xcf\x11\xe0"

#: ZIP 成员加密标志位（通用位 0）。
_ZIP_ENCRYPTED_FLAG = 0x1

#: 成员读取的流式窗口大小。
_READ_CHUNK_SIZE = 64 * 1024

_PACKAGE_CONTENT_TYPES = "[Content_Types].xml"
_PACKAGE_RELS = "_rels/.rels"

#: 会破坏正文读取顺序或内容版本的最小结构集合（按提取语义命名）。
_UNSUPPORTED_STRUCTURE_PARAMS = {
    "ins": "revision",
    "del": "revision",
    "moveFrom": "revision",
    "moveTo": "revision",
    "altChunk": "alt_chunk",
    "sdt": "content_control",
}


def _split_tag(tag: str) -> tuple[str, str]:
    """把 ``{namespace}local`` 形式的标签拆为 (namespace, local)。"""
    if tag.startswith("{"):
        namespace, _, local = tag[1:].partition("}")
        return namespace, local
    return "", tag


def _normalize_rel_target(base_dir: str, target: str) -> str | None:
    """把包内关系目标解析为规范 ZIP 成员路径；逃逸或绝对目标返回 None。"""
    if target.startswith("/"):
        normalized = posixpath.normpath(target)
    else:
        normalized = posixpath.normpath(posixpath.join(base_dir, target) if base_dir else target)
    if normalized.startswith("..") or normalized.startswith("/"):
        return None
    return normalized


def _corrupt() -> AttachmentParseError:
    return AttachmentParseError(
        CONTENT_UNREADABLE,
        "附件损坏或不是有效的 Word 文档（.docx），请重新导出后上传",
        params={"reason": "corrupt_package"},
    )


def _parse_rels(data: bytes, config: AttachmentParserConfig) -> list[tuple[str, str, str]]:
    """解析一个 .rels part，返回 (Type, Target, TargetMode) 列表。"""
    try:
        root = DefusedElementTree.fromstring(
            data,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except (ParseError, ValueError) as exc:
        raise _corrupt() from exc
    relationships: list[tuple[str, str, str]] = []
    for element in root.iter():
        _, local = _split_tag(element.tag)
        if local != "Relationship":
            continue
        relationships.append(
            (
                element.get("Type", ""),
                element.get("Target", ""),
                element.get("TargetMode", "Internal"),
            ),
        )
        if len(relationships) > config.max_docx_members:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "附件结构过于复杂，请缩小文件后重新上传",
                params={"reason": "member_count_limit"},
            )
    return relationships


class DocxAttachmentParser:
    """DOCX 正文提取器：包校验 + 流式 XML 提取 + 覆盖范围 warning。"""

    @property
    def producer(self) -> str:
        """解析实现的稳定身份。"""
        return "docx_extract"

    @property
    def producer_version(self) -> str:
        """提取/定位规则版本；改变规则或依赖时必须提升。"""
        return "1"

    def parse(
        self,
        raw: bytes,
        *,
        config: AttachmentParserConfig,
        source_raw_revision: int,
        source_raw_hash: str,
        clock: Callable[[], float] | None = None,
    ) -> ParsedAttachmentContent:
        """把 DOCX 包 bytes 转为带段落级 locator 的正文文字表示。"""
        if len(raw) > config.max_raw_bytes:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "文件超过大小上限，请缩小后重新上传",
                params={"reason": "raw_input_limit"},
            )
        budget = (
            ParseBudget(config.parse_budget_seconds, clock)
            if clock
            else ParseBudget(config.parse_budget_seconds)
        )

        if raw.startswith(_CFB_MAGIC):
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "附件已加密，请解密后另存为 .docx 重新上传",
                params={"reason": "encrypted_package"},
            )

        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except (zipfile.BadZipFile, OSError) as exc:
            # ZIP 打开失败只说明容器无效，不能断言为加密。
            raise _corrupt() from exc

        with archive:
            members = self._precheck_directory(archive, config)
            content_types = self._read_member(archive, members, _PACKAGE_CONTENT_TYPES, config)
            package_rels = self._read_member(
                archive,
                members,
                _PACKAGE_RELS,
                config,
                required=False,
            )
            main_part = self._resolve_main_part(content_types, package_rels, members, config)
            document_xml = self._read_member(archive, members, main_part, config)
            covered_warnings = self._collect_covered_warnings(
                archive,
                members,
                main_part,
                config,
            )

        builder = AttachmentContentBuilder(
            content_format="plain_text",
            source_raw_revision=source_raw_revision,
            source_raw_hash=source_raw_hash,
            config=config,
        )
        image_count, numbered_count = self._extract_body(document_xml, builder, config, budget)
        for message_key in covered_warnings:
            builder.add_warning(message_key)
        if image_count:
            builder.add_warning("docx_images_ignored", {"count": image_count})
        if numbered_count:
            builder.add_warning("docx_auto_numbering_ignored", {"count": numbered_count})

        budget.check()
        return builder.build(producer=self.producer, producer_version=self.producer_version)

    # ------------------------------------------------------------------
    # 包级校验（ZIP 目录预检 + 主文档定位）
    # ------------------------------------------------------------------

    def _precheck_directory(
        self,
        archive: zipfile.ZipFile,
        config: AttachmentParserConfig,
    ) -> dict[str, zipfile.ZipInfo]:
        """目录预检：成员数、路径寻址、加密标志、声明大小与压缩比。"""
        infos = archive.infolist()
        if len(infos) > config.max_docx_members:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "附件结构过于复杂，请缩小文件后重新上传",
                params={"reason": "member_count_limit"},
            )
        members: dict[str, zipfile.ZipInfo] = {}
        total_declared = 0
        for info in infos:
            name = info.filename
            if name in members:
                raise AttachmentParseError(
                    CONTENT_UNREADABLE,
                    "附件损坏或不是有效的 Word 文档（.docx），请重新导出后上传",
                    params={"reason": "duplicate_member"},
                )
            # 限定内部 part 寻址：拒绝路径逃逸、绝对路径与反斜杠分隔。
            if (
                name.startswith("/")
                or "\\" in name
                or any(segment == ".." for segment in name.split("/"))
            ):
                raise AttachmentParseError(
                    CONTENT_UNREADABLE,
                    "附件损坏或不是有效的 Word 文档（.docx），请重新导出后上传",
                    params={"reason": "unsafe_member_path"},
                )
            if info.flag_bits & _ZIP_ENCRYPTED_FLAG:
                raise AttachmentParseError(
                    CONTENT_UNREADABLE,
                    "附件已加密，请解密后另存为 .docx 重新上传",
                    params={"reason": "encrypted_package"},
                )
            if info.file_size > config.max_docx_member_uncompressed_bytes:
                raise AttachmentParseError(
                    RESOURCE_LIMIT,
                    "文件超过大小上限，请缩小后重新上传",
                    params={"reason": "member_size_limit"},
                )
            if info.compress_size == 0 and info.file_size > 0:
                raise AttachmentParseError(
                    CONTENT_UNREADABLE,
                    "附件损坏或不是有效的 Word 文档（.docx），请重新导出后上传",
                    params={"reason": "compression_ratio_limit"},
                )
            if info.file_size > info.compress_size * config.max_docx_compression_ratio:
                raise AttachmentParseError(
                    CONTENT_UNREADABLE,
                    "附件损坏或不是有效的 Word 文档（.docx），请重新导出后上传",
                    params={"reason": "compression_ratio_limit"},
                )
            total_declared += info.file_size
            members[name] = info
        if total_declared > config.max_docx_package_uncompressed_bytes:
            raise AttachmentParseError(
                RESOURCE_LIMIT,
                "文件超过大小上限，请缩小后重新上传",
                params={"reason": "package_size_limit"},
            )
        return members

    def _read_member(
        self,
        archive: zipfile.ZipFile,
        members: dict[str, zipfile.ZipInfo],
        name: str,
        config: AttachmentParserConfig,
        *,
        required: bool = True,
    ) -> bytes:
        """按声明上限流式读取单个成员，并对实际读取字节计数。"""
        info = members.get(name)
        if info is None:
            if required:
                raise AttachmentParseError(
                    CONTENT_UNREADABLE,
                    "附件损坏或不是有效的 Word 文档（.docx），请重新导出后上传",
                    params={"reason": "missing_part"},
                )
            return b""
        chunks: list[bytes] = []
        remaining = config.max_docx_member_uncompressed_bytes
        with archive.open(info) as stream:
            while True:
                chunk = stream.read(min(_READ_CHUNK_SIZE, remaining + 1))
                if not chunk:
                    break
                remaining -= len(chunk)
                if remaining < 0:
                    raise AttachmentParseError(
                        RESOURCE_LIMIT,
                        "文件超过大小上限，请缩小后重新上传",
                        params={"reason": "member_size_limit"},
                    )
                chunks.append(chunk)
        return b"".join(chunks)

    def _resolve_main_part(
        self,
        content_types: bytes,
        package_rels: bytes,
        members: dict[str, zipfile.ZipInfo],
        config: AttachmentParserConfig,
    ) -> str:
        """定位主文档 part，并验证其为非宏 Word 文档。"""
        office_document_types = {f"{ns}/{_REL_OFFICE_DOCUMENT}" for ns in _REL_NS}
        office_targets = [
            (target, mode)
            for rel_type, target, mode in _parse_rels(package_rels, config)
            if rel_type in office_document_types
        ]
        if not office_targets:
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "附件不是 Word 文档（.docx），请确认文件格式后重新上传",
                params={"reason": "not_word_document"},
            )
        target, mode = office_targets[0]
        if mode == "External":
            raise _corrupt()
        main_part = _normalize_rel_target("", target)
        if main_part is None or main_part not in members:
            raise _corrupt()

        overrides = self._content_type_overrides(content_types, config)
        main_content_type = overrides.get(f"/{main_part}")
        if main_content_type == _CT_MACRO_MAIN_DOCUMENT or _CT_VBA_PROJECT in (overrides.values()):
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "不支持包含宏的 Word 文档，请另存为不带宏的 .docx 后上传",
                params={"reason": "macro_document"},
            )
        if main_content_type != _CT_MAIN_DOCUMENT:
            raise AttachmentParseError(
                CONTENT_UNREADABLE,
                "附件不是 Word 文档（.docx），请确认文件格式后重新上传",
                params={"reason": "not_word_document"},
            )
        return main_part

    def _content_type_overrides(
        self,
        content_types: bytes,
        config: AttachmentParserConfig,
    ) -> dict[str, str]:
        """解析 [Content_Types].xml 的 Override 表（PartName → ContentType）。"""
        try:
            root = DefusedElementTree.fromstring(
                content_types,
                forbid_dtd=True,
                forbid_entities=True,
                forbid_external=True,
            )
        except (ParseError, ValueError) as exc:
            raise _corrupt() from exc
        overrides: dict[str, str] = {}
        for element in root.iter():
            _, local = _split_tag(element.tag)
            if local != "Override":
                continue
            part_name = element.get("PartName", "")
            if not part_name:
                raise _corrupt()
            overrides[part_name] = element.get("ContentType", "")
            if len(overrides) > config.max_docx_members:
                raise AttachmentParseError(
                    RESOURCE_LIMIT,
                    "附件结构过于复杂，请缩小文件后重新上传",
                    params={"reason": "member_count_limit"},
                )
        return overrides

    def _collect_covered_warnings(
        self,
        archive: zipfile.ZipFile,
        members: dict[str, zipfile.ZipInfo],
        main_part: str,
        config: AttachmentParserConfig,
    ) -> list[str]:
        """按主文档关系识别首轮未覆盖的内容（页眉页脚/脚注尾注/批注）。"""
        directory, _, filename = main_part.rpartition("/")
        rels_name = f"{directory}/_rels/{filename}.rels" if directory else f"_rels/{filename}.rels"
        rels_data = self._read_member(archive, members, rels_name, config, required=False)
        if not rels_data:
            return []
        covered: set[str] = set()
        for rel_type, _target, _mode in _parse_rels(rels_data, config):
            ns, _, local = rel_type.rpartition("/")
            if ns not in _REL_NS:
                continue
            if local in {_REL_HEADER, _REL_FOOTER}:
                covered.add("docx_headers_footers_ignored")
            elif local in {_REL_FOOTNOTES, _REL_ENDNOTES}:
                covered.add("docx_footnotes_endnotes_ignored")
            elif local == _REL_COMMENTS:
                covered.add("docx_comments_ignored")
        return sorted(covered)

    # ------------------------------------------------------------------
    # 正文流式提取（Store 锁外执行；深度/节点/预算在检查点累计）
    # ------------------------------------------------------------------

    def _extract_body(
        self,
        document_xml: bytes,
        builder: AttachmentContentBuilder,
        config: AttachmentParserConfig,
        budget: ParseBudget,
    ) -> tuple[int, int]:
        """流式遍历 document.xml，返回 (未覆盖图片数, 自动编号段落计数)。

        状态说明：

        - ``suppress_depth``：位于 drawing/pict/object 内部时正文不参与提取，
          其中的段落/表格/修订不改变外部状态；
        - ``field_stack``：复杂字段 begin→separate 段的显示结果不输出，
          separate→end 段为文档保存的字段显示结果；
        - ``unit_count``：正文单元（body 段落 + 表格行）序号，计入空元素。
        """
        depth = 0
        nodes = 0
        suppress_depth = 0
        image_count = 0
        numbered_count = 0
        unit_count = 0
        tc_depth = 0
        field_stack: list[str] = []
        paragraph_chars: list[str] = []
        paragraph_numbered = False
        cell_paragraph_texts: list[str] = []
        row_cell_texts: list[str] = []

        def emit_unit(unit_text: str, *, numbered: bool) -> None:
            """把一个正文单元（段落文字行或表格行）写入 builder。"""
            nonlocal unit_count, numbered_count
            if unit_count > 0:
                builder.append_text("\n")
            start = builder.char_count
            builder.append_text(unit_text)
            unit_count += 1
            if numbered:
                numbered_count += 1
            if unit_text:
                builder.add_locator(
                    kind=LOCATOR_KIND_PARAGRAPH,
                    number=unit_count,
                    start=start,
                    end=start + len(unit_text),
                )

        def field_visible() -> bool:
            return all(state == "post" for state in field_stack)

        try:
            events = DefusedElementTree.iterparse(
                io.BytesIO(document_xml),
                events=("start", "end"),
                forbid_dtd=True,
                forbid_entities=True,
                forbid_external=True,
            )
            for event, element in events:
                budget.check()
                uri, local = _split_tag(element.tag)
                w_local = local if uri in _W_NS else None

                if event == "start":
                    depth += 1
                    nodes += 1
                    if depth > config.max_xml_depth:
                        raise AttachmentParseError(
                            RESOURCE_LIMIT,
                            "附件结构过于复杂，请缩小文件后重新上传",
                            params={"reason": "xml_depth_limit"},
                        )
                    if nodes > config.max_xml_nodes:
                        raise AttachmentParseError(
                            RESOURCE_LIMIT,
                            "附件结构过于复杂，请缩小文件后重新上传",
                            params={"reason": "xml_node_limit"},
                        )

                    if suppress_depth == 0:
                        # 未接受修订、altChunk、内容控件与 AlternateContent 整体失败。
                        if w_local in _UNSUPPORTED_STRUCTURE_PARAMS:
                            raise AttachmentParseError(
                                CONTENT_UNREADABLE,
                                "附件包含暂不支持的内容（如修订、嵌套表格或内容控件），"
                                "请简化文档后重新上传",
                                params={
                                    "reason": "unsupported_structure",
                                    "structure": _UNSUPPORTED_STRUCTURE_PARAMS[w_local],
                                },
                            )
                        if uri == _NS_MC and local == "AlternateContent":
                            raise AttachmentParseError(
                                CONTENT_UNREADABLE,
                                "附件包含暂不支持的内容（如修订、嵌套表格或内容控件），"
                                "请简化文档后重新上传",
                                params={
                                    "reason": "unsupported_structure",
                                    "structure": "alternate_content",
                                },
                            )

                    if w_local == "fldChar":
                        field_type = element.get(f"{{{_W_NS_TRANSITIONAL}}}fldCharType") or (
                            element.get(f"{{{_W_NS_STRICT}}}fldCharType")
                        )
                        if field_type == "begin":
                            field_stack.append("pre")
                        elif field_type == "separate" and field_stack:
                            field_stack[-1] = "post"
                    elif w_local in {"drawing", "pict", "object"}:
                        if suppress_depth == 0:
                            image_count += 1
                        suppress_depth += 1
                    elif suppress_depth == 0 and w_local == "tbl" and tc_depth > 0:
                        raise AttachmentParseError(
                            CONTENT_UNREADABLE,
                            "附件包含暂不支持的内容（如修订、嵌套表格或内容控件），"
                            "请简化文档后重新上传",
                            params={
                                "reason": "unsupported_structure",
                                "structure": "nested_table",
                            },
                        )
                    elif suppress_depth == 0 and w_local == "tc":
                        tc_depth += 1
                        cell_paragraph_texts = []
                    elif suppress_depth == 0 and w_local == "numPr":
                        paragraph_numbered = True
                    continue

                # ---- end 事件 ----
                depth -= 1
                if w_local == "t":
                    if suppress_depth == 0 and field_visible() and element.text:
                        paragraph_chars.append(element.text)
                elif w_local == "tab":
                    if suppress_depth == 0 and field_visible():
                        paragraph_chars.append("\t")
                elif w_local in {"br", "cr"}:
                    if suppress_depth == 0 and field_visible():
                        paragraph_chars.append("\n")
                elif w_local == "fldChar":
                    # 只在字段结束符上弹栈；begin/separate 的 end 事件不改变状态，
                    # 否则 begin→separate 段的 w:t（非合规文档）会被误放行。
                    field_type = element.get(f"{{{_W_NS_TRANSITIONAL}}}fldCharType") or (
                        element.get(f"{{{_W_NS_STRICT}}}fldCharType")
                    )
                    if field_type == "end" and field_stack:
                        field_stack.pop()
                elif w_local in {"drawing", "pict", "object"}:
                    suppress_depth -= 1
                elif suppress_depth > 0:
                    pass  # 未覆盖区域内部的段落/单元格/表格行不改变外部状态
                elif w_local == "p":
                    numbered = paragraph_numbered
                    paragraph_numbered = False
                    paragraph_text = "".join(paragraph_chars)
                    paragraph_chars = []
                    if tc_depth > 0:
                        cell_paragraph_texts.append(paragraph_text)
                        if numbered:
                            numbered_count += 1
                    else:
                        emit_unit(paragraph_text, numbered=numbered)
                elif w_local == "tc":
                    tc_depth -= 1
                    row_cell_texts.append("\n".join(cell_paragraph_texts))
                    cell_paragraph_texts = []
                elif w_local == "tr":
                    # 表格行降级为文字行：单元格间 TAB，行作为整体记段落级 locator。
                    emit_unit("\t".join(row_cell_texts), numbered=False)
                    row_cell_texts = []
                element.clear()
        except (ParseError, ValueError) as exc:
            raise _corrupt() from exc

        return image_count, numbered_count


__all__ = ["DocxAttachmentParser"]
