"""DOCX 正文提取器与真实 ZIP/XML 依赖的集成测试（计划 15.4 节）。

被测协作边界：``DocxAttachmentParser`` 使用真实 ``zipfile`` 与
``defusedxml`` 处理内存中构造的 DOCX 包样本；覆盖正文顺序、段落级
定位、覆盖范围 warning、拒绝路径与包/XML 资源限制。
"""

import io
import zipfile

import pytest

from hivememory.system.config.attachments import AttachmentParserConfig
from hivememory.system.services.attachments import (
    CONTENT_UNREADABLE,
    RESOURCE_LIMIT,
    AttachmentParseError,
    DocxAttachmentParser,
)

_W_NS_TRANSITIONAL = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W_NS_STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main"
_NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_NS_V = "urn:schemas-microsoft-com:vml"
_NS_MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
_CT_MAIN = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
_CT_MACRO_MAIN = "application/vnd.ms-word.document.macroEnabled.main+xml"
_REL_OFFICE_DOCUMENT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
)
_REL_HEADER = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"


def _make_docx(
    body_xml: str,
    *,
    namespace: str = _W_NS_TRANSITIONAL,
    content_type: str = _CT_MAIN,
    extra_overrides: str = "",
    extra_package_rels: str = "",
    document_rels: str | None = None,
    main_part: str = "word/document.xml",
) -> bytes:
    """构造一个最小但结构完整的 DOCX 包。"""
    content_types = (
        '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/{main_part}" ContentType="{content_type}"/>'
        f"{extra_overrides}</Types>"
    )
    package_rels = (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{_REL_OFFICE_DOCUMENT}" Target="{main_part}"/>'
        f"{extra_package_rels}</Relationships>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr(
            main_part,
            (
                '<?xml version="1.0"?>'
                f'<w:document xmlns:w="{namespace}"'
                f' xmlns:r="{_NS_R}" xmlns:wp="{_NS_WP}" xmlns:v="{_NS_V}" xmlns:mc="{_NS_MC}">'
                f"<w:body>{body_xml}</w:body></w:document>"
            ),
        )
        if document_rels is not None:
            directory, _, filename = main_part.rpartition("/")
            rels_path = (
                f"{directory}/_rels/{filename}.rels" if directory else f"_rels/{filename}.rels"
            )
            archive.writestr(rels_path, document_rels)
    return buffer.getvalue()


def _parse(
    raw: bytes,
    **limit_overrides,
):
    return DocxAttachmentParser().parse(
        raw,
        config=AttachmentParserConfig(**limit_overrides),
        source_raw_revision=1,
        source_raw_hash="raw-hash",
    )


def _para(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


def test_minimal_package_extracts_paragraph_with_locator() -> None:
    """捕获最小 DOCX 的正文与段落级定位漂移。"""
    result = _parse(_make_docx(_para("Hello")))

    assert result.text == "Hello"
    assert result.content_object["locators"] == [
        {"kind": "paragraph", "number": 1, "start": 0, "end": 5},
    ]
    assert result.content_object["warnings"] == []
    assert (result.producer, result.producer_version) == ("docx_extract", "1")
    assert result.content_format == "plain_text"


def test_strict_namespace_document_is_recognized() -> None:
    """捕获只按 local name 匹配导致的 Strict 文档解析失败或误判。"""
    result = _parse(_make_docx(_para("严格"), namespace=_W_NS_STRICT))

    assert result.text == "严格"


def test_run_whitespace_and_breaks_are_preserved() -> None:
    """捕获 run 间空白丢失、TAB/换行/分页符转换不符合 7.4 规则。"""
    body = (
        "<w:p>"
        "<w:r><w:t>multi </w:t></w:r>"
        '<w:r><w:t xml:space="preserve"> spaced</w:t><w:tab/><w:t>after</w:t></w:r>'
        "<w:r><w:br/><w:t>line2</w:t></w:r>"
        '<w:r><w:br w:type="page"/><w:t>page2</w:t></w:r>'
        "</w:p>"
    )
    result = _parse(_make_docx(body))

    assert result.text == "multi  spaced\tafter\nline2\npage2"


def test_paragraph_table_paragraph_interleaves_in_document_order() -> None:
    """捕获先读完全部段落再读表格的顺序缺陷或单元格/行分隔符错误。"""
    body = (
        _para("Para One")
        + "<w:tbl>"
        + "<w:tr><w:tc><w:p><w:r><w:t>A1</w:t></w:r></w:p></w:tc>"
        + "<w:tc><w:p><w:r><w:t>B1</w:t></w:r></w:p></w:tc></w:tr>"
        + "<w:tr><w:tc><w:p><w:r><w:t>A2</w:t></w:r></w:p></w:tc>"
        + "<w:tc><w:p><w:r><w:t>B2</w:t></w:r></w:p></w:tc></w:tr>"
        + "</w:tbl>"
        + _para("Para Two")
    )
    result = _parse(_make_docx(body))

    # 行间 LF、单元格间 TAB；行作为整体记段落级 locator，序号计入空元素。
    assert result.text == "Para One\nA1\tB1\nA2\tB2\nPara Two"
    assert [item["number"] for item in result.content_object["locators"]] == [1, 2, 3, 4]


def test_empty_paragraphs_are_counted_in_locator_numbering() -> None:
    """捕获空段落未计入序号导致定位错位。"""
    body = _para("first") + "<w:p/>" + _para("last")
    result = _parse(_make_docx(body))

    assert result.text == "first\n\nlast"
    assert result.content_object["locators"] == [
        {"kind": "paragraph", "number": 1, "start": 0, "end": 5},
        {"kind": "paragraph", "number": 3, "start": 7, "end": 11},
    ]


def test_hyperlink_display_text_is_kept_without_visiting_target() -> None:
    """捕获超链接目标地址进入正文或显示文字丢失。"""
    body = '<w:p><w:hyperlink r:id="rId5"><w:r><w:t>link text</w:t></w:r></w:hyperlink></w:p>'
    result = _parse(_make_docx(body))

    assert result.text == "link text"


def test_field_results_are_kept_and_instructions_never_emitted() -> None:
    """捕获字段指令进入正文或字段结果丢失。"""
    complex_field = (
        "<w:p>"
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        "<w:r><w:instrText> TOC \\o </w:instrText></w:r>"
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        "<w:r><w:t>cached</w:t></w:r>"
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        "</w:p>"
    )
    simple_field = '<w:p><w:fldSimple w:instr=" PAGE "><w:r><w:t>7</w:t></w:r></w:fldSimple></w:p>'
    result = _parse(_make_docx(complex_field + simple_field))

    assert result.text == "cached\n7"


def test_nonconformant_field_instruction_runs_are_suppressed() -> None:
    """捕获 begin→separate 之间非合规的 w:t 指令文本泄入正文。"""
    body = (
        "<w:p>"
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        "<w:r><w:t>instruction-as-text</w:t></w:r>"
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        "<w:r><w:t>result</w:t></w:r>"
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        "</w:p>"
    )
    result = _parse(_make_docx(body))

    assert result.text == "result"


def test_headings_and_list_items_keep_plain_text_without_markdown_markers() -> None:
    """捕获为标题/列表臆造 Markdown 标记或编号。"""
    body = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>标题一</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        "<w:r><w:t>列表项</w:t></w:r></w:p>"
    )
    result = _parse(_make_docx(body))

    assert result.text == "标题一\n列表项"
    assert result.content_object["warnings"] == [
        {"message_key": "docx_auto_numbering_ignored", "params": {"count": 1}},
    ]


def test_uncovered_content_yields_stable_deduped_warnings() -> None:
    """捕获页眉/脚注/批注/图片的 warning 缺失、重复或顺序不稳定。"""
    document_rels = (
        '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="h1" Type="{_REL_HEADER}" Target="header1.xml"/>'
        '<Relationship Id="f1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/>'
        '<Relationship Id="c1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>'
        "</Relationships>"
    )
    body = (
        _para("带图正文")
        + "<w:p><w:r><w:drawing><wp:inline/></w:drawing></w:r></w:p>"
        + "<w:p><w:r><w:drawing><wp:inline/></w:drawing></w:r></w:p>"
    )
    result = _parse(_make_docx(body, document_rels=document_rels))

    assert result.text == "带图正文\n\n"
    assert result.content_object["warnings"] == [
        {"message_key": "docx_comments_ignored", "params": {}},
        {"message_key": "docx_footnotes_endnotes_ignored", "params": {}},
        {"message_key": "docx_headers_footers_ignored", "params": {}},
        {"message_key": "docx_images_ignored", "params": {"count": 2}},
    ]


def test_image_only_document_fails_as_empty_content() -> None:
    """捕获只包含未覆盖内容的文档伪装成空正文成功。"""
    body = "<w:p><w:r><w:drawing><wp:inline/></w:drawing></w:r></w:p>"

    with pytest.raises(AttachmentParseError) as error:
        _parse(_make_docx(body))
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "empty_content",
    )


def test_textbox_content_inside_picture_is_not_extracted() -> None:
    """捕获 VML 文本框内部文字泄入正文。"""
    body = (
        "<w:p><w:r><w:t>outside</w:t></w:r>"
        "<w:r><w:pict><v:shape><v:textbox><w:txbxContent>"
        "<w:p><w:r><w:t>inside box</w:t></w:r></w:p>"
        "</w:txbxContent></v:textbox></v:shape></w:pict></w:r></w:p>"
    )
    result = _parse(_make_docx(body))

    assert result.text == "outside"


@pytest.mark.parametrize(
    ("body_xml", "structure"),
    [
        ("<w:p><w:ins><w:r><w:t>new</w:t></w:r></w:ins></w:p>", "revision"),
        ("<w:p><w:del><w:r><w:delText>old</w:delText></w:r></w:del></w:p>", "revision"),
        ('<w:p><w:altChunk r:id="rId2"/></w:p>', "alt_chunk"),
        (
            "<w:p><w:sdt><w:sdtPr/><w:sdtContent><w:r><w:t>boxed</w:t></w:r></w:sdtContent></w:sdt></w:p>",
            "content_control",
        ),
        (
            '<w:p><mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">'
            "<mc:Choice Requires='wps'><w:r><w:t>choice</w:t></w:r></mc:Choice></mc:AlternateContent></w:p>",
            "alternate_content",
        ),
        (
            "<w:tbl><w:tr><w:tc><w:tbl><w:tr><w:tc>"
            + _para("nested")
            + "</w:tc></w:tr></w:tbl></w:tc></w:tr></w:tbl>",
            "nested_table",
        ),
    ],
)
def test_unsupported_structures_fail_whole_document(body_xml: str, structure: str) -> None:
    """捕获不支持结构被静默跳过或以截断正文冒充成功。"""
    with pytest.raises(AttachmentParseError) as error:
        _parse(_make_docx(body_xml))
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "unsupported_structure",
    )
    assert error.value.params["structure"] == structure


def test_renamed_non_word_zip_is_rejected_as_not_word_document() -> None:
    """捕获改后缀的普通 ZIP 或文本文件被当作有效 Word 文档。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "not a word document")
    raw = buffer.getvalue()

    with pytest.raises(AttachmentParseError) as error:
        _parse(raw)
    assert error.value.params["reason"] in {"not_word_document", "missing_part"}


def test_corrupt_zip_bytes_fail_as_content_unreadable() -> None:
    """捕获损坏容器被解析成功或错误断言为加密。"""
    with pytest.raises(AttachmentParseError) as error:
        _parse(b"PK\x03\x04 definitely not a zip")
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "corrupt_package",
    )


def test_encrypted_containers_are_detected_without_guessing() -> None:
    """捕获 CFB 魔数与 ZIP 加密标志两条路径都稳定识别加密。"""
    # CFB 魔数（OOXML 加密文档的实际容器形态）。
    with pytest.raises(AttachmentParseError) as cfb_error:
        _parse(b"\xd0\xcf\x11\xe0\xa1\x1b\xae\x1b\x1a\x1e" + b"\x00" * 32)
    assert cfb_error.value.params["reason"] == "encrypted_package"

    # 改写中央目录项的通用加密标志位（general purpose bit 0）；
    # 目录预检必须先于任何成员读取识别加密。
    plain = _make_docx(_para("x"))
    raw = bytearray(plain)
    scan = 0
    patched = 0
    while True:
        scan = raw.find(b"PK\x01\x02", scan)
        if scan < 0:
            break
        raw[scan + 8] |= 0x01
        patched += 1
        scan += 4
    assert patched >= 1  # 前置断言：包内存在中央目录项可供置位

    with pytest.raises(AttachmentParseError) as encrypted_error:
        _parse(bytes(raw))
    assert encrypted_error.value.params["reason"] == "encrypted_package"


def test_macro_document_is_rejected_with_save_as_hint() -> None:
    """捕获宏文档被当作普通 .docx 提取。"""
    macro_extra = '<Override PartName="/word/vbaProject.bin" ContentType="application/vnd.ms-office.vbaProject"/>'
    with pytest.raises(AttachmentParseError) as error:
        _parse(_make_docx(_para("x"), content_type=_CT_MACRO_MAIN, extra_overrides=macro_extra))
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "macro_document",
    )


def test_entity_declaration_is_rejected() -> None:
    """捕获 DTD/实体展开未被 defusedxml 拒绝。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0"?>'
                '<!DOCTYPE Types [<!ENTITY xxe "leak">]>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                f'<Override PartName="/word/document.xml" ContentType="{_CT_MAIN}"/></Types>'
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0"?><Relationships '
                'xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Id="r1" Type="{_REL_OFFICE_DOCUMENT}" '
                'Target="word/document.xml"/></Relationships>'
            ),
        )
        archive.writestr(
            "word/document.xml",
            (
                f'<w:document xmlns:w="{_W_NS_TRANSITIONAL}"><w:body>{_para("x")}</w:body></w:document>'
            ),
        )

    with pytest.raises(AttachmentParseError) as error:
        _parse(buffer.getvalue())
    assert error.value.category == CONTENT_UNREADABLE


@pytest.mark.parametrize(
    ("limits", "expected_reason"),
    [
        ({"max_docx_members": 2}, "member_count_limit"),
        ({"max_docx_member_uncompressed_bytes": 10}, "member_size_limit"),
    ],
)
def test_package_size_limits_are_enforced(limits: dict, expected_reason: str) -> None:
    """捕获成员数与单成员大小的小配额边界未被强制执行。"""
    with pytest.raises(AttachmentParseError) as error:
        _parse(_make_docx(_para("hello world, enough bytes")), **limits)
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        expected_reason,
    )


def test_compression_ratio_anomaly_is_rejected_as_corrupt() -> None:
    """捕获声明压缩比异常（ZIP 炸弹特征）未被目录预检拒绝。"""
    with pytest.raises(AttachmentParseError) as error:
        _parse(
            _make_docx(_para("hello world, enough bytes")),
            max_docx_compression_ratio=1,
        )
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "compression_ratio_limit",
    )


def test_xml_depth_and_node_limits_are_enforced() -> None:
    """捕获深层嵌套 XML 与节点数上限未在遍历检查点生效。"""
    deep = "<w:p>" * 8 + "<w:r><w:t>deep</w:t></w:r>" + "</w:p>" * 8
    with pytest.raises(AttachmentParseError) as depth_error:
        _parse(_make_docx(deep), max_xml_depth=4)
    assert depth_error.value.params["reason"] == "xml_depth_limit"

    many = "".join(_para(f"p{i}") for i in range(10))
    with pytest.raises(AttachmentParseError) as node_error:
        _parse(_make_docx(many), max_xml_nodes=12)
    assert node_error.value.params["reason"] == "xml_node_limit"


def test_parse_budget_exceeded_with_injectable_clock() -> None:
    """捕获 DOCX 解析预算检查点失效或测试依赖真实等待。"""

    class _SteppingClock:
        def __init__(self, step: float) -> None:
            self.step = step
            self.now = 0.0

        def __call__(self) -> float:
            self.now += self.step
            return self.now

    with pytest.raises(AttachmentParseError) as error:
        DocxAttachmentParser().parse(
            _make_docx(_para("budget")),
            config=AttachmentParserConfig(parse_budget_seconds=5.0),
            source_raw_revision=1,
            source_raw_hash="h",
            clock=_SteppingClock(step=10.0),
        )
    assert (error.value.category, error.value.params["reason"]) == (
        RESOURCE_LIMIT,
        "parse_budget_exceeded",
    )


def test_duplicate_member_names_fail_as_corrupt() -> None:
    """捕获重复成员名的包通过目录预检。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            (
                '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                f'<Override PartName="/word/document.xml" ContentType="{_CT_MAIN}"/></Types>'
            ),
        )
        archive.writestr(
            "_rels/.rels",
            (
                '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'<Relationship Id="r1" Type="{_REL_OFFICE_DOCUMENT}" Target="word/document.xml"/></Relationships>'
            ),
        )
        document = f'<w:document xmlns:w="{_W_NS_TRANSITIONAL}"><w:body>{{}}</w:body></w:document>'
        archive.writestr("word/document.xml", document.format(_para("one")))
        # 同名重复成员；zipfile 写入侧会告警，与被测行为无关。
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("word/document.xml", document.format(_para("two")))

    with pytest.raises(AttachmentParseError) as error:
        _parse(buffer.getvalue())
    assert error.value.params["reason"] == "duplicate_member"


def test_member_path_escape_fails_as_corrupt() -> None:
    """捕获 ../ 路径逃逸成员通过寻址校验。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.xml", "<xml/>")

    with pytest.raises(AttachmentParseError) as error:
        _parse(buffer.getvalue())
    assert (error.value.category, error.value.params["reason"]) == (
        CONTENT_UNREADABLE,
        "unsafe_member_path",
    )
