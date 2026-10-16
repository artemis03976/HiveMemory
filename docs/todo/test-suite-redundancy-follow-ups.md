---
title: 测试套件冗余收敛后续
status: todo
owner: project
scope: test-suite-ownership-and-redundancy-follow-ups
code_paths:
  - tests/
related_docs:
  - docs/governance/testing/test-design-standards.md
last_reviewed: 2026-10-16
---

# 测试套件冗余收敛后续

## 问题与证据

[测试设计规范](../governance/testing/test-design-standards.md)第 4 节要求每条不变量只在归属层拥有权威测试、参数化维度对应不同分支、不写类型层与墓碑测试、测试文件不以计划命名。2026-10-16 按该节完成了第一轮收敛：快速集由 3035 个用例降至 2863 个，收集总数由 3085 个降至 2911 个，参数化展开由 624 个降至 486 个，行与分支覆盖没有新增缺失。第一轮处理了展开最多的访问矩阵、身份入口守卫文件、旧 Profile 权限键的下游测试、必填参数测试、以计划或阶段命名的文件，以及感知与轨迹摘要的平行重复。

以下事项在第一轮中确认存在，但超出当时范围或需要 owner 决定：

1. **感知测试的主类型归类矛盾**：`tests/unit/patchouli/services/test_perception.py` 与 `tests/integration/patchouli/test_perception_flush_chain.py` 组装同一组真实组件（Engine、Store、WorkingSet、Journal，仅 relay 与 bus 为替身），是同一边界分处两个目录。第一轮只删除了平行重复，两个文件的剩余场景仍分属不同主类型。
2. **同类文件对尚未排查**：第一轮只在感知中确认了“unit 与 integration 使用同一组装”的情况，其他子系统是否存在同样的文件对尚未逐一核对。
3. **`integration/mtp` 下以 `_e2e` 命名的文件**：`test_koakuma_update_e2e.py` 与 `test_koakuma_write_e2e.py` 位于集成目录，文件名与主类型矛盾。
4. **测试目录缺少 `__init__.py`**：`tests/unit/prompts`、`tests/unit/core`、`tests/integration/system` 等 9 个含测试文件的目录不是包，其中的测试模块依赖文件名在全部非包目录中唯一；按被测模块命名的通用文件名（如 `test_contracts.py`）在这些目录中会冲突。第一轮只为 `tests/unit/gateway` 补充了 `__init__.py`。
5. **参数化矩阵未逐项复核**：第一轮按展开数量与名称特征筛查，其余展开不少于 3 个的参数化函数尚未逐一核对其维度是否对应不同分支。

原列出的“`MTPTraceReducer` 没有生产调用方”已于 2026-10-16 处理：门面与其测试删除，生产链路一直直接使用 `ActionReducer` 与 `TraceReducer`，门面独有的 dict 输入与孤立结果用例迁入 `ActionReducer` 测试。

[测试设计规范](../governance/testing/test-design-standards.md)第 13 节登记的 Patchouli application 薄转发测试仍是独立遗留项，不在本 Todo 重复。

## 影响范围

只涉及 `tests/` 与引用测试路径的当前文档。

## 完成条件

- 第 1、2 项：同一组装的测试只保留在一个主类型目录，或在文件说明中写明两者边界的实际差异；
- 第 3 项：文件按被测行为改名，并同步引用它们的文档；
- 第 4 项：含测试文件的目录统一为包，或在规范中写明不需要的理由；
- 第 5 项：剩余参数化函数逐一核对，收敛结果对比覆盖率确认没有丢失分支。
