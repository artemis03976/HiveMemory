---
title: Engineering Governance
status: current
owner: project
scope: cross-version-engineering-governance
last_reviewed: 2026-08-13
---

# Engineering Governance

本目录保存跨版本持续生效的工程治理主题。它回答“项目长期必须达到什么质量门槛、当前成熟度在哪里、哪些工作包仍未排期”，不描述某个版本准备如何实现一项功能，也不替代当前架构、子系统设计、公共契约或 ADR。

## 当前治理主题

### Reliability

- [耐久性与故障恢复](./reliability/durability-and-recovery.md)：状态耐久性分级、恢复边界、schema 与 reconciliation 门槛；
- [幂等性与重试](./reliability/idempotency-and-retry.md)：operation identity、重复结果、模糊失败与业务重放边界。

### Security

- [身份隔离与执行安全](./security/identity-and-execution-safety.md)：身份传播、授权所有权、缓存/运行时隔离及不可信执行边界。

### Data Model

- [数据模型可变性治理](./data-model/mutability.md)：模型角色、冻结深度、聚合所有权与跨边界投影规则。

### Testing

- [测试设计规范](./testing/test-design-standards.md)：测试编写的分层、断言、mock、隔离与命名规则，以及必须避免的无效测试反模式。

## 调研基线

[Baselines](./baselines/README.md) 保存已经完成的 point-in-time 清单和风险证据。基线用于说明治理判断建立在什么事实之上，不代表对应治理目标已经实现，也不作为当前系统行为的唯一来源。

## 与其他目录的边界

- `architecture/` 和各子系统目录描述当前已经生效的设计；
- `contracts/` 定义调用方当前可以依赖的跨边界语义；
- `architecture/decisions/` 保存已经接受的重要选择及其理由；
- `plans/` 只保存绑定明确版本或里程碑、可以独立验收的实施切片；
- `ideas/` 保存尚未形成排期承诺的功能和研究方向；
- `todo/` 保存范围较小、排期灵活的缺陷和技术债。

## 治理主题的使用规则

治理文档可以维护成熟度等级、长期目标、未排期工作包和升级门槛，但不得：

- 使用未来时设计冒充当前能力；
- 维护某个版本的逐文件实施步骤；
- 复制当前架构或公共契约的完整描述；
- 仅因一项工作被列入治理主题，就宣称它已经进入 Roadmap 承诺。

当某个治理工作包具备明确版本、依赖闭包、迁移方案和验收出口后，应从治理文档中提取独立 Plan。实现完成后更新当前设计与契约、归档 Plan，再回写治理主题的成熟度状态。
