---
title: ADR-0005 Unified Actor Authentication Gateway and Workspace-Owned Authorization
status: accepted
owner: project
scope: actor-authentication-workspace-admission-and-operation-authorization
decided_at: 2026-09-19
applies_to: v0.7.0-access-boundary-baseline
last_reviewed: 2026-09-19
---

# ADR-0005：统一 Actor 认证网关与 Workspace 持有的授权

> 本文记录 v0.7.0 A1 落地的访问边界决策及理由。设计的完整当前事实见 [Workspace 架构](../workspace.md)第 4 节、[错误模型](../../contracts/error-model.md)第 4.4 节与 [System 应用服务](../../system/application-services.md)；实施历史见[归档的 A1 计划](../../archive/plans/v0.7.0-a1-workspace-access-boundary.md)。

## Context

v0.7.0 A1 之前，Workspace 访问由 `LocalTrustedAdmissionService` 承担：`principal_id → operations` 的来源级映射同时承担接入与操作许可，再以"actor user 等于 workspace owner"代替 Workspace 准入。该结构有三个问题：

1. **准入结果与校验责任拆散**：签发与逐次校验分属不同对象，"context 是否仍然有效"没有权威承载者；
2. **来源级授权无法表达 Actor/Workspace 范围**：principal 拿到的 operation 集合对所有 Workspace 生效，无法表达"Actor researcher 在 notes 可检索、在 archive 只读"，也无法拒绝同一来源下的未登记 Actor；
3. **context 绑定单次 operation**：每个操作都要重新申请 context，等价于把认证做成逐次授权，来源认证被反复重放。

实施过程经过两轮收敛（`ebce0f15` 重新实现统一网关与两类登记后，又删除了 grant/有效期锚点/受控工厂/凭据自检协议，把签发有效性收回 guard），最终形成下述决定。

## Decision

1. **一个对外认证入口，两段所有权**。System 持有 Actor 接入登记（来源、adapter、身份解析规则）与唯一认证网关；Workspace 访问基础设施持有准入判定、签发生命周期与行为白名单。网关在内部顺序完成 Principal authentication 与 Workspace authentication（第二项委托注入的 guard），配置所有者不同不制造第二个对外认证入口。
2. **context 是按对象身份的最小准入结果**。`WorkspaceAccessContext` 只公开不可变的 `identity_scope`，不携带 principal、访问记录、白名单或单次 operation；guard 用私有弱引用表记录"本实例实际签发的对象"及到期时间。复制、同值重建、序列化重建或跨实例的同值对象不继承准入。拒绝基于签发事实本身，而不是凭据内嵌的防篡改字段。
3. **operation 目录与白名单分离**。`WorkspaceOperation` 枚举表达"系统有哪些操作、哪个方法需要哪个操作"（代码契约）；某 Actor 在某 Workspace 获准的集合只由访问注册表表达（授权配置）。每次动作由公共 application 调用共享行为检查逐次授权，方法所需 operation 由方法绑定决定；不同 operation 互不隐含，新增 operation 不自动加入白名单。
4. **调用来源不穿越认证边界**。`CallerPrincipal` 只在网关完成接入认证；认证成功后的 context 与 Patchouli 提交 API 均不携带来源，交互与生成链沿用 Patchouli 既有的内部来源记录。
5. **首版为进程内本地配置**：TTL 与单调时钟注入 guard；`close()` 使既有 context 与新认证一并失效；不建设热更新、持久化凭据或管理 API。外部凭据协议归计划 B，生产消费者接线归 A6。

## Consequences

- 授权配置表达力落在 Actor+Workspace 键上：同一 principal 服务多个 Actor、同一 Actor 在不同 Workspace 有不同白名单都是一等能力；未知来源、未登记 Actor、空白名单、未许可 operation 分别有稳定拒绝语义。
- guard 成为准入与行为授权的共同权威：两个运行实例即使复用同一条配置记录也不互认 context，这使"运行实例"成为授权生命周期的天然边界，但也意味着 context 不能跨进程或跨实例传递，调用侧失效后必须重新认证。
- 每次动作的授权检查发生在公共 application 层，内部领域步骤不再重复授权；新增公共方法必须同步绑定 operation 并补行为测试，否则默认拒绝。
- 迁移期保留裸 scope 兼容分支（管理 CRUD、检索、Profile、Topic、附件上传等既有调用方），清单冻结在 `access_consumption.py`，A6 切换生产消费者后删除；兼容窗口关闭前，生产 HTTP 请求实际不经过网关认证。

## Alternatives

- **保留 per-principal operation 授予**（`trusted_principals` 演进）：被拒绝——无法表达 Actor/Workspace 维度的准入与白名单，且来源级配置天然越权放大。
- **grant/有效期锚点/凭据自检协议**（第一轮实现）：被拒绝——grant 把签发凭据复制进 context 形成第二份身份与权限数据，自检协议让凭据"自己证明自己"；改为 guard 按对象身份跟踪签发后，伪造与复用语义更简单且拒绝点唯一。
- **两个对外认证入口**（来源认证服务 + Workspace admission 服务）：被拒绝——调用方需分别访问两个服务才能得到可用 context，失败阶段割裂；收敛后调用侧只依赖一个入口。
- **把行为白名单固化进 context**（签发时快照）：被拒绝——白名单必须跟随有效注册配置逐次生效，固化会造成"进入时许可、检查时已变更"的陈旧授权。

## Status

Accepted（v0.7.0 A1 基线）。生产消费者切换、shutdown 关闭时机与兼容分支退出由 [A6 计划](../../plans/v0.7.0-a6-actor-adapters-and-integration.md)收口；外部凭据协议由计划 B 承接。

## Related documents

- [Workspace 架构](../workspace.md)第 4 节（当前事实）
- [错误模型](../../contracts/error-model.md)第 4.4 节（阶段拒绝语义）
- [子系统公共契约](../../contracts/subsystem-contracts.md)第 3.5 节（访问上下文契约）
- [归档的 A1 计划](../../archive/plans/v0.7.0-a1-workspace-access-boundary.md)（实施历史与约束承接清单）
