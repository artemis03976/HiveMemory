# HiveMemory 技术架构更新文档
## 主题：MTP 回填策略与角色隔离重构 (Role Separation Injection)

**文档状态**: Active (执行中)
**适用阶段**: MTP 稳定性优化
**核心重构模块**: `engines.kernel.PatchouliKernel`
**关联协议**: Memory Tool Protocol (MTP) v1.0

---

### 1. 背景与问题定义 (Background & Problem Statement)

#### 1.1 异常现象
在真实的连续多轮对话测试中，当 Agent 在第一轮成功使用了 MTP 指令（如 `⟪ READ | mem_01 ⟫`）并接收到系统回填的 `<mtp_response>` 后，**在第二轮或后续对话中，Agent 极易产生严重的“幻觉 (Hallucination)”**。
具体表现为：Agent 不再输出 MTP 指令去真实地调用系统，而是自行伪造（捏造）出一段带有 `<mtp_response>...</mtp_response>` 标签的虚假执行结果。

#### 1.2 根本原因：自回归污染 (Autoregressive Contamination)
原有的 MTP 注入策略采用了 **Prompt Prefilling (预填充)** 的方式，即直接将 `<mtp_response>` 字符串追加到了 Agent 当前的 `role: "assistant"` 消息尾部。

由于现代 LLM 属于自回归模型（Autoregressive Models），其本质是“根据上文模仿并预测下一个 Token”。当 LLM 在历史记录中看到 `assistant` 角色输出了 `<mtp_response>` 格式时，**模型在逻辑上产生了“越权错觉”**，误以为这种 XML 标签格式及其内部的数据是由它自己生成的，从而在后续回合中主动模仿这一行为。

---

### 2. 核心架构变更 (Core Architectural Changes)

为了彻底根除自回归污染，同时保留 MTP 协议“不依赖原生 Function Calling”和“维护心流”的轻量化特性，系统将回填策略由 **“同角色尾部续写 (Single-Message Append)”** 升级为 **“角色隔离注入 (Role-Separated Injection)”**。

#### 2.1 核心策略：物理隔离执行结果
在 LLM 因触发 `stop=["⟫"]` 而暂停后，系统不再修改当前 `assistant` 消息的生成内容（除补全闭合符号外）。
MTP 的执行结果将被封装为一条**全新的、独立的消息**，以 **`role: "user"`**（或特定的 `tool`/`system` 角色，为保证全模型兼容性，推荐使用 `user` 承载系统反馈）注入到对话历史中。

---

### 3. 消息流时序对比 (Message Flow Reconstruction)

#### ❌ 废弃的旧时序 (v1.2 之前)
所有内容混杂在同一条 Assistant 消息中，导致边界模糊。
```json[
  {"role": "user", "content": "请检查数据库配置。"},
  {
    "role": "assistant", 
    "content": "好的，我查阅一下配置。\n⟪ READ | mem_config ⟫\n<mtp_response>\nHost: 127.0.0.1\n</mtp_response>\n根据读取到的配置，主机 IP 是 127.0.0.1..."
  }
]
```

#### ✅ 全新的新时序 (v1.3)
动作与反馈被物理阻断，模型能清晰感知到环境（System/User）的介入。
```json[
  {"role": "user", "content": "请检查数据库配置。"},
  {
    "role": "assistant", 
    "content": "好的，我查阅一下配置。\n⟪ READ | mem_config ⟫"
  },
  {
    "role": "user", 
    "content": "[System MTP Execution Result]\n<mtp_response status='success'>\nHost: 127.0.0.1\n</mtp_response>"
  },
  {
    "role": "assistant", 
    "content": "根据读取到的配置，主机 IP 是 127.0.0.1..."
  }
]
```

---

### 4. 代码实现指南 (Implementation Guide)

此变更主要集中在 `PatchouliKernel` 处理 MTP 中断的递归循环逻辑中。

#### 4.1 Kernel 生成循环重构片段
```python
# engines/kernel.py -> _recursive_generation_loop(self, history)

# ... 前置生成逻辑 (直到命中 stop=["⟫"]) ...

if finish_reason == "stop": 
    # 1. 获取 LLM 生成的半截文本 (含 ⟪)
    generated_text = full_content
    
    # 2. 解析出具体的 MTP 指令
    command_str = generated_text.split("⟪")[-1] + "⟫"
    
    # ===[关键变更点 1：闭合 Assistant 消息] ===
    # 将包含 ⟪...⟫ 的完整动作记录在 assistant 名下，并彻底终结该消息
    history.append({
        "role": "assistant", 
        "content": generated_text + "⟫"
    })
    
    # 3. 交由 Koakuma 执行指令 (通过 SystemBus)
    mtp_result_xml = await self.bus.request("koakuma.execute", command=command_str)
    
    # ===[关键变更点 2：作为新的 User 消息注入反馈] ===
    # 明确声明这是系统反馈，防止模型误解
    system_feedback = f"[System MTP Execution Result]\n{mtp_result_xml}"
    history.append({
        "role": "user", 
        "content": system_feedback
    })
    
    # 4. 进入下一层递归，LLM 将基于新的 History (最后一条是 user) 顺滑续写
    current_depth += 1
    continue 
```

---

### 5. 附带收益与下游影响 (Downstream Impacts)

本次重构虽然是为了修复幻觉 Bug，但为整个系统的架构带来了意想不到的红利：

1.  **感知层 (Perception Layer) 的极大减负**：
    *   **旧版**：感知层的 `MTPLogParser` 需要用复杂的正则表达式，从长篇的 Assistant 文本中剔除 `<mtp_response>...</mtp_response>` 噪音。
    *   **新版**：由于 MTP 的执行结果现在全都被隔离在独立的 `role: "user"`（且带有 `[System MTP Execution Result]` 抬头）消息中。感知层在构建 `LogicalBlock` 时，只需简单地**丢弃或单独归档这些特定的 User 消息**，提取出的 `clean_response` 将天然纯净。

2.  **更广的模型兼容性 (Model Compatibility)**：
    *   部分开源小模型（如 Qwen, Llama 家族）对长上下文中的多角色混杂理解力较弱。标准化的 `User -> Assistant -> User -> Assistant` 轮替阵型，完美契合了所有 Instruct 模型的 SFT 预训练数据分布，指令遵循能力将得到显著提升。