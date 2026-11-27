---
title: 短期记忆
status: current
owner: patchouli-workspace-system
scope: short-term-memory-mechanism-topic-progress-lifecycle-folding-and-read-out
code_paths:
  - src/hivememory/patchouli/services/perception.py
  - src/hivememory/patchouli/services/topic_working_set.py
  - src/hivememory/memory_library/stores.py
  - src/hivememory/memory_library/topic_log.py
  - src/hivememory/memory_library/adapters/short_term.py
  - src/hivememory/core/models/topic.py
  - src/hivememory/engines/perception/
  - src/hivememory/workspace/capability/root/topic.py
  - src/hivememory/workspace/capability/actor/memory.py
  - src/hivememory/workspace/resolution/memory.py
  - src/hivememory/engines/memory_compiler/
  - src/hivememory/system/services/import_bus/
  - src/hivememory/mcp_server/server.py
  - integrations/hivememory/connector/
related_contracts:
  - docs/contracts/mcp.md
  - docs/contracts/subsystem-contracts.md
  - docs/contracts/routes-and-events.md
related_docs:
  - docs/patchouli/perception.md
  - docs/patchouli/generation.md
  - docs/workspace/sessions.md
  - docs/alice/agent-runtime.md
  - docs/system/import-bus.md
last_reviewed: 2026-11-27
---

# 短期记忆

本文介绍 HiveMemory 的短期记忆：HiveMemory 怎样把正在进行的工作与对话记成“话题”，怎样记下每个话题进行到哪一步，什么时候把话题整理成可以检索的正式记忆，以及 agent 怎样在需要时把话题读回来。

短期记忆是基于话题机制的一套上下文控制工程，由几部分组合而成：

- **话题路由**：决定一轮交互归入哪个话题；
- **进展记录**：记下话题进行到哪一步；
- **折叠**：材料变长时压缩较早的部分；
- **沉淀与退役**：决定话题什么时候整理进中期记忆、什么时候离开；
- **读回**：让 agent 在需要时拿到话题的进展。

它能对 agent 的上下文干预到什么程度，取决于接入方式（第 1 节）：controller 模式下 agent 由 HiveMemory 自己运行，记忆系统本身就是对 agent 的上下文管理；plugin 模式下上下文属于外部 harness，短期记忆只能提供信息，由 agent 自己来读。

读完之后，你应当能回答这几个问题：一轮对话结束后去了哪里；话题什么时候被整理；agent 回到一件做了一半的事时，能看到什么。

## 1. 要解决的问题

agent 与用户的对话会越来越长。模型的上下文有上限，所以 harness（Claude Code、Codex 这类运行 agent 的程序）会压缩较早的内容；压缩会丢细节，很长的上下文里模型的注意力也会衰减。结果是：一件做到一半的事，过一会儿 agent 就记不清了。这就是短期记忆要解决的问题。

记忆要对 agent 起作用，最直接的方式是把信息放进它的上下文，短期、中期与长期记忆都是如此。HiveMemory 能做到什么程度，取决于接入方式：

- **controller 模式**：agent 由 HiveMemory 自己运行（当前是 Alice），它的上下文由 HiveMemory 构建。会话资源原样保存全部轮次，每次执行从中构建一份预算受限的上下文，太长时由 Alice 压缩；这一轮开始前检索到的中期记忆也直接放进上下文。这时记忆系统本身就是对 agent 的上下文管理。
- **plugin 模式**：HiveMemory 以插件接入外部 harness，对话的上下文属于 harness，由 harness 自己压缩。HiveMemory 是被动的一方，不改写、也不替换这份上下文，只能提供信息，让 agent 自己来读。

两种模式共有的，是每一轮对话结束后都会交给 HiveMemory。HiveMemory 把这些轮次按内容归进**话题**，保管完整的原文，记下进展，并在 agent 回来时让它读到。目前话题只以 agent 读取的方式回到前台，controller 模式下话题的进展还没有进入 Alice 的上下文（第 9 节）。

为什么按内容而不是按会话组织：会话按时间顺序排列，而且会被压缩；话题按内容组织，原文完整。“先讨论 A，做了十几轮 B，再回到 A”时，回到 A 的 agent 可以从 A 的话题里拿回前台已经丢掉或混在一起的东西。一个话题可以是 agent 的一项工作，也可以是一段讨论或闲聊。

## 2. 一个例子

以 Claude Code 为例，看一个话题从出现到被读回的全过程。下面的时间取自随附配置。

1. 用户在 Claude Code 里说“帮我给登录页加上‘记住我’”。这一轮结束后，HiveMemory 插件把这一轮交给 HiveMemory：用户的消息、agent 的思考、每次工具调用与结果、最后的回复。
2. HiveMemory 用一个模型判断这一轮属于哪个话题。话题池里没有相关的话题，于是新建话题「登录页的记住我」，把这一轮放进去。接下来几轮也被判到这个话题。
3. 用户去吃饭，这个话题 30 分钟没有新的交互。HiveMemory 读这几轮的原文，写下一条**进展记录**，例如“已在登录表单加上复选框，后端 token 的有效期还没改”。这一条指回它覆盖的那几轮。
4. 话题一共停了 1 小时。HiveMemory 把这个话题的全部交互交给记忆生成，整理成可以检索的正式记忆，例如一条决定“记住我的有效期定为 30 天”。这一步叫**沉淀**。沉淀之后话题仍留在话题池里。
5. 用户回来接着做。会话越来越长，Claude Code 压缩了会话。压缩之后，插件在会话开头放进一句**指针**：“HiveMemory：可以读取 topic:… 查看这个会话所在的话题进行到哪一步。”
6. agent 读这个句柄，看到这个话题的进展记录，以及还没有整理的最近几轮的节选；需要细节时再读其中一条，得到这一条的全文与它覆盖的原文。
7. 第二天，用户开了一个新会话，说“继续昨天那个记住我的功能”。agent 读 `topic:current`。HiveMemory 判断这一轮属于「登录页的记住我」，返回这个话题的进展。
8. 话题池是有容量的（每个 Workspace 5 个）。池满时要建新话题，最久没用的话题先沉淀还没沉淀的交互，再离开话题池。这一步叫**退役**。

## 3. 基本概念

- **一轮交互**：用户的一条消息，加上 agent 为回应它做的一切：思考、工具调用与结果、子 agent 交回的消息，直到这一轮的回复。HiveMemory 按发生的先后保存一轮的全部内容。
- **话题**：关于同一件事的若干轮交互，可以来自多个会话。每一轮进入哪个话题由 HiveMemory 判断（第 5 节）。话题只是短期的：它最终会整理进中期记忆，然后离开。
- **中期记忆**：HiveMemory 的正式记忆库，agent 通过检索与读取使用它。
- **话题池**：当前保存着的话题，每个 Workspace 最多 5 个。
- **话题的进展**：回答“话题进行到哪一步”。它由两部分组成：已经整理好的进展记录，以及进展记录还没有覆盖到的原始交互。从来没写过进展记录的话题，进展就是它的原文。
- **进展记录**：话题停下一段时间后写下的条目。每一条写这段时间里话题有了什么进展，并指回它覆盖的那几轮原文。只往后加，旧的条目不改。
- **话题停下多久**：话题有多久没有新的交互。写进展记录与沉淀都看它，只是等待的时间不同。收到一轮交互、为写入或读当前话题确定这一轮的话题、HiveMemory 重启读回，都会重新计时；读话题、写进展记录与沉淀不会。
- **交互记录的推进进度**：话题里的交互只按顺序追加。折叠、写进展记录与沉淀各自记下自己处理到了第几轮，彼此独立推进、互不等待；一段交互已经整理到哪一步，就看这几个进度（第 10.2 节）。
- **折叠**：话题的原文太长时，把较早的轮次压成摘要，供系统内部读材料的模型读取：记忆生成，以及写进展记录。它和前台的会话压缩是同一类事，只是在系统内部进行。原文不删。
- **沉淀**：把话题的交互交给记忆生成，整理成中期记忆。话题留在池中，以后还可以再沉淀。
- **退役**：话题离开话题池。退役前先把还没沉淀的交互沉淀掉。
- **删除**：用户在话题池里丢弃一个话题：不沉淀，也不生成记忆。
- **句柄**：读话题用的标识。`topic:<话题 id>` 读进展记录的索引，`topic:<话题 id>#<n>` 读第 n 条进展记录的全文与原文，`topic:current` 读这一轮所在的话题。
- **指针**：会话压缩之后，插件放进会话的一句话，告诉 agent 读哪个句柄。

## 4. 话题的一生

### 4.1 建立

话题在第一次需要时才建立，不预先建立。需要它的时刻有两种：

- 这一轮要写入记忆。写入需要一个话题来提供背景材料：记忆生成会读这个话题的内容。plugin 模式下，agent 读 `topic:current` 时也要先有这一轮的话题。
- 这一轮交给话题时，目标是一个新话题。

建立新话题时如果池已满，先让最久没用的话题退役（第 4.6 节）。

有时话题建立了，却一轮交互也没收到：例如为写入建立之后，这一轮被用户停止了，或者外部 harness 的这一轮没有交回来。这样的空话题不会作为候选话题出现；它停下超过沉淀的等待时间，或者池满时，直接退役，不生成记忆。系统不另外清理它，也不撤销已经发生的写入。

### 4.2 接收交互

每一轮结束后，这一轮被封装好交给 Patchouli（HiveMemory 中负责记忆的部分），追加到目标话题的末尾：

- controller 模式下由任务进程提交，它在 Alice 执行完这一轮之后提交；
- plugin 模式下由 Import Bus 提交，它是外部 harness 导入轮次的入口。

两条路径交来的内容形式相同，并共用同一个提交入口：

- **同一轮不会写两次**：提交按这一轮的交互标识去重。追加是提交的最后一步，所以一轮要么没有写进话题，要么已经写进并且成功。
- **话题在中途消失**：目标话题如果恰好在提交之前退役或被删除，这一轮的提交会失败，提交方重新判断话题后再交。

### 4.3 写进展记录

**什么时候写**：话题停下 30 分钟后写一条。之所以不每轮都写，是因为交互连续进行时，前台自己的上下文就够用；真正需要话题进展的，是停下之后回来的人，或者另一个会话。

**怎么写**：模型看两样东西：已有的进展记录，以及还没整理的那部分交互，然后写出一段总结，作为新的一条追加到末尾。还没整理的交互如果开头已经被折叠，被折叠的那几轮只给覆盖它们的折叠摘要，之后给原文。这样写进展记录和记忆生成读的是同一个工作视图，输入长度都由折叠控制（第 4.4 节）。

- 看旧条目是为了让新的一条接得上前文，只作参考，不改写它们。只看新原文时没有上下文，常会写出“请修改后提交”这种离开前文就读不懂的短句。
- 不每次从头重写整份进展记录：那样只能得到一句不断翻新、越来越笼统的概括，前后两次也对不上，成本也更高。

**写不成时**：模型出错、超时或输出不合格，这次就不写，也不用规则凑一条。进展记录只增不改，写坏的一条无法改回。同一段内容写失败后，要等话题有新的交互才再试，免得模型服务故障时反复重试。没有配置模型时不写进展记录。

写下一条不会让话题重新计时，写完后没有新交互就不会再写。进展记录只给前台读，不作为记忆生成的材料。

### 4.4 折叠

话题的原文一直在增长，而系统内部读这份材料的模型——记忆生成，以及写进展记录——上下文都有上限。折叠是控制这份材料长度的唯一手段：它和前台 agent 的会话压缩是同一类事，区别只在它是系统内部的处理。这些模型读的都是话题的**工作视图**：已折叠部分的摘要，加上折叠之后的原文。

- **什么时候折叠**：工作视图超过 token 阈值（随附配置为 131072）时，把除最近 2 轮之外的轮次折叠掉。阈值要小于这些模型的上下文窗口；它们都用 `patchouli.llm` 的同一个模型，所以只需要和一个窗口比。
- **每次只写这一次的摘要**：一次折叠只为这一次被折叠的轮次写摘要，追加在此前的摘要之后，此前的摘要不会被改写。工作视图里的折叠摘要，就是各次折叠的摘要按顺序连起来。道理和进展记录相同：反复整份重写，细节会一点点漂移。
- **一次的摘要写什么**：
  - 一行标题：覆盖第几轮到第几轮；
  - 模型写的五栏：已定的决定及理由；被否掉的方案、失败的尝试与纠正；确认的事实与结论；用户的偏好与要求；涉及的关键文件及其作用。空的栏不写；
  - 代码从原文直接取出的部分，不经过模型：每一轮的用户消息（截到 200 字）、出现过的文件路径与记忆 alias、失败的工具调用（工具名加结果的第一行）、用到的工具。

  摘要不写“做到哪了、接下来做什么”，那是进展记录的事。用户的原话与文件路径交给代码，是因为它们不需要判断，又最容易在模型的概括里丢掉。
- **写不成时**：模型不可用、出错、超时或输出不合格时，这一次的摘要只有标题和代码取出的部分，并注明缺了模型写的部分；折叠照常完成，否则工作视图会一直涨。此前的摘要不受影响。
- **原文不删**：被折叠的轮次仍留在话题里，沉淀时全部原文照样存进记忆的来源记录。只是读材料的模型在较早的部分只看得到摘要。
- **只为材料服务**：折叠回答的是“读材料的模型需要保留什么”，与话题进行到哪一步无关。它不产出话题的进展，也不触发沉淀。折叠摘要不给前台，也不给 Gateway：话题池、话题选择器与话题路由看的是进展记录。
- **不拖慢交互**：折叠在一轮写进话题之后单独排队进行，这一轮的提交不等它；同一话题的下一轮要等折叠结束。

### 4.5 沉淀

**什么时候沉淀**：话题停下 1 小时，并且上次沉淀之后有新的交互时，HiveMemory 把话题的全部交互（不只是新增的部分）连同折叠摘要交给记忆生成。

- 话题沉淀之后留在池里，计时不变，所以同一段停顿不会反复沉淀；有了新交互重新计时，以后再沉淀。
- 停下只沉淀、不退役，是因为用户常常断断续续地回到同一件事。停下就让话题离开，会把一件连续的事切成几个话题。

**记忆怎样生成**有两种方式，由用户选择：

- **一次性生成**：一次模型调用提取出多条记忆，再靠查重去掉重复；
- **馆长**：一个会使用工具的 agent，先看这个话题之前生成过的记忆，再检索库中相关的记忆，只写两者都没有覆盖的内容。它更准，但更慢、开销更大。

**多次沉淀**：同一个话题会沉淀多次。每次生成的记忆都记在话题上，叫做**本话题生成的记忆**，下一次沉淀能看到它们，从而避免重复。

**完成之后才记进度**：一次沉淀完成后，才记下“沉淀到了第几轮”。如果在交给记忆生成时就记，进程一旦崩溃，话题会以为已经沉淀过，生成任务却已经丢了，这段交互就再也不会被沉淀。代价是：记忆已经写下、还没来得及记进度时进程崩溃，重启后会再沉淀一次，重复的内容交给查重处理。生成失败或被用户取消，也算这次沉淀完成，不会自动重试。

**入口不筛选**：话题收到的每一轮都进入沉淀。一轮有没有长期价值，由记忆生成在完整材料上判断。入口只看得到用户的消息，而“继续”“请提交”这类短消息背后常常是一整轮实现。

### 4.6 退役与删除

**退役**发生在两种情况下，退役前都会先沉淀还没沉淀的交互：

- **池满**：要建新话题时，最久没用、且当前没有被占用的话题让出位置；
- **手动**：用户在话题池里对话题执行“沉淀并退役”。

退役之后，话题从话题池里消失，agent 也读不到它。它的日志会保留到最后一次沉淀完成：这样最后这次沉淀仍能看到之前生成的记忆，完成后日志才删除。

**删除**是话题池里的“丢弃”：话题和日志立刻删除，不沉淀。已经交给记忆生成的沉淀照常完成。

HiveMemory 关闭时，话题不会因此退役。

## 5. 一轮交互属于哪个话题

**由谁判断**：每一轮只属于一个话题。HiveMemory 用 Gateway（HiveMemory 分析输入的部分）的话题路由来判断：一次模型调用，从当前的话题池里选一个，或者决定新建。

**什么时候判断**：在第一次需要时判断，每轮最多一次。

- **controller 模式**：Gateway 在这一轮开始时就判断，因为它本来就要先分析这一轮的输入。指向已有话题时，在执行之前确认这个话题还在；指向新话题时，第一次写入时建立，没有写入就在这一轮交给话题时建立。
- **plugin 模式**：HiveMemory 要等这一轮结束才收到它，但 agent 可能在一轮中间就写入记忆、更新记忆或读 `topic:current`。所以在这些操作与这一轮结束后的导入中，哪个先到，就由哪个触发判断，之后都沿用这个结果。HiveMemory 不在每一轮开始时单独判断话题，因为插件不一定能在每一轮开头介入。

**plugin 模式下怎样沿用**：agent 调用写入、更新，或带 `topic:current` 的读取之前，插件会把这一轮的会话标识、轮次标识、用户消息和之前几轮的文本补进调用参数。HiveMemory 由此推出这一轮的标识，记下“这一轮属于哪个话题”；这一轮导入时直接使用它，不再判断。这份记录按轮保存，而不是按会话：上一轮的导入可能晚于下一轮的写入才到达。

**几种特殊情况**：

- **已经写进话题**：这一轮已经写进话题之后，才有一次写入到达，例如后台运行的子 agent 晚到。这时直接使用这一轮实际所在的话题。
- **正在写进话题**：这一轮正在写进话题时，让调用方稍后重试（503 `apply_pending`），免得重新判断后落到别的话题。
- **判断失败**：例如模型不可用。这次写入或读取直接失败（`gateway_unavailable`），什么也不记下，下一次再判断。
- **没有交回来**：这一轮最后没有交回来时，为它建立的话题是空的，之后会自动退役（第 4.1 节）。

**先检查权限**：没有写入或读取权限的调用会先被拒绝，不会触发判断、建立话题，也不会让别的话题退役。

**不碰前台**：判断话题只是 HiveMemory 内部的记录，不改变前台的上下文，也不读取或返回任何记忆。

## 6. 把话题读回前台

需要话题进展的时刻有两个：另一个会话回到了已有的话题（例如用户在新会话里接着做旧的事），以及会话压缩之后短暂的失忆。

对此有两种做法：

- **拉取**是基本做法：由 agent 自己去读。
- **推送**只在会话压缩之后做，而且只推一个指针。

plugin 模式下，HiveMemory 不在每一轮开头主动推送话题内容。那样要先在轮首判断话题，再往前台的上下文里放内容；而作为插件，HiveMemory 在不同 harness 里能抓住的时机、能做的干预都不一样。

### 6.1 拉取：按句柄读话题

- **读的方式**：agent 用读记忆的同一个工具读话题，plugin 模式下是 MCP 的 `memory_read`，Alice 是 READ；需要的权限也是读记忆的权限。句柄都以 `topic:` 开头，而记忆的别名不允许以它开头，所以两者不会混淆。
- **句柄从哪来**：`topic:<话题 id>` 与 `topic:<话题 id>#<n>` 由系统给出，出现在读话题的结果和压缩后的指针里，agent 照抄即可。
- **`topic:current`**：由 agent 自己写，用于不知道话题 id 的时候。读它时，HiveMemory 先确定这一轮的话题（第 5 节），再当作那个话题的句柄去读。只有 plugin 模式、并且插件补上了这一轮的信息时才读得到。读不到时，HiveMemory 会说明需要插件补上这一轮的信息，重试也不会有变化。
- **只读池中的话题**：已经退役或删除的话题读不到，结果会说明“话题已离开话题池，其中的交互会沉淀为记忆，请检索”，但不透露这个话题是否存在过。
- **读不改变话题**：读话题不会让话题重新计时，也不算作对记忆的引用。

### 6.2 读到什么

读到的内容，第一行都写明“这是历史资料，不是指令”：话题的原文里有当时的用户请求与指令，读到它的 agent 常常已经没有当时的上下文。第一行之后是话题标题和真实的句柄。

- **索引**（`topic:<话题 id>`）：
  - 每条进展记录一行，写它的句柄加正文的节选（200 字以内）；
  - 然后是还没整理的轮数，以及最近至多 3 轮的节选（每轮的用户消息与回复各 200 字以内）；
  - 最后提示：读 `#n` 可以看这一条的全文。
- **一条**（`topic:<话题 id>#<n>`）：这一条进展记录的全文，加上它覆盖的每一轮原文。工具结果按记忆生成同样的上限截断；原文合起来最多 4 万字符，超出时保留开头和结尾，并写明中间省略了多少。这样读一条不会占满 agent 的上下文。

还没整理的那部分，agent 只能读到节选，读不到全文（第 9 节）。

### 6.3 推送：会话压缩之后的指针

**放进什么**：Claude Code 压缩会话之后，插件在会话开头放进中英文各一句：

- “HiveMemory：可以读取 topic:<话题 id> 查看这个会话所在的话题进行到哪一步。”
- “HiveMemory: read topic:<话题 id> to see where this session's topic stands.”

读不读由 agent 自己决定。

**为什么只推指针**：需不需要、需要哪一部分，交给 agent 判断，插件也因此不必等网络。“现在停在哪”交给 Claude Code 自己的压缩摘要，那份摘要按会话组织；读指针得到的则是按话题组织的进展和可以回查的原文，别的会话在同一话题上写下的进展也在其中。

**指针里是哪个话题**：插件记下的、这个会话最近一轮所在的话题。如果压缩发生在一轮中间，这一轮还没有导入，指向的就是上一轮所在的话题。

**不放或照放的情况**：

- 这个会话还没有任何一轮导入成功时，不放指针，只在插件日志里记一行；
- 话题已经离开话题池时，照样放指针，agent 读到的是“请检索”的说明。

**不影响 Claude Code**：插件只读本地记下的状态，不联网，任何失败都只记进日志。

**其他情况**：Codex、pi 与 DeepSeek Harness 不推送。controller 模式下，Alice 的上下文里没有话题，也不能读当前话题。

## 7. 保存与重启

**每个话题一个日志文件**：日志只追加（默认在工作目录的 `.hivememory/topics/<话题 id>.jsonl`）。话题上发生的每件事，发生时就写一行：建立、收到一轮、一次折叠、一次沉淀完成、生成的记忆、一条进展记录、退役。

之所以不整体保存快照，是因为话题本来就只增长：快照每次都要重写整个话题，而且只能在正常关闭时写出，崩溃就全部丢失。

**崩溃与断电**：进程崩溃不会丢失已经写下的行；断电可能丢掉最后几行，因为写入时不强制刷盘。

**关闭时**：

1. 先停掉话题的维护：进行中的折叠和进展记录被取消，不留下写了一半的结果；
2. 再把已经收到的交互写完；
3. 话题不在关闭时沉淀或退役；
4. 记忆生成队列中，已经开始写记忆的沉淀会写完，还没开始的取消；写入、更新与合并任务保存下来，下次启动后继续。

**重启时**，HiveMemory 在接受新的交互之前读回全部日志，话题池恢复成关闭前的样子：

- 话题停下多久从重启时重新计时；
- 上次没有完成的沉淀，之后的沉淀扫描会重新做；
- 已经退役、但最后一次沉淀没完成的话题，启动后重新发起这次沉淀；
- 日志最后一行如果只写了一半，丢掉这半行；中间有坏行的日志整个改名放到一边，这个话题不读回。
- 旧版本每次折叠存的是累积的整份摘要；读回这样的记录时，它之前各条折叠记录的摘要不再计入，连起来的结果与原来相同，日志本身不改写。

**话题的 id**：全局唯一，所属的 Workspace 写在日志第一行。

## 8. 必须守住的规则

- **会话与话题互不控制**：会话只记下每一轮落在哪个话题；话题不读会话。压缩会话不影响话题里的原文，折叠也不替前台缩短会话。删除或沉淀话题不会删除会话。plugin 模式下 HiveMemory 不保存外部会话，插件只记下最近一轮所在的话题。
- **三件事各管各的**：三者之间没有联系。前台压缩了某段内容，不说明话题已经收到它，更不说明已经生成了记忆。
  - 会话压缩：由前台的上下文预算触发，回答“下一次推理需要什么”；
  - 折叠：由话题内容超过阈值触发，回答“读材料的模型需要保留什么”；
  - 进展记录：由话题停下触发，回答“进行到哪一步”。
- **Alice 的压缩不进话题**：Alice 自己的压缩摘要不进入话题，也不进入记忆生成。
- **plugin 模式下，话题到前台只有读**：推送只有指针；内容只经 agent 自己的读取进入前台，读取要经过认证与读取权限，而且不改变话题。读 `topic:current` 可能会为这一轮确定话题，这和写入时一样，只是 HiveMemory 内部的归属记录，也会让话题重新计时。
- **原文、视图与记忆是三件独立的事**：较早的轮次被折叠，不等于原文删了，也不等于已经生成了记忆。agent 主动写入的记忆已经生成了，相关的交互仍然留在话题里。
- **维护不改变结果**：话题的维护不改变权限、写入接纳或持久化的结果；运行时事件只用于观测。

## 9. 已知限制

- **未整理的部分只有节选**：索引里还没整理的交互，只给出最近至多 3 轮、每轮 200 字的节选，没有办法读到全文。话题停下不到 30 分钟时还没有任何进展记录，被带回这个话题的 agent 只读得到这几行节选。
- **进展记录只在话题停下时写**：一轮之内不写，会话切到别的话题时也不写；每一条覆盖完整的几轮。旧条目全部保留，并全部作为写下一条的输入，长话题的输入会越来越长。
- **话题只增长**：退役之前保存全部原文，内存、日志与沉淀材料都随轮数增长；重启时要重放全部日志；每次沉淀都保存一份覆盖全部交互的来源记录，多次沉淀的记录互相重叠。
- **写日志是同步的**：会占用事件循环，长的一轮可能有几 MB。
- **崩溃时丢失的东西**：记忆生成任务与写入登记只在正常关闭时保存，崩溃时会丢失；读回的话题靠重新沉淀补上，重新沉淀可能复述已经写下的记忆，靠查重处理。沉淀的记忆生成期间，话题池仍显示“有未沉淀的交互”。附件在关闭时清理，重启后提升附件按尽力而为处理，取不到就跳过。
- **降级**：降级到不认识话题日志的旧版本时，日志既不读也不删；再次升级会读回降级之前的日志，降级期间沉淀过的交互会再沉淀一次。
- **折叠的局限**：
  - 折叠只按轮数保留最近几轮，不保证保留部分低于阈值；单个很大的一轮会让工作视图一直超过阈值，之后每来一轮就折一次；
  - 一次折叠的输入太大时，模型更可能超时或超出窗口，这一次只剩代码取出的部分；
  - 被折叠的轮次在记忆生成与写进展记录里只剩摘要：话题长时间连续工作、中间没有停下时，进展记录对被折叠的那几轮只能依据摘要来写；
  - 折叠摘要只增长：每次折叠追加一份，不再压缩；代码取出的部分也没有总量上限。
- **维护占用提交队列**：折叠与写进展记录各占用交互提交队列的一个执行槽（共 4 个），一次最多 60 秒；其间同一话题新到的交互要等待。
- **并发建话题**：同时建两个新话题时，容量检查有一个短暂的竞态窗口。
- **未覆盖的模式**：controller 模式下话题的进展还没有进入 Alice 的上下文，Alice 也不能读当前话题；Codex、pi 与 DeepSeek Harness 压缩之后不推送。

调整话题的生命周期时，要同时检查记忆生成、来源记录、Import Bus、关闭与重启：“什么时候丢弃交互记录”是数据耐久性的边界，不只是一个摘要算法的参数。

## 10. 参考

本节面向维护代码的读者，列出上文概念在代码中的名字、话题上记录的位置、配置与观测。

### 10.1 代码里的名字

| 概念 | 代码 |
|:---|:---|
| 话题（读取值） | `TopicData`；一条进展记录是 `TopicProgressEntry`，一次折叠是 `TopicFold`（`summary` 只存这一次的摘要，`TopicData.fold_summary` 按顺序拼接）；话题池界面读的快照是 `TopicSnapshot` |
| 一轮交互在话题里的形式 | `LogicalBlock`（这一轮的事件集加 block id、时间与 token 估算） |
| 编排话题的一生 | `PerceptionFamiliar`（`patchouli/services/perception.py`），模块说明里有逐个触发的处理与占用规则 |
| 话题池、最久没用的顺序与占用 | `TopicWorkingSet`；每个驻留话题一条 `ResidentTopic`；同一话题的占用叫 lease |
| 日志与存储 | `ShortTermMemoryStore`（每件事一个写入方法），套用记录的函数在 `memory_library/topic_log.py`，文件由 `JsonlShortTermStorage` 读写 |
| 写进展记录、写折叠摘要 | `ProgressRecordWriter`；`LLMFoldSummaryWriter`（模型以 JSON 写五栏：`decisions`、`rejected`、`facts`、`preferences`、`files`）与 `RuleFoldSummaryWriter`（只有标题与代码取出的部分）；两者共用 `material.render_numbered_turn` 渲染轮次 |
| 旧版本写下的折叠记录 | 日志 `fold` 记录的 `covers_earlier_folds`：缺省为真，表示旧版本写下的累积摘要，读回时清空它之前各条折叠记录的摘要，拼接结果与旧的工作视图相同；新写的记录为假 |
| 沉淀交给记忆生成的材料 | `TopicSettleMaterial` |
| 为一轮确定话题 | root API 的 `RootTopicService.determine_topic`；plugin 模式经 `ImportBusService.place_turn` |
| 读话题句柄、换掉 `topic:current` | `ActorMemoryService.resolve_references`；句柄的格式化与解析是 `format_topic_handle` / `parse_topic_handle` |
| 读到的文字 | `get_topic_read_text`（中英文各一份） |

### 10.2 话题上记的位置

每个位置都是“处理到第几轮”，彼此独立推进，都不切分材料：沉淀总是交出全部交互，进展记录只决定下一条从哪一轮写起。

| 位置 | 含义 | 保存在 |
|:---|:---|:---|
| `interaction_count` | 收到了几轮 | 日志 |
| `folded_until` | 折叠到第几轮，工作视图从这里开始 | 日志 |
| `recorded_until` | 进展记录写到第几轮 | 日志 |
| `settled_until` | 已经完成的沉淀覆盖到第几轮 | 日志 |
| `settling_until` | 已交给记忆生成、还没完成的沉淀覆盖到第几轮 | 只在进程内 |
| `progress_failed_until` | 上一次写进展记录失败时要覆盖到第几轮 | 只在进程内 |

### 10.3 配置

| 键 | 缺省 | 含义 |
|:---|:---|:---|
| `patchouli.perception.progress_record_after_seconds` | 1800 | 话题停下多久后写一条进展记录 |
| `patchouli.perception.settle_after_seconds` | 3600 | 话题停下多久后沉淀 |
| `patchouli.perception.max_resident_topics` | 5 | 每个 Workspace 的话题池容量 |
| `patchouli.perception.topic_dir` | `.hivememory/topics` | 话题日志的目录 |
| `patchouli.perception.engine.fold_token_threshold` | 32768（随附配置 131072） | 工作视图超过它时折叠；要小于 `patchouli.llm` 模型的窗口，启动时不校验 |
| `patchouli.perception.engine.fold_retain_recent_blocks` | 2 | 折叠时保留的最近轮数，至少 1 |
| `patchouli.perception.engine.summary_writer.type` | `simple`（随附配置 `llm`） | `llm` 用 `patchouli.llm` 的模型写折叠摘要与进展记录；`simple` 的折叠摘要只有标题与代码取出的部分，不写进展记录 |
| `patchouli.perception.engine.summary_writer.summary_timeout_seconds` | 60 | 每次写摘要的时限 |
| `patchouli.generation.settle_mode` | `one_shot` | 沉淀时的记忆生成方式：`one_shot` 一次性生成，`librarian` 馆长 |
| `scheduler.tasks.perception_settle_interval_seconds`、`perception_progress_record_interval_seconds` | 30 | 两个扫描的间隔，各有 `enable_*` 开关 |

### 10.4 观测

- **运行时事件**：
  - 新建话题发 `topic.created`；
  - 每次沉淀或退役发 `topic.settled`，在交给记忆生成时发出，不等沉淀完成；
  - 删除话题发 `topic.deleted`；
  - 折叠与写进展记录不发话题事件，出现在工作队列的 `work.*` 事件中。
- **启动读回**：一次 `patchouli.restore_topic_pool` 操作，报告读回的话题数、待重新沉淀的退役话题数与放到一边的日志数；有日志放到一边时级别为 error。
- **日志原因码**：`progress_record_failed`、`topic_fold_failed`、`fold_summary_no_model`、`fold_summary_model_failed`、`topic_log_partial_line`、`topic_log_unreadable`。
- **压缩后的指针**：没有放指针时，插件日志写 `topic pointer not injected`。

### 10.5 代码与测试入口

- 编排：[`patchouli/services/perception.py`](../../src/hivememory/patchouli/services/perception.py)、[`patchouli/services/topic_working_set.py`](../../src/hivememory/patchouli/services/topic_working_set.py)
- 模型与句柄：[`core/models/topic.py`](../../src/hivememory/core/models/topic.py)
- 存储：[`memory_library/stores.py`](../../src/hivememory/memory_library/stores.py)、[`memory_library/topic_log.py`](../../src/hivememory/memory_library/topic_log.py)、[`memory_library/adapters/short_term.py`](../../src/hivememory/memory_library/adapters/short_term.py)
- 写作者：[`engines/perception/`](../../src/hivememory/engines/perception/)
- 确定话题：[`workspace/capability/root/topic.py`](../../src/hivememory/workspace/capability/root/topic.py)、[`system/services/import_bus/`](../../src/hivememory/system/services/import_bus/)
- 读话题：[`workspace/capability/actor/memory.py`](../../src/hivememory/workspace/capability/actor/memory.py)、[`workspace/resolution/memory.py`](../../src/hivememory/workspace/resolution/memory.py)
- 压缩之后的指针：[`integrations/hivememory/connector/claude_code/hooks.py`](../../integrations/hivememory/connector/claude_code/hooks.py)
- 测试：
  - 感知与话题：[感知集成测试](../../tests/integration/patchouli/test_perception.py)、[话题访问链](../../tests/integration/patchouli/test_topic_access_chain.py)、[话题模型](../../tests/unit/core/models/test_topic.py)
  - 存储：[短期存储](../../tests/unit/memory_library/test_short_term_store.py)、[话题日志](../../tests/unit/memory_library/test_topic_log.py)、[WorkingSet](../../tests/unit/patchouli/services/test_topic_working_set.py)
  - 写作者：[进展记录写作者](../../tests/unit/engines/perception/test_progress_record.py)、[折叠摘要](../../tests/unit/engines/perception/test_fold_summary.py)
  - 确定话题与读取：[确定话题](../../tests/unit/workspace/capability/root/test_topic.py)、[actor 读取](../../tests/unit/workspace/capability/actor/test_memory.py)、[MCP 记忆访问](../../tests/integration/system/test_mcp_memory_access.py)
  - 插件：[Claude Code hooks](../../tests/unit/connectors/test_claude_code_hooks.py)
