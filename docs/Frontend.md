# HiveMemory UI/UX 设计文档 (v1.0)
## 主题：面向 AIOS 的可视化仪表盘 (The AIOS Visual Dashboard)

### 1. 设计理念与目标 (Design Philosophy)

HiveMemory 系统在后端已经突破了传统“聊天机器人”的框架，演进为一套具备完整生命周期、段页式内存管理（MMU）和专属通信协议（MTP）的 AI 操作系统内核。因此，其前端界面绝不能仅仅是一个简单的对话框。

本阶段前端设计的核心挑战在于：**如何在一个界面中，既呈现出强大、极客的系统底座，又保持用户日常对话的流畅与专注？** 为此，我们确立了以下三大设计目标：

#### 1.1 核心目标：构建“透明、沉浸、可观测”的 AIOS 仪表盘
HiveMemory 的前端不仅是给普通用户使用的“聊天窗口（Chat Interface）”，更是给系统管理员/开发者使用的“控制台（Dashboard）”。
*   **透明性 (Transparency)**：系统在后台做了什么（比如偷偷把话题切到了上一个项目，或者正在检索数据库），必须以某种形式让用户感知到，不能做“黑盒”。
*   **沉浸感 (Immersion)**：复杂的后台操作必须是安静的，不能用弹窗或大段代码阻断用户的思维流。
*   **可观测性 (Observability)**：对于开发者，必须提供随时查看底层数据流（如 SystemBus 广播的事件、MTP 的原始 XML 响应）的渠道。

#### 1.2 机器态与人类态分离 (Separation of Machine and Human States)
这是针对 MTP 协议（Memory Tool Protocol）专属的 UI 降维打击策略。

*   **痛点**：Agent 实际输出的文本类似于 `根据我的推理，⟪ RUN | sys_write_file | path="x.py" ⟫ <mtp_response status="success">...</mtp_response> 写入成功。` 如果直接展示，阅读体验极差。
*   **解法**：前端解析器（Markdown Renderer）必须充当“翻译官”。
    *   **人类态视图 (Human View)**：主聊天区。将 `⟪...⟫` 和 `<...>` 的“机器态”内容彻底隐藏，替换为一个小巧的、可点击的 **“交互式状态卡片 (Action Badge)”**（如：`🛠️ 帕秋莉正在修改文件...`）。用户看到的是连贯的、干净的对话。
    *   **机器态视图 (Machine View)**：提供“查看原文”或右侧边栏的“日志终端 (Terminal)”，允许高级用户随时展开卡片，偷窥底层原生的 MTP 字符串，满足极客的掌控欲。

#### 1.3 沉浸式心流体验 (Immersive Flow via Glassmorphism)
*   **痛点**：传统的企业级后台（如各种多栏面板）往往带有强烈的机械感和压迫感，边框僵硬，不适合长时间的创意型对话。
*   **解法**：全面拥抱 **Glassmorphism（毛玻璃质感）**。
    *   消除实体边框（Solid Borders），使用带透明度的背景和模糊层（Backdrop Blur）来划分区域。
    *   当底层的动态背景（如流动的极光渐变）透过 UI 面板隐约显现时，不仅能在视觉上打破沉闷，更能通过模糊度的深浅，自然地向用户暗示信息层级（距离越近的悬浮窗越清晰，距离越远的底板越模糊）。
    *   这种设计语言深刻契合了“HiveMemory 在幕后无形运转，却又随时响应”的魔法感（Patchouli 的人设）。

### 2. 视觉系统与主题规范 (Visual System & Theming)

为了在 React 项目中高效落地我们的设计理念，我们采用 **Tailwind CSS** 作为原子化样式框架，并结合 **Shadcn UI** 作为组件基础。本节定义了系统的色彩、质感、排版以及组件的主题覆盖策略。

#### 2.1 质感规范：毛玻璃效应 (Glassmorphism Guidelines)

毛玻璃质感是 HiveMemory UI 的灵魂。为了避免滥用导致界面杂乱或性能下降，我们确立以下“毛玻璃使用最佳实践”：

1.  **分层原则 (Layering Principle)**
    *   不要在毛玻璃上叠加毛玻璃，这会导致模糊计算过度，使界面显得浑浊且降低渲染性能。
    *   **底层 (Background)**：流动的色彩或深色暗纹。
    *   **中层 (Panels & Sidebars)**：应用重度模糊（`backdrop-blur-md` 或 `backdrop-blur-lg`）的面板。
    *   **顶层 (Cards & Inputs)**：在面板内部的组件，使用轻度模糊（`backdrop-blur-sm`）或带透明度的纯色。

2.  **核心 Tailwind 类名组合 (The Glass Recipe)**
    在开发组件时，统一使用以下 Tailwind 类名组合来实现毛玻璃效果：
    *   **主面板/侧边栏 (Heavy Glass)**:
        `bg-background/60 backdrop-blur-lg border border-white/10 shadow-xl`
    *   **对话气泡/浮动卡片 (Light Glass)**:
        `bg-muted/40 backdrop-blur-sm border border-white/5`
    *   **悬浮输入框 (Omni-Input)**:
        `bg-background/80 backdrop-blur-md border border-border shadow-2xl`

3.  **对比度与可读性 (Contrast & Readability)**
    *   毛玻璃背景极易影响文字可读性。必须确保前景文本（`text-foreground`）具有足够的高对比度。
    *   避免在包含细小文字的区域（如代码块内部）使用毛玻璃。

#### 2.2 色彩调色板 (Color Palette via OKLCH/HSL)

Shadcn UI 依赖 CSS 变量来进行主题控制。我们将在 `globals.css` 中定义基于 **深色模式优先 (Dark Mode First)** 的主题变量。

推荐使用具有魔法感和科技感的色调：**星云紫 (Nebula Purple) 与 深渊蓝 (Abyss Blue)**。

**`globals.css` 核心变量示例 (深色模式)**:
```css
@layer base {
  :root {
    /* 默认背景：极深的蓝紫色调 */
    --background: 240 10% 3.9%;
    --foreground: 0 0% 98%;

    /* 魔法紫作为主色调 (Primary) */
    --primary: 262 80% 50%; /* e.g., Tailwind purple-600 */
    --primary-foreground: 210 40% 98%;

    /* MTP 指令/内核信息的专属次要色调 (Secondary/Muted) */
    --secondary: 240 3.7% 15.9%;
    --secondary-foreground: 0 0% 98%;

    /* 卡片与面板的底色，保留透明度空间 */
    --card: 240 10% 3.9%;
    --card-foreground: 0 0% 98%;

    /* 柔和的边框，配合毛玻璃 */
    --border: 240 3.7% 15.9%;
    --ring: 240 4.9% 83.9%;
  }
}
```
*(注：开发时可借助 Shadcn Theme Generator 等工具微调这些变量，以获得最佳视觉效果)*。

#### 2.3 排版与字体 (Typography)

排版是区分“玩具”与“专业生产力工具”的关键。对于 Agent 系统，文本的易读性直接影响交互效率。

1.  **字体栈 (Font Stack)**
    *   **UI 界面与自然语言 (Sans-serif)**: `Inter`, `system-ui`, `-apple-system`。清晰、现代。
    *   **MTP 协议日志与代码 (Monospace)**: `JetBrains Mono`, `Fira Code`。等宽字体能完美呈现协议的 `⟪` `⟫` 定界符和 JSON 结构，增强“极客感”。

2.  **Tailwind Typography 插件配置**
    对话区域 (Chat Stream) 往往包含复杂的 Markdown。我们将使用 `@tailwindcss/typography` 插件（`prose` 类）来统一渲染样式。
    *   **定制 `prose`**：覆盖默认的 `prose` 样式，使其适应深色毛玻璃背景。例如，调整 `prose-headings` 的颜色，消除代码块（`pre`）的背景色（让代码块本身呈现毛玻璃透视感或使用统一的深色底）。

#### 2.4 Shadcn UI 组件定制策略 (Component Customization Strategy)

Shadcn UI 采用“复制源码到项目”的模式（而非 npm 安装），这赋予了我们极大的定制自由度。

1.  **卡片组件 (`Card`)**
    修改 `components/ui/card.tsx`，在默认的 `className` 中混入我们的毛玻璃配方（`bg-background/60 backdrop-blur-md`），使其默认呈现透明质感。
2.  **折叠面板 (`Accordion` / `Collapsible`)**
    这是渲染 **MTP 动作卡片** 的核心组件。定制其动画效果（使用 Framer Motion 或 CSS Transitions），使其在展开/收起内核执行日志时（如 `<mtp_response>` 内容）平滑不突兀。
3.  **滚动条 (Scrollbar)**
    原生的滚动条会破坏毛玻璃的沉浸感。必须全局定制基于 Webkit 的极窄、半透明的滚动条样式。

### 3. 宏观布局架构 (Macro Layout Architecture)

本系统采用现代化桌面级应用的 **三栏/四栏自适应弹性布局 (Resizable Layout)**。为了兼顾日常对话的沉浸感和作为 AIOS 控制台的专业度，我们将空间划分为 L1 到 L4 四个主要层级。

#### 3.1 L1：全局导航栏 (Global Nav Bar - 左侧最边缘)
*   **定位**：系统的一级菜单，控制核心模块的切换，采用极窄的图标列设计。
*   **布局**：分为顶部功能区与底部系统区，图标垂直排列。
*   **内容规划**：
    *   **Top (核心模块)**：
        *   💬 **对话台 (Chat/Terminal)**：进入主对话界面（默认）。
        *   📚 **记忆花园 (Memory Garden)**：进入 Librarian 的卡片式知识库管理界面。
        *   🤖 **智能体中心 (Agents - 待定/灰色)**：预留给未来多智能体系统（Agent 列表或编排面板）。
    *   **Bottom (辅助与设置)**：
        *   🌓 **主题切换 (Theme Toggle)**：一键切换深色/浅色/跟随系统（深色模式下强化毛玻璃的极光质感）。
        *   ⌨️ **内核终端 (Kernel Console)**：快捷唤出一个全局悬浮的底部/右侧抽屉，用于快速查看底层日志。
        *   ⚙️ **全局设置 (Settings)**：API Key 配置、Qdrant 连接设置、实验性功能开关。
        *   👤 **用户账户 (User Profile)**：（预留）。

#### 3.2 L2：左侧边栏 (Context Sidebar)
*   **定位**：当前选中核心模块下的二级菜单或上下文列表。参考 Cherry Studio 的多标签页设计，以提升空间利用率。
*   **设计形式**：顶部放置 `Tabs`（标签页切换），下方为对应的列表内容。
*   **Tabs 规划**：
    1.  **🗂️ 话题池 (Topics - MMU 视图)**：
        *   **核心功能**：这不仅是“历史记录”，更是感知层维护的 `Active Topics`。
        *   **UI 细节**：当前激活的话题高亮；可以添加微小的状态指示器（如：🟢 活跃、🌙 休眠/已 Swap-out）；支持手动点击“垃圾桶”或“存档”按钮触发 `flush`。
    2.  **⚙️ 模型/Worker 设定 (Config)**：
        *   当前使用的话题或 Agent 的快速设定，如：系统提示词微调、Temperature 调整。
    3.  **🤖 Agent 列表 (Agents - 未来规划)**：
        *   当多智能体上线后，可以在此快速切换或拖拽不同的 Agent 参与当前话题。

#### 3.3 L3：主工作区 (Main Workspace)
*   **定位**：用户视线的绝对焦点，沉浸式对话流的发生地。
*   **布局特点**：
    *   **对话流 (Chat Stream)**：占据主体空间。在 MTP 执行时，隐藏原始 XML，渲染为精美的状态折叠卡片（如：`[帕秋莉正在查阅记忆...]`）。
    *   **全能输入框 (Omni-Input Area)**：
        *   吸底悬浮设计，背景略带透明度。
        *   上方/内部集成辅助功能条：`[➕ 附件/文件]`、`[# 引用特定记忆原子]`、`[@ 呼叫特定 Agent]`、`[清除上下文]`。
        *   自适应高度的 `Textarea`，支持多行代码舒适输入。

#### 3.4 L4：右侧附属面板 (Kernel Vision / Inspector)
*   **定位**：系统的“透视镜”，满足极客需求，默认折叠/吸附在右侧，按需**增量展开**。
*   **设计动机**：如果不将这些内核信息隔离，主工作区会被系统级的 Prompt 和参考资料淹没。
*   **展开逻辑**：点击 L3 中的某条 MTP 执行卡片，或点击界面右上角的“边栏”按钮即可滑出。
*   **内容规划**：
    *   **板块 A：RAG 检索菜单 (Context Menu)**：
        *   展示当前话题回合被 Retrieval Familiar 预检索出的记忆原子卡片（Alias + 摘要）。让用户知道 AI 当前参考了什么。
    *   **板块 B：执行详情与终端 (Execution Details)**：
        *   当展开特定的 `⟪ RUN ⟫` 或 `⟪ READ ⟫` 动作时，这里展示原始的 `<mtp_response>` 内容（长文本、代码执行的 Standard Output 等）。

---

### 给前端开发的建议 (Implementation Tips)

使用 Shadcn UI 实现这个布局时，可以重点关注以下组件：

1.  **整体骨架**：强烈建议使用 **`Resizable`** 组件（底层基于 `react-resizable-panels`）。这能让用户自由拖拽 L2 和 L4 面板的宽度，甚至将它们完全折叠隐藏，完美复刻 Cherry Studio 的丝滑体验。
2.  **L2 的标签页**：使用 **`Tabs`** 组件，它可以很方便地在“话题池”和“模型设置”之间无缝切换。
3.  **L4 的右侧面板**：如果你不希望它一直占据屏幕空间，可以将其实现为 **`Sheet`**（从右侧滑出的抽屉）或者配置了折叠状态的 `ResizablePanel`。
4.  **输入框**：使用 **`Textarea`** 组件，并结合 `react-textarea-autosize` 库，让它能像主流 Chat 软件一样，随着用户的输入自动增高，而不是出现难看的内部滚动条。

---

### 4. 核心功能视图详解 (Detailed View Design)

本节详细定义 L1 到 L4 四个主要板块的具体功能、UI 元素及交互逻辑。

#### 4.1 L1：全局导航栏 (Global Nav Bar)
*   **形态**：极窄侧边栏，背景采用强模糊 (`backdrop-blur-xl`)，色彩略深于主背景。
*   **交互**：Hover 图标时显示 Tooltip（文字提示）；点击图标平滑切换主视图。
*   **顶部功能区 (Top)**：
    *   **💬 对话台 (Chat)**：默认激活。图标：气泡。
    *   **📚 记忆花园 (Memory Garden)**：图标：书本或大脑。
    *   **🤖 智能体市场 (Agent Hub)**：（置灰预留）未来管理多智能体的入口。
*   **底部辅助区 (Bottom)**：
    *   **🌓 主题 (Theme)**：点击在“深色/浅色/系统”间循环切换。
    *   **⌨️ 内核日志 (Kernel Console)**：快捷键（如 `Ctrl + ~`）也可唤出。图标：终端图标。
    *   **⚙️ 设置 (Settings)**：弹出全局设置的 Modal。
    *   **👤 用户 (User)**：头像，点击展开账户管理 Dropdown。

#### 4.2 L2：左侧边栏 (Context Sidebar)
*   **形态**：可通过拖拽改变宽度，背景使用中度模糊 (`backdrop-blur-lg`)。
*   **Tabs 结构**：使用 Shadcn UI 的 `Tabs` 组件，样式精简为纯文字或带下划线的样式。
    *   **Tab 1: 🗂️ 话题池 (Topics / MMU State)**
        *   **功能**：展示感知层（Perception Layer）中的活跃/休眠话题。
        *   **卡片设计**：每个 Topic 为一个可点击的卡片。
            *   **主标题**：话题摘要（State Summary 的前20个字）。
            *   **副标题**：最后活跃时间（如 "10 mins ago"）。
            *   **状态标识**：
                *   🟢 绿点：当前正在使用的 Active Topic。
                *   🟡 黄点：驻留内存但未激活的 Topic。
                *   ⚪️ 灰点：已被 Swap-out 归档的旧 Topic。
        *   **交互**：Hover 时显示操作图标（如 `[💾 手动归档/Flush]`, `[🗑️ 删除]`）。
    *   **Tab 2: ⚙️ 模型配置 (Agent Config)**
        *   **功能**：调整当前 Agent 的参数。
        *   **元素**：Model 下拉菜单（GPT-4o, Claude 等）、Temperature 滑块、Max Tokens 输入框、System Prompt 微调文本框。

#### 4.3 L3：主工作区 (Main Workspace)
*   **形态**：占据屏幕主体，背景使用轻度模糊（透出底层极光渐变），营造沉浸感。
*   **4.3.1 对话流 (Chat Stream)**
    *   **气泡样式**：
        *   User：右侧对齐，主色调背景（如半透明紫色），无边框。
        *   Assistant：左侧对齐，透明背景，清晰的文本排版。
    *   **MTP 动作卡片 (Action Trace Cards)**：
        *   **渲染逻辑**：当流式输出中检测到 `⟪ RUN | sys_write_file ⟫` 时，**隐藏该文本**，替换渲染为一个 UI 组件（例如基于 `Accordion`）。
        *   **UI 表现**：
            *   *执行中*：显示微调动画卡片，如 `[ ⚙️ 帕秋莉正在执行 sys_write_file... ]`。
            *   *执行成功*：变成绿色小勾，如 `[ ✅ 文件写入成功 ]`。点击可展开查看详细日志。
*   **4.3.2 全能输入框 (Omni-Input Area)**
    *   **位置**：吸附在对话流底部，悬浮设计，四周有阴影。
    *   **内部结构**：
        *   **功能条 (Top Bar)**：
            *   `[➕ 附件]` 按钮：上传文件供读取。
            *   `[# 引用记忆]` 按钮：输入 `#` 号唤出下拉列表，快捷检索并引用特定的 Memory Alias。
        *   **输入区 (Textarea)**：使用 `react-textarea-autosize`，输入代码时自动长高。
        *   **发送区 (Bottom Right)**：发送按钮（可带入回车发送的快捷键提示 `Enter`，换行 `Shift+Enter`）。

#### 4.4 L4：右侧附属面板 (Kernel Vision / Inspector)
*   **设计动机**：AIOS 的工作流比传统聊天复杂得多。L4 是为高级用户和开发者设计的透视镜，展示隐藏在自然语言背后的 MTP 机制和 RAG 状态。
*   **展开交互**：
    *   可以通过 L1 的“内核终端”图标呼出。
    *   点击 L3 中的“MTP 动作卡片”时，自动展开 L4 并定位到对应日志。
*   **形态**：右侧滑出的抽屉（Sheet）或可拖拽的 Resizable Panel，背景使用强对比的深色主题（偏极客风），带轻微毛玻璃效果。
*   **Tab 1: 记忆上下文 (Context Menu)**
    *   **功能**：展示 *Retrieval Familiar*（检索使魔）在当前回合预检索出的记忆列表。
    *   **UI 元素**：
        *   标题：“当前话题参考记忆 (Top K)”。
        *   卡片列表：每张卡片显示 `Alias`（如 `fact_project_env`）、`Tags`（如 `#python` `#config`）和 `Summary`。
        *   **交互**：点击卡片可查看该原子的完整 Payload 并在必要时复制 ID 供手动输入 MTP 指令时使用。
*   **Tab 2: 执行终端 (Execution Log/Terminal)**
    *   **功能**：可视化 SystemBus 的事件流和 *Koakuma*（小恶魔）的执行细节。
    *   **UI 元素**：
        *   类似 VS Code Terminal 的黑底绿字/白字风格区。
        *   **日志条目**：
            *   `[SYSTEM] TheEye routed query to Topic T_05.`
            *   `[MTP_PARSE] Detected instruction: ⟪ READ | [mem_01] ⟫`
            *   `[KOAKUMA] Executing READ... Success (45ms).`
            *   `[INJECT] <mtp_response>...</mtp_response>` (支持长文本折叠/展开)。
    *   **交互**：提供“清空日志”、“自动滚动到底部”、“过滤错误 (Errors Only)”的控制按钮。

### 5. 关键交互与微动效 (Interaction & Micro-animations)

#### 5.1 智能路由感知 (The Routing Blink)
*   **设计动机**：当 *TheEye* 通过 Agentic Routing 将用户的新输入匹配到了后台的非当前话题（例如从“做菜”切回了“写代码”）时，必须让用户感知到“系统的注意力已经自动跳转”。
*   **交互表现**：
    *   左侧边栏 (L2) 对应的话题卡片会自动滚动到可视区域内。
    *   该卡片产生一次短暂的 **“呼吸高亮” (Breathing Glow)**：卡片背景色瞬间提亮（如添加 `bg-primary/20`），伴随轻微的白光边框闪烁（`ring-2 ring-primary/50`），然后平滑过渡到正常的“选中态”。
    *   **心理暗示**：就像接线员在总机上帮你切断了一根线，插到了另一根线上，既明确又不抢戏。

#### 5.2 MTP 机器态的“呼吸感” (MTP Execution States)
*   **设计动机**：*Koakuma* 在执行 `⟪ RUN ⟫` 或 `⟪ READ ⟫` 等耗时操作（沙箱运行、网络请求）时，主界面的对话流不能显得“死机”，需要有明确的生命力表现。
*   **状态流转与动效**：
    *   **解析态 (Parsing)**：文字流中刚生成 `⟪ RUN...` 时，立即折叠为 UI 卡片。
    *   **执行态 (Executing)**：卡片显示 Loading 图标（如旋转的虚线圈 `lucide-react/Loader2`），文字显示为“*帕秋莉正在执行指令...*”，卡片施加无限循环的柔和呼吸动效（`animate-pulse`）。
    *   **完成态 (Resolved)**：执行完毕时，通过 Framer Motion 实现平滑的形态转换（Layout Transition）。图标变为绿色的对勾（✅）或红色的警告（❌）。
    *   **展开交互 (Expand)**：用户点击卡片查看 `<mtp_response>` 日志时，面板必须像手风琴（Accordion）一样平滑推开下方的内容，拒绝生硬的瞬间拉伸。

#### 5.3 记忆沉淀的暗线反馈 (Background Archiving Cues)
*   **设计动机**：*Librarian* 往往在话题 Swap-out 或收到 `⟪ WRITE ⟫` 指令后，在完全异步的后台生成“记忆原子”。这种“冷链路”不能打断用户的“热链路”聊天，但需要给予“记性很好”的系统暗示。
*   **交互表现**：
    *   使用 Shadcn UI 的 `Sonner` 或 `Toast` 组件。
    *   当 Librarian 成功写入向量库后，在屏幕右下角静默滑出一个轻量级的毛玻璃 Toast（如：`✨ 提炼了一条新记忆：[项目配置更新]`）。
    *   （高级视觉特效）可选：当话题被自动存档时，左侧栏的话题卡片可以化作几点光斑（粒子特效）“飞”向全局导航栏 (L1) 的【📚 记忆花园】图标，图标随之产生微小的水波纹点按反馈（Scale bump）。

#### 5.4 丝滑的生成流与打字机光标 (Smooth Streaming & Cursor)
*   **设计动机**：由于包含了 MTP 协议的解析与隐藏，大模型吐出的 Token 并不是1:1映射到屏幕上的，直接渲染会出现跳跃感。
*   **交互表现**：
    *   **虚拟光标 (Virtual Cursor)**：在最后渲染出的文本末尾，跟随一个具有闪烁动画的光标块（`w-2 h-5 bg-primary animate-pulse`）。这比等待 Markdown 渲染完毕突然冒出一大段文字要有生命力得多。
    *   **自动滚动 (Auto-scroll)**：当生成流超过当前屏幕高度时，通过 `scrollIntoView({ behavior: 'smooth' })` 确保光标始终保持在视口下方约 20% 的位置，而不是死死贴在屏幕最底边，减轻用户的视觉压迫。

#### 5.5 毛玻璃的物理交互逻辑 (Glassmorphism Physics)
*   **设计动机**：毛玻璃质感不应仅仅是静态的图片，它应该响应用户的交互，体现出“物理层级”。
*   **交互表现**：
    *   **悬浮聚焦 (Hover / Focus)**：当鼠标悬浮在某个话题卡片、按钮或输入框上时，不仅背景透明度发生微小变化（如 `hover:bg-white/10`），其**模糊度 (Blur)** 也应有细微增加（如从 `blur-sm` 变成 `blur-md`），产生一种“物体被拉近”的景深感。
    *   **面板拖拽 (Panel Resizing)**：在使用 `Resizable` 拖拽调整 L2 或 L4 面板宽度时，拖拽手柄（Handle）在 Hover 时显示为一条高亮的极细光线，拖拽过程中不应卡顿，底层的毛玻璃渲染需保持 60fps。

### 6. 技术栈映射与实施指南 (Tech Stack & Implementation Guide)

为了兼顾开发效率与极客级的交互体验，前端项目将全面拥抱现代 React 生态。本章规定了核心依赖库的选择以及复杂交互的具体实现方案。

#### 6.1 核心依赖库清单 (Core Dependencies)

| 功能领域 | 推荐库 / 技术 | 选择理由 |
| :--- | :--- | :--- |
| **底层框架** | `React` | 提供 React 生态的全部功能，包括组件化、状态管理、事件处理等。 |
| **原子化样式** | `Tailwind CSS v4` + `clsx` + `tailwind-merge` | 快速构建毛玻璃效果，实现高度定制化的设计系统。 |
| **UI 组件库** | `shadcn/ui` | 无头组件（Headless UI），源码级控制，不绑定特定的样式框架。 |
| **动画引擎** | `framer-motion` | 声明式动画库，处理复杂的布局过渡（Layout Transitions）和微动效的最佳选择。 |
| **图标库** | `lucide-react` | Shadcn 默认搭配，线条干净，符合现代化极简风格。 |
| **输入框适配** | `react-textarea-autosize` | 解决原生 Textarea 无法随内容平滑自动增高的痛点。 |
| **Markdown 渲染** | `react-markdown` + `rehype-highlight` + `@tailwindcss/typography` | 高质量渲染大模型输出的文本与代码块，并完美融入暗色毛玻璃主题。 |

---

#### 6.2 Shadcn UI 组件映射表 (Component Mapping)

在搭建第 3 章定义的 **四栏布局 (L1-L4)** 时，直接使用以下 Shadcn UI 组件作为积木：

1.  **全局宏观布局 (The OS Layout)**
    *   使用 **`Resizable`** 组件 (`react-resizable-panels`)。
    *   *实现*：用 `<ResizablePanelGroup direction="horizontal">` 包裹 L2 (Topic Sidebar)、L3 (Main Chat) 和 L4 (Kernel Vision)。通过 `<ResizableHandle withHandle />` 实现像 VS Code 或 Cherry Studio 那样丝滑的拖拽调整宽度，并利用自带的持久化功能（`onLayout` 存入 `localStorage`）记住用户的偏好。
2.  **左侧栏/右侧栏切换**
    *   使用 **`Tabs`** (`<TabsList>`, `<TabsTrigger>`, `<TabsContent>`) 实现在“话题池”、“模型设置”或“内核日志”间的无缝切换。
3.  **MTP 动作卡片 (Action Trace Cards)**
    *   使用 **`Collapsible`** 组件。它比普通的 Accordion 更灵活，适合用来做“折叠态显示 Loading/成功，展开态显示 XML 日志”的 UI 逻辑。
4.  **感知层归档提示 (Background Cues)**
    *   使用 **`Sonner`** (Shadcn 中的现代化 Toast 组件)。当帕秋莉在后台完成 `Swap-out` 或 `WRITE` 操作时，在屏幕右下角弹出精美的毛玻璃通知。
5.  **滚动处理**
    *   对话流区域必须使用 **`ScrollArea`** 组件替换原生浏览器的滚动条，以保证毛玻璃背景的视觉连贯性。

---

#### 6.3 关键技术实现方案 (Key Technical Implementations)

**1. 极致毛玻璃样式的 Tailwind 封装**
为了避免在代码中反复书写冗长的类名，建议在全局配置或统一样式文件中封装一个公共类：
```css
/* globals.css */
@layer utilities {
  .glass-panel {
    @apply bg-background/60 backdrop-blur-xl border border-white/10 shadow-2xl;
  }
  .glass-card {
    @apply bg-muted/40 backdrop-blur-md border border-white/5 shadow-sm hover:bg-muted/50 transition-colors;
  }
}
```
*   *注意*：要确保 html/body 的背景是一张流动的动态壁纸或极光渐变图，否则 `backdrop-blur` 将失去视觉意义。

**2. 沉浸式全能输入框 (Omni-Input) 实现**
传统的 `<input>` 或 `<textarea>` 无法满足 Coding 场景的需求。我们结合 `react-textarea-autosize` 打造心流输入框：
```tsx
import TextareaAutosize from 'react-textarea-autosize';

// 在 Shadcn UI 风格下的包装
<div className="relative glass-panel rounded-2xl p-2 flex flex-col focus-within:ring-2 ring-primary">
  <div className="flex gap-2 pb-2 border-b border-white/10 mb-2">
     {/* 挂载工具栏：附件、引用记忆 */}
     <Button variant="ghost" size="icon"><PaperclipIcon /></Button>
     <Button variant="ghost" size="icon"><HashIcon /></Button>
  </div>
  <TextareaAutosize
    maxRows={10} // 限制最大高度，超过则内部滚动
    className="w-full resize-none bg-transparent outline-none p-2 text-foreground"
    placeholder="向帕秋莉提问，或输入 / 唤出指令..."
  />
</div>
```

**3. MTP 卡片的平滑过渡 (Framer Motion 动画)**
当 `⟪ RUN ⟫` 指令从“执行中”变为“完成”并展开时，DOM 的高度会发生变化。使用 Framer Motion 的 `AnimatePresence` 和 `motion.div` 来消除生硬的跳变。

```tsx
import { motion, AnimatePresence } from "framer-motion";

export function MtpActionCard({ status, logContent }) {
  const[isOpen, setIsOpen] = useState(false);

  return (
    <motion.div layout className="glass-card rounded-lg p-3 my-2 cursor-pointer" onClick={() => setIsOpen(!isOpen)}>
      <motion.div layout className="flex items-center gap-2">
        {status === 'pending' ? <Loader2 className="animate-spin text-primary" /> : <CheckCircle className="text-green-500" />}
        <span className="text-sm font-medium">
          {status === 'pending' ? '帕秋莉正在查阅记忆...' : '查阅完成'}
        </span>
      </motion.div>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden mt-2 text-xs font-mono text-muted-foreground"
          >
            {/* 这里展示原始的 <mtp_response> XML 内容 */}
            {logContent}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
```

**4. 解决“打断流”的打字机光标问题**
由于 MTP 协议要求拦截 LLM 生成（见内核设计部分），前端收到的流（Stream）可能会出现卡顿。
*   *解决策略*：不要直接将后端的 chunk 瞬间渲染上去。在前端维护一个极小的 `Queue (队列)`，以恒定的速率（如每 20ms 出队一个 Token）渲染字符，并在末尾加上一个闪烁的光标 `<span className="animate-pulse bg-primary w-2 h-4 inline-block ml-1" />`。这能完美掩盖后端因为执行 MTP 而产生的零点几秒的网络延迟。
