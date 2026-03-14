# HiveMemory UI/UX 设计文档 (v1.2)
## 专篇：七曜魔法主题与水晶质感重构 (The Seven Luminaries & Crystal UI)

**文档状态**: Active (执行中)
**核心思想**: 以帕秋莉·诺蕾姬的设定为灵感，摒弃单调的冷色系暗黑模式，引入“日月交辉”与“五行水晶”的概念，构建兼具魔法感与生产力工具克制感的前端视觉体系。

---

### 1. 核心视觉理念 (Core Visual Concept)

#### 1.1 从“死黑毛玻璃”到“水晶通透感”
*   **痛点**：早期的深色模式背景过暗，导致毛玻璃（Glassmorphism）失去折射光源，组件显得像实心的塑料块。
*   **解法**：大幅度提亮背景的“极光（Aurora）”层，增加色彩的多样性与饱和度；同时降低所有 UI 面板（如 `glass-sidebar`）的不透明度（Opacity），并增加内部高光投影（`inset box-shadow`），使其呈现出**内部发光的水晶质感**。

#### 1.2 “七曜魔法”色彩映射 (The Semantic Mapping)
将色彩管理赋予物理和状态隐喻：
*   **日与月 (Sun & Moon)**：作为全局的宏观环境光与核心强调色。
*   **五行 (木火土金水)**：作为组件的状态指示色（Success, Error, Pending 等）。

@theme {
  /* ... 保持背景和基础紫色不变 ... */
  
  /* 七曜魔法 Semantic Colors (调整了 HSL 以适应暗色毛玻璃的通透感) */
  --color-magic-water: hsl(190 90% 50%);  /* 青蓝 - 运行中 */
  --color-magic-wood: hsl(150 80% 45%);   /* 翡翠 - 成功 */
  --color-magic-fire: hsl(340 80% 60%);   /* 绯红 - 错误/破坏 */
  --color-magic-earth: hsl(30 20% 50%);   /* 暖灰 - 休眠/历史 */
  --color-magic-metal: hsl(40 90% 55%);   /* 琥珀 - 警告/待机 */
  
  --color-magic-moon: hsl(260 80% 65%);   /* 亮紫 - 主题高光 */
  --color-magic-sun: hsl(45 100% 60%);    /* 亮金 - 核心指引 */
}

---

### 2. 全局环境光定义 (The Canvas: Sun & Moon)

宏观背景不再是单一的冷色调，而是引入对角线式的冷暖碰撞，形成视觉张力。

*   **🌙 月 (The Moon)**：
    *   **色调**：星云紫 (Nebula Purple) 与 深渊蓝 (Abyss Blue)。
    *   **位置**：占据背景的左下与核心大面积区域。代表系统深不可测的记忆库与沉稳的思考逻辑。
*   **☀️ 日 (The Sun)**：
    *   **色调**：琥珀金 (Amber Gold) 与 晨曦微红。
    *   **位置**：点缀于背景的右上角高光区。代表用户活跃的指令、当前的焦点与系统的温度。

**实现指引 (`body::before` 背景动画更新)**:
```css
body::before {
  /* ... */
  background:
    /* 左下：月之深邃 (紫) */
    radial-gradient(circle at 15% 85%, hsl(265 70% 40% / 0.25) 0%, transparent 50%),
    /* 中部：魔法之渊 (蓝) */
    radial-gradient(circle at 50% 50%, hsl(225 80% 50% / 0.15) 0%, transparent 60%),
    /* 右上：日之高光 (琥珀金) */
    radial-gradient(circle at 85% 15%, hsl(40 90% 55% / 0.2) 0%, transparent 40%);
  opacity: 0.8; /* 提高基础透明度，确保玻璃有光可透 */
}
```

---

### 3. 五行功能色定义 (Semantic Crystals)

将金、木、水、火、土映射为具体 UI 组件的交互状态。在毛玻璃体系下，状态色必须遵循**“极低透明度底色 (`bg-color/10`) + 高亮边框/文字 (`border-color/30 text-color`)”**的“发光晶体”原则，绝不可使用高饱和度的实心色块。

| 元素 | 色调名称 | 状态语义 | 适用组件场景示例 | Tailwind / CSS 变量值 (HSL) |
| :--- | :--- | :--- | :--- | :--- |
| **💧 水** | **湛蓝 / 青蓝** | 流动、执行中、网络请求 | MTP `[正在执行]` 卡片、流式打字机光标、Loading 动画 | `--color-magic-water: 190 90% 50%` <br> `(Cyan-400)` |
| **🌲 木** | **翡翠 / 荧绿** | 生长、完成、成功、激活 | MTP `[执行成功]` 卡片、当前活跃的 Topic 绿点指示器 | `--color-magic-wood: 150 80% 45%` <br> `(Emerald-400)` |
| **🔥 火** | **绯红 / 玫红** | 破坏、警告、报错、删除 | MTP `[执行中断]` 卡片、Topic 卡片的删除按钮、系统 Error Toast | `--color-magic-fire: 340 80% 60%` <br> `(Rose-400)` |
| **🪨 土** | **暖灰 / 陶土** | 沉淀、历史、休眠、基础 | 已 Swap-out 的归档 Topic 灰点、未激活的 Tabs 文字、历史时间戳 | `--color-magic-earth: 30 20% 50%` <br> `(Stone-400)` |
| **⚡ 金** | **白金 / 明黄** | 警示、注意、挂起、待机 | 处于 Dormant 状态的 Topic 黄点、需要用户确认的危险写入拦截 | `--color-magic-metal: 40 90% 55%` <br> `(Amber-400)` |

---

### 4. 部分关键组件的色彩重塑 (Component Reshaping)

结合上述色彩体系，对当前 UI 中视觉表现较弱的组件进行定向“整容”。

#### 4.1 用户气泡 (User Bubble) 的升华
*   **当前问题**：较深的纯紫色过于暗沉，与背景顺色。
*   **重塑方案**：引入帕秋莉标志性的**粉紫 (Magenta/Fuchsia)**。将其从主紫调向暖色调倾斜，不仅代表人类指令的温度，更能与背景偏冷的蓝紫色形成极佳的前后景对比。
*   **具体样式**：
    `bg-gradient-to-br from-fuchsia-600/30 to-purple-600/10 border-fuchsia-500/20 shadow-[0_4px_15px_rgba(192,38,211,0.15)] text-white`

#### 4.2 MTP 指令卡片的“晶体化”
*   **当前问题**：类似早期的深灰色实心终端框，破坏了阅读心流。
*   **重塑方案**：根据 MTP 的执行状态，应用对应的“五行”色彩。彻底移除不透明的灰色背景底。
*   **具体样式 (以 Success/木 为例)**：
    `bg-emerald-900/20 backdrop-blur-md border border-emerald-500/30 text-emerald-300 shadow-[inset_0_0_12px_rgba(16,185,129,0.1)]`
    *(注：`inset` 阴影是实现“内部发光魔法石”质感的关键点。)*
