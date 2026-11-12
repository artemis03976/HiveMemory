---
title: Frontend Application Shell and Visual System
status: current
owner: frontend
scope: navigation-layout-theme-and-deployment
code_paths:
  - frontend/src/App.tsx
  - frontend/src/components/GlobalNavBar.tsx
  - frontend/src/components/ChatLayout.tsx
  - frontend/src/components/common/
  - frontend/src/index.css
  - frontend/src/stores/ui/theme.ts
  - frontend/src/stores/ui/toastStore.ts
  - frontend/src/transports/connectionStatus.ts
  - frontend/src/hooks/useDialog.ts
  - frontend/vite.config.ts
  - src/hivememory/server/app.py
related_contracts:
  - docs/contracts/routes-and-events.md
last_reviewed: 2026-11-11
---

# 前端应用壳与视觉系统

应用壳把 HiveMemory 的多个观察和管理视角放进同一张工作台。它不试图用一套路由层模拟多个独立产品，而是让用户在 Chat、记忆、Agent 和配置之间快速切换，同时保留主题与面板布局偏好。当前设计优先服务桌面开发与个人使用，尚未把 URL 导航、账户边界或移动端布局作为已交付能力。

## 1. 当前结构

```text
App
  ├─ DynamicToast（右上角）
  ├─ GlobalNavBar（固定 64px）
  └─ 主区域
       ├─ ConnectionBanner（连不上后端或凭据失效时出现）
       └─ 当前页面
            ├─ ChatLayout
            │    ├─ ContextSidebar（会话列表，264px）
            │    ├─ ChatPane（剩余宽度）
            │    └─ WorkspaceVision（引用记忆 / 记忆任务 / 模型配置，360px）
            ├─ MemoryLibrary
            ├─ AgentManagement
            ├─ RunView（运行）
            └─ SettingsPanel
```

`activeNavTab` 决定五个主页面中的哪一个被渲染。这里没有 React Router，也没有按 URL 恢复页面、浏览器前进后退或页面级深链接。

### 1.1 导航栏与头像菜单

导航栏上方是品牌标记与对话、记忆图书馆、Agent、运行四个页面，底部是系统设置、主题切换与头像菜单；“运行”打开[全局运行页](./run-view.md)。导航栏的按钮都带 `aria-label`，页面按钮与主题切换按钮在悬停或键盘聚焦时还会在右侧出现同名提示；当前页面带 `aria-current="page"`，并以品牌渐变的竖条标出。主题切换按钮按当前实际显示的深浅色切到另一种，选择随即固定为深色或浅色。

头像是本地的品牌样式标记，不加载外部图片。没有账户系统，因此不显示用户名；点击打开外观菜单，菜单里是深色、浅色、跟随系统三个主题选项和“外观设置”入口（跳到设置页“常用”中的外观一节）。点击菜单外部或按 Esc 关闭。

### 1.2 离开设置页

设置页有未保存的更改时，经导航栏离开设置页之前先弹出确认；刷新或关闭窗口时由浏览器的离开提示拦下（`beforeunload`）。是否有未保存的更改记在不持久化的 `stores/ui/unsavedSettings.ts` 中，由设置页写入、`App` 在切换页面时读取。

### 1.3 Chat 三栏与断点

Chat View 不是旧设计中的可拖拽四栏：左侧会话栏 264px，右栏固定 360px、不随标签变宽，中间栏使用剩余空间。`react-resizable-panels` 尚未被源码使用，不能把依赖存在等同于拖拽布局已经落地。

`ChatLayout` 按窗口宽度决定两栏的位置：

| 窗口宽度 | 会话栏 | 右栏 |
|:---|:---|:---|
| 1280px 及以上 | 行内 | 行内 |
| 1024–1279px | 行内 | 抽屉 |
| 1024px 以下 | 抽屉 | 抽屉 |

行内的两栏各有一个收起按钮，收起后由中间栏头部的按钮展开：左侧是图标按钮，右侧是写着“引用记忆”的按钮（本轮有引用时带条数）。放不下的一栏不在行内显示，同一个按钮改为打开抽屉，抽屉从对应一侧覆盖在中间栏上，点遮罩、抽屉里的关闭按钮或按 Esc 关闭。收起状态持久化在 `chat-ui-store` 中，只在行内生效；抽屉的开合是组件内的临时状态，不改动它。运行时事件流在 Chat 页挂载时连接，右栏收为抽屉时仍接收记忆任务的事件。

桌面客户端的窗口最小为 1024×640（[桌面客户端](./desktop-client.md)），1024px 以下只会出现在浏览器中。

### 1.4 页面头部

`components/common/PageHeader.tsx` 提供统一的页面头部：左侧是标题、一条品牌渐变短线和一行说明，右侧放本页的操作。记忆图书馆与系统设置使用完整的 `PageHeader`，运行页使用紧凑的 `PageTitle`；Chat 的中间栏头部是会话标题与轮数。

## 2. 为什么采用同屏工作台

HiveMemory 的价值不仅在最终回答，还在回答依赖了哪些记忆、触发了哪些动作、后台是否仍在沉淀。本设计因此把三种视角放在同一水平空间：

- 左侧回答“本次对话处于哪个会话”；
- 中央回答“人正在与哪个 Agent 交互”；
- 右侧回答“系统正在引用、生成什么，这一轮用什么模型”。

侧栏可以收起，是为了让日常对话保持沉浸；它们始终可返回，又使调试不必跳到另一套管理工具。两栏固定宽度，窗口放不下时改为抽屉而不是继续压缩，这样中间栏的正文与输入栏不会被挤到无法阅读。

## 3. 视觉系统

### 3.1 主色与点缀

紫色是主色，用于选中、焦点与主要操作；樱花粉（`accent`）是第二强调色，只作点缀（如 `#` 话题标签、头像图标），不表示任何状态。紫到粉的品牌渐变（`bg-brand`）只用在少数焦点上：品牌标记、导航当前项的竖条、每个视图里最主要的一个按钮、页面标题的装饰线。大面积的背景与卡片保持中性，对话、Agent 与设置的主区域只铺一层很淡的径向氛围色（`bg-atmo`）。

### 3.2 状态色与记忆类型色

状态色只表示状态，`index.css` 中有四个：

| 令牌 | 主要语义 |
|:---|:---|
| `magic-water` | 进行中、流式、连接 |
| `magic-wood` | 完成、成功、生效 |
| `magic-fire` | 失败、错误、删除 |
| `magic-metal` | 等待、警告、需要注意 |

失败色是偏橙的红，与樱花粉的色相拉开，使“失败”不会被看成装饰。状态色作为低透明度底色、圆点与文字提示，而不是高饱和度实心色块；状态总是以圆点加文字呈现。若同一业务状态在不同页面被映射为不同颜色，颜色就失去了帮助用户识别状态的意义。

记忆类型另有一组七色的 `type-*` 令牌（水、木、金、火、月、日、土），只上在记忆类型的图标块上：图标着色、底为同色的浅底，类型名是中性文字，卡片背景与边框不带类型色（`components/memory/MemoryTypeTile.tsx`）。类型是图标块、状态是圆点，两者靠形状区分；反思类型保留偏粉的红，与失败色不同。

### 3.3 表面、文字与主题

组件的颜色取自语义令牌，只有品牌渐变上的文字用白色：

- 表面分层：`background`、五级 `surface-container-*`，输入框的 `field`，代码块的 `code`，对话框与抽屉的遮罩 `scrim`；
- 文字分三级：`fg`（正文）、`fg-2`（次要）、`fg-3`（弱，用于标签与时间）；
- 边框分两级：`line` 与 `line-strong`；`ink` 是叠加用的前景色，配合透明度做悬停底色与细分隔；
- 主要按钮与当前项的光晕 `shadow-glow`、浮层阴影 `shadow-pop`。

面板与卡片都用不透明的表面令牌，不做毛玻璃面板。深色主题是近黑偏紫的底色；浅色主题有独立的调色板，不由深色反转而来：底色是带一点薰衣草色的白，主色、樱花粉与状态色都取更深的一档以保证白底上的对比度，不用光晕，改用轻阴影区分层级。

主题可选深色、浅色或跟随系统（`stores/ui/theme.ts`），选项持久化在 `chat-ui-store` 中：新安装默认跟随系统，已经保存为深色或浅色的保持不变。跟随系统时 `App` 订阅 `prefers-color-scheme`，桌面客户端中由 Electron 按 Windows 的设置给出；解析出的深浅色在 `useLayoutEffect` 中写入 `<html data-theme>`，避免首屏闪烁，同时切换 `color-scheme`，让原生控件与滚动条随主题变化。切换主题的入口有三处：导航栏的主题按钮、头像菜单与设置页“常用”中的外观一节。

### 3.4 字体与文字

Inter（正文）与 Manrope（标题）以 `@fontsource-variable/inter`、`@fontsource-variable/manrope` 随前端构建打包，不从网络加载，桌面端离线时字体不变；中文回退到系统字体（Microsoft YaHei UI、PingFang SC、Noto Sans SC），等宽字体用 Cascadia Mono、Consolas。界面文案用中文，标识符与协议名原样显示；层级靠字号、字重与颜色区分，不用全大写，最小字号 12px；等宽字体只用于标识符、代码与日志，并原样显示大小写。

时间统一由 `utils/dateTime.ts` 格式化：今天只显示时分，今年显示月日与时分，更早的显示完整日期；列表中一周以内的时间用相对时间（“3 分钟前”）。随时间变化的显示以 `hooks/useNow.ts` 给出的当前时间为准、按固定间隔刷新，不在渲染中直接读时钟。

### 3.5 动效是状态迁移的视觉投影

前端的 glow、pulse、展开收束和完成提示并不是为了给等待过程添加“生命感”。它们应当投影后端已经发生的状态迁移：例如 `done.topic_id` 确认本轮话题后才能显示回复的话题标注，收到 `mtp_start` 后才能开始 MTP 动作提示，`mtp_result`、`done` 或对应 RuntimeEvent 到达后才能收束为完成。若事件尚未到达，界面应保持准备中、等待或不确定，而不能用恒速 token 队列和循环动画掩盖超时、断流或未完成终态。

这条原则也约束未来的视觉增强：任何“记忆已完成”“任务已落地”或“系统已切换话题”的提示，都必须能追溯到结构化事件或业务响应；动画本身永远不能成为业务成功的证据。

## 4. 组件体系

当前组件为项目内自有 React 实现，样式主要由 Tailwind CSS 4 class 与第 3 节的令牌组成。`components/common/` 放跨页面复用的部件：页面头部、确认框、开关与输入框、滑块、Markdown 渲染、全局提示与连接状态条。Motion 用于消息、卡片与弹层的入场和展开；Lucide 提供图标；Markdown 由 react-markdown、remark/rehype 与 highlight.js 渲染。

旧设计稿曾提出 Shadcn UI，但当前依赖和源码都没有使用 Shadcn/Radix。若未来引入新的组件基础，必须先明确它如何继承现有 token、主题和交互语义，而不是让两套视觉与状态规范并行。

### 4.1 提示与连接状态条

全局提示（`components/common/DynamicToast.tsx`、`stores/ui/toastStore.ts`）放在右上角、页面头部之下，不压住页面标题。同一条提示（文字与类型都相同）在显示期间只出现一次，重复的调用直接忽略，避免同一错误弹出两条。未指定时长时按类型取值：成功 2 秒、信息 3 秒、警告 4 秒、错误 6 秒；错误提示另有关闭按钮。

连不上后端或管理凭据失效时，主区域顶部出现连接状态条（`components/common/ConnectionBanner.tsx`），写明原因与处理办法：连不上后端时提示确认后端正在运行，恢复连接后自动消失；凭据失效时提示刷新页面，并带“刷新页面”按钮。两者同时成立时只显示“连不上后端”。各页面的局部错误在这时只写“数据暂不可用”，不再各自重复连接类的原因；连接状况如何得出见[状态、持久化与传输](./state-and-transports.md#32-连接状况与局部错误)。

### 4.2 键盘与读屏

- 键盘焦点有可见的主色描边（`:focus-visible`）。
- 确认框、注入记忆与记忆详情三个对话框经 `hooks/useDialog.ts` 处理键盘：打开时焦点移入面板，Tab 只在对话框内循环，Esc 关闭最上层的一个，关闭后焦点回到打开之前的位置。
- 以 `role="button"` 呈现的卡片（记忆卡片与列表项、子 Agent 卡片）可以用 Tab 到达，Enter 或空格打开（`utils/activateOnKey.ts`），卡片内部的按钮与输入框各自处理按键。
- 图标按钮带 `aria-label`；开关 `Toggle` 是带 `aria-checked` 的 `role="switch"`；全局提示与连接状态条以 `role="status"` 播报。
- 系统设置为“减少动效”时，CSS 的动画与过渡（脉动、闪烁的光标、流动的边框）缩短到几乎为零；Motion 驱动的入场与展开动画由应用根部的 `MotionConfig reducedMotion="user"` 一并去掉。

## 5. 开发与部署形态

| 形态 | 前端入口 | 后端入口 | 连接方式 |
|:---|:---|:---|:---|
| 本地开发 | `127.0.0.1:5173` | `localhost:8769` | Vite 将相对 `/api` 代理到后端，并代理 WebSocket |
| Docker/整合部署 | `localhost:8000` | 同源 `localhost:8000` | FastAPI 挂载 `frontend/dist`，API、SSE 与 WebSocket 同源 |

生产构建只有在 `HIVEMEMORY_SERVE_FRONTEND=true` 且构建目录存在时才由 FastAPI 提供。静态服务对非 API 路径执行 SPA fallback，并对请求路径做 realpath/commonpath 检查；API 未匹配仍返回 404，不会被 `index.html` 吞掉。

前端的 HTTP、SSE 与 WebSocket 默认连接页面自身的 origin。`VITE_BACKEND_ORIGIN` 让三者一起直连另一个 origin，但直连的请求不带管理凭据，非同源部署仍需要外部反向代理（[状态与传输](./state-and-transports.md#4-origin-与代理)第 4 节）。

## 6. 不变量与矛盾检查

- 页面导航若新增一个 tab，必须同时存在真实渲染分支、可恢复状态和无障碍名称；
- 颜色只经语义令牌使用，新增的颜色在深浅两套主题下都要有取值；不在局部组件随意重新定义同名状态色；
- 状态色只表示状态，樱花粉与品牌渐变不表示任何状态，记忆类型色只用在类型图标块上；
- 连接类的原因只由顶部的连接状态条说明，页面局部不重复；
- 机器态信息通过结构化卡片和侧栏渐进披露，不回流为普通正文噪声；
- 前端部署口径必须区分开发端口 `5173/8769` 与 Docker 同源端口 `8000`；
- 新的后端 origin 配置只有覆盖 HTTP、SSE 与 WebSocket 全部传输后，才能被称为完整的分离部署支持；
- 任何“可拖拽、响应式、账户头像、终端页面”描述必须有当前组件和交互证据，不能只依据依赖、图标或设计稿。

## 7. 当前限制

- 无客户端路由、深链接和浏览器历史集成；
- 没有账户系统，头像只是外观菜单的入口；
- 断点只按 1440 与 1024 两种宽度核对过：Chat 页在 1280 与 1024 两处收起侧栏，短期话题池在 1280 以下收窄列表；1024 以下、超宽屏下的内容宽度，以及 Agent 页与设置页固定宽度的左栏都没有系统处理；
- 动效没有统一的时长与缓动，入场动画只有部分组件使用 Motion；上述细节与响应速度登记在 [Todo：前端细节优化](../todo/frontend-ui-polish.md)；
- 没有已配置的前端组件测试、视觉回归或可访问性自动化门槛，`frontend/tests/` 覆盖会话 store 的行为与主题解析、提示去重、连接状况等纯逻辑，不渲染组件；
- `react-resizable-panels` 留在依赖中但未使用，属于实现与依赖清理问题。

Chat 内部交互见 [Chat 视图](./chat-view.md)，状态与部署连接细节见[状态、持久化与传输](./state-and-transports.md)。
