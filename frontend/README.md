# HiveMemory AIOS Frontend

面向 AI Agent 的可视化操作系统仪表盘 - 基于 React + TypeScript + Vite 构建

## 项目概述

HiveMemory 前端是一个专为 AI 操作系统设计的现代化仪表盘，采用 Glassmorphism（毛玻璃）设计风格，提供沉浸式的用户体验。

## 技术栈

- **框架**: React 19 + TypeScript
- **构建工具**: Vite 7
- **样式**: Tailwind CSS v4 + 自定义 Glassmorphism 效果
- **UI 组件**:
  - react-resizable-panels - 可调整大小的面板布局
  - lucide-react - 图标库
  - framer-motion - 动画引擎
- **Markdown 渲染**: react-markdown + rehype-highlight
- **代码高亮**: highlight.js

## 核心功能

### 四栏布局架构 (L1-L4)

#### L1: 全局导航栏 (Global Nav Bar)
- 位置：左侧最边缘，极窄图标列
- 功能：
  - 💬 对话台 (Chat)
  - 📚 记忆花园 (Memory Garden)
  - 🤖 智能体中心 (Agents - 预留)
  - 🌓 主题切换
  - ⌨️ 内核终端
  - ⚙️ 全局设置
  - 👤 用户账户

#### L2: 上下文侧边栏 (Context Sidebar)
- 可调整宽度的侧边栏
- 标签页切换：
  - **话题池 (Topics)**: 显示 MMU 状态，活跃/休眠话题管理
  - **配置 (Config)**: 模型设置、Temperature、Max Tokens

#### L3: 主工作区 (Main Workspace)
- 对话流展示
- MTP 动作卡片可视化
- 全能输入框 (Omni-Input)
  - 支持附件上传
  - 记忆引用 (#)
  - 自适应高度

#### L4: 内核视图 (Kernel Vision)
- 可折叠的右侧面板
- 标签页：
  - **Context**: RAG 检索的记忆原子展示
  - **Terminal**: SystemBus 事件流和执行日志

## 设计特色

### Glassmorphism 毛玻璃效果
- 深色模式优先设计
- 星云紫 (Nebula Purple #8B5CF6) + 深渊蓝 (Abyss Blue #1E1B4B) 配色
- 极光渐变动态背景
- 三层模糊策略：
  - 底层：流动色彩背景
  - 中层：重度模糊面板 (`backdrop-blur-xl`)
  - 顶层：轻度模糊卡片 (`backdrop-blur-sm`)

### MTP 协议可视化
- 机器态与人类态分离
- 动作卡片状态流转：
  - ⏳ 解析中 (Parsing)
  - ⚙️ 执行中 (Executing) - 带呼吸动效
  - ✅ 成功 (Success)
  - ❌ 失败 (Error)
- 可展开查看原始 MTP 响应

### 动画与交互
- Framer Motion 平滑过渡
- 智能路由感知 - 话题切换时的呼吸高亮
- 自定义滚动条
- 支持 `prefers-reduced-motion` 无障碍访问

## 开发指南

### 安装依赖

```bash
cd frontend
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:5173

### 构建生产版本

```bash
npm run build
```

### 代码检查

```bash
npm run lint
```

## 项目结构

```
frontend/
├── src/
│   ├── components/          # React 组件
│   │   ├── GlobalNavBar.tsx      # L1 全局导航
│   │   ├── ContextSidebar.tsx    # L2 上下文侧边栏
│   │   ├── MainWorkspace.tsx     # L3 主工作区
│   │   ├── KernelVision.tsx      # L4 内核视图
│   │   ├── ChatMessage.tsx       # 聊天消息组件
│   │   ├── MtpActionCard.tsx     # MTP 动作卡片
│   │   └── OmniInput.tsx         # 全能输入框
│   ├── lib/
│   │   └── utils.ts         # 工具函数 (cn)
│   ├── types/
│   │   └── index.ts         # TypeScript 类型定义
│   ├── App.tsx              # 主应用组件
│   ├── main.tsx             # 入口文件
│   └── index.css            # 全局样式
├── public/                  # 静态资源
├── index.html              # HTML 模板
├── vite.config.ts          # Vite 配置
├── tailwind.config.js      # Tailwind 配置
└── tsconfig.json           # TypeScript 配置
```

## TypeScript 类型系统

项目使用严格的 TypeScript 类型检查，确保后端数据字段快速迭代时前端能立即定位问题。

主要类型定义：
- `MtpAction` - MTP 协议动作
- `Topic` - 话题/MMU 状态
- `Message` - 聊天消息
- `MemoryAtom` - 记忆原子
- `SystemEvent` - 系统事件
- `AgentConfig` - Agent 配置

## 无障碍访问 (Accessibility)

- 所有交互元素支持键盘导航
- ARIA 标签完整
- 表单字段带有 id/name 属性
- 支持 `prefers-reduced-motion`
- 文本对比度符合 WCAG 4.5:1 标准

## 浏览器兼容性

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## 性能优化

- Vite 快速热更新
- 代码分割和懒加载
- 优化的 Tailwind CSS (JIT 模式)
- 自定义滚动条减少重绘

## 后续开发计划

- [ ] 连接后端 API
- [ ] WebSocket 实时通信
- [ ] 多智能体管理界面
- [ ] 记忆花园可视化
- [ ] 主题切换功能
- [ ] 国际化支持

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
