- [ ] 完善后端日志信息流显示的log信息（分span、增加log信息丰富度等）
- [ ] 令记忆花园搜索框支持基于全局记忆库的搜索，而非当前的本地形式

## 身份管理优化

### 方案三：前端增加 UserStore（长期架构优化）

**目标**：为未来的多用户登录功能做准备，前端拥有完整的身份管理能力。

**实现步骤**：

1. **创建 UserStore**
   - 文件路径：`frontend/src/stores/userStore.ts`
   - 功能：管理当前用户的 `userId`，支持持久化
   - 提供 `setUserId` 方法用于切换用户（调试/多用户场景）

2. **集成到 ChatStore**
   - 在 `sendMessage` 中从 `useUserStore` 获取 `userId`
   - 优先级：`options.user_id` > `userStore.userId` > `DEFAULT_USER_ID`

3. **UI 支持（可选）**
   - 在开发模式下提供用户切换面板
   - 用于调试不同用户的记忆隔离

**优点**：
- ✅ 为未来的用户登录功能做准备
- ✅ 前端有完整的身份管理能力
- ✅ 可以在 UI 中切换用户（调试用）

**前置条件**：
- ✅ 方案一已完成（统一常量文件）

**参考实现**：

```typescript
// frontend/src/stores/userStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_USER_ID } from '@/constants/identity';

interface UserStore {
  userId: string;
  setUserId: (id: string) => void;
}

export const useUserStore = create<UserStore>()(
  persist(
    (set) => ({
      userId: DEFAULT_USER_ID,
      setUserId: (id: string) => set({ userId: id }),
    }),
    { name: 'user-store' }
  )
);
```

```typescript
// frontend/src/stores/chatStore.ts 修改
import { useUserStore } from '@/stores/userStore';

sendMessage: async (content: string, options = {}) => {
  const userId = useUserStore.getState().userId;
  const requestBody = {
    message: content,
    user_id: options.user_id || userId,
    agent_id: options.agent_id || state.currentAgentId,
    // ...
  };
  // ...
}
```

**注意事项**：
- 当前阶段如果没有多用户需求，可以暂缓实现
- 实现时需要考虑与后端认证系统的集成
- 需要处理用户切换时的状态清理（消息、话题等）
