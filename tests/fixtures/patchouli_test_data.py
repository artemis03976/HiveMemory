"""
HiveMemory Patchouli Stage 1 集成测试数据 Fixtures

提供专门用于 PatchouliAgent 集成测试的对话数据。

设计原则:
    - 模拟真实生产环境的对话长度
    - 覆盖多种语义距离场景（远、较远、较近、近）
    - 支持 Chatbot 和 Agent 两种对话流
    - 每条消息超过短文本阈值 (50 tokens) 以避免强制吸附

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any, Optional


def estimate_tokens(text: str) -> int:
    """
    估算文本的 Token 数量

    规则：
    - 中文 1 token ≈ 2 字符
    - 英文 1 token ≈ 4 字符
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return (chinese_chars // 2) + (other_chars // 4)


# ========== 话题 1: 数据科学对话（基线话题） ==========

DATA_SCIENCE_CONVERSATION = [
    {
        "role": "user",
        "content": """我想学习数据可视化，你能介绍一下Python中常用的数据可视化库吗？
包括Matplotlib、Seaborn、Plotly这些库的区别和使用场景，
最好能有一些实际应用的例子。我主要做数据分析工作，
需要制作各种统计图表和交互式可视化。""",
        "estimated_tokens": 130,
    },
    {
        "role": "assistant",
        "content": """Python数据可视化库主要有以下几个：

1. Matplotlib
- 基础可视化库，其他库的底层依赖
- 适合绘制静态图表，灵活性高但代码相对繁琐
- 适用场景：快速原型、自定义复杂图表

2. Seaborn
- 基于Matplotlib的高级API，提供更美观的默认样式
- 专为统计绘图设计，支持复杂的多变量可视化
- 适用场景：统计分析、热力图、小提琴图等

3. Plotly
- 交互式可视化库，支持缩放、悬停、筛选等交互
- 可输出为HTML或嵌入Web应用
- 适用场景：仪表板、探索性数据分析

实际应用示例：
```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载示例数据
tips = sns.load_dataset("tips")

# 绘制小提琴图
sns.violinplot(x="day", y="total_bill", data=tips)
plt.show()
```""",
        "estimated_tokens": 420,
    },
    {
        "role": "user",
        "content": """这些可视化库中，哪个最适合做实时数据监控？
我需要监控服务器的性能指标，比如CPU使用率、内存占用、
网络流量等，需要实时更新的图表。""",
        "estimated_tokens": 100,
    },
    {
        "role": "assistant",
        "content": """对于实时数据监控，我推荐以下方案：

1. Plotly Dash
- 专为实时数据监控设计的Web框架
- 支持自动刷新、实时回调
- 可以快速构建完整的监控仪表板

2. Streamlit
- 更简单的实时监控方案
- 自动重新渲染，代码量极少
- 适合快速原型开发

3. Bokeh
- 高性能交互式可视化
- 支持流式数据更新
- 适合大规模数据监控

对于服务器性能监控，建议使用Plotly Dash配合WebSocket实现实时更新：
```python
import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval', interval=1000)
])

@app.callback(Output('live-graph', 'figure'),
              Input('interval', 'n_intervals'))
def update_graph(n):
    # 获取实时数据
    cpu_data = get_cpu_usage()
    # 更新图表
    return create_figure(cpu_data)
```""",
        "estimated_tokens": 380,
    },
    {
        "role": "user",
        "content": """我想深入了解Seaborn的统计图表功能。比如小提琴图、
箱线图、热力图这些图表各自适合什么场景？
在数据分布比较偏斜的情况下，应该怎么选择可视化方式？""",
        "estimated_tokens": 105,
    },
    {
        "role": "assistant",
        "content": """Seaborn的统计图表各有特点：

1. 小提琴图 (Violin Plot)
- 结合了箱线图和核密度估计
- 适合展示数据分布的形状
- 特别适合比较多个类别的分布

2. 箱线图 (Box Plot)
- 展示中位数、四分位数、异常值
- 适合快速了解数据分布概貌
- 对偏斜数据不那么敏感

3. 热力图 (Heatmap)
- 适合展示相关性矩阵
- 可视化二维数据的密度
- 常用于特征相关性分析

对于偏斜数据：
- 使用小提琴图可以看到完整的分布形状
- 考虑对数变换后绘图
- 使用箱线图配合散点图展示个体数据点

示例代码：
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 生成偏斜数据
skewed_data = np.exp(np.random.randn(100))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.violinplot(data=skewed_data, ax=axes[0])
sns.boxplot(data=skewed_data, ax=axes[1])
plt.show()
```""",
        "estimated_tokens": 360,
    },
]

# ========== 话题 2: Web 开发（较近话题，相似度 0.6-0.8） ==========

WEB_DEVELOPMENT_CONVERSATION = [
    {
        "role": "user",
        "content": """我在做Web开发，想了解一下前端状态管理的最佳实践。
Redux、MobX、Zustand这些状态管理库有什么区别？
在什么场景下选择哪个比较好？我的项目是一个中型的电商管理系统。""",
        "estimated_tokens": 115,
    },
    {
        "role": "assistant",
        "content": """前端状态管理库对比：

1. Redux
- 成熟稳定，生态系统完善
- 样板代码较多，学习曲线陡峭
- 适用场景：大型企业级应用、需要可预测状态管理

2. MobX
- 基于响应式编程，代码简洁
- 自动追踪依赖，更新效率高
- 适用场景：中大型应用、复杂状态关系

3. Zustand
- 轻量级，API简洁
- 无需Provider，使用方便
- 适用场景：中小型应用、快速开发

对于中型电商管理系统，我推荐 Zustand：
```javascript
import create from 'zustand'

const useStore = create((set) => ({
  products: [],
  cart: [],
  addProduct: (product) => set((state) => ({
    products: [...state.products, product]
  }))
}))
```""",
        "estimated_tokens": 350,
    },
    {
        "role": "user",
        "content": """Zustand听起来不错，那它能和React配合使用吗？
有没有关于持久化存储的方案？比如把状态保存到localStorage
或者IndexedDB里，这样用户刷新页面后数据还在。""",
        "estimated_tokens": 110,
    },
    {
        "role": "assistant",
        "content": """Zustand 和 React 配合非常简单，而且有官方的持久化中间件：

1. 基础 React 集成
```javascript
import create from 'zustand'

const useStore = create((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 }))
}))

// 在组件中使用
function Counter() {
  const { count, increment } = useStore()
  return <button onClick={increment}>{count}</button>
}
```

2. localStorage 持久化
```javascript
import { persist } from 'zustand/middleware'

const useStore = create(
  persist(
    (set) => ({
      user: null,
      login: (user) => set({ user }),
      logout: () => set({ user: null })
    }),
    { name: 'user-storage' } // localStorage key
  )
)
```

3. IndexedDB 持久化（大数据量）
```javascript
import { persist } from 'zustand/middleware'

const useStore = create(
  persist(
    (set) => ({ /* ... */ }),
    {
      name: 'large-storage',
      storage: {
        getItem: (name) => {
          // 从 IndexedDB 读取
          return getFromIndexedDB(name)
        },
        setItem: (name, value) => {
          // 写入 IndexedDB
          saveToIndexedDB(name, value)
        }
      }
    }
  )
)
```""",
        "estimated_tokens": 420,
    },
]

# ========== 话题 3: 游戏开发（较远话题，相似度 0.3-0.6） ==========

GAME_DEVELOPMENT_CONVERSATION = [
    {
        "role": "user",
        "content": """我想用Unity做一款2D平台跳跃游戏，
需要了解物理引擎的设置和碰撞检测的实现方法。
玩家角色需要能够跳跃、攀墙、还有二段跳能力，
请问我该如何配置Rigidbody2D和Collider？""",
        "estimated_tokens": 110,
    },
    {
        "role": "assistant",
        "content": """Unity 2D平台游戏物理设置：

1. Rigidbody2D配置
- Gravity Scale: 2.5-3（调整跳跃手感）
- Linear Drag: 0（空中不减速）
- Angular Drag: 0（禁止旋转）
- Collision Detection: Continuous（防止穿透）

2. Collider2D设置
- 使用BoxCollider2D作为主角碰撞体
- 设置Material为ZeroFriction防止墙面摩擦
- 勾选Used By Effector以配合 Effector2D

3. 二段跳实现
```csharp
public class PlayerController : MonoBehaviour
{
    public int maxJumps = 2;
    private int jumpCount;
    private Rigidbody2D rb;

    void Update() {
        if (IsGrounded()) jumpCount = 0;
        if (Input.GetKeyDown(KeyCode.Space) && jumpCount < maxJumps) {
            rb.velocity = Vector2.up * jumpForce;
            jumpCount++;
        }
    }
}
```""",
        "estimated_tokens": 340,
    },
    {
        "role": "user",
        "content": """关于角色移动，我想实现平滑的加速和减速效果，
而不是瞬间达到最大速度。另外，如何在Unity中制作
可破坏的地形？比如砖块被打碎后消失。""",
        "estimated_tokens": 95,
    },
    {
        "role": "assistant",
        "content": """1. 平滑移动实现

使用Mathf.Lerp或Vector2.Lerp实现平滑加速：
```csharp
public float acceleration = 10f;
public float maxSpeed = 5f;

void FixedUpdate() {
    float input = Input.GetAxis("Horizontal");
    float targetSpeed = input * maxSpeed;

    // 平滑加速
    currentSpeed = Mathf.Lerp(currentSpeed, targetSpeed, acceleration * Time.deltaTime);
    rb.velocity = new Vector2(currentSpeed, rb.velocity.y);
}
```

2. 可破坏地形实现

方法一：Tilemap + Tile Exchange
```csharp
// 碰撞时交换Tile
private void OnCollisionEnter2D(Collision2D collision) {
    Vector3 hitPosition = Vector3.zero;
    foreach (ContactPoint2D hit in collision.contacts) {
        hitPosition.x = hit.point.x - 0.01f * hit.normal.x;
        hitPosition.y = hit.point.y - 0.01f * hit.normal.y;
        tilemap.SetTile(tilemap.WorldToCell(hitPosition), null);
    }
}
```

方法二：可破坏预制体
```csharp
public class DestructibleBlock : MonoBehaviour {
    public GameObject debrisPrefab;

    public void DestroyBlock() {
        Instantiate(debrisPrefab, transform.position, Quaternion.identity);
        Destroy(gameObject);
    }
}
```""",
        "estimated_tokens": 380,
    },
]

# ========== 话题 4: 烹饪食谱（完全不相关话题，相似度 < 0.3） ==========

COOKING_RECIPE_CONVERSATION = [
    {
        "role": "user",
        "content": """请教一道家常菜——红烧茄子的做法。
我想要那种茄子软烂入味、表皮微微焦香的效果，
不知道是先炸还是先蒸？调料方面生抽和老抽怎么配比？""",
        "estimated_tokens": 95,
    },
    {
        "role": "assistant",
        "content": """红烧茄子的最佳做法：

【食材准备】
- 长茄子2根（去皮切滚刀块）
- 蒜末、葱花、小米辣
- 生抽2勺、老抽半勺、蚝油1勺、糖1小勺

【关键步骤】
1. 茄子先撒盐腌制10分钟，挤出水分
2. 少油煎炸至表面金黄（不需要大量油炸）
3. 留底油爆香蒜末和小米辣

【调味顺序】
- 茄子回锅
- 加入生抽+老抽+蚝油+糖
- 少量清水焖煮2分钟
- 大火收汁撒葱花

【秘诀】
- 腌制挤出水分可以减少吸油
- 煎炸比蒸的口感更香
- 老抽只用一点上色，多了会发苦""",
        "estimated_tokens": 280,
    },
    {
        "role": "user",
        "content": """"茄子怎么做才不变黑？我每次做茄子都发黑，
看着就没食欲。还有，除了红烧茄子，
你有没有其他茄子的做法推荐？比如鱼香茄子？""",
        "estimated_tokens": 85,
    },
    {
        "role": "assistant",
        "content": """茄子不变黑的秘诀：

1. 腌制后挤水（最重要）
2. 炸前裹薄淀粉
3. 用深色容器盛装
4. 做好趁热吃

【鱼香茄子做法】

食材：茄子、肉丝、葱蒜姜、泡椒

鱼香汁配方：
- 糖1勺、醋1勺、生抽1勺
- 老抽少许、淀粉水适量

步骤：
1. 茄子切条，撒盐腌10分钟后挤干
2. 裹淀粉炸至金黄捞出
3. 炒香肉丝、蒜姜、泡椒
4. 倒入鱼香汁煮开
5. 放入茄子快速翻炒即可

【其他茄子做法】

蒜蓉蒸茄子：
- 茄子切条蒸熟
- 蒜蓉爆香，加生抽、蚝油
- 淋在茄子上，撒葱花

肉末茄子：
- 先炒肉末出油
- 加入茄子丁炒软
- 调味后焖煮5分钟""",
        "estimated_tokens": 320,
    },
]

# ========== 边界测试数据 ==========

# 低于阈值（< 50 tokens）
BELOW_THRESHOLD_TEXT = "这是一个测试文本，用于验证阈值边界的判定。" * 9  # 约 180 字符

# 刚好在阈值上（≈ 50 tokens）
AT_THRESHOLD_TEXT = "这是一个测试文本，用于验证阈值边界的判定。" * 10  # 约 200 字符

# 高于阈值（> 50 tokens）
ABOVE_THRESHOLD_TEXT = "这是一个测试文本，用于验证阈值边界的判定。" * 11  # 约 220 字符

# ========== Token 溢出测试长对话 ==========

# 生成 20 轮长对话用于触发 Token 溢出（测试时阈值设为 2048）
LONG_CONVERSATION_FOR_OVERFLOW = []
for i in range(20):
    LONG_CONVERSATION_FOR_OVERFLOW.extend([
        {
            "role": "user",
            "content": f"""这是第{i+1}轮对话。我想深入了解Python的高级特性，
包括装饰器、生成器、上下文管理器、元类等内容。
请详细讲解它们的原理和使用场景，
并给出实际的代码示例帮助我理解。

我想了解装饰器是如何在不修改原函数代码的情况下，
为函数添加额外功能的。生成器是如何实现惰性求值的，
它和普通函数有什么区别。上下文管理器的with语句
是如何自动管理资源的。元类又是如何动态修改类定义的。

这些都是Python中非常强大的特性，
掌握它们可以让我的代码更加简洁、高效、易维护。""" * 2,
            "estimated_tokens": 250,
        },
        {
            "role": "assistant",
            "content": f"""很高兴为您解答。关于Python高级特性的讲解：

装饰器（Decorator）本质上是一个函数，它接受一个函数作为参数，
返回一个新的函数。装饰器可以在不修改原函数代码的情况下，
为函数添加额外的功能，如日志记录、性能计时、权限验证等。

装饰器的实现原理：
1. 定义一个接受函数作为参数的函数
2. 在内部定义一个包装函数
3. 包装函数中调用原函数并添加额外逻辑
4. 返回包装函数

生成器（Generator）是一种特殊的迭代器，使用yield关键字
可以暂停函数执行并保存状态，下次调用时从上次位置继续。
生成器具有惰性求值的特点，可以节省内存。

生成器的优势：
1. 惰性计算，只在需要时才生成值
2. 节省内存，不需要一次性生成所有值
3. 可以表示无限序列

上下文管理器（Context Manager）使用with语句，
可以自动管理资源的获取和释放，如文件操作、锁的获取等。
实现上下文管理器需要定义__enter__和__exit__方法。

元类（Metaclass）是创建类的类，可以拦截类的创建过程，
实现动态修改类定义的功能。元类是Python中最强大的特性之一，
但也最容易滥用，建议谨慎使用。

代码示例：
```python
# 装饰器示例
def timing_decorator(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{{func.__name__}} took {{time.time()-start:.2f}}s")
        return result
    return wrapper

# 生成器示例
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 上下文管理器示例
class FileManager:
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
```

这些特性在实际开发中非常实用，掌握它们
可以让你的Python代码更加优雅和高效。""" * 2,
            "estimated_tokens": 550,
        },
    ])

# ========== Agent 工具调用测试场景 ==========

# 注意：由于 PatchouliAgent.add_message() API 限制，
# 工具调用信息需要通过特殊方式传递
# 这里提供 OpenAI 格式的原始消息，供底层测试使用

AGENT_TOOL_CALL_SCENARIO = {
    "description": "单工具调用场景：查询天气并给出建议",
    "messages": [
        {
            "role": "user",
            "content": "帮我查询一下北京的天气，如果温度低于10度，推荐一些保暖的穿搭建议。",
        },
        {
            "role": "assistant",
            "content": "我需要先查询北京的天气信息，然后根据温度推荐合适的穿搭。",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "北京"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature": 5, "condition": "晴", "humidity": 45}'
        },
        {
            "role": "assistant",
            "content": """北京目前气温5度，天气晴朗，湿度45%。由于温度较低，建议您穿着保暖内衣、毛衣或厚卫衣，外加羽绒服或厚棉衣。下身建议穿保暖裤或加绒裤子，脚部穿保暖袜子和保暖鞋。外出时记得戴帽子和手套保护手脚。""",
        }
    ]
}

AGENT_MULTI_TOOL_SCENARIO = {
    "description": "多工具调用场景：分析代码复杂度并给出优化建议",
    "messages": [
        {
            "role": "user",
            "content": "帮我分析这段Python代码的时间复杂度，然后给出优化建议。代码是一个双重循环的列表查找函数。",
        },
        {
            "role": "assistant",
            "content": "我需要先使用代码分析工具来计算时间复杂度。",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "analyze_complexity",
                    "arguments": '{"code": "def find_duplicates(lst):\\n    result = []\\n    for i in range(len(lst)):\\n        for j in range(i+1, len(lst)):\\n            if lst[i] == lst[j]:\\n                result.append(lst[i])\\n    return result"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"time_complexity": "O(n^2)", "space_complexity": "O(n)", "hotspots": ["nested_loop"]}'
        },
        {
            "role": "assistant",
            "content": "根据分析，这段代码的时间复杂度是O(n²)，主要瓶颈在嵌套循环。现在让我获取优化建议。",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "suggest_optimization",
                    "arguments": '{"time_complexity": "O(n^2)", "pattern": "nested_search"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"suggestion": "使用哈希表（集合）可以将查找复杂度从O(n)降低到O(1)", "optimized_code": "def find_duplicates(lst):\\n    seen = set()\\n    return [x for x in lst if x in seen or seen.add(x)]"}'
        },
        {
            "role": "assistant",
            "content": """经过分析，这段代码的时间复杂度为O(n²)，空间复杂度为O(n)。主要性能瓶颈在于嵌套循环中的重复查找操作。

优化建议：使用哈希表（集合）存储已见过的元素，可以将查找操作的复杂度从O(n)降低到O(1)，从而将整体时间复杂度优化到O(n)。

优化后的代码：
```python
def find_duplicates(lst):
    seen = set()
    return [x for x in lst if x in seen or seen.add(x)]
```

这样只需要遍历列表一次，效率大幅提升。""",
        }
    ]
}

# ========== 话题距离测试场景配置 ==========

TOPIC_DISTANCE_TEST_SCENARIOS = {
    "far": {
        "name": "远距离话题（数据科学 -> 烹饪）",
        "base_topic": DATA_SCIENCE_CONVERSATION[:2],
        "new_topic": COOKING_RECIPE_CONVERSATION,
        "expected_similarity": "< 0.3",
        "expected_flush": True,
        "expected_reason": "SEMANTIC_DRIFT",
    },
    "medium_far": {
        "name": "较远话题（数据科学 -> 游戏开发）",
        "base_topic": DATA_SCIENCE_CONVERSATION[:2],
        "new_topic": GAME_DEVELOPMENT_CONVERSATION,
        "expected_similarity": "0.3 - 0.6",
        "expected_flush": True,
        "expected_reason": "SEMANTIC_DRIFT",
    },
    "near": {
        "name": "较近话题（数据科学 -> Web开发）",
        "base_topic": DATA_SCIENCE_CONVERSATION[:2],
        "new_topic": WEB_DEVELOPMENT_CONVERSATION,
        "expected_similarity": "0.6 - 0.8",
        "expected_flush": False,
        "expected_reason": None,
    },
    "same": {
        "name": "相同话题（数据科学连续对话）",
        "base_topic": DATA_SCIENCE_CONVERSATION[:2],
        "new_topic": DATA_SCIENCE_CONVERSATION[2:4],
        "expected_similarity": ">= 0.8",
        "expected_flush": False,
        "expected_reason": None,
    },
}


# ========== 导出 ==========
__all__ = [
    # 话题对话数据
    "DATA_SCIENCE_CONVERSATION",
    "WEB_DEVELOPMENT_CONVERSATION",
    "GAME_DEVELOPMENT_CONVERSATION",
    "COOKING_RECIPE_CONVERSATION",
    # 边界测试数据
    "BELOW_THRESHOLD_TEXT",
    "AT_THRESHOLD_TEXT",
    "ABOVE_THRESHOLD_TEXT",
    # Token 溢出测试数据
    "LONG_CONVERSATION_FOR_OVERFLOW",
    # Agent 工具调用场景
    "AGENT_TOOL_CALL_SCENARIO",
    "AGENT_MULTI_TOOL_SCENARIO",
    # 话题距离测试配置
    "TOPIC_DISTANCE_TEST_SCENARIOS",
    # 辅助函数
    "estimate_tokens",
]
