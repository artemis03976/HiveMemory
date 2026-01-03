"""
HiveMemory 感知层测试数据 Fixtures

提供真实长度的测试对话数据，避免因短消息频繁触发短文本强吸附。

设计原则:
    - 用户消息: 60-150 tokens (约 240-600 中文字符)
    - 助手回复: 200-500 tokens (约 800-2000 中文字符)
    - 确保消息超过短文本阈值 (50 tokens) 以测试语义��似度判定

作者: HiveMemory Team
版本: 1.0.0
"""

from typing import List, Dict, Any, Optional

# ========== 话题 1: Python 编程（连续话题，用于测试语义吸附） ==========

PYTHON_CONVERSATION = [
    {
        "role": "user",
        "content": """请帮我详细讲解一下Python中的装饰器模式，包括它的语法糖用法、带参数的装饰器、
以及类装饰器的实现方式。我想了解装饰器的实际应用场景，比如日志记录、
性能测试、权限验证等。最好能有具体的代码示例。""",
        "estimated_tokens": 120,
    },
    {
        "role": "assistant",
        "content": """Python装饰器是一个强大的功能，它允许你在不修改函数代码的情况下增强函数的功能。
���饰器本质上是一个接受函数作为参数并返回新函数的高阶函数。

一、基础装饰器语法
使用@符号作为语法糖，相当于my_func = decorator(my_func)。这种方式让代码更加简洁和可读。

二、带参数的装饰器
通过三层嵌套函数实现，外层函数接收参数，中间层返回装饰器，内层是包装函数。
这样可以让装饰器接受自定义配置参数。

三、类装饰器
使用类的__call__方法实现，类实例可以像函数一样被调用。这种方式适合需要维护状态的装饰器。

四、实际应用场景
1. 日志记录：自动记录函数调用信息、参数、返回值和执行时间
2. 性能测试：计算函数执行时间，用于性能分析
3. 权限验证：检查用户是否有权限执行某些操作
4. 缓存：缓存函数结果避免重复计算（如functools.lru_cache）
5. 重试机制：自动重试失败的函数

下面是具体的代码示例...""",
        "estimated_tokens": 400,
    },
    {
        "role": "user",
        "content": """那你能给我写一个具体的装饰器例子吗？比如用于计算函数执行时间的装饰器，
并且要支持异步函数。我想知道如何在装饰器中获取被装饰函数的参数和返回值，
以及如何处理函数抛出异常的情况。""",
        "estimated_tokens": 100,
    },
    {
        "role": "assistant",
        "content": """好的，我来写一个支持同步和异步函数的计时装饰器。这个装饰器需要能够：
1. 检测被装饰的函数是同步还是异步
2. 正确处理函数的参数和返回值
3. 捕获并记录异常
4. 使用functools.wraps保留原函数的元数据

代码实现如下：

```python
import asyncio
import functools
import time
from typing import Callable, Any

def timer(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper_sync(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")

    @functools.wraps(func)
    async def wrapper_async(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")

    if asyncio.iscoroutinefunction(func):
        return wrapper_async
    else:
        return wrapper_sync
```

关于参数和返回值：装饰器通过*args和**kwargs传递所有参数，函数的返回值会被捕获并原样返回。
对于异常处理，我们在finally块中记录时间，确保无论是否抛出异常都能正确计时。""",
        "estimated_tokens": 350,
    },
    {
        "role": "user",
        "content": """这个异步装饰器怎么处理函数的参数和返回值？如果被装饰的函数抛出异常怎么办？
另外，能不能给我展示一下如何在装饰器中添加日志记录功能？""",
        "estimated_tokens": 85,
    },
    {
        "role": "assistant",
        "content": """关于参数和返回值处理：装饰器通过*args和**kwargs透传所有参数，使用asyncio.iscoroutinefunction
来判断函数类型，返回值直接return即可。如果是异步函数，使用await等待结果。

异常处理方面，我们可以捕获异常并记录：

```python
def timer_with_logging(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper_async(*args, **kwargs):
        start_time = time.perf_counter()
        logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            end_time = time.perf_counter()
            logger.info(f"{func.__name__} completed in {end_time - start_time:.4f}s")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            logger.error(f"{func.__name__} failed after {end_time - start_time:.4f}s: {e}")
            raise  # 重新抛出异常
```

这样既记录了调用信息，又不会改变函数的异常行为。""",
        "estimated_tokens": 300,
    },
]

# ========== 话题 2: 机器学习（用于测试语义漂移） ==========

ML_CONVERSATION = [
    {
        "role": "user",
        "content": """我现在想了解机器学习中的梯度下降算法。请详细讲解批量梯度下降、
随机梯度下降和小批量梯度下降的区别和各自的优缺点。
我想知道在不同场景下如何选择合适的梯度下降变体，
以及如何处理学习率调整的问题。""",
        "estimated_tokens": 110,
    },
    {
        "role": "assistant",
        "content": """梯度下降是优化神经网络最常用的算法之一。让我详细比较这三种变体：

一、批量梯度下降（Batch Gradient Descent）
- 每次迭代使用所有训练样本计算梯度
- 优点：梯度估计稳定，收敛轨迹平滑，保证收敛到凸优化问题的全局最优
- 缺点：计算成本高，大数据集无法使用，内存占用大
- 适用于：小数据集，凸优化问题

二、随机梯度下降（Stochastic Gradient Descent）
- 每次迭代只使用一个样本计算梯度
- 优点：更新速度快，可在线学习，跳出局部最优的能力强
- 缺点：梯度噪声大，收敛轨迹震荡，难以稳定收敛
- 适用于：大规模数据集，在线学习场景

三、小批量梯度下降（Mini-Batch Gradient Descent）
- 每次迭代使用一小批样本（如32、64、128）
- 综合了BGD和SGD的优点，是深度学习中最常用的方法
- 可以利用矩阵运算加速，同时保持一定的梯度稳定性

学习率调整策略：
1. 学习率衰减：随训练逐渐降低，如指数衰减、步进衰减
2. AdaGrad：自适应学习率，对频繁参数使用较小学习率
3. RMSprop：引入动量项，解决AdaGrad学习率递减过快问题
4. Adam：结合动量和自适应学习率，是最常用的优化器

实际使用中，Adam通常是首选，因为它对超参数不太敏感，收敛快且稳定。""",
        "estimated_tokens": 450,
    },
]

# ========== 话题 3: 烹饪（完全不相关话题） ==========

COOKING_CONVERSATION = [
    {
        "role": "user",
        "content": """我想学习做红烧肉，请告诉我详细的步骤和配料。包括如何让肉色红亮、
肥而不腻的技巧，以及不同地区的做法差异。
我想知道用冰糖还是白糖更好，炒糖色的火候如何掌握。""",
        "estimated_tokens": 95,
    },
    {
        "role": "assistant",
        "content": """红烧肉是一道经典的中式菜肴，不同地区有不同的做法。下面是详细步骤：

一、食材准备
- 五花肉500克（肥瘦相间最好，三层肉最佳）
- 冰糖30克（冰糖比白糖颜色更亮，口感更好）
- 生抽3勺、老抽1勺、料酒2勺
- 葱2根、姜3片、八角2个、桂皮1小段
- 香叶2片（可选）

二、制作步骤
1. 肉切块，冷水下锅焯水去血沫，捞出沥干
2. 热锅放少许油，下冰糖小火炒至枣红色（关键步骤！）
3. 下肉块快速翻炒上色，每块都要裹上糖色
4. 加葱姜八角桂皮爆香
5. 加料酒、生抽、老抽调色调味
6. 加开水没过肉块（一定要开水！），大火烧开转小火
7. 炖煮40-50分钟至软烂
8. 大火收汁即可，收汁时勤翻动避免糊锅

三、技巧要点
- 炒糖色要用小火，耐心炒到枣红色，不要炒焦会有苦味
- 加水一定要加开水，冷水会让肉突然收缩变柴
- 最后收汁要勤翻动，让每一块肉都裹上汤汁
- 可以加少许醋，帮助肉质软烂

不同地区做法：湖南加辣椒，上海偏甜，北方加豆腐结一起炖。""",
        "estimated_tokens": 350,
    },
]

# ========== 边界测试数据 ==========

# 刚好在阈值上（50 tokens ≈ 200 中文字符）
AT_THRESHOLD_TEXT = "这是一个测试文本，用于验证阈值边界的判定。" * 10  # 约 200 字符

# 低于阈值（49 tokens ≈ 196 中文字符）
BELOW_THRESHOLD_TEXT = "这是一个测试文本，用于验证阈值边界的判定。" * 9  # 约 180 字符

# 高于阈值（51 tokens ≈ 204 中文字符）
ABOVE_THRESHOLD_TEXT = "这是一个测试文本，用于验证阈值边界的判定。" * 11  # 约 220 字符

# ========== Token 溢出测试用长文本 ==========

LONG_TEXT_BLOCK = """这是一段用于测试Token溢出的长文本。在实际应用中，Agent 可能需要处理非常长的对话内容。
当我们处理长对话时，需要确保Token计数和溢出检测机制能够正确工作。这段文本需要足够长，以便在累积多个块后能够触发Token溢出阈值。

Python 是一种高级编程语言，具有简洁明了的语法。它支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。Python 的设计哲学强调代码的可读性和简洁的语法（尤其是使用空格缩进划分代码块，而非使用大括号或关键字）。相比于 C++ 或 Java，Python 让开发者能够用更少的代码行表达概念。

在数据处理和科学计算领域，Python 有丰富的库支持。NumPy 提供了对大型多维数组和矩阵的支持，以及数学函数库。Pandas 提供了数据分析和数据操作的强大工具。Matplotlib 和 Seaborn 用于数据可视化。SciPy 则提供了科学计算的工具集。

在 Web 开发方面，Django 和 Flask 是两个流行的 Web 框架。Django 是一个全功能的 Web 框架，包含了 ORM、模板引擎、表单处理等完整的功能。Flask 则是一个微框架，更加灵活，开发者可以根据需要选择各种扩展。

在机器学习和人工智能领域，Python 是最流行的编程语言。TensorFlow、PyTorch、Scikit-learn 等库为开发者提供了强大的工具。这些库使得开发者能够快速构建和训练神经网络模型。

在自动化和脚本编写方面，Python 也是首选。它的语法简洁，标准库丰富，使得开发者能够快速完成各种自动化任务。""" * 5  # 重复5次以获得足够的长度

# ========== Tool Call 测试场景 ==========

TOOL_CALL_SCENARIOS = {
    "single_tool": {
        "user_query": "帮我查询一下北京的天气，如果温度低于10度，推荐一些保暖的穿搭建议。",
        "thought": "用户需要两个步骤：首先获取北京的天气信息，然后根据温度推荐合适的穿搭",
        "tool_name": "get_weather",
        "tool_args": {"location": "北京"},
        "tool_output": '{"temperature": 5, "condition": "晴", "humidity": 45}',
        "assistant_response": "北京目前气温5度，天气晴朗，湿度45%。由于温度较低，建议您穿着保暖内衣、毛衣或厚卫衣，外加羽绒服或厚棉衣。下身建议穿保暖裤或加绒裤子，脚部穿保暖袜子和保暖鞋。外出时记得戴帽子和手套保护手脚。",
    },
    "multi_tool": {
        "user_query": "帮我分析这段代码的时间复杂度，然后给出优化建议",
        "steps": [
            {
                "thought": "首先需要使用代码分析工具计算时间复杂度",
                "tool_name": "analyze_complexity",
                "tool_output": '{"time_complexity": "O(n^2)", "space_complexity": "O(n)", "hotspots": ["nested_loop"]}'
            },
            {
                "thought": "接下来使用优化工具给出改进建议",
                "tool_name": "suggest_optimization",
                "tool_output": '{"suggestion": "使用哈希表可以将查找复杂度从O(n)降低到O(1)", "optimized_code": "..."}'
            }
        ],
        "assistant_response": "经过分析，这段代码的时间复杂度为O(n²)，空间复杂度为O(n)。主要性能瓶颈在于嵌套循环中的重复查找操作。优化建议：使用哈希表（字典）存储中间结果，可以将查找操作的复杂度从O(n)降低到O(1)，从而将整体时间复杂度优化到O(n)。具体的优化代码已经生成，您可以参考上面工具返回的建议进行修改。",
    },
}

# ========== 辅助函数 ==========

def create_stream_message(
    message_type: str,
    content: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[Dict[str, Any]] = None,
    tool_result: Optional[str] = None,
) -> Dict[str, Any]:
    """
    创建 StreamMessage 所需的数据字典

    Args:
        message_type: 消息类型
        content: 消息内容
        tool_name: 工具名称（可选）
        tool_args: 工具参数（可选）
        tool_result: 工具结果（可选）

    Returns:
        Dict: 消息数据字典
    """
    data = {
        "message_type": message_type,
        "content": content,
    }
    if tool_name:
        data["tool_name"] = tool_name
    if tool_args:
        data["tool_args"] = tool_args
    if tool_result:
        data["tool_result"] = tool_result
    return data


def get_conversation_tokens(conversation: List[Dict[str, Any]]) -> int:
    """计算对话的总 Token 数"""
    return sum(msg.get("estimated_tokens", estimate_tokens(msg["content"])) for msg in conversation)


# ========== 导出 ==========

__all__ = [
    # 对话数据
    "PYTHON_CONVERSATION",
    "ML_CONVERSATION",
    "COOKING_CONVERSATION",
    # 边界测试数据
    "AT_THRESHOLD_TEXT",
    "BELOW_THRESHOLD_TEXT",
    "ABOVE_THRESHOLD_TEXT",
    "LONG_TEXT_BLOCK",
    # Tool Call 场景
    "TOOL_CALL_SCENARIOS",
    # 辅助函数
    "estimate_tokens",
    "create_stream_message",
    "get_conversation_tokens",
]
