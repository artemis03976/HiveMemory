"""
HiveMemory Pipeline E2E Tests

经真实 HiveMemorySystem（真实 LLM + Qdrant）的公开进程入口验证完整链路。

包含:
    - test_agent_loop_e2e.py: 单 agent 执行循环与 MTP 工具调用
    - test_chat_run_e2e.py: chat 闭环、检索注入与 MTP WRITE 落库
    - test_sub_agent_call_e2e.py: CALL 子代理流式契约与星型拓扑约束
"""
