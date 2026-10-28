"""连接器的公共骨架，与具体 harness 无关。

骨架只要求 harness 把会话写进可读、只追加、条目按行有序的会话记录；
记录的解析与轮次切分由各 harness 的读取器实现 ``TranscriptReader``。
"""
