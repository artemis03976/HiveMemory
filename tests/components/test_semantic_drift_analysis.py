"""
HiveMemory Semantic Drift Analysis Test

功能：
1. 加载 perception_test_data.py 中的测试对话数据
2. 使用真实的 Embedding 和 Reranker 模型计算相似度
3. 模拟感知层的语义吸附与漂移检测逻辑
4. 输出详细的相似度报告，辅助阈值调优

使用模型：
- Embedding: perception 配置指定的模型 (默认 all-MiniLM-L6-v2)
- Reranker: retrieval 配置指定的模型 (默认 bge-reranker-v2-m3)

运行方式：
    pytest tests/components/test_semantic_drift_analysis.py -s
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pytest
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 导入配置和基础设施
from hivememory.patchouli.config import load_app_config, SemanticAdsorberConfig
from hivememory.infrastructure.embedding import get_perception_embedding_service
from hivememory.infrastructure.rerank import get_flag_reranker_service

# 导入测试数据
from tests.fixtures.perception_test_data import (
    DATA_SCIENCE_CONVERSATION,
    WEB_DEVELOPMENT_CONVERSATION,
    GAME_DEVELOPMENT_CONVERSATION,
    COOKING_RECIPE_CONVERSATION,
)

console = Console(record=True)

class TestSemanticAnalyzer:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self, request):
        # 加载配置
        app_config = load_app_config()
        adsorber_config = app_config.perception.engine.adsorber
        
        # 初始化 Embedding 服务
        console.print("[cyan]正在加载 Embedding 模型...[/cyan]")
        embedding_service = get_perception_embedding_service(
            app_config.embedding.perception
        )
        
        # 初始化 Reranker 服务
        console.print("[cyan]正在加载 Reranker 模型...[/cyan]")
        reranker_service = get_flag_reranker_service(
            app_config.retrieval.retriever.reranker
        )
        
        # 绑定到类，使所有实例可用
        request.cls.app_config = app_config
        request.cls.adsorber_config = adsorber_config
        request.cls.embedding_service = embedding_service
        request.cls.reranker_service = reranker_service
        request.cls.ema_alpha = adsorber_config.ema_alpha

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的 Embedding 余弦相似度"""
        vec1 = self.embedding_service.encode(text1)
        vec2 = self.embedding_service.encode(text2)
        return self.embedding_service.compute_cosine_similarity(vec1, vec2)

    def calculate_rerank_score(self, query: str, doc: str) -> float:
        """计算 Reranker 分数"""
        # FlagRerankerService.compute_score 接受 List[List[str]]，返回 List[float]
        # 注意：第二个参数是 batch_size，不能传字符串
        scores = self.reranker_service.compute_score([[query, doc]])
        return scores[0] if scores else 0.0

    def analyze_conversation_group(self, name: str, conversation: List[Dict[str, Any]]):
        """分析组内对话的连贯性（模拟 EMA 漂移检测）"""
        console.print(Panel(f"[bold green]组内分析: {name}[/bold green]", expand=False))
        
        # Write to report file
        report_path = r"C:\Users\29305\Projects\HiveMemory\docs\semantic_drift_data_report.md"
        print(f"Writing report to {report_path}")
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n### 组内分析: {name}\n\n")
            f.write("| 轮次 | Query (Rewritten) | 上一轮 Query | Embedding Sim | Reranker Score | EMA Sim |\n")
            f.write("|---|---|---|---|---|---|\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("轮次", style="dim", width=6)
        table.add_column("Query (Rewritten)", width=40)
        table.add_column("上一轮 Query", width=40)
        table.add_column("Embedding Sim", justify="right")
        table.add_column("Reranker Score", justify="right")
        table.add_column("EMA Sim", justify="right")
        
        ema_sim = 1.0  # 初始 EMA
        prev_query = None
        
        queries = []
        for msg in conversation:
            if msg["role"] == "user":
                query = msg.get("rewritten_query") or msg["content"]
                queries.append(query)

        for i, query in enumerate(queries):
            emb_sim = 0.0
            rerank_score = 0.0
            
            if prev_query:
                emb_sim = self.calculate_similarity(prev_query, query)
                rerank_score = self.calculate_rerank_score(query, prev_query)
                
                # 更新 EMA (模拟 SemanticBoundaryAdsorber 逻辑)
                ema_sim = (self.ema_alpha * emb_sim) + ((1 - self.ema_alpha) * ema_sim)
            
            # Add to rich table
            table.add_row(
                f"T{i+1}",
                query[:38] + "..." if len(query) > 38 else query,
                (prev_query[:38] + "..." if len(prev_query) > 38 else prev_query) if prev_query else "N/A",
                f"{emb_sim:.4f}" if prev_query else "-",
                f"{rerank_score:.4f}" if prev_query else "-",
                f"{ema_sim:.4f}",
            )
            
            # Write to report file
            report_path = r"C:\Users\29305\Projects\HiveMemory\docs\semantic_drift_data_report.md"
            with open(report_path, "a", encoding="utf-8") as f:
                prev_q_str = prev_query.replace("\n", " ") if prev_query else "N/A"
                curr_q_str = query.replace("\n", " ")
                emb_str = f"{emb_sim:.4f}" if prev_query else "-"
                rerank_str = f"{rerank_score:.4f}" if prev_query else "-"
                f.write(f"| T{i+1} | {curr_q_str} | {prev_q_str} | {emb_str} | {rerank_str} | {ema_sim:.4f} |\n")

            prev_query = query
            
        console.print(table)
        console.print("\n")

    def analyze_inter_group_similarity(self, group1_name: str, group1: List[Dict], group2_name: str, group2: List[Dict]):
        """分析组间话题切换的相似度"""
        console.print(Panel(f"[bold yellow]组间切换分析: {group1_name} -> {group2_name}[/bold yellow]", expand=False))
        
        # 获取 Group 1 最后一个 Query
        last_query_g1 = None
        for msg in reversed(group1):
            if msg["role"] == "user":
                last_query_g1 = msg.get("rewritten_query") or msg["content"]
                break
                
        # 获取 Group 2 第一个 Query
        first_query_g2 = None
        for msg in group2:
            if msg["role"] == "user":
                first_query_g2 = msg.get("rewritten_query") or msg["content"]
                break
                
        if not last_query_g1 or not first_query_g2:
            console.print("[red]无法提取 Query 进行比较[/red]")
            return

        emb_sim = self.calculate_similarity(last_query_g1, first_query_g2)
        rerank_score = self.calculate_rerank_score(first_query_g2, last_query_g1)
        
        # 判定建议
        high_threshold = self.adsorber_config.semantic_threshold_high
        low_threshold = self.adsorber_config.semantic_threshold_low
        
        if emb_sim >= high_threshold:
            verdict = "强吸附 (Adsorb)"
        elif emb_sim < low_threshold:
            verdict = "强制漂移 (Drift)"
        else:
            verdict = "灰色区域 (Grey Area)"

        table = Table(show_header=True)
        table.add_column("类型", style="bold")
        table.add_column("内容")
        table.add_row("Group 1 Last Query", last_query_g1)
        table.add_row("Group 2 First Query", first_query_g2)
        table.add_row("Embedding Similarity", f"[bold cyan]{emb_sim:.4f}[/bold cyan]")
        table.add_row("Reranker Score", f"[bold cyan]{rerank_score:.4f}[/bold cyan]")
        
        console.print(table)
        console.print(f"当前阈值判定: {verdict} (High: {high_threshold}, Low: {low_threshold})")
        console.print("\n")
        
        # Write to report file
        report_path = r"C:\Users\29305\Projects\HiveMemory\docs\semantic_drift_data_report.md"
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n### 组间切换分析: {group1_name} -> {group2_name}\n\n")
            f.write(f"- **Group 1 Last Query**: {last_query_g1}\n")
            f.write(f"- **Group 2 First Query**: {first_query_g2}\n")
            f.write(f"- **Embedding Similarity**: {emb_sim:.4f}\n")
            f.write(f"- **Reranker Score**: {rerank_score:.4f}\n")
            f.write(f"- **判定结果**: {verdict} (High: {high_threshold}, Low: {low_threshold})\n")

    def test_run_analysis(self):
        """运行完整分析"""
        # Clear/Create report file
        report_path = r"C:\Users\29305\Projects\HiveMemory\docs\semantic_drift_data_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 语义漂移分析数据报告\n\n")
            f.write("本报告包含主要测试对话数据的组内与组间语义相似度分析数据。\n")
        
        # 1. 组内连贯性分析
        self.analyze_conversation_group("数据科学 (基线)", DATA_SCIENCE_CONVERSATION)
        self.analyze_conversation_group("Web 开发", WEB_DEVELOPMENT_CONVERSATION)
        self.analyze_conversation_group("游戏开发", GAME_DEVELOPMENT_CONVERSATION)
        self.analyze_conversation_group("烹饪食谱", COOKING_RECIPE_CONVERSATION)
        
        # 2. 组间切换分析
        # 近距离: 数据科学 -> Web 开发
        self.analyze_inter_group_similarity(
            "数据科学", DATA_SCIENCE_CONVERSATION,
            "Web 开发", WEB_DEVELOPMENT_CONVERSATION
        )
        
        # 中距离: 数据科学 -> 游戏开发
        self.analyze_inter_group_similarity(
            "数据科学", DATA_SCIENCE_CONVERSATION,
            "游戏开发", GAME_DEVELOPMENT_CONVERSATION
        )
        
        # 远距离: 数据科学 -> 烹饪食谱
        self.analyze_inter_group_similarity(
            "数据科学", DATA_SCIENCE_CONVERSATION,
            "烹饪食谱", COOKING_RECIPE_CONVERSATION
        )
        
        # 保存报告
        console.save_text("docs/semantic_drift_data_report.txt")
