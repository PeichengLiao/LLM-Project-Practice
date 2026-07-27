"""
测试 Agent 解析逻辑
"""
import pytest
from unittest.mock import MagicMock
from src.agent.agent import ReActAgent, AgentStep, AgentResult


class TestAgentStepDataclass:
    def test_agent_step_creation(self):
        step = AgentStep(
            round_num=1,
            thought="需要搜索库存信息",
            action="search_documents",
            action_input="SKU12345 库存",
            observation="库存200件...",
        )
        assert step.round_num == 1
        assert step.action == "search_documents"

    def test_agent_result_creation(self):
        result = AgentResult(
            answer="SKU12345的库存是200件",
            steps=[AgentStep(round_num=1)],
            total_rounds=3,
        )
        assert "SKU12345" in result.answer
        assert result.total_rounds == 3
        assert len(result.steps) == 1


class TestReActAgentParsing:
    """测试 Agent 输出的解析逻辑"""

    def setup_method(self):
        self.agent = ReActAgent(
            llm_client=MagicMock(),
            tool_registry=MagicMock(),
            max_rounds=3,
        )

    def test_parse_final_answer(self):
        """测试解析 Final Answer 格式"""
        response = """Thought: 我现在有足够的信息回答用户问题。
Final Answer: ## SKU12345 库存分析
当前库存 200 件，可支撑 5 天销售。"""
        step = self.agent._parse_response(response, round_num=2)
        assert step.action == "Final Answer"
        assert "SKU12345" in step.thought

    def test_parse_action(self):
        """测试解析 Action 格式"""
        response = """Thought: 需要搜索SKU12345的库存信息。
Action: search_documents
Action Input: SKU12345 库存余量"""
        step = self.agent._parse_response(response, round_num=1)
        assert step.action == "search_documents"
        assert step.action_input == "SKU12345 库存余量"
        assert "SKU12345" in step.thought

    def test_parse_without_action(self):
        """测试解析无 Action 的输出"""
        response = """Thought: 让我想想怎么做。"""
        step = self.agent._parse_response(response, round_num=1)
        assert step.action == ""
        assert "想想" in step.thought

    def test_extract_final_answer(self):
        """测试提取 Final Answer"""
        response = "Final Answer: 这是最终答案。"
        result = self.agent._extract_final_answer(response)
        assert "最终答案" in result

    def test_extract_final_answer_fallback(self):
        """测试提取失败时回退"""
        response = "Thought: 这是思考内容。"
        result = self.agent._extract_final_answer(response)
        assert result is not None
        assert "思考" in result
