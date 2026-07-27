"""
测试 Agent 工具集
"""
import pytest
from src.agent.tools import Tool, ToolRegistry, create_calculator_tool


class TestTool:
    def test_tool_creation(self):
        """测试工具创建"""
        def dummy_func(query: str) -> str:
            return f"结果: {query}"

        tool = Tool(
            name="test_tool",
            description="测试工具",
            func=dummy_func,
            parameters={"query": "查询参数"},
        )
        assert tool.name == "test_tool"
        assert "test_tool" in tool.to_prompt_description()

    def test_tool_run(self):
        """测试工具执行"""
        def echo(msg: str) -> str:
            return msg

        tool = Tool(name="echo", description="回显工具", func=echo)
        result = tool.run(msg="Hello")
        assert result == "Hello"

    def test_tool_run_with_list_result(self):
        """测试列表结果格式化"""
        def get_list() -> list:
            return ["项目1", "项目2", "项目3"]

        tool = Tool(name="list_tool", description="返回列表", func=get_list)
        result = tool.run()
        assert "项目1" in result
        assert "项目2" in result

    def test_tool_run_error(self):
        """测试工具执行出错"""
        def failing_func():
            raise ValueError("测试错误")

        tool = Tool(name="fail_tool", description="会失败的工具", func=failing_func)
        result = tool.run()
        assert "出错" in result or "错误" in result


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_and_get(self):
        """测试注册和获取"""
        tool = Tool(name="my_tool", description="描述", func=lambda: "ok")
        self.registry.register(tool)
        assert self.registry.get("my_tool") == tool
        assert "my_tool" in self.registry.list_tools()

    def test_get_nonexistent(self):
        """测试获取不存在的工具"""
        with pytest.raises(ValueError):
            self.registry.get("nonexistent")

    def test_execute(self):
        """测试执行工具"""
        def greet(name: str) -> str:
            return f"你好, {name}!"

        tool = Tool(name="greet", description="问候", func=greet)
        self.registry.register(tool)
        result = self.registry.execute("greet", name="小明")
        assert result == "你好, 小明!"

    def test_all_descriptions(self):
        """测试获取所有工具描述"""
        tool1 = Tool(name="t1", description="工具1", func=lambda: "")
        tool2 = Tool(name="t2", description="工具2", func=lambda: "")
        self.registry.register(tool1)
        self.registry.register(tool2)
        desc = self.registry.get_all_descriptions()
        assert "t1" in desc
        assert "t2" in desc


class TestCalculatorTool:
    def setup_method(self):
        self.tool = create_calculator_tool()

    def test_basic_calc(self):
        """测试基本计算"""
        result = self.tool.run(expression="100 + 50")
        assert "150" in result

    def test_complex_calc(self):
        """测试复杂表达式"""
        result = self.tool.run(expression="(200 - 50) / 30")
        assert "5" in result

    def test_invalid_expression(self):
        """测试非法表达式"""
        result = self.tool.run(expression="__import__('os').system('ls')")
        assert "错误" in result or "不允许" in result
