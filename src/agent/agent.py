"""
ReAct Agent 实现

=== 大白话解释 ===
这是整个项目中最"智能"的部分。Agent 不是简单地"你问我答"，
而是一个"会思考 + 会动手 + 会自我纠正"的 AI。

## ReAct 模式原理

ReAct = Reasoning（推理）+ Acting（行动）

Agent 的每轮循环做三件事：
1. Thought（思考）: 我现在知道什么？还需要知道什么？
2. Action（行动）: 调用哪个工具？传什么参数？
3. Observation（观察）: 工具返回了什么？这对回答有什么帮助？

每轮循环 Agent 都会看到之前所有的 Thought → Action → Observation 记录，
这叫做"上下文窗口"。Agent 是带着"记忆"在工作的。

## 一个完整的 Agent 运行过程示例

用户问: "我们仓库里 SKU12345 的库存够不够？如果不够，推荐哪个替代品？"

Round 1:
  Thought: 我需要先查 SKU12345 的库存情况
  Action: search_documents("SKU12345 库存余量 安全库存")
  Observation: [文档1] SKU12345: 库存200件, 安全库存50件, 日销量约30件

Round 2:
  Thought: 库存200件, 安全库存50件, 日销30件。大约够卖 (200-50)/30 = 5天。
           用户问"够不够"——如果定义是"够卖一周"那就不够。
           但我还需要知道"不够时推荐什么替代品"
           让我查一下替代品规则
  Action: search_documents("SKU12345 替代品 缺货推荐 供应链")
  Observation: [文档3] 当SKU12345缺货时，推荐替代品SKU67890（功能相似，价格略高10%）

Round 3:
  Thought: 我现在有了完整信息。计算一下库存够卖几天。
  Action: calculate("(200-50)/30")
  Observation: 计算结果: (200-50)/30 = 5.0

Final Answer:
  根据系统数据，SKU12345 当前库存 200 件，安全库存 50 件，日均销量约 30 件。
  剩余可售库存可支撑约 **5 天**。根据公司补货周期（7个工作日），当前库存
  **可能不够**覆盖到新货到达。

  如果库存不足，推荐替代品 **SKU67890**，功能相似但价格略高约 10%。

## 面试考点
Q: Agent 和 RAG 的本质区别是什么？
A: RAG 是"单跳"（一次检索→一次回答），Agent 是"多跳"（多轮检索+工具调用+推理）。
   Agent 能处理"先查A，根据A的结果决定查B"这种多步任务。

Q: Agent 的最大问题是什么？
A: ① Token 消耗大（每轮循环都调一次 LLM，上下文越来越长）
   ② 可能陷入死循环（找不到答案一直搜，不知道放弃）
   ③ 工具调用可能出错（传错参数、误解工具用途）
   ④ 本质上是概率模型，不能保证每次都做正确决策
   面试时主动说这些"坑"比只讲优点强得多。

Q: 怎么防止 Agent 陷入死循环？
A: 设置 max_rounds（最大循环轮数）+ 每轮检查"有没有新的信息进来"。
   如果连续两轮结果一样，说明搜不到了，该放弃了。
"""

import re
import json
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field


@dataclass
class AgentStep:
    """记录 Agent 一步操作的完整信息"""
    round_num: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class AgentResult:
    """Agent 运行的最终结果"""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    total_rounds: int = 0
    total_tokens: int = 0  # 估算了消耗的 token 数


class ReActAgent:
    """
    ReAct Agent

    使用示例:
        agent = ReActAgent(llm_client, tool_registry)
        result = agent.run("SKU12345库存够不够？")

        # 查看 Agent 的推理过程（面试展示用）
        for step in result.steps:
            print(f"Round {step.round_num}: {step.thought} → {step.action}")
    """

    # Agent 的系统提示词
    SYSTEM_PROMPT = """你是一个智能助手，能够使用工具来获取信息并回答用户问题。

## 工作方式
你按照以下格式逐步完成任务:

Thought: 分析当前情况，决定下一步需要做什么
Action: 要使用的工具名称（必须是下面列出的工具之一）
Action Input: 传递给工具的参数

当你收集到足够的信息后，用以下格式给出最终答案:
Thought: 我现在有足够的信息来回答用户的问题了
Final Answer: [你的回答，使用 Markdown 格式]

## 注意事项
1. 每次只能调用一个工具，等工具返回结果后再决定下一步
2. 如果第一次搜索没找到相关信息，尝试换不同的关键词再搜
3. 如果连续两次都找不到相关信息，就告诉用户找不到，不要浪费循环
4. 不要编造信息——只能使用工具返回的实际数据
5. 如果工具返回了错误，分析错误原因并尝试修正参数后再试
6. 最多进行 {max_rounds} 轮操作

## 可用工具
{tools_description}"""

    def __init__(self, llm_client, tool_registry, max_rounds: int = 5):
        """
        参数:
            llm_client: LLMClient 实例（用于调用 LLM）
            tool_registry: ToolRegistry 实例（管理可用工具）
            max_rounds: 最大循环轮数，防止 Agent 陷入死循环
        """
        self.llm = llm_client
        self.tools = tool_registry
        self.max_rounds = max_rounds

    def run(self, query: str, verbose: bool = False) -> AgentResult:
        """
        运行 Agent

        参数:
            query: 用户问题
            verbose: 是否打印每轮详情（开发调试时开）

        返回:
            AgentResult，包含最终答案和每一步的推理过程
        """
        # 构建初始 messages
        system_prompt = self.SYSTEM_PROMPT.format(
            max_rounds=self.max_rounds,
            tools_description=self.tools.get_all_descriptions(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        steps: List[AgentStep] = []

        for round_num in range(1, self.max_rounds + 1):
            if verbose:
                print(f"\n{'='*50}\n🔄 Round {round_num}/{self.max_rounds}\n{'='*50}")

            # 1. 让 LLM 思考下一步
            response = self.llm.chat(messages, temperature=0.1)
            messages.append({"role": "assistant", "content": response})

            # 2. 解析 LLM 的输出
            step = self._parse_response(response, round_num)
            steps.append(step)

            if verbose:
                print(f"💭 Thought: {step.thought}")
                if step.action:
                    print(f"🔧 Action: {step.action}({step.action_input})")

            # 3. 如果模型决定给出最终答案
            if step.action == "Final Answer" or step.observation == "FINAL":
                if verbose:
                    print("✅ Agent 完成任务")
                return AgentResult(
                    answer=step.thought,  # 此时 thought 字段存的是 Final Answer
                    steps=steps,
                    total_rounds=round_num,
                )

            # 4. 如果模型要调用工具
            if step.action and step.action != "Final Answer":
                observation = self._execute_tool(step.action, step.action_input)
                step.observation = observation

                if verbose:
                    print(f"👁️  Observation: {observation[:200]}...")

                # 将工具返回结果加入对话历史
                # 格式: "Observation: [工具返回内容]"
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

            # 5. 如果模型没有给出有效输出
            if not step.action:
                # 可能 LLM 没按格式来，提醒它
                messages.append({
                    "role": "user",
                    "content": "请按照格式继续：先 Thought 分析，然后 Action + Action Input，或给出 Final Answer。"
                })

        # 达到最大轮数，强制结束
        final_messages = messages + [{
            "role": "user",
            "content": "已达到最大操作轮数。请基于目前收集到的信息给出你的最佳回答（Final Answer）。"
        }]
        final_response = self.llm.chat(final_messages, temperature=0.1)
        # 尝试提取 Final Answer
        final_answer = self._extract_final_answer(final_response)

        return AgentResult(
            answer=final_answer or final_response,
            steps=steps,
            total_rounds=self.max_rounds,
        )

    def _parse_response(self, response: str, round_num: int) -> AgentStep:
        """
        解析 LLM 的 ReAct 格式输出

        期望格式:
        Thought: 用户问的是库存情况，我需要搜索SKU12345的库存数据
        Action: search_documents
        Action Input: SKU12345 库存余量

        或者:
        Thought: 我现在有足够的信息了
        Final Answer: [最终回答]

        设计决策: 为什么不用 JSON 格式而用纯文本？
        开源的 ReAct 模型（如 Llama、Qwen）训练时用的就是这种文本格式，
        它们更熟悉这种写法。强行用 JSON 反而可能导致格式错误。
        """
        step = AgentStep(round_num=round_num)

        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)',
                                  response, re.DOTALL)
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # 检查是否是 Final Answer
        final_match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
        if final_match:
            step.thought = final_match.group(1).strip()
            step.action = "Final Answer"
            return step

        # 提取 Action
        action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response)
        if action_match:
            step.action = action_match.group(1).strip()

        # 提取 Action Input
        input_match = re.search(r'Action Input:\s*(.+)', response, re.DOTALL)
        if input_match:
            step.action_input = input_match.group(1).strip()

        return step

    def _execute_tool(self, action: str, action_input: str) -> str:
        """
        执行工具调用

        这里有个重要的工程决策: 工具执行的容错
        - 如果工具不存在: 返回明确的错误信息（而不是抛异常），
          让 Agent 能读到错误并尝试其他工具或修正
        - 如果工具执行失败: 同样返回错误信息 + 建议

        为什么不让 Agent 直接崩溃:
        Agent 可能输出错别字（search_docments 少个 u）或传错参数。
        如果直接崩溃，整个流程就断了。返回错误信息让 Agent 自我修正。
        """
        try:
            return self.tools.execute(action, query=action_input)
        except ValueError as e:
            available = ", ".join(self.tools.list_tools())
            return f"错误: {str(e)}\n可用工具: {available}\n请检查工具名称拼写并重试。"
        except Exception as e:
            return f"错误: {str(e)}\n请检查参数格式并重试。"

    def _extract_final_answer(self, response: str) -> Optional[str]:
        """从文本中提取 Final Answer"""
        match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 备选: 尝试提取 Thought 后的内容作为答案
        match = re.search(r'Thought:\s*(.+)', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None
