import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Callable

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from tools import (
    load_tools_definition,
    execute_tool,
    calculator,
    csv_query,
    policy_lookup,
    text_analyzer,
)


DEFAULT_LLM_CONFIGS = {
    "deepseek": {
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "doubao": {
        "model": "doubao-seed-2-0-mini-260428",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key_env": "DOUBAO_API_KEY",
    },
    "openai": {
        "model": "gpt-5.5",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


def create_llm_client(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    if OpenAI is None:
        return None, ""
    cfg = DEFAULT_LLM_CONFIGS.get(provider, {}).copy()
    if api_key:
        cfg["api_key"] = api_key
    else:
        env_name = cfg.get("api_key_env", "")
        cfg["api_key"] = os.environ.get(env_name, "")

    if base_url:
        cfg["base_url"] = base_url
    if model:
        cfg["model"] = model

    if not cfg.get("api_key"):
        return None, cfg.get("model", "")

    try:
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"],
        )
        return client, cfg.get("model", "")
    except Exception:
        return None, cfg.get("model", "")


def build_system_prompt(tools_def: List[Dict[str, Any]], disable_tools: bool = False) -> str:
    tools_desc = []
    valid_names = []
    for t in tools_def:
        name = t["name"]
        valid_names.append(name)
        desc = t["description"]
        input_schema = json.dumps(t["inputSchema"], ensure_ascii=False, indent=2)
        tools_desc.append(f"工具名称: {name}\n工具描述: {desc}\n输入参数Schema:\n{input_schema}")
    tools_text = "\n\n----------\n\n".join(tools_desc)

    if disable_tools:
        return f"""你是一个AI助手。当前模式为：禁用工具，直接回答。
请直接根据用户的问题给出清晰准确的回答。
你必须以严格JSON格式输出：
{{
  "task_understanding": "对用户任务的理解",
  "need_tool": false,
  "next_action": "final_answer",
  "tool_input": {{
    "answer": "你的最终回答内容"
  }},
  "reason": "无需调用工具，直接回答",
  "is_done": true
}}
"""
    return f"""你是一个智能 Agent 规划器（Planner），负责根据用户任务选择合适的工具或直接回答。
你必须严格按照JSON格式输出决策，且只能使用规定的字段。

## 可用工具列表
{tools_text}

## 决策规则
1. 首先仔细理解用户任务和上下文（历史工具输出）。
2. 判断需要调用什么工具，或者直接回答（final_answer），或者信息不足需要追问（ask_clarification）。
3. next_action 的合法取值只有：{', '.join(valid_names)}。
4. tool_input 必须严格匹配对应工具的 inputSchema，字段名和类型必须一致。
5. 如果信息不足以完成任务（例如缺少具体事件、缺少数据），不要猜测，选择 ask_clarification 并提出具体追问问题。
6. 如果选择 final_answer，tool_input.answer 中要写完整的最终回答。
7. 如果已经有足够工具结果，应该综合分析并选择 final_answer。

## 输出JSON格式（绝对严格遵守，不能包含任何额外文字或Markdown）
{{
  "task_understanding": "对用户任务的理解，一句话总结",
  "need_tool": true或false,
  "next_action": "工具名称 或 final_answer 或 ask_clarification",
  "tool_input": {{ 必须符合所选工具的inputSchema }},
  "reason": "一句话说明选择原因",
  "is_done": true或false（只有final_answer或ask_clarification时为true）
}}

注意：need_tool 只有在 next_action 是四个实际工具（calculator/csv_query/policy_lookup/text_analyzer）时才为 true；其他情况下为 false。
"""


def build_user_prompt(
    task: str,
    history: List[Dict[str, Any]],
    step: int,
) -> str:
    history_text = ""
    if history:
        history_parts = []
        for i, h in enumerate(history):
            part = f"""Step {h.get('step', i+1)}:
- Agent判断: {h.get('task_understanding', '')}
- 选择动作: {h.get('next_action', '')}
- 是否需要工具: {h.get('need_tool', False)}
- 工具输入: {json.dumps(h.get('tool_input', {}), ensure_ascii=False)}
- 工具输出: {json.dumps(h.get('tool_output', {}), ensure_ascii=False)}
- 选择理由: {h.get('reason', '')}
"""
            history_parts.append(part)
        history_text = "## 历史执行过程\n" + "\n".join(history_parts) + "\n\n"

    return f"""## 当前用户任务
{task}

{history_text}## 当前状态
这是第 {step} 步决策。请基于上述信息选择下一步动作，严格以JSON输出。
"""


def call_llm_planner(
    client: Optional[OpenAI],
    model: str,
    messages: List[Dict[str, str]],
) -> Optional[str]:
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            temperature=0,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[planner] LLM call failed: {e}")
        return None


def parse_decision(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None


def validate_decision(decision: Dict[str, Any], tools_def: List[Dict[str, Any]]) -> Tuple[bool, str]:
    required_fields = ["task_understanding", "need_tool", "next_action", "tool_input", "reason", "is_done"]
    for f in required_fields:
        if f not in decision:
            return False, f"缺少字段: {f}"
    valid_names = [t["name"] for t in tools_def]
    if decision["next_action"] not in valid_names:
        return False, f"非法 next_action: {decision['next_action']}, 合法值: {valid_names}"
    if not isinstance(decision["tool_input"], dict):
        return False, "tool_input 必须是对象"
    return True, "ok"


def _fallback_decision_for_task(
    task: str,
    history: List[Dict[str, Any]],
    tools_def: List[Dict[str, Any]],
) -> Dict[str, Any]:
    task_lower = task.lower()
    steps_done = len(history)

    csv_called = any(h.get("next_action") == "csv_query" for h in history)
    policy_called = any(h.get("next_action") == "policy_lookup" for h in history)
    calc_called = any(h.get("next_action") == "calculator" for h in history)
    text_called = any(h.get("next_action") == "text_analyzer" for h in history)

    keywords_calc = ["增长率", "比例", "平均", "百分比", "多少倍", "增长了", "减少了", "+", "-", "*", "/"]
    keywords_csv = ["媒体案例", "案例数据", "media_cases", "负面评论率", "最高", "排序", "筛选", "统计", "阅读量", "平台"]
    keywords_policy = ["课程资料", "规范", "政策", "伦理", "能不能", "是否可以", "应该", "规则", "ai_news"]
    keywords_text = ["情绪倾向", "关键词", "风险", "未经证实", "分析文本", "文本分析"]
    keywords_vague = ["这个事件", "该事件", "分析这个", "帮我分析一下"]

    has_tool_hint = (
        any(k in task for k in keywords_calc)
        or any(k in task for k in keywords_csv)
        or any(k in task for k in keywords_policy)
        or any(k in task for k in keywords_text)
    )
    lacks_specific_event = not any(k in task for k in ["高校", "地铁", "主播", "未成年", "商场"])

    is_vague = any(k in task for k in keywords_vague) and lacks_specific_event and not has_tool_hint

    if is_vague and steps_done == 0:
        return {
            "task_understanding": f"用户希望分析某个事件的舆论趋势，但没有提供具体事件内容或数据",
            "need_tool": False,
            "next_action": "ask_clarification",
            "tool_input": {
                "question": "请问您指的是哪个具体事件呢？例如：1）高校AI摄像头；2）地铁人脸识别；3）AI主播播报新闻；4）平台限制未成年人时长；5）商场客流识别。或者您可以直接描述事件和数据。"
            },
            "reason": "任务中没有明确指出是哪个事件，缺少具体分析对象，需要追问",
            "is_done": True
        }

    if steps_done == 0:
        if any(k in task for k in keywords_calc) and not calc_called:
            nums = re.findall(r"\d+(?:\.\d+)?", task)
            expr = ""
            if "增长率" in task and len(nums) >= 2:
                last, curr = float(nums[-2]), float(nums[-1])
                expr = f"({curr}-{last})/{last}*100"
            elif "比例" in task and len(nums) >= 2:
                a, b = float(nums[0]), float(nums[1])
                expr = f"{a}/{b}*100"
            else:
                expr = " ".join(nums) if nums else "0"
            return {
                "task_understanding": f"用户需要进行数学计算：{task[:40]}",
                "need_tool": True,
                "next_action": "calculator",
                "tool_input": {"expression": expr or "0"},
                "reason": "任务涉及数学运算，使用 calculator 工具",
                "is_done": False
            }

        has_csv_key = any(k in task for k in keywords_csv)
        has_text_key = any(k in task for k in keywords_text)
        has_policy_key = any(k in task for k in keywords_policy)

        if has_csv_key and not csv_called:
            op = "all"
            col = ""
            if "负面评论率最高" in task or ("最高" in task and "负面" in task):
                op, col = "max", "negative_rate"
            elif "正面最高" in task or ("最高" in task and "正面" in task):
                op, col = "max", "positive_rate"
            elif "最高阅读量" in task or "最多阅读" in task:
                op, col = "max", "views"
            elif "排序" in task:
                op, col = "sort", "views"
            return {
                "task_understanding": f"用户需要查询/分析媒体案例CSV数据：{task[:40]}",
                "need_tool": True,
                "next_action": "csv_query",
                "tool_input": {"operation": op, "column": col, "order": "desc"},
                "reason": "任务涉及媒体案例数据分析，使用 csv_query 工具",
                "is_done": False
            }

        if has_text_key and not text_called and not has_csv_key:
            return {
                "task_understanding": f"用户需要分析文本内容（情绪/关键词/风险）：{task[:40]}",
                "need_tool": True,
                "next_action": "text_analyzer",
                "tool_input": {"text": task, "analysis_type": "all"},
                "reason": "任务涉及文本语义分析，使用 text_analyzer 工具",
                "is_done": False
            }

        if has_policy_key and not policy_called and (not has_csv_key or csv_called) and (not has_text_key or text_called):
            return {
                "task_understanding": f"用户需要查询AI新闻伦理规范或课程政策：{task[:40]}",
                "need_tool": True,
                "next_action": "policy_lookup",
                "tool_input": {"query": task},
                "reason": "任务涉及政策或伦理规范查询，使用 policy_lookup 工具",
                "is_done": False
            }

        if "算法推荐" in task or "是什么" in task or "解释" in task:
            return {
                "task_understanding": f"用户需要知识性直接回答：{task[:40]}",
                "need_tool": False,
                "next_action": "final_answer",
                "tool_input": {
                    "answer": "算法推荐是一种根据用户的历史行为、兴趣偏好、人口属性等特征，通过算法自动计算并向用户推送其可能感兴趣的内容（如新闻、商品、视频等）的信息分发技术。"
                },
                "reason": "这是一个常识性问题，不需要调用工具，可以直接回答",
                "is_done": True
            }

    if steps_done >= 1:
        last_out = history[-1].get("tool_output", {})
        last_action = history[-1].get("next_action", "")

        if last_action == "csv_query" and "负面" in task and not text_called:
            rows = last_out.get("result", [])
            event_desc = ""
            if rows:
                event_desc = rows[0].get("event", "")
            if event_desc:
                return {
                    "task_understanding": f"已找到目标事件，现在需要分析其描述文本的情绪倾向",
                    "need_tool": True,
                    "next_action": "text_analyzer",
                    "tool_input": {"text": event_desc, "analysis_type": "all"},
                    "reason": "下一步需要对事件描述进行文本情绪和风险分析",
                    "is_done": False
                }

        if (last_action == "text_analyzer" or text_called) and "伦理" in task and not policy_called:
            return {
                "task_understanding": f"已完成文本分析，现在需要对照AI新闻伦理规范判断是否合规",
                "need_tool": True,
                "next_action": "policy_lookup",
                "tool_input": {"query": task},
                "reason": "需要根据政策规则进行合规判断，使用 policy_lookup 工具",
                "is_done": False
            }

        answer_parts = []
        for h in history:
            act = h.get("next_action", "")
            out = h.get("tool_output", {})
            if act == "calculator" and "result" in out:
                answer_parts.append(f"计算结果：{out['result']:.2f}")
            if act == "csv_query" and "result" in out:
                rows = out["result"]
                if rows:
                    row0 = rows[0]
                    if "event" in row0:
                        answer_parts.append(f"媒体案例结果：事件='{row0.get('event','')}'，平台={row0.get('platform','')}，负面率={row0.get('negative_rate','')}")
            if act == "policy_lookup" and "relevant" in out:
                rels = out["relevant"][:3]
                answer_parts.append("相关政策条款：" + "；".join(r.get("content","") for r in rels))
            if act == "text_analyzer":
                answer_parts.append(f"文本情绪：{out.get('sentiment','')}，关键词：{out.get('keywords',[])}")

        final_answer = "\n".join(answer_parts) if answer_parts else "任务已完成（无可用结果输出）。"
        return {
            "task_understanding": "已有足够工具结果，综合输出最终回答",
            "need_tool": False,
            "next_action": "final_answer",
            "tool_input": {"answer": final_answer},
            "reason": "已经通过工具获取足够信息，可以给出最终回答",
            "is_done": True
        }

    return {
        "task_understanding": f"用户提问：{task[:50]}",
        "need_tool": False,
        "next_action": "final_answer",
        "tool_input": {"answer": "这是一个通用回答（规则回退）。建议配置LLM以获得更准确的智能决策。"},
        "reason": "规则无法明确分类，直接回答（回退策略）",
        "is_done": True
    }


def planner_step(
    task: str,
    history: List[Dict[str, Any]],
    step: int,
    tools_def: List[Dict[str, Any]],
    client: Optional[OpenAI],
    model: str,
    disable_tools: bool = False,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(tools_def, disable_tools=disable_tools)
    user_prompt = build_user_prompt(task, history, step)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    decision = None
    raw_response = None
    use_llm = (client is not None)

    if use_llm:
        for attempt in range(2):
            raw_response = call_llm_planner(client, model, messages)
            parsed = parse_decision(raw_response) if raw_response else None
            if parsed:
                valid, msg = validate_decision(parsed, tools_def)
                if valid:
                    decision = parsed
                    break
                else:
                    messages.append({"role": "assistant", "content": raw_response or "{}"})
                    messages.append({
                        "role": "user",
                        "content": f"你的决策JSON校验失败：{msg}。请重新严格按照要求输出合法JSON。"
                    })

    if decision is None:
        decision = _fallback_decision_for_task(task, history, tools_def)
        valid, msg = validate_decision(decision, tools_def)
        if not valid:
            decision = {
                "task_understanding": task,
                "need_tool": False,
                "next_action": "final_answer",
                "tool_input": {"answer": "（系统内部校验错误）无法生成合法决策。"},
                "reason": "fallback validation failed, 强制 final_answer",
                "is_done": True
            }

    decision["_raw_llm"] = raw_response
    decision["_used_llm"] = use_llm
    return decision


def run_agent(
    task: str,
    max_steps: int = 5,
    provider: str = "deepseek",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    disable_tools: bool = False,
    llm_client_creator_for_text: Optional[Callable] = None,
    on_step_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    tools_def_all = load_tools_definition()
    tools_def = tools_def_all["tools"]
    tool_names_real = [t["name"] for t in tools_def if t["name"] not in ("final_answer", "ask_clarification")]

    client, model_name = create_llm_client(provider, api_key, base_url, model)

    history: List[Dict[str, Any]] = []
    called_tools = set()
    completion_type = "unknown"

    for step in range(1, max_steps + 1):
        decision = planner_step(
            task=task,
            history=history,
            step=step,
            tools_def=tools_def,
            client=client if not disable_tools else None,
            model=model_name,
            disable_tools=disable_tools,
        )

        next_action = decision["next_action"]
        need_tool = decision["need_tool"]
        tool_input = decision["tool_input"]

        if next_action in tool_names_real:
            tool_output = execute_tool(next_action, tool_input, llm_client_creator=llm_client_creator_for_text)
            called_tools.add(next_action)
        elif next_action == "final_answer":
            tool_output = {"answer": tool_input.get("answer", "")}
        elif next_action == "ask_clarification":
            tool_output = {"question": tool_input.get("question", "")}
        else:
            tool_output = {"error": "unexpected next_action"}

        step_record = {
            "step": step,
            "task_understanding": decision.get("task_understanding", ""),
            "need_tool": need_tool,
            "next_action": next_action,
            "tool_input": tool_input,
            "tool_output": tool_output,
            "reason": decision.get("reason", ""),
            "is_done": decision.get("is_done", False),
            "_raw_llm": decision.get("_raw_llm"),
            "_used_llm": decision.get("_used_llm"),
        }
        history.append(step_record)
        if on_step_callback:
            try:
                on_step_callback(step_record)
            except Exception:
                pass

        if next_action == "final_answer":
            completion_type = "direct" if len(called_tools) == 0 else ("single" if len(called_tools) == 1 else "multi")
            break
        if next_action == "ask_clarification":
            completion_type = "clarification"
            break

    else:
        if history:
            completion_type = "max_steps"

    not_called_tools = [n for n in tool_names_real if n not in called_tools]

    final_answer = ""
    clarification_q = ""
    if history:
        last = history[-1]
        if last["next_action"] == "final_answer":
            final_answer = last["tool_output"].get("answer", "")
        elif last["next_action"] == "ask_clarification":
            clarification_q = last["tool_output"].get("question", "")

    return {
        "task": task,
        "history": history,
        "called_tools": sorted(called_tools),
        "not_called_tools": not_called_tools,
        "completion_type": completion_type,
        "final_answer": final_answer,
        "clarification_question": clarification_q,
        "total_steps": len(history),
    }
