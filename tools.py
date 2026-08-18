import os
import json
import csv
import re
import math
from typing import Any, Dict, List, Optional, Callable


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TOOLS_JSON_PATH = os.path.join(BASE_DIR, "tools.json")


def load_tools_definition() -> Dict[str, Any]:
    with open(TOOLS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def calculator(input_params: Dict[str, Any]) -> Dict[str, Any]:
    expression = input_params.get("expression", "")
    if not expression:
        return {"error": "expression is required"}
    try:
        allowed_chars = set("0123456789+-*/().%^ ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "expression contains invalid characters"}
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "math": math,
        }
        expr = expression.replace("^", "**")
        result = eval(expr, safe_dict, {})
        return {"result": float(result)}
    except Exception as e:
        return {"error": f"calculation failed: {str(e)}"}


def _read_csv() -> List[Dict[str, Any]]:
    csv_path = os.path.join(DATA_DIR, "media_cases.csv")
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in ("case_id", "views", "comments"):
                    try:
                        parsed[k] = int(v)
                    except ValueError:
                        parsed[k] = v
                elif k in ("positive_rate", "negative_rate"):
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def csv_query(input_params: Dict[str, Any]) -> Dict[str, Any]:
    operation = input_params.get("operation", "all")
    column = input_params.get("column", "")
    order = input_params.get("order", "desc")
    condition = input_params.get("condition", "")
    limit = input_params.get("limit", None)

    try:
        rows = _read_csv()
        result = rows
        summary = ""

        if operation == "filter" and condition:
            match = re.match(r"(\w+)\s*[=:]\s*(.+)", condition.strip())
            if match:
                col, val = match.group(1), match.group(2).strip().strip("'\"")
                result = [r for r in rows if str(r.get(col, "")) == val]
                summary = f"筛选条件 {col}={val}，共 {len(result)} 条结果"
            else:
                summary = "条件格式解析失败，返回全部数据"
                result = rows

        elif operation == "sort":
            if column and column in rows[0] if rows else False:
                reverse = (order == "desc")
                result = sorted(rows, key=lambda r: r.get(column, 0), reverse=reverse)
                summary = f"按 {column} {order} 排序"
            else:
                summary = "排序列名无效，返回原始顺序"

        elif operation == "max":
            if column and rows:
                max_row = max(rows, key=lambda r: r.get(column, float("-inf")))
                result = [max_row]
                summary = f"按 {column} 取最大值的记录"
            else:
                summary = "max: 列名无效或数据为空"
                result = []

        elif operation == "min":
            if column and rows:
                min_row = min(rows, key=lambda r: r.get(column, float("inf")))
                result = [min_row]
                summary = f"按 {column} 取最小值的记录"
            else:
                summary = "min: 列名无效或数据为空"
                result = []

        elif operation == "stats":
            if rows:
                stats = {}
                numeric_cols = []
                for col in rows[0].keys():
                    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
                    if vals:
                        numeric_cols.append(col)
                        stats[col] = {
                            "count": len(vals),
                            "sum": sum(vals),
                            "avg": sum(vals) / len(vals),
                            "max": max(vals),
                            "min": min(vals),
                        }
                summary = f"统计数值列: {numeric_cols}"
                result = [stats]
            else:
                summary = "无数据"
                result = []

        elif operation == "group":
            if column:
                groups = {}
                for r in rows:
                    key = str(r.get(column, ""))
                    if key not in groups:
                        groups[key] = {"count": 0, "records": []}
                    groups[key]["count"] += 1
                    groups[key]["records"].append(r)
                summary = f"按 {column} 分组，共 {len(groups)} 组"
                result = [{k: {"count": v["count"], "sample": v["records"][:3]} for k, v in groups.items()}]
            else:
                summary = "缺少分组列名"
                result = []
        else:
            summary = f"全部数据共 {len(rows)} 条"
            result = rows

        if limit and isinstance(limit, int) and limit > 0:
            result = result[:limit]

        return {"result": result, "summary": summary}

    except Exception as e:
        return {"error": f"csv_query failed: {str(e)}", "result": [], "summary": ""}


def _read_policy() -> str:
    policy_path = os.path.join(DATA_DIR, "ai_news_policy.md")
    with open(policy_path, "r", encoding="utf-8") as f:
        return f.read()


def policy_lookup(input_params: Dict[str, Any]) -> Dict[str, Any]:
    query = input_params.get("query", "")
    try:
        policy_text = _read_policy()
        lines = [line.strip() for line in policy_text.strip().splitlines() if line.strip()]
        relevant = []
        if query:
            query_lower = query.lower()
            keywords = re.findall(r"[\w\u4e00-\u9fff]+", query_lower)
            for line in lines:
                line_lower = line.lower()
                score = 0
                for kw in keywords:
                    if kw and kw in line_lower:
                        score += 1
                if score > 0:
                    relevant.append({"content": line, "score": score})
            relevant = sorted(relevant, key=lambda x: x["score"], reverse=True)
        if not relevant:
            relevant = [{"content": line, "score": 0} for line in lines]
        return {"policy": policy_text, "relevant": relevant}
    except Exception as e:
        return {"error": f"policy_lookup failed: {str(e)}", "policy": "", "relevant": []}


def _call_llm_for_text_analyzer(
    text: str,
    analysis_type: str,
    llm_client_creator: Optional[Callable] = None
) -> Dict[str, Any]:
    if llm_client_creator is None:
        return {
            "sentiment": "中性",
            "sentiment_score": 0.0,
            "keywords": [],
            "risk_terms": [],
            "unverified_claims": [],
            "summary": "无LLM客户端，返回默认分析结果",
            "note": "LLM不可用，使用启发式规则分析。请配置LLM以获得更准确结果。"
        }
    try:
        client, model = llm_client_creator()
        prompt = f"""你是一个文本分析专家。请分析以下文本并以严格JSON格式返回结果。

文本内容：
{text}

分析类型：{analysis_type}

请严格返回以下JSON结构（不要包含任何额外文字）：
{{
  "sentiment": "正面|负面|中性",
  "sentiment_score": "-1到1之间的浮点数",
  "keywords": ["关键词1", "关键词2"],
  "risk_terms": ["风险表述1", "风险表述2"],
  "unverified_claims": ["未经证实的说法1", "未经证实的说法2"],
  "summary": "简要的分析摘要"
}}
"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            temperature=0,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        return {
            "sentiment": "中性",
            "sentiment_score": 0.0,
            "keywords": [],
            "risk_terms": [],
            "unverified_claims": [],
            "summary": f"LLM分析失败: {str(e)}",
            "error": str(e)
        }


def _heuristic_text_analysis(text: str, analysis_type: str) -> Dict[str, Any]:
    text_lower = text.lower()
    positive_words = ["支持", "赞同", "好", "棒", "优秀", "正面", "利好", "满意", "hope", "good", "great"]
    negative_words = ["反对", "抗议", "差", "糟糕", "负面", "担忧", "质疑", "争议", "风险", "bad", "terrible", "worry"]

    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)

    if pos_count > neg_count:
        sentiment = "正面"
        score = min(1.0, (pos_count - neg_count) * 0.2)
    elif neg_count > pos_count:
        sentiment = "负面"
        score = max(-1.0, (pos_count - neg_count) * 0.2)
    else:
        sentiment = "中性"
        score = 0.0

    cn_chars = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
    from collections import Counter
    kw_counter = Counter(cn_chars)
    keywords = [kw for kw, _ in kw_counter.most_common(5)]

    risk_patterns = ["隐私", "泄露", "风险", "危险", "争议", "举报", "违法", "侵权", "歧视"]
    risk_terms = [p for p in risk_patterns if p in text]

    unverified_patterns = ["据说", "网传", "据称", "据传闻", "有消息称", "疑似", "可能"]
    unverified_claims = [p for p in unverified_patterns if p in text]
    if unverified_claims:
        unverified_claims.append("文中包含可能未经证实的传闻表述")

    summary = f"基于规则分析：情绪{sentiment}，关键词{len(keywords)}个，风险表述{len(risk_terms)}个"
    return {
        "sentiment": sentiment,
        "sentiment_score": float(score),
        "keywords": keywords,
        "risk_terms": risk_terms,
        "unverified_claims": unverified_claims,
        "summary": summary
    }


def text_analyzer(
    input_params: Dict[str, Any],
    llm_client_creator: Optional[Callable] = None
) -> Dict[str, Any]:
    text = input_params.get("text", "")
    analysis_type = input_params.get("analysis_type", "all")
    if not text:
        return {"error": "text is required"}

    if llm_client_creator is not None:
        result = _call_llm_for_text_analyzer(text, analysis_type, llm_client_creator)
    else:
        result = _heuristic_text_analysis(text, analysis_type)

    return result


TOOL_REGISTRY: Dict[str, Callable] = {
    "calculator": calculator,
    "csv_query": csv_query,
    "policy_lookup": policy_lookup,
    "text_analyzer": text_analyzer,
}


def execute_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
    llm_client_creator: Optional[Callable] = None
) -> Dict[str, Any]:
    if tool_name == "final_answer":
        return {"answer": tool_input.get("answer", "")}
    if tool_name == "ask_clarification":
        return {"question": tool_input.get("question", "")}
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"unknown tool: {tool_name}"}
    fn = TOOL_REGISTRY[tool_name]
    try:
        if tool_name == "text_analyzer":
            return fn(tool_input, llm_client_creator=llm_client_creator)
        return fn(tool_input)
    except Exception as e:
        return {"error": f"tool execution exception: {str(e)}"}
