import os
import sys
import json
import time
from typing import Optional

import streamlit as st

st.set_page_config(
    page_title="工具选择 Agent 演示",
    page_icon="🤖",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from tools import load_tools_definition, execute_tool
from planner import (
    run_agent,
    create_llm_client,
    DEFAULT_LLM_CONFIGS,
)


EXAMPLE_TASKS = [
    ("（请选择示例任务）", ""),
    ("① 不调用工具：请用一句话解释什么是算法推荐。", "请用一句话解释什么是算法推荐。"),
    ("② 单工具调用：计算阅读量增长率", "某新闻账号上周阅读量 85000，本周阅读量 123000，增长率是多少？"),
    ("③ 查询本地资料：AI 采访引语规范", "根据课程资料，AI 生成的采访对象引语能不能直接写进新闻稿？"),
    ("④ 多工具调用：负面率最高事件 + 情绪 + 伦理", "请找出媒体案例中负面评论率最高的事件，分析该事件描述的情绪倾向，并根据AI新闻伦理规则判断是否符合规范。"),
    ("⑤ 信息不足：分析事件舆论趋势", "帮我分析这个事件的舆论趋势。"),
]


COMPLETION_LABELS = {
    "direct": "✅ 直接回答（未调用工具）",
    "single": "🛠 单工具完成",
    "multi": "🔗 多工具完成",
    "clarification": "❓ 信息不足，追问用户",
    "max_steps": "⚠ 达到最大步数",
    "unknown": "❔ 未知",
}

COMPLETION_COLORS = {
    "direct": "blue",
    "single": "green",
    "multi": "orange",
    "clarification": "red",
    "max_steps": "gray",
    "unknown": "gray",
}


def init_sidebar() -> dict:
    with st.sidebar:
        st.title("⚙️ 配置")

        st.markdown("---")
        st.subheader("🧠 模型设置")
        provider = st.selectbox(
            "选择 Provider",
            ["deepseek", "doubao", "openai", "custom"],
            index=0,
        )

        if provider == "custom":
            cfg = {
                "model": st.text_input("模型名称 (Model)", value="custom-model"),
                "base_url": st.text_input("Base URL", value="https://api.example.com/v1"),
                "api_key": st.text_input("API Key", type="password", value=os.environ.get("CUSTOM_API_KEY", "")),
            }
        else:
            default = DEFAULT_LLM_CONFIGS.get(provider, {})
            env_key = default.get("api_key_env", "")
            env_val = os.environ.get(env_key, "")

            with st.expander(f"修改 {provider} 参数（可选）", expanded=False):
                model = st.text_input("模型名称", value=default.get("model", ""), key=f"m_{provider}")
                base_url = st.text_input("Base URL", value=default.get("base_url", ""), key=f"b_{provider}")
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    value=env_val,
                    key=f"k_{provider}",
                    help=f"可设置环境变量 {env_key}",
                )
            cfg = {"model": model, "base_url": base_url, "api_key": api_key} if provider != "custom" else None
            if provider != "custom":
                cfg = {
                    "model": st.session_state.get(f"m_{provider}", default.get("model", "")),
                    "base_url": st.session_state.get(f"b_{provider}", default.get("base_url", "")),
                    "api_key": st.session_state.get(f"k_{provider}", env_val),
                }

        st.markdown("---")
        st.subheader("🧰 工具箱")
        tools_def_all = load_tools_definition()
        tool_real = [t for t in tools_def_all["tools"] if t["name"] not in ("final_answer", "ask_clarification")]
        special = [t for t in tools_def_all["tools"] if t["name"] in ("final_answer", "ask_clarification")]

        for t in tool_real:
            with st.expander(f"🔧 {t['name']}", expanded=False):
                st.markdown(f"**描述**: {t['description']}")
                st.markdown("**输入参数:**")
                st.json(t["inputSchema"])
                st.markdown("**输出参数:**")
                st.json(t["outputSchema"])
        for t in special:
            with st.expander(f"🎯 {t['name']}", expanded=False):
                st.markdown(f"**描述**: {t['description']}")

        st.markdown("---")
        if st.button("🔄 重置会话", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("run_") or key == "result":
                    del st.session_state[key]
            st.rerun()

        return {
            "provider": provider,
            **cfg,
            "tools_def_all": tools_def_all,
        }


def step_visual(step_record: dict):
    step_num = step_record["step"]
    next_action = step_record["next_action"]
    need_tool = step_record["need_tool"]
    is_done = step_record["is_done"]

    icon = "🤔"
    if next_action == "final_answer":
        icon = "✅"
    elif next_action == "ask_clarification":
        icon = "❓"
    elif need_tool:
        icon = "🛠"

    if is_done:
        title_color = "#4CAF50"
    elif need_tool:
        title_color = "#FF9800"
    else:
        title_color = "#2196F3"

    st.markdown(
        f"<div style='padding:10px 16px;background:{title_color}15;border-left:4px solid {title_color};border-radius:6px;margin-bottom:12px;'>"
        f"<span style='font-weight:bold;font-size:16px;'>{icon} Step {step_num} — {next_action}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**📝 Agent 判断**")
        st.info(step_record["task_understanding"])
        st.markdown("**🔀 是否需要工具**")
        st.success("✅ 需要" if need_tool else "❌ 不需要")
        st.markdown("**💡 选择理由**")
        st.write(step_record["reason"])

    with col2:
        st.markdown("**📥 工具输入**")
        if step_record["tool_input"]:
            st.json(step_record["tool_input"])
        else:
            st.caption("（无）")
        st.markdown("**📤 工具输出**")
        tool_out = step_record["tool_output"]
        if isinstance(tool_out, dict) and "result" in tool_out and isinstance(tool_out["result"], list) and len(tool_out["result"]) > 3:
            with st.expander(f"共 {len(tool_out['result'])} 条记录，点击展开"):
                st.json(tool_out)
        else:
            st.json(tool_out if tool_out else {"info": "（无输出）"})

    st.markdown("---")


def main():
    st.title("🤖 工具选择 Agent 演示")
    st.caption("演示大模型如何根据任务判断：是否调用工具 / 调用哪个工具 / 连续调用多个工具 / 直接回答 / 信息不足追问")

    cfg = init_sidebar()
    tools_def_all = cfg.get("tools_def_all")

    st.markdown("---")

    with st.container():
        st.subheader("📋 选择示例任务（或手动输入）")
        example_idx = st.selectbox(
            "内置示例",
            range(len(EXAMPLE_TASKS)),
            format_func=lambda i: EXAMPLE_TASKS[i][0],
            key="example_select",
        )

        default_text = EXAMPLE_TASKS[example_idx][1]
        user_task = st.text_area(
            "✍️ 用户任务",
            value=default_text,
            placeholder="请输入您的任务描述...",
            height=110,
            key="user_task_input",
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            mode = st.radio(
                "🎛 模式",
                ["自动选择工具", "禁用工具，仅让模型直接回答"],
                index=0,
                horizontal=True,
            )
        with col_b:
            max_steps = st.slider("🔢 最大步数", min_value=1, max_value=10, value=6, step=1)
        with col_c:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            run_clicked = st.button("🚀 运行 Agent", type="primary", use_container_width=True)

    disable_tools = (mode == "禁用工具，仅让模型直接回答")

    result_placeholder = st.container()
    workflow_placeholder = st.container()

    if run_clicked:
        if not user_task.strip():
            st.error("❌ 请先输入用户任务")
            return

        with st.spinner("🤖 Agent 思考中..."):
            status_box = st.empty()
            progress_bar = st.progress(0)

            steps_list = []

            def on_step(rec):
                steps_list.append(rec)
                progress_bar.progress(min(1.0, rec["step"] / max(max_steps, 1)))
                status_box.caption(f"已完成 Step {rec['step']}：选择了 {rec['next_action']} — {rec['reason'][:30]}")

            provider = cfg["provider"]
            api_key = cfg["api_key"]
            base_url = cfg["base_url"]
            model = cfg["model"]

            def llm_creator_for_text():
                c, m = create_llm_client(provider, api_key, base_url, model)
                return c, m

            t0 = time.time()
            result = run_agent(
                task=user_task,
                max_steps=max_steps,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                disable_tools=disable_tools,
                llm_client_creator_for_text=llm_creator_for_text,
                on_step_callback=on_step,
            )
            elapsed = time.time() - t0
            progress_bar.progress(1.0)
            status_box.caption(f"✅ 运行完成，总耗时 {elapsed:.2f}s，共 {result['total_steps']} 步")
            st.session_state["result"] = result
            time.sleep(0.3)
            status_box.empty()

    result = st.session_state.get("result")
    if result is None:
        with workflow_placeholder:
            st.info("👈 请选择示例任务或输入任务后，点击【🚀 运行 Agent】开始演示")
        return

    with result_placeholder:
        ctype = result["completion_type"]
        label = COMPLETION_LABELS.get(ctype, ctype)
        color = COMPLETION_COLORS.get(ctype, "gray")

        st.markdown("## 📊 运行结果汇总")
        cols = st.columns([2, 1, 1, 1])
        with cols[0]:
            st.metric("任务完成方式", label)
        with cols[1]:
            st.metric("总步数", f"{result['total_steps']} / {max_steps}")
        with cols[2]:
            st.metric("调用工具数", len(result["called_tools"]))
        with cols[3]:
            st.metric("未调用工具数", len(result["not_called_tools"]))

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("**🔧 本次调用过的工具**")
            if result["called_tools"]:
                st.success("  ".join([f"✅ {t}" for t in result["called_tools"]]))
            else:
                st.warning("（本次未调用任何工具）")
        with col_c2:
            st.markdown("**🔒 本次未调用的工具**")
            if result["not_called_tools"]:
                st.info("  ".join([f"⏭ {t}" for t in result["not_called_tools"]]))
            else:
                st.success("（所有工具都被调用过）")

        st.markdown("---")
        st.subheader("🎯 最终输出")
        if ctype == "clarification":
            st.error("### ❓ 信息不足，Agent 需要追问：")
            st.write(result["clarification_question"])
        elif ctype == "max_steps":
            st.warning("### ⚠ 达到最大步数：")
            if result["final_answer"]:
                st.write(result["final_answer"])
            else:
                st.caption("（Agent在最大步数内未能生成最终回答）")
        else:
            st.success("### ✅ 最终回答：")
            st.markdown(result["final_answer"])

    with workflow_placeholder:
        st.markdown("---")
        st.subheader("🔄 工作流详情（每一步）")
        for step_rec in result["history"]:
            step_visual(step_rec)


if __name__ == "__main__":
    main()
