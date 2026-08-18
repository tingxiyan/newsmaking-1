# 🤖 工具选择 Agent 演示

一个基于 Streamlit + OpenAI-compatible API 的 "工具选择 Agent" 演示系统，用于直观展示大模型如何根据任务：
- 判断 **是否需要调用工具**
- 判断 **调用哪个工具**
- 判断 **是否需要连续调用多个工具**
- 或者 **直接回答**
- 或者 **信息不足时追问用户**

完整呈现了 Agent 的 **规划（Planner） + 工具选择 + 工具调用 + 过程可视化** 全流程。

---

## 📁 项目结构

```
tool_selection_agent/
├── app.py              # Streamlit 网页运行模块（前端可视化）
├── planner.py          # LLM + 规划模块（核心 Planner 决策）
├── tools.py            # 工具实现模块
├── tools.json          # 工具定义文件（Schema）
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例
├── .gitignore
├── README.md           # 本说明文档
├── tests/
│   └── test_agent.py   # Agent 全流程测试用例
└── data/
    ├── media_cases.csv # 媒体案例数据
    └── ai_news_policy.md  # AI 新闻伦理政策
```

---

## 🧰 工具箱一览

| 工具名 | 作用 | 典型场景 |
|---|---|---|
| `calculator` | 执行数学计算 | 增长率、比例、平均数、百分比、加减乘除等 |
| `csv_query` | 读取分析 `data/media_cases.csv` | 排序、筛选、分组、找最大/最小值、统计 |
| `policy_lookup` | 查询 `data/ai_news_policy.md` | AI 新闻伦理、课程规范、采访引用规则 |
| `text_analyzer` | 文本分析（关键词/情绪/风险/未证实说法） | 情绪倾向、关键词提取、风险表述识别 |
| `final_answer` | Agent 决策动作：不再调用工具，直接输出最终回答 | 信息足够或无需工具时 |
| `ask_clarification` | Agent 决策动作：信息不足，向用户追问 | 缺少具体事件、缺少数据、问题模糊时 |

> 注：`final_answer` 和 `ask_clarification` 是 **决策动作** 而非功能性工具。

工具定义详见 [`tools.json`](./tools.json)，包含详细的 `inputSchema` 和 `outputSchema`。

---

## 🚀 快速开始

### 1️⃣ 安装依赖

建议使用 Python 3.10+：

```bash
cd tool_selection_agent
pip install -r requirements.txt
```

### 2️⃣ 配置 API Key

```bash
# 复制示例
cp .env.example .env   # Windows: copy .env.example .env

# 编辑 .env，填入至少一个 Provider 的 Key
DEEPSEEK_API_KEY=sk-xxxx
DOUBAO_API_KEY=xxxx
OPENAI_API_KEY=sk-xxxx
```

> 💡 **无需 Key 也能运行**：系统内置了一套 **规则回退 Planner**（Fallback），没有配置 LLM API Key 时会自动使用启发式规则进行决策，保证演示和测试可独立运行。

### 3️⃣ 启动 Web 演示

```bash
streamlit run app.py
```

浏览器会自动打开页面（默认 http://localhost:8501 ）。

---

## 🎮 内置示例任务

页面顶部提供 5 个典型示例下拉框，一键注入：

| # | 示例 | 期望完成方式 |
|---|---|---|
| ① | "请用一句话解释什么是算法推荐。" | **直接回答**（不调用工具） |
| ② | "某新闻账号上周 85000 阅读量，本周 123000，增长率是多少？" | **单工具**（calculator） |
| ③ | "根据课程资料，AI 生成的采访对象引语能不能直接写进新闻稿？" | **单工具**（policy_lookup） |
| ④ | "找出负面评论率最高的媒体事件，分析该事件描述的情绪倾向，并根据 AI 新闻伦理规则判断是否符合规范。" | **多工具**（csv_query → text_analyzer → policy_lookup） |
| ⑤ | "帮我分析这个事件的舆论趋势。" | **信息不足追问**（ask_clarification） |

---

## 🧪 运行测试（全流程 Agent 决策验证）

项目包含完整的 pytest 测试，覆盖 5 个示例任务的 **全程决策**，而非仅测试单个工具：

```bash
cd tool_selection_agent
python -m pytest tests/test_agent.py -v
```

或运行详细输出：
```bash
python -m pytest tests/test_agent.py -v -s
```

**测试覆盖点**：
- `test_direct_answer`：直接回答，不调用工具
- `test_single_tool_calculator`：单工具 calculator
- `test_single_tool_policy`：单工具 policy_lookup
- `test_multi_tool_workflow`：多工具 csv → text → policy 链路
- `test_ask_clarification`：信息不足触发 ask_clarification
- `test_invalid_tool_fallback`：Planner 校验非法工具并回退
- `test_max_steps_stop`：达到最大步数安全停止
- `test_disable_tools_mode`：禁用工具模式下全部直接回答

---

## 🧠 Planner 决策详解

### 决策输出 JSON 格式

每一步 Planner 严格输出：
```json
{
  "task_understanding": "对用户任务的理解",
  "need_tool": true,
  "next_action": "calculator",
  "tool_input": {
    "expression": "(123000-85000)/85000*100"
  },
  "reason": "任务涉及数学计算（增长率）",
  "is_done": false
}
```

### 合法性校验

系统会对模型输出的 JSON 做严格校验：
- 字段完整性（6 个必填字段）
- `next_action` 是否属于合法工具列表
- `tool_input` 是否为对象
- **校验失败自动重试一次**，仍失败则进入规则回退 Planner（Fallback）

### Agent 运行循环

```
接收用户任务
    ↓
Planner 判断下一步（LLM JSON 强制输出）
    ↓
如果 next_action 是 4 个实际工具之一 → 执行工具并保存 输入/输出/理由
    ↓
更新状态 → 再次调用 Planner 判断
    ↓
直到 final_answer / ask_clarification / 达到最大步数
```

---

## 🛠 如何自定义工具

增加新工具只需 **3 步**：

### Step 1：在 `tools.json` 追加定义

```json
{
  "name": "weather_lookup",
  "description": "查询城市实时天气",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名"}
    },
    "required": ["city"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "temperature": {"type": "float"},
      "condition": {"type": "string"}
    }
  }
}
```

### Step 2：在 `tools.py` 中实现函数并注册

```python
def weather_lookup(input_params: Dict[str, Any]) -> Dict[str, Any]:
    city = input_params.get("city", "")
    # 你的实现逻辑...
    return {"temperature": 26.5, "condition": "晴"}

TOOL_REGISTRY["weather_lookup"] = weather_lookup
```

### Step 3：完成！

Planner 的 System Prompt 会自动读取 `tools.json` 并把新工具的 Schema 告知 LLM，无需修改 planner.py 或 app.py。

> 提示：如果你希望规则回退 Planner 也能识别这个新工具，可在 `planner.py` 的 `_fallback_decision_for_task()` 函数中添加对应的关键词分支（可选）。

---

## 🎨 页面功能说明

### 左侧栏
- **模型选择**：DeepSeek / doubao / OpenAI / 自定义，可随时改 Key、模型名、URL
- **工具箱列表**：展示每个工具的描述、输入/输出 Schema

### 主页面
1. **示例任务下拉框** → 一键注入 5 个演示任务
2. **用户任务输入框** → 自由输入
3. **模式选择**：自动选择工具 / 禁用工具仅直接回答
4. **最大步数滑块**（1-10）
5. **工作流步骤可视化**：每步展示 Step、Agent 判断、是否需要工具、选择的工具、工具输入、工具输出、选择理由
6. **最终回答区域**
7. **工具调用统计**：本次调用了哪些 / 没调用哪些
8. **任务完成方式徽章**：直接回答 / 单工具 / 多工具 / 信息不足追问

---

## 🔌 支持的 LLM Provider

都使用 OpenAI 兼容接口，支持 JSON 模式：

| Provider | 默认模型 | Base URL | 环境变量 Key |
|---|---|---|---|
| DeepSeek | `deepseek-v4-flash` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` |
| 豆包（火山方舟） | `doubao-seed-2-0-mini-260428` | `https://ark.cn-beijing.volces.com/api/v3` | `DOUBAO_API_KEY` |
| OpenAI | `gpt-5.5` | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| 自定义 | 用户填写 | 用户填写 | `CUSTOM_API_KEY` |

调用方式示例：
```python
client = OpenAI(api_key="...", base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=messages,
    response_format={"type": "json_object"},
    extra_body={"thinking": {"type": "disabled"}}
)
```

---

## 📌 设计亮点

1. **LLM + 规则双轨制**：无 API Key 时规则回退 Planner 自动接管，保证可离线演示和测试。
2. **Schema 驱动工具定义**：新增工具只改 `tools.json` + `tools.py`，Planner 自动感知。
3. **严格 JSON 校验 + 自动重试**：防止模型输出格式错误导致 Agent 崩溃。
4. **完整步骤记忆**：每步的输入、输出、理由全量保存，方便可视化复盘。
5. **禁用工具模式**：用于对比 "直接回答 vs 借助工具" 的效果差异。
6. **最大步数安全机制**：防止无限循环。

Enjoy building your Tool-Calling Agent! 🎉
