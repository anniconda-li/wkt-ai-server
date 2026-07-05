# Minimal AI Chat Backend

一个最小可运行的 AI 聊天后端示例，使用 Python 3.11、FastAPI 和 OpenAI-compatible API。

当前版本已经支持：

- `/chat` 聊天接口
- 模型 streaming 输出
- 按设备隔离的最近 10 轮内存 memory
- 本地文物知识卡查询
- 文物追问上下文继承
- 简单 tool 调用机制
- OpenAI / DeepSeek / 其他 OpenAI-compatible 服务商
- Windows PowerShell 下的终端测试客户端

项目没有引入 LangChain、Dify 等框架，代码量尽量少，方便学习后端调用 LLM 的基本链路。

## 项目结构

```text
.
├── main.py           # FastAPI app，定义 /health 和 /chat
├── router.py         # 组装上下文、memory、tool 判断、流式转发
├── sessions.py       # 按 device_id 管理设备会话和 memory
├── artifacts.py      # 加载本地文物知识卡
├── llm.py            # 封装 OpenAI-compatible streaming 调用
├── tools.py          # 示例 tool：get_device_status()
├── data/artifacts/   # 5 件核心文物的本地 JSON 知识卡
├── chat_cli.py       # 终端聊天客户端，方便本地测试
├── requirements.txt  # Python 依赖
├── .env.example      # 环境变量示例，不放真实 key
└── .gitignore        # 忽略 .env、.venv、缓存文件
```

## 实现流程

一次聊天请求的完整流程：

```text
用户输入消息
  -> main.py 接收 POST /chat
  -> sessions.py 获取该 device 的会话
  -> router.py 构造 messages
  -> router.py 如果识别到文物名称，则加入本地知识卡
  -> router.py 如果是“它/这件/继续讲”等追问，则继承该设备上一件文物
  -> router.py 判断是否需要调用 tool
  -> llm.py 调用模型 API，stream=True
  -> router.py 一边转发 token，一边收集完整回复
  -> main.py 使用 StreamingResponse 返回给客户端
  -> router.py 把本轮 user/assistant 存入 memory
```

核心点：

- `main.py` 负责 HTTP 接口。
- `router.py` 负责业务编排。
- `sessions.py` 负责设备上下文和短期记忆。
- `artifacts.py` 负责加载和查询本地文物知识卡。
- `llm.py` 负责模型调用。
- `tools.py` 负责外部工具能力。
- `chat_cli.py` 只是测试客户端，不参与后端核心逻辑。

## 环境准备

本机推荐使用 `uv` 创建虚拟环境：

```powershell
cd D:\develop\Projects\AI-Box
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

如果 `.venv` 已存在但不可用，可以删除后重新创建。

## 配置模型

复制或编辑 `.env`：

```powershell
notepad .env
```

DeepSeek 示例：

```env
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-v4-flash
OPENAI_BASE_URL=https://api.deepseek.com
```

OpenAI 官方示例：

```env
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1
```

说明：

- `.env` 是本地私密配置，已经被 `.gitignore` 忽略。
- `.env.example` 只放示例，不要写真实 key。
- 如果修改了 `.env`，需要重启 uvicorn。

## 启动后端

```powershell
cd D:\develop\Projects\AI-Box
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --env-file .env
```

看到类似输出说明启动成功：

```text
Uvicorn running on http://127.0.0.1:8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

正常返回：

```json
{"status":"ok"}
```

## 推荐测试方式

Windows PowerShell 手写 JSON 很容易遇到中文编码和引号转义问题，所以推荐使用内置终端客户端：

```powershell
cd D:\develop\Projects\AI-Box
.\.venv\Scripts\python.exe chat_cli.py
```

进入后直接输入：

```text
Terminal chat client. Type /exit to quit.

You> 你好，你是谁呀？
AI> 你好呀！我是 ...

You> 我叫李庆博，请记住我的名字
AI> 好的 ...

You> 你还记得我叫什么吗？
AI> 当然记得，你叫李庆博 ...
```

指定设备会话：

```powershell
.\.venv\Scripts\python.exe chat_cli.py --device walkie-01
```

退出：

```text
You> /exit
```

也可以单次调用：

```powershell
.\.venv\Scripts\python.exe chat_cli.py --device walkie-01 "what is the device status?"
```

## 手动 HTTP 测试

如果想直接用 PowerShell 调接口，推荐强制 UTF-8：

```powershell
$msg = Read-Host "请输入消息"
$json = @{ message = $msg } | ConvertTo-Json -Compress
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body $bytes
```

如果使用 `curl.exe`，Windows PowerShell 可以用 `--%` 停止后续参数解析：

```powershell
curl.exe --% -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" --data-raw "{\"message\":\"hello\"}"
```

## API

### GET /health

健康检查。

返回：

```json
{"status":"ok"}
```

### POST /chat

聊天接口，返回 `text/plain` 流式文本。

请求：

```json
{
  "message": "hello",
  "device": "walkie-01"
}
```

响应：

```text
Hello! How can I help you today?
```

### GET /sessions

查看当前后端里有哪些设备 session。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/sessions
```

### GET /sessions/{device_id}

查看某台设备的上下文和 memory。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/sessions/walkie-01
```

### POST /sessions/{device_id}/clear

清空某台设备的短期 memory。

```powershell
Invoke-RestMethod -Method POST http://127.0.0.1:8000/sessions/walkie-01/clear
```

### GET /artifacts

查看本地文物知识卡列表。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/artifacts
```

当前知识卡包含：

- `yingguo_jade_eagle`：应国玉鹰
- `bronze_he_dragon_knob_lidded`：盘龙钮带盖铜盉
- `denggong_gui`：邓公簋
- `black_glaze_blue_splash_tripod_washer`：黑釉蓝斑花口三足洗
- `shuyao_chuilin_sheng_ding`：束腰垂鳞纹升鼎

### GET /artifacts/{artifact_id}

查看某件文物的完整知识卡。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/artifacts/yingguo_jade_eagle
```

## 本地文物知识卡

文物资料放在 `data/artifacts/*.json` 中。每张卡片目前包含：

```json
{
  "id": "yingguo_jade_eagle",
  "name": "应国玉鹰",
  "aliases": ["玉鹰", "白玉线雕鹰", "应国鹰形玉器"],
  "category": "玉器",
  "period": "西周晚期",
  "material": "白玉",
  "visual_keywords": ["鹰形", "鸟形", "白玉", "线雕"],
  "recognition_features": ["整体呈鹰展翅飞翔状"],
  "facts": ["应国玉鹰又称白玉线雕鹰，是西周晚期玉器。"],
  "guide_notes": ["后端维护用讲解提示，不直接传给模型"],
  "source_note": "公开网络资料补充，后续仍建议用馆方正式资料复核展陈口径。",
  "source_urls": [
    {
      "title": "平顶山博物馆文物解码",
      "url": "https://www.news.cn/shuhua/20211116/a840ac11f5d243ccb34457ef307d8ae8/c.html"
    }
  ],
  "data_status": "public_web_enriched"
}
```

这一层的设计目的：

```text
把文物事实掌握在后端
LLM 只负责把已知事实组织成讲解语言
```

当前卡片已经根据公开网络资料做了第一轮补充，主要参考新华网/人民日报的《平顶山博物馆文物解码》和河南省文物局页面。后续做正式景区项目时，仍建议用馆方正式资料复核展陈口径。

字段分层很重要：

- `facts`：会传给 LLM，是游客可以直接听到的事实。
- `visual_keywords` 和 `recognition_features`：后续图像识别和本地匹配会用到。
- `guide_notes`：后端维护用提示，当前不会传给模型。
- `source_urls` 和 `source_note`：资料溯源和复核提示，当前不会传给模型。
- `data_status`：标记资料状态，例如 `public_web_enriched`。

注意：传给模型的事实文本必须是游客可接受的表达，不要出现“种子数据卡”“内部配置”“讲解时可以”这类工程或运营内部措辞。

如果修改了 `data/artifacts/*.json`，当前建议重启 uvicorn。因为 `artifacts.py` 使用 `lru_cache` 缓存知识卡，重启后才会重新加载 JSON。

当用户消息中明确提到某件文物名称或别名时，`router.py` 会把匹配到的知识卡加入 LLM 上下文。例如用户问：

```text
讲讲应国玉鹰
```

后端会匹配 `应国玉鹰`，并把 `yingguo_jade_eagle` 的知识卡拼进 messages。后续接入图像识别后，识别出的 `latest_artifact_id` 也会走同一套知识卡。

当用户明确提到某件文物时，`router.py` 会把该文物 id 保存到当前设备的 `session.latest_artifact_id`。如果下一轮用户没有再说文物名，而是问：

```text
它对平顶山市有什么重要意义吗？
继续讲讲它的故事
这件东西是做什么用的？
```

后端会判断这类句子像追问，并把 `latest_artifact_id` 对应的知识卡再次加入 LLM 上下文。这样回答不是单纯依赖上一轮 assistant 的文本记忆，而是明确继承“当前设备正在看的那件文物”。

回答生成有一个重要约束：模型输出就是设备端直接展示给游客的内容。因此 prompt 明确要求模型不要输出“讲解时可以这样带”“你可以引导游客观察”这类内部指导语，而是直接生成游客可听可看的讲解文本。知识卡里的 `guide_notes` 只作为后端维护资料保留，当前不会传给模型，避免模型把内部讲解建议原样说给游客。

## Memory 实现

当前 memory 已经按 `device_id` 隔离。`sessions.py` 中维护一个进程内字典：

```python
sessions: dict[str, DeviceSession] = {}
```

每台设备都有自己的 `DeviceSession`：

```python
@dataclass
class DeviceSession:
    device_id: str
    memory: list[dict[str, str]]
    latest_image_id: str | None
    latest_artifact_id: str | None
    latest_vision_description: str | None
    last_answer: str | None
    upload_generation: int
```

每次模型完整回复结束后，会向对应设备 session 保存：

- 当前用户消息
- 当前助手回复

最多保留最近 10 轮，也就是 20 条 message：

```python
del session.memory[:-MAX_MEMORY_ROUNDS * 2]
```

注意：

- memory 只在当前 Python 进程内有效。
- 重启服务后 memory 会清空。
- 当前版本已经按 `device_id` 隔离 memory。
- `latest_artifact_id` 也按 `device_id` 隔离，用来承接“它/这件/继续讲”等文物追问。
- 后续接 ESP32 时，`walkie-01`、`walkie-02` 应该各自使用自己的 device id。

## Tool 调用实现

`router.py` 中通过简单 if/else 判断是否需要调用 tool：

```python
def should_call_tool(user_message: str) -> bool:
    text = user_message.lower()
    return "status" in text or "device" in text
```

当用户消息包含 `status` 或 `device` 时，会调用：

```python
get_device_status()
```

`tools.py` 目前返回模拟设备状态：

```json
{
  "device_id": "esp32-demo-001",
  "device_type": "ESP32",
  "online": true,
  "temperature_c": 26.8,
  "humidity_percent": 47.2
}
```

这个例子的重点不是 ESP32 本身，而是演示：

```text
用户问题 -> 后端调用外部工具 -> 把工具结果拼进 LLM 上下文 -> 模型用自然语言回答
```

以后可以把这个 tool 换成：

- 查询数据库
- 查询订单状态
- 查询真实设备
- 调用公司内部 API
- 搜索知识库

## Streaming 实现

`llm.py` 中开启 streaming：

```python
stream = await client.chat.completions.create(
    model=get_model(),
    messages=messages,
    stream=True,
    temperature=0.7,
)
```

然后逐块读取模型厂商返回的 chunk：

```python
async for chunk in stream:
    token = chunk.choices[0].delta.content
    if token:
        yield token
```

`main.py` 使用 FastAPI 的 `StreamingResponse` 把这些 token 转发给客户端：

```python
return StreamingResponse(
    chat_stream(user_message),
    media_type="text/plain; charset=utf-8",
)
```

所以流式输出的本质是：

```text
模型厂商分块返回 -> 后端分块接收 -> 后端分块转发给客户端
```

## 常见问题

### 1. POST /chat 返回 JSON decode error

通常是 PowerShell 中 JSON 引号被转义坏了。推荐使用：

```powershell
.\.venv\Scripts\python.exe chat_cli.py
```

或者使用上面的 UTF-8 `Invoke-RestMethod` 示例。

### 2. 中文变成问号或乱码

这是 Windows PowerShell 5.1 常见编码问题。手动请求时把 body 转成 UTF-8 bytes：

```powershell
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
```

终端客户端 `chat_cli.py` 已经处理了 UTF-8。

### 3. LLM_ERROR APITimeoutError

说明后端收到了请求，但连接模型 API 超时。检查：

- `.env` 是否在项目目录 `D:\develop\Projects\AI-Box`
- `OPENAI_API_KEY` 是否是真实 key
- `OPENAI_BASE_URL` 是否正确
- 当前网络是否能访问模型服务商
- 修改 `.env` 后是否重启了 uvicorn

### 4. 修改 .env 后不生效

重启 uvicorn：

```powershell
Ctrl+C
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --env-file .env
```

## 当前限制

- 没有数据库
- 没有用户登录
- 没有持久化用户系统
- 没有前端页面
- tool 是 mock 数据
- memory 重启后丢失

这些限制是刻意保留的，目的是让这个项目作为最小学习版本，先看清 LLM 后端的基本骨架。
