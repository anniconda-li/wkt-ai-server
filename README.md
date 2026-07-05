# Minimal AI Chat Backend

一个最小可运行的 AI 聊天后端示例，使用 Python 3.11、FastAPI 和 OpenAI-compatible API。

当前版本已经支持：

- `/chat` 聊天接口
- `/camera/upload` JPEG 图片上传接口
- 模型 streaming 输出
- 按设备隔离的最近 10 轮内存 memory
- 本地文物知识卡查询
- 文物追问上下文继承
- DashScope Qwen-VL 图片识别
- 后端内部文物上下文模拟接口
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
├── vision.py         # 保存相机图片、模拟视觉识别结果
├── vision_llm.py     # 调用 DashScope Qwen-VL 做候选约束识别
├── llm.py            # 封装 OpenAI-compatible streaming 调用
├── tools.py          # 示例 tool：get_device_status()
├── data/artifacts/   # 5 件核心文物的本地 JSON 知识卡
├── uploads/          # 本地上传图片目录，git 忽略
├── samples/camera/   # ESP32 实拍测试图片
├── chat_cli.py       # 终端聊天客户端，方便本地测试
├── camera_upload_cli.py # 终端图片上传测试客户端
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
  -> router.py 如果该设备有最新视觉描述，则作为辅助视觉上下文加入
  -> router.py 判断是否需要调用 tool
  -> llm.py 调用模型 API，stream=True
  -> router.py 一边转发 token，一边收集完整回复
  -> main.py 使用 StreamingResponse 返回给客户端
  -> router.py 把本轮 user/assistant 存入 memory
```

一次图片上传的完整流程：

```text
设备或本地测试脚本 POST /camera/upload?device=walkie-01
  -> main.py 读取 raw JPEG body
  -> vision.py 校验 Content-Type、大小和 JPEG 文件头
  -> vision.py 保存到 uploads/{device_id}/{image_id}.jpg
  -> main.py 如果带 artifact_id，则当作模拟识别结果
  -> main.py 如果没带 artifact_id 且配置了 VISION，则调用 vision_llm.py
  -> vision_llm.py 将图片和 5 件候选文物发给 Qwen-VL，要求返回 JSON
  -> sessions.py 写入 latest_image_id / latest_artifact_id / latest_vision_description
  -> 返回 ready
```

核心点：

- `main.py` 负责 HTTP 接口。
- `router.py` 负责业务编排。
- `sessions.py` 负责设备上下文、当前文物和短期记忆。
- `artifacts.py` 负责加载和查询本地文物知识卡。
- `vision.py` 负责图片保存和人工模拟识别结果。
- `vision_llm.py` 负责调用视觉模型，当前支持 DashScope / 百炼 OpenAI-compatible 接口。
- `llm.py` 负责模型调用。
- `tools.py` 负责外部工具能力。
- `chat_cli.py` 只是测试客户端，不参与后端核心逻辑。
- `camera_upload_cli.py` 是图片上传测试客户端，不参与后端核心逻辑。

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

DeepSeek 文本讲解模型：

```env
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_MODEL=deepseek-v4-flash
OPENAI_BASE_URL=https://api.deepseek.com
LOG_LEVEL=INFO
```

百炼统一配置。视觉识别、后续 ASR、后续 TTS 可以共用这一个 key：

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

百炼视觉识别：

```env
VISION_PROVIDER=dashscope
VISION_MODEL=qwen-vl-plus
VISION_MIN_CONFIDENCE=0.60
```

一般不需要单独写 `VISION_API_KEY` 或 `VISION_BASE_URL`，它们会默认复用 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。只有当视觉模型想换到另一个服务商或另一个百炼 endpoint 时，才需要覆盖：

```env
# VISION_API_KEY=another-api-key
# VISION_BASE_URL=https://another-compatible-endpoint/v1
```

后续 ASR / TTS 会使用这些变量，目前只是先写入配置：

```env
ASR_PROVIDER=dashscope
ASR_MODEL=paraformer-realtime-v2

TTS_PROVIDER=dashscope
TTS_MODEL=qwen3-tts-flash
TTS_VOICE=Cherry
```

当前分工：

```text
DeepSeek / OPENAI_MODEL      -> 文本讲解和问答
Qwen-VL / VISION_MODEL       -> 图片识别，输出 artifact_id
Paraformer / ASR_MODEL       -> 语音转文字，后续接
Qwen3-TTS / TTS_MODEL        -> 文字转语音，后续接
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
- `LOG_LEVEL=INFO` 会输出阶段耗时日志，测试性能时建议保留。
- 如果修改了 `.env`，需要重启 uvicorn。

## 架构约定

当前项目仍然保持最小后端结构，暂时不引入复杂分层。新增文件遵守一个原则：文件名直接对应职责。

- `main.py`：HTTP 路由入口，只做请求校验、调用业务函数、组织响应。
- `router.py`：聊天编排，负责 memory、tool、知识卡和 LLM streaming。
- `llm.py`：文本模型调用，不放业务逻辑。
- `vision.py`：图片上传校验、保存、人工模拟识别结果。
- `vision_llm.py`：视觉模型调用和识别结果 JSON 解析。
- `artifacts.py`：本地文物知识卡加载和查询。
- `sessions.py`：设备级内存状态。
- `chat_cli.py` / `camera_upload_cli.py`：本地测试客户端，不参与后端核心链路。

注释策略：只在容易误解的边界处写注释，例如“为什么 raw JPEG body”“为什么 `artifact_id` 会跳过视觉模型”。普通赋值和显而易见的代码不加重复注释，避免后面维护时注释和代码打架。

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

## 耗时日志

测试性能时，把启动后端的 PowerShell 窗口留着看日志。`LOG_LEVEL=INFO` 时，会输出每个关键阶段的耗时，单位是毫秒。

图片上传链路会看到类似日志：

```text
2026-07-05 18:20:01 INFO [ai_box.main] camera.upload.start device=walkie-01 manual_artifact=False use_vision=True content_type=image/jpeg
2026-07-05 18:20:01 INFO [ai_box.main] camera.upload.stage read_body_ms=1.2 bytes=14088
2026-07-05 18:20:01 INFO [ai_box.main] camera.upload.stage save_image_ms=3.4 image_id=... size_bytes=14088
2026-07-05 18:20:03 INFO [vision_llm] vision.recognition.stage api_call_ms=1820.5
2026-07-05 18:20:03 INFO [ai_box.main] camera.upload.stage recognition_ms=1845.7 mode=vision_llm predicted_artifact_id=yingguo_jade_eagle accepted=True confidence=0.86
2026-07-05 18:20:03 INFO [ai_box.main] camera.upload.done device=walkie-01 image_id=... latest_artifact_id=yingguo_jade_eagle total_ms=1860.2
```

聊天链路会看到：

```text
2026-07-05 18:20:10 INFO [router] chat.stage build_messages_ms=0.6 device=walkie-01 messages=4 latest_artifact_id=yingguo_jade_eagle
2026-07-05 18:20:11 INFO [router] chat.stage first_token_ms=760.4 device=walkie-01
2026-07-05 18:20:15 INFO [router] chat.done device=walkie-01 output_chars=218 total_ms=4730.8
```

重点看这几个字段：

- `api_call_ms`：视觉模型本身耗时，通常是图片识别最慢的部分。
- `recognition_ms`：整个识别阶段耗时，包含模型调用和本地解析。
- `first_token_ms`：聊天首 token 延迟，决定用户感觉“等了多久才开始说话”。
- `total_ms`：接口总耗时。

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

使用真实视觉模型识别 ESP32 实拍样例图。注意：这里不要传 `--artifact-id`，后端会调用 `VISION_MODEL`：

```powershell
.\.venv\Scripts\python.exe camera_upload_cli.py `
  samples\camera\yingguo_jade_eagle_esp32.jpg `
  --device walkie-01
```

另一张样例图：

```powershell
.\.venv\Scripts\python.exe camera_upload_cli.py `
  samples\camera\shuyao_chuilin_sheng_ding_esp32.jpg `
  --device walkie-01
```

只保存图片、不调用视觉模型：

```powershell
.\.venv\Scripts\python.exe camera_upload_cli.py `
  samples\camera\yingguo_jade_eagle_esp32.jpg `
  --device walkie-01 `
  --no-vision
```

也可以用人工标注对照模式。传了 `--artifact-id` 后，不会调用视觉模型，而是直接使用你指定的文物 id：

```powershell
.\.venv\Scripts\python.exe camera_upload_cli.py `
  samples\camera\yingguo_jade_eagle_esp32.jpg `
  --device walkie-01 `
  --artifact-id yingguo_jade_eagle `
  --vision-description "ESP32 实拍图：浅色玉质鹰形器，呈展翅姿态。"
```

人工标注另一张样例图：

```powershell
.\.venv\Scripts\python.exe camera_upload_cli.py `
  samples\camera\shuyao_chuilin_sheng_ding_esp32.jpg `
  --device walkie-01 `
  --artifact-id shuyao_chuilin_sheng_ding `
  --vision-description "ESP32 实拍图：青铜升鼎，双耳外撇，三足，器身有复杂纹饰。"
```

上传后再用同一个设备 id 聊天：

```powershell
.\.venv\Scripts\python.exe chat_cli.py --device walkie-01
```

```text
You> 这是什么？
AI> 这是……
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

### POST /sessions/{device_id}/artifact-context

后端内部模拟接口，用来直接设置某台设备“当前正在看的文物”。这一步相当于以后 `/camera/upload` 识别成功后的结果写入，但现在不需要 ESP32，也不需要真的上传图片。

请求：

```json
{
  "artifact_id": "yingguo_jade_eagle",
  "image_id": "local-test-001",
  "vision_description": "画面中是一件浅色玉质鹰形器，呈展翅姿态，可见线雕羽翼。"
}
```

PowerShell 示例：

```powershell
$body = @{
  artifact_id = "yingguo_jade_eagle"
  image_id = "local-test-001"
  vision_description = "画面中是一件浅色玉质鹰形器，呈展翅姿态，可见线雕羽翼。"
} | ConvertTo-Json -Compress

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/sessions/walkie-01/artifact-context" `
  -Method POST `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

返回：

```json
{
  "status": "ready",
  "device_id": "walkie-01",
  "latest_artifact_id": "yingguo_jade_eagle",
  "latest_artifact_name": "应国玉鹰",
  "latest_image_id": "local-test-001",
  "latest_vision_description": "画面中是一件浅色玉质鹰形器，呈展翅姿态，可见线雕羽翼。",
  "upload_generation": 1
}
```

然后就可以直接问：

```powershell
.\.venv\Scripts\python.exe chat_cli.py --device walkie-01
```

```text
You> 这是什么？
AI> 这是应国玉鹰，又称白玉线雕鹰……

You> 它对平顶山市有什么重要意义？
AI> 它是平顶山“鹰城”文化的重要象征……
```

### POST /camera/upload

相机图片上传接口。接口会保存 JPEG，并根据请求参数决定识别方式：

- 不传 `artifact_id`：调用 `VISION_MODEL`，当前推荐 `qwen-vl-plus`。
- 传 `artifact_id`：跳过视觉模型，把它当作人工指定的模拟识别结果，方便对照测试。
- 传 `use_vision=false`：只保存图片，不识别，并清空该设备旧的 `latest_artifact_id`。

如果没有配置 `VISION_API_KEY` 或 `DASHSCOPE_API_KEY`，不传 `artifact_id` 时也会退化成“只保存图片、不识别”。

接口形态特意使用 raw JPEG body，而不是 multipart。这样以后 ESP32 C 端更容易对齐：HTTP body 直接发送 JPEG 字节即可。

真实视觉识别请求：

```text
POST /camera/upload?device=walkie-01
Content-Type: image/jpeg

<JPEG bytes>
```

人工标注对照请求：

```text
POST /camera/upload?device=walkie-01&artifact_id=yingguo_jade_eagle
Content-Type: image/jpeg

<JPEG bytes>
```

PowerShell 示例：

```powershell
$imageBytes = [System.IO.File]::ReadAllBytes("D:\test\artifact.jpg")

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/camera/upload?device=walkie-01" `
  -Method POST `
  -ContentType "image/jpeg" `
  -Body $imageBytes
```

返回示例：

```json
{
  "status": "ready",
  "device_id": "walkie-01",
  "image": {
    "image_id": "20260705T120000000000Z_abcd1234",
    "filename": "20260705T120000000000Z_abcd1234.jpg",
    "path": "D:\\develop\\Projects\\AI-Box\\uploads\\walkie-01\\20260705T120000000000Z_abcd1234.jpg",
    "size_bytes": 123456,
    "content_type": "image/jpeg"
  },
  "recognition": {
    "mode": "vision_llm",
    "artifact_id": "yingguo_jade_eagle",
    "artifact_name": "应国玉鹰",
    "predicted_artifact_id": "yingguo_jade_eagle",
    "confidence": 0.86,
    "accepted": true,
    "evidence": ["浅色玉质", "鹰形", "展翅", "线雕羽翼"]
  },
  "latest_artifact_id": "yingguo_jade_eagle",
  "latest_image_id": "20260705T120000000000Z_abcd1234",
  "upload_generation": 1
}
```

上传成功后，可以直接用同一个设备 id 追问：

```powershell
.\.venv\Scripts\python.exe chat_cli.py --device walkie-01
```

```text
You> 这是什么？
AI> 这是应国玉鹰……
```

如果视觉模型返回 `unknown`，或者置信度低于 `VISION_MIN_CONFIDENCE`，接口仍会返回 `ready`，但不会写入 `latest_artifact_id`，避免误把不确定图片当成某件文物。

项目里已经放了两张 ESP32 实拍样例图：

- `samples/camera/yingguo_jade_eagle_esp32.jpg`
- `samples/camera/shuyao_chuilin_sheng_ding_esp32.jpg`

它们是测试夹具，不是知识库事实。现在可以直接不传 `artifact_id` 测真实视觉识别，也可以传 `artifact_id` 做人工标注对照。

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
  "reference_images": [
    {
      "title": "白玉线雕鹰（图①）",
      "source_page": "https://www.news.cn/shuhua/20211116/a840ac11f5d243ccb34457ef307d8ae8/c.html",
      "credit": "图片由平顶山博物馆提供"
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
- `reference_images`：公开图片来源记录，供后续人工核对、参考图整理或检索素材使用，当前不会传给模型。
- `data_status`：标记资料状态，例如 `public_web_enriched`。

注意：传给模型的事实文本必须是游客可接受的表达，不要出现“种子数据卡”“内部配置”“讲解时可以”这类工程或运营内部措辞。

如果修改了 `data/artifacts/*.json`，当前建议重启 uvicorn。因为 `artifacts.py` 使用 `lru_cache` 缓存知识卡，重启后才会重新加载 JSON。

当用户消息中明确提到某件文物名称或别名时，`router.py` 会把匹配到的知识卡加入 LLM 上下文。例如用户问：

```text
讲讲应国玉鹰
```

后端会匹配 `应国玉鹰`，并把 `yingguo_jade_eagle` 的知识卡拼进 messages。`/camera/upload` 调用视觉模型识别出的 `latest_artifact_id` 也会走同一套知识卡。

当用户明确提到某件文物时，`router.py` 会把该文物 id 保存到当前设备的 `session.latest_artifact_id`。如果下一轮用户没有再说文物名，而是问：

```text
它对平顶山市有什么重要意义吗？
继续讲讲它的故事
这件东西是做什么用的？
```

后端会判断这类句子像追问，并把 `latest_artifact_id` 对应的知识卡再次加入 LLM 上下文。这样回答不是单纯依赖上一轮 assistant 的文本记忆，而是明确继承“当前设备正在看的那件文物”。

如果想绕过图片上传和视觉模型，仍然可以使用：

```text
POST /sessions/{device_id}/artifact-context
```

手动设置 `latest_artifact_id` 和 `latest_vision_description`。这相当于后端内部模拟“拍照识别成功”，适合做对照测试。

现在 `/camera/upload` 已经具备同样的状态写入能力：先保存 JPEG；如果没有人工 `artifact_id`，就调用 `vision_llm.py`，把图片和 5 件候选文物的视觉特征发给 `VISION_MODEL`，并解析模型输出的 `artifact_id / confidence / evidence / vision_description`。

视觉识别只使用候选文物的名称、别名、类别、材质、`visual_keywords` 和 `recognition_features`。历史事实和讲解文本仍由聊天阶段使用，避免视觉模型把“讲故事”混进识别任务。

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

### 5. uploads 里的图片打不开

正常通过 `/camera/upload` 上传的文件应该能作为普通 `.jpg` 打开。如果打不开，通常说明保存进去的 body 不是完整 JPEG。

检查文件大小：

```powershell
Get-ChildItem -Recurse uploads | Select-Object FullName,Length
```

如果只有十几字节，基本就是测试时写入的假图片，或者请求 body 不是图片字节。当前接口已经检查：

- `Content-Type` 必须是 `image/jpeg`
- 文件大小不能太小
- 文件必须有 JPEG 开头 `FF D8 FF`
- 文件必须有 JPEG 结束标记 `FF D9`

PowerShell 上传时要用：

```powershell
$imageBytes = [System.IO.File]::ReadAllBytes("D:\test\artifact.jpg")
```

不要把图片路径字符串、Base64 文本、JSON 或普通字符串当成 body 发给 `/camera/upload`。

## 当前限制

- 没有数据库
- 没有用户登录
- 没有持久化用户系统
- 没有前端页面
- tool 是 mock 数据
- memory 重启后丢失

这些限制是刻意保留的，目的是让这个项目作为最小学习版本，先看清 LLM 后端的基本骨架。
