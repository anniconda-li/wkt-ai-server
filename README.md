# wkt-ai-server

对讲设备平台的独立 AI 后端，使用 Python 3.11、FastAPI 和 OpenAI-compatible API。

## 项目身份与边界

- 项目名、本地目录名、GitHub 仓库名和 Docker 镜像名统一为 `wkt-ai-server`。
- Compose 服务名固定为 `ai`，容器名为 `wkt-ai-server`。
- 本仓库只负责现有 AI 语音、ASR、TTS、WAV 分片和相机分析能力。
- 本仓库保持独立 Git 仓库、独立 GitHub 仓库、独立 Docker 镜像和独立容器，不与对讲服务或 OTA 服务合并。
- 本地父目录 `wkt-platform` 只用于归档四个独立 Git 仓库，不初始化 Git，也不是 monorepo：

  ```text
  wkt-platform/
  ├── wkt-intercom-server
  ├── wkt-ai-server
  ├── wkt-ota-server
  └── wkt-deploy
  ```

- `wkt-deploy` 负责跨服务部署编排；本仓库只提供 `ai` 服务自己的镜像、端口、健康检查和运行契约。
- ESP-IDF 固件项目 `walkie-talkiev1` 位于 `wkt-platform` 之外，完全独立于本服务。
- 本次名称规范化不改变现有 API 路径、端口、协议或请求响应格式。

当前版本已经支持：

- `/chat` 聊天接口
- `/camera/upload` JPEG 图片上传接口
- `/ai/*` ESP32 语音分片上传、ASR、轮询结果、按需拉取 WAV 协议
- 本地文本 -> LLM -> TTS -> ESP32 WAV 测试链路
- 模型 streaming 输出
- 按设备隔离的最近 10 轮内存 memory
- 本地文物知识卡查询
- 文物追问上下文继承
- DashScope Qwen-VL 图片识别
- DashScope Paraformer 语音识别
- DashScope / OpenAI-compatible TTS 输出设备标准 WAV
- 后端内部文物上下文模拟接口
- 简单 tool 调用机制
- OpenAI / DeepSeek / 其他 OpenAI-compatible 服务商
- Windows PowerShell 下的终端测试客户端

项目没有引入 LangChain、Dify 等框架，代码量尽量少，方便学习后端调用 LLM 的基本链路。

## 项目结构

```text
.
├── main.py           # FastAPI app，定义 HTTP 接口
├── ai_protocol.py    # ESP32 /ai 语音协议、session、分片、取消状态
├── asr.py            # 调用 DashScope Paraformer，把设备 WAV 转成文字
├── pipeline.py       # 本地文本 -> LLM -> TTS -> 设备 WAV 流水线
├── tts.py            # 调用 TTS 并统一输出 ESP32 标准 WAV
├── wav_utils.py      # 设备 WAV 格式校验和测试 WAV 生成
├── router.py         # 组装上下文、memory、tool 判断、流式转发
├── output_format.py  # 清理 Markdown，生成设备端可直接显示的纯文本
├── text_normalize.py # ASR 文本拼音近似纠错，修正文物名同音错字
├── sessions.py       # 按 device_id 管理设备会话和 memory
├── artifacts.py      # 加载本地文物知识卡
├── vision.py         # 保存相机图片、模拟视觉识别结果
├── vision_llm.py     # 调用 DashScope Qwen-VL 做候选约束识别
├── llm.py            # 封装 OpenAI-compatible streaming 调用
├── tools.py          # 示例 tool：get_device_status()
├── data/artifacts/   # 5 件核心文物的本地 JSON 知识卡
├── uploads/          # 本地上传图片目录，git 忽略，每台设备默认只保留最近 10 张
├── outputs/          # 本地生成回复 WAV 目录，git 忽略
├── samples/camera/   # ESP32 实拍测试图片
├── chat_cli.py       # 终端聊天客户端，方便本地测试
├── camera_upload_cli.py # 终端图片上传测试客户端
├── local_text_ask_cli.py # 本地文本问答 + TTS 测试客户端
├── ai_wav_cli.py     # 模拟 ESP32 /ai 协议上传 WAV 并拉取回复
├── requirements.txt  # Python 依赖
├── pyproject.toml    # wkt-ai-server 项目元数据
├── Dockerfile       # 独立镜像构建配置
├── compose.yaml     # 本地 Compose 配置，服务名 ai
├── tests/           # 不访问外部模型服务的冒烟测试
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
  -> router.py 如果该设备已有 latest_artifact_id，则默认继承当前文物
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
  -> vision.py 清理该设备目录下较旧图片，默认只保留最近 10 张
  -> main.py 如果带 artifact_id，则当作模拟识别结果
  -> main.py 如果没带 artifact_id 且配置了 VISION，则调用 vision_llm.py
  -> vision_llm.py 将图片和 5 件候选文物发给 Qwen-VL，要求返回 JSON
  -> sessions.py 写入 latest_image_id / latest_artifact_id / latest_vision_description
  -> 返回 ready
```

一次 ESP32 语音请求的协议流程：

```text
设备 POST /ai/start，提交 device、language 和可选 audio_format
  -> ai_protocol.py 创建语音 session，返回 session 和 chunk_size
设备 POST /ai/upload 分片上传 PCM WAV 或 AOP1 裸 Opus 帧
  -> ai_protocol.py 按 session + offset 组装请求音频
  -> AOP1 在服务端校验并封装成标准 Ogg/Opus，同时解码 WAV 供静音判断和 ASR 回退
设备 POST /ai/finish
  -> 后端立即返回当前状态，并在后台处理 ASR / LLM / TTS
设备 POST /ai/result_info 每秒轮询
  -> 文本先完成时返回 answer_text
  -> TTS 完成时返回 reply_wav_size
设备 POST /ai/result_chunk 按 offset/len 拉取原始 WAV bytes
```

一次本地文本 pipeline 的流程：

```text
本地输入文本
  -> pipeline.py 调用 router.py / llm.py 生成 answer_text
  -> tts.py 调用 TTS 服务
  -> wav_utils.py 将返回音频统一成 16k/16-bit/mono PCM WAV
  -> 保存到 outputs/replies/{device_id}/
```

核心点：

- `main.py` 负责 HTTP 接口。
- `ai_protocol.py` 负责 ESP32 语音协议状态机。
- `asr.py` 负责调用 DashScope Qwen3-ASR，并在需要时回退到 Paraformer。
- `opus_packets.py` 负责校验设备 AOP1 裸 Opus 帧并封装成标准 Ogg/Opus。
- `pipeline.py` 负责本地文本到回答和 TTS WAV 的完整流水线。
- `tts.py` 负责调用 TTS，并把返回音频统一成设备要求的 WAV。
- `wav_utils.py` 负责校验设备要求的 WAV 格式。
- `router.py` 负责业务编排。
- `output_format.py` 负责把模型回复规整成设备端可直接显示和朗读的纯文本。
- `text_normalize.py` 负责在文本进入 LLM 前修正文物名同音错字。
- `sessions.py` 负责设备上下文、当前文物和短期记忆。
- `artifacts.py` 负责加载和查询本地文物知识卡。
- `vision.py` 负责图片保存和人工模拟识别结果。
- `vision_llm.py` 负责调用视觉模型，当前支持 DashScope / 百炼 OpenAI-compatible 接口。
- `llm.py` 负责模型调用。
- `tools.py` 负责外部工具能力。
- `chat_cli.py` 只是测试客户端，不参与后端核心逻辑。
- `camera_upload_cli.py` 是图片上传测试客户端，不参与后端核心逻辑。
- `ai_wav_cli.py` 是设备语音协议测试客户端，不参与后端核心逻辑。

## 环境准备

本机推荐使用 `uv` 创建虚拟环境：

```powershell
cd C:\path\to\wkt-ai-server
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

百炼统一配置。视觉识别、ASR、TTS 可以共用这一个 key：

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

百炼视觉识别：

```env
VISION_PROVIDER=dashscope
VISION_MODEL=qwen3.6-flash-2026-04-16
VISION_ENABLE_THINKING=false
VISION_MIN_CONFIDENCE=0.60
MAX_SAVED_IMAGES_PER_DEVICE=10
```

一般不需要单独写 `VISION_API_KEY` 或 `VISION_BASE_URL`，它们会默认复用 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。只有当视觉模型想换到另一个服务商或另一个百炼 endpoint 时，才需要覆盖：

```env
# VISION_API_KEY=another-api-key
# VISION_BASE_URL=https://another-compatible-endpoint/v1
```

百炼 ASR 和 TTS：

```env
ASR_PROVIDER=dashscope
ASR_MODEL=qwen3-asr-flash-2026-02-10
ASR_FALLBACK_MODEL=paraformer-realtime-v2
ASR_LANGUAGE=zh
ASR_ENABLE_ITN=false
ASR_TIMEOUT_SECONDS=60
# ASR_API_KEY 默认复用 DASHSCOPE_API_KEY
ASR_FRAME_BYTES=3200
ASR_FRAME_SLEEP_SECONDS=0
ASR_ARTIFACT_PHONETIC_THRESHOLD=0.86

TTS_PROVIDER=dashscope
TTS_API_STYLE=dashscope_qwen
TTS_MODEL=qwen3-tts-flash-2025-11-27
TTS_VOICE=Cherry
TTS_LANGUAGE_TYPE=Chinese
TTS_TIMEOUT_SECONDS=120
```

当前设备在 `/ai/finish` 后处理完整请求音频（PCM WAV 或 AOP1/Opus），因此主 ASR 使用同步短音频模型。主模型请求失败时会自动回退到 `ASR_FALLBACK_MODEL`；将该变量留空可以关闭回退。Qwen3-ASR 单文件限制为 5 分钟、10 MB，超过设备协议约束的音频会在调用前失败。

`TTS_API_KEY` 默认复用 `DASHSCOPE_API_KEY`。Qwen3-TTS 默认请求百炼公共 API：

```text
https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

一般不需要配置 `TTS_BASE_URL`。如果以后换私有 endpoint，可以用 `TTS_ENDPOINT` 完整覆盖。

真实 TTS 可能返回 WAV、MP3 或其他常见音频。后端会优先使用 `ffmpeg` 统一转成 ESP32 要的格式：

```text
WAV / PCM s16le / 16000 Hz / mono
```

如果 `ffmpeg` 不在 PATH，可以在 `.env` 里指定：

```env
FFMPEG_BIN=D:\tools\ffmpeg\bin\ffmpeg.exe
```

当前分工：

```text
DeepSeek / OPENAI_MODEL      -> 文本讲解和问答
Qwen3-VL / VISION_MODEL            -> 图片识别，输出 artifact_id
Qwen3-ASR / ASR_MODEL              -> 完整音频语音转文字
Paraformer / ASR_FALLBACK_MODEL    -> 主 ASR 失败时回退
Qwen3-TTS / TTS_MODEL              -> 文字转语音，输出 ESP32 标准 WAV
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
cd C:\path\to\wkt-ai-server
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

## Docker / Compose 本地验证

镜像名、服务名和容器名已经固定，下面的命令只用于本地或测试环境，不会部署生产环境：

```powershell
docker build -t wkt-ai-server:local .
docker compose up --build ai
```

Compose 保持现有 `8000` 端口和 `.env` 配置方式。详细说明见 [`docs/deployment.md`](docs/deployment.md)。

供 `wkt-deploy` 使用的固定契约摘要：镜像 `wkt-ai-server`、容器 `wkt-ai-server`、Compose 服务 `ai`、容器端口 `8000`、健康检查 `GET /health`、持久化目录 `/app/uploads` 和 `/app/outputs`。全部路由、环境变量、请求体限制、超时和单实例状态约束以 [`docs/deployment.md`](docs/deployment.md) 为准。

### GHCR 正式镜像发布

- 普通 push 和 pull request 只运行测试，不发布正式镜像。
- 推送严格符合 `vX.Y.Z` 的 Git 标签时发布 GHCR 镜像；例如 Git 标签 `v0.1.0` 对应镜像标签 `0.1.0`。
- 每次发布同时生成 `sha-*` 镜像标签。
- 正式镜像地址为 `ghcr.io/anniconda-li/wkt-ai-server:<version>`；部署时禁止使用 `latest`。

## 耗时日志

测试性能时，把启动后端的 PowerShell 窗口留着看日志。`LOG_LEVEL=INFO` 时，会输出每个关键阶段的耗时，单位是毫秒。

图片上传链路会看到类似日志：

```text
2026-07-05 18:20:01 INFO [wkt_ai_server.main] camera.upload.start device=walkie-01 manual_artifact=False use_vision=True content_type=image/jpeg
2026-07-05 18:20:01 INFO [wkt_ai_server.main] camera.upload.stage read_body_ms=1.2 bytes=14088
2026-07-05 18:20:01 INFO [wkt_ai_server.main] camera.upload.stage save_image_ms=3.4 image_id=... size_bytes=14088
2026-07-05 18:20:03 INFO [vision_llm] vision.recognition.stage api_call_ms=1820.5
2026-07-05 18:20:03 INFO [wkt_ai_server.main] camera.upload.stage recognition_ms=1845.7 mode=vision_llm predicted_artifact_id=yingguo_jade_eagle accepted=True confidence=0.86
2026-07-05 18:20:03 INFO [wkt_ai_server.main] camera.upload.done device=walkie-01 image_id=... latest_artifact_id=yingguo_jade_eagle total_ms=1860.2
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
cd C:\path\to\wkt-ai-server
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

本地跑通“文本 -> LLM -> TTS -> ESP32 WAV”完整链路，不需要 ASR，也不需要设备端：

```powershell
.\.venv\Scripts\python.exe local_text_ask_cli.py `
  --device walkie-01 `
  --artifact-id yingguo_jade_eagle `
  "这是什么"
```

成功后会输出回答文本，并在 `outputs/replies/walkie-01/` 下生成回复 WAV。生成的 WAV 会被校验为：

```text
WAV / PCM / 16000 Hz / 16-bit / mono / little-endian
```

如果只想先测 LLM，不生成语音：

```powershell
.\.venv\Scripts\python.exe local_text_ask_cli.py `
  --device walkie-01 `
  --artifact-id yingguo_jade_eagle `
  --no-tts `
  "这是什么"
```

如果暂时不想请求真实 TTS，可以在 `.env` 里打开 mock TTS：

```env
AI_ENABLE_MOCK_TTS=true
```

这会生成符合设备格式的静音 WAV，只用于测试文件格式和分片读取，不代表真实语音合成效果。

如果要测试真实 TTS，请确保 `.env` 里是：

```env
AI_ENABLE_MOCK_TTS=false
```

不需要配置 `TTS_BASE_URL`。如果真实 TTS 返回 MP3 或非 PCM WAV，建议安装 `ffmpeg`，或者在 `.env` 里设置 `FFMPEG_BIN`。

完整模拟 ESP32 语音协议：

```powershell
.\.venv\Scripts\python.exe ai_wav_cli.py `
  D:\path\to\request.wav `
  --device walkie-01
```

`request.wav` 必须是设备同款格式：

```text
WAV / PCM / 16000 Hz / 16-bit / mono / little-endian
```

`ai_wav_cli.py` 会依次调用 `/ai/start`、`/ai/upload`、`/ai/finish`、`/ai/result_info`，如果 TTS 成功，还会调用 `/ai/result_chunk` 下载回复音频，默认保存到 `outputs/device_protocol/{device}/`。

如果你手头暂时没有录音 WAV，可以先用真实 TTS 生成一句“这是什么”当测试输入：

```powershell
$env:AI_MOCK_LLM_TEXT = "这是什么"
.\.venv\Scripts\python.exe local_text_ask_cli.py --device walkie-01 "生成测试请求音频"
Remove-Item Env:\AI_MOCK_LLM_TEXT
```

然后把上一步输出的 `reply_wav_path` 作为 `ai_wav_cli.py` 的输入。这个方法只是本地联调用，真实设备会直接上传麦克风录到的 WAV。

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

- 不传 `artifact_id`：调用 `VISION_MODEL`，当前使用 `qwen3.6-flash-2026-04-16`，关闭思考并要求 JSON 输出。
- 传 `artifact_id`：跳过视觉模型，把它当作人工指定的模拟识别结果，方便对照测试。
- 传 `use_vision=false`：只保存图片，不识别，并清空该设备旧的 `latest_artifact_id`。

如果没有配置 `VISION_API_KEY` 或 `DASHSCOPE_API_KEY`，不传 `artifact_id` 时也会退化成“只保存图片、不识别”。

为了避免服务器磁盘一直增长，后端会按设备清理旧图片。默认每台设备只保留最近 10 张上传图片，可通过 `.env` 调整：

```env
MAX_SAVED_IMAGES_PER_DEVICE=10
```

清理只针对 `uploads/{device_id}/` 下的 `.jpg` 图片，不会影响语音请求和回复 WAV。

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
    "path": "C:\\path\\to\\wkt-ai-server\\uploads\\walkie-01\\20260705T120000000000Z_abcd1234.jpg",
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

视觉识别不是强制五选一，但也不会因为 ESP32 图片稍微模糊就直接放弃。`vision_llm.py` 会要求模型先判断图片里有没有可辨认文物主体；如果还能看出器型、材质、轮廓、纹饰、孔洞、足、耳、钮、翅形等关键特征，并且这些特征能匹配某件候选文物，就应该返回该文物并给出合理置信度。只有空照片、黑屏、严重过曝、桌面、墙面、手指、普通环境，或者无法和任何候选文物建立关键特征匹配时，才返回 `artifact_id: "unknown"`。后端会保存图片和视觉描述，但不会把 `unknown` 写成当前文物上下文。

项目里已经放了两张 ESP32 实拍样例图：

- `samples/camera/yingguo_jade_eagle_esp32.jpg`
- `samples/camera/shuyao_chuilin_sheng_ding_esp32.jpg`

它们是测试夹具，不是知识库事实。现在可以直接不传 `artifact_id` 测真实视觉识别，也可以传 `artifact_id` 做人工标注对照。

### POST /ai/start

ESP32 语音协议入口。请求上行兼容旧的 PCM WAV 和新的 AOP1 裸 Opus 帧；回复下行仍为 PCM WAV。

请求：

```json
{"device":"walkie-02","language":"zh","audio_format":"opus_packets_v1"}
```

响应：

```json
{"ok":true,"session":"abc123","audio_format":"opus_packets_v1","chunk_size":32768}
```

后端会用 `device` 绑定语音 session。后续 `/ai/upload`、`/ai/finish`、`/ai/result_info`、`/ai/result_chunk` 即使只传 `session` 也能找到设备上下文；如果设备端额外带 `device=walkie-02`，后端会校验它和 session 里的设备是否一致。

### POST /ai/upload

上传请求音频分片。`audio_format` 省略时按旧的 `pcm_wav` 处理；新设备使用 `opus_packets_v1`。

```text
POST /ai/upload?session=abc123&device=walkie-02&index=0&offset=0&total=156024
Content-Type: application/vnd.wkt.opus-packets

<AOP1 bytes chunk>
```

响应：

```json
{"ok":true}
```

后端按 `session + offset` 写入文件，分片不必与 Opus 包边界对齐。AOP1 最大 256 KiB，格式为 24 字节小端文件头，之后循环保存 `uint16 packet_len + raw Opus packet`。固定参数为 16 kHz、单声道、320 samples/20 ms，每包最大 1275 字节，最长 3000 帧/60 秒。

AOP1 会额外解码一份 WAV 用于静音检测和 Paraformer 回退，因此传统 Python 启动也需要本机 `PATH` 中存在 `ffmpeg`，或通过 `FFMPEG_BIN` 指定；Docker 镜像已经安装。

旧设备的 PCM WAV 最大约 2.1 MB，并继续使用 `application/octet-stream`：

```text
Container: WAV
Codec: PCM
Sample rate: 16000 Hz
Bit depth: 16-bit
Channels: mono
Endian: little-endian
```

### POST /ai/finish

通知上传完成并启动后台处理。这个接口会快速返回，设备端即使遇到响应超时，也可以继续轮询 `/ai/result_info`。

```text
POST /ai/finish?session=abc123&device=walkie-02
```

当前版本的处理策略：

- 如果 WAV 或 AOP1 无效，返回 `failed`。
- 如果音频太短或接近静音，返回 `no_speech`，`answer_text` 默认为“我没有听清，请再说一遍。”。
- AOP1 会无损封装为 Ogg/Opus；Qwen3-ASR 直接接收 Ogg，Paraformer 回退使用服务端解码的 WAV。
- ASR 文本进入 LLM 前会经过 `text_normalize.py` 的文物名拼音近似纠错，例如“英国玉婴”“应国玉英”会修正为“应国玉鹰”，再进行本地知识卡匹配。
- 如果 ASR 配置错误或服务调用失败，返回 `failed`，并通过 `answer_text` 给设备一段可显示的错误提示，避免设备一直轮询。
- 如果配置了 `AI_MOCK_ASR_TEXT`，后端会跳过真实 ASR，用这段文本调用现有 LLM 编排。
- LLM 生成 `answer_text` 后，后端会异步调用 TTS，并把结果统一保存为 16k/16-bit/mono PCM WAV。
- 如果 `AI_ENABLE_MOCK_TTS=true`，会跳过真实 TTS，生成一个符合设备格式的静音 WAV，用来测试 `result_chunk`。

本地联调可以在 `.env` 里临时加：

```env
AI_MOCK_ASR_TEXT=这是什么
AI_MOCK_LLM_TEXT=这是一个本地模拟回答，用来测试设备端轮询和音频拉取。
AI_ENABLE_MOCK_TTS=true
```

### POST /ai/result_info

设备轮询结果。

```text
POST /ai/result_info?session=abc123&device=walkie-02
Content-Type: application/json

{}
```

文本先就绪时：

```json
{
  "ok": true,
  "session": "abc123",
  "device": "walkie-02",
  "status": "text_ready",
  "asr_text": "这是什么",
  "answer_text": "这是应国玉鹰……",
  "audio_ready": false,
  "reply_wav_ready": false,
  "reply_wav_size": 0,
  "tts_status": "pending",
  "tts_error": null
}
```

语音就绪时：

```json
{
  "ok": true,
  "session": "abc123",
  "status": "audio_ready",
  "answer_text": "这是应国玉鹰……",
  "audio_ready": true,
  "reply_wav_ready": true,
  "reply_wav_size": 12844,
  "tts_status": "done",
  "tts_error": null
}
```

静音或空语音：

```json
{
  "ok": true,
  "session": "abc123",
  "status": "no_speech",
  "asr_text": "",
  "answer_text": "我没有听清，请再说一遍。",
  "audio_ready": false,
  "reply_wav_ready": false,
  "reply_wav_size": 0,
  "tts_status": "skipped",
  "tts_error": null
}
```

TTS 失败但文本保留：

```json
{
  "ok": true,
  "session": "abc123",
  "status": "audio_failed",
  "answer_text": "这是应国玉鹰……",
  "audio_ready": false,
  "reply_wav_ready": false,
  "reply_wav_size": 0,
  "tts_status": "failed",
  "tts_error": "TTS HTTP 401: invalid api key"
}
```

`no_speech`、`audio_failed`、`cancelled`、`failed` 都是终态，设备端不应该继续轮询到 300 秒超时。

### POST /ai/result_chunk

按需拉取回复 WAV。响应 body 是原始 WAV 字节，不是 JSON，也不是 Base64。

```text
POST /ai/result_chunk?session=abc123&device=walkie-02&offset=0&len=32768
Content-Type: application/json

{}
```

后端按 `offset + len` 精确读取，最后一片由设备按剩余长度请求。只有 `status == "audio_ready"` 且 `audio_ready == true` 时才应该调用。

### POST /ai/cancel

通用取消，表示用户放弃当前这次 AI 任务。

```text
POST /ai/cancel?session=abc123&device=walkie-02
```

后端会标记：

```json
{
  "ok": true,
  "session": "abc123",
  "status": "cancelled",
  "audio_ready": false,
  "reply_wav_ready": false,
  "reply_wav_size": 0,
  "tts_status": "cancelled"
}
```

取消后的旧 session 结果不会写成可播放音频。第三方 API 调用如果已经发出，当前版本不强杀，但结果回来后会根据 `cancel_requested` 丢弃。

### POST /ai/stop_audio

只停止回复音频播放或拉取，不取消整次问答，不清空 `answer_text`。

```text
POST /ai/stop_audio?session=abc123&device=walkie-02
```

如果 TTS 还没开始或正在生成，后端会把 `tts_status` 标记为 `stopped`。如果音频已经生成，后端不会删除文本回答；设备端仍应以本地播放状态为准。

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

当用户明确提到某件文物时，`router.py` 会把该文物 id 保存到当前设备的 `session.latest_artifact_id`。当 `/camera/upload` 视觉识别成功时，也会把识别出的文物 id 写入同一个字段。

后续同一个 `device_id` 再调用 `/chat` 时，只要 `session.latest_artifact_id` 存在，后端默认就会把当前文物知识卡拼进 LLM 上下文。也就是说，聊天阶段不再依赖“这又是什么”这种句式是否命中追问关键词，而是把“当前设备正在看的文物”当作设备状态的一部分。

所以这些问法都会承接最新图片：

```text
它对平顶山市有什么重要意义吗？
继续讲讲它的故事
这件东西是做什么用的？
这又是什么？
这个又是啥？
```

如果用户明确说出另一件文物名称，`router.py` 会优先匹配新文物并更新 `latest_artifact_id`。这样回答不是单纯依赖上一轮 assistant 的文本记忆，而是明确继承“当前设备正在看的那件文物”。

如果想绕过图片上传和视觉模型，仍然可以使用：

```text
POST /sessions/{device_id}/artifact-context
```

手动设置 `latest_artifact_id` 和 `latest_vision_description`。这相当于后端内部模拟“拍照识别成功”，适合做对照测试。

现在 `/camera/upload` 已经具备同样的状态写入能力：先保存 JPEG；如果没有人工 `artifact_id`，就调用 `vision_llm.py`，把图片和 5 件候选文物的视觉特征发给 `VISION_MODEL`，并解析模型输出的 `artifact_id / confidence / evidence / vision_description`。

视觉识别只使用候选文物的名称、别名、类别、材质、`visual_keywords` 和 `recognition_features`。历史事实和讲解文本仍由聊天阶段使用，避免视觉模型把“讲故事”混进识别任务。

回答生成有一个重要约束：模型输出就是设备端直接展示给游客的内容。因此 prompt 明确要求模型不要输出“讲解时可以这样带”“你可以引导游客观察”这类内部指导语，而是直接生成游客可听可看的讲解文本。知识卡里的 `guide_notes` 只作为后端维护资料保留，当前不会传给模型，避免模型把内部讲解建议原样说给游客。

为了适配 ESP32 / LVGL 小屏显示，后端不会把模型原始 Markdown 直接给设备。`output_format.py` 会在非流式问答链路里清理 `**`、标题、列表、链接、代码符号和转义换行，并把 `answer_text` 规整成纯文本。TTS 也使用这份清洗后的文本，避免把 Markdown 符号读出来。

此外，`router.py` 对两类高频入口做了确定性兜底：

- 用户问“你是谁”时，直接回答“我是平顶山市博物馆的 AI 讲解助手”，不让模型绕开身份问题。
- 当前设备没有识别到文物时，用户说“介绍一下吧”“讲讲吧”“这是什么”等模糊追问，后端会提示先拍展品照片或说出文物名称，不会编造“今天特展”或不存在的当前展品。

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

- `.env` 是否在项目目录 `C:\path\to\wkt-ai-server`
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
- `/ai/*` 的 session 仍保存在内存里，服务重启后会丢失
- `AI_ENABLE_MOCK_TTS=true` 生成的是测试静音 WAV；关闭后才会调用真实 TTS

这些限制是刻意保留的，目的是让这个项目作为最小学习版本，先看清 LLM 后端的基本骨架。
