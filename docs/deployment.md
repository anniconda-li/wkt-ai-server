# wkt-ai-server 部署契约

本文档是 `wkt-deploy` 集成本服务时的固定契约。它只描述 `wkt-ai-server`，不部署生产环境，也不包含对讲或 OTA 服务。

## 仓库与职责边界

本地父目录不是 Git 仓库，下面四个目录各自拥有独立 Git 历史：

```text
wkt-platform/
├── wkt-intercom-server
├── wkt-ai-server
├── wkt-ota-server
└── wkt-deploy
```

`wkt-ai-server` 负责 AI 语音、ASR、LLM、TTS、WAV 分片上传和下载、相机上传与分析，以及自己的 Docker 镜像。跨服务 Compose、反向代理、TLS、生产密钥和主机配置由 `wkt-deploy` 管理。

## 固定运行身份

| 配置项 | 固定值 |
| --- | --- |
| 项目与 GitHub 仓库 | `wkt-ai-server` |
| Docker 镜像 | `wkt-ai-server` |
| Compose 服务 | `ai` |
| 容器 | `wkt-ai-server` |
| 容器端口 | `8000/tcp` |
| 启动命令 | `python -m uvicorn main:app --host 0.0.0.0 --port 8000` |
| 健康检查 | `GET http://127.0.0.1:8000/health`，期望 `200` 和 `{"status":"ok"}` |

镜像自带健康检查：启动宽限期 10 秒，每 30 秒检查一次，单次超时 5 秒，连续失败 3 次判定不健康。

## 构建和启动

```powershell
docker build -t wkt-ai-server:local .
docker run --rm `
  --name wkt-ai-server `
  --env-file .env `
  -p 8000:8000 `
  -v wkt-ai-uploads:/app/uploads `
  -v wkt-ai-outputs:/app/outputs `
  wkt-ai-server:local
```

本仓库的 `compose.yaml` 用于本地验证：

```powershell
docker compose up --build ai
```

Compose 默认读取 `.env`。只做无秘密的配置展开验证时，可以将 `AI_ENV_FILE` 指向 `.env.example`：

```powershell
$env:AI_ENV_FILE = ".env.example"
docker compose config
```

`AI_ENV_FILE` 只控制 Compose 读取哪个环境文件，不是应用环境变量。

## 持久化目录和进程模型

| 容器路径 | 用途 | 部署要求 |
| --- | --- | --- |
| `/app/uploads` | 相机 JPEG、JPEG 分片、相机 SQLite、WAI1 SQLite/临时文件、设备上传 WAV、处理后的 WAV/ROP1 | 必须持久化；生产环境使用命名卷或宿主机受控目录 |
| `/app/outputs` | 本地文本/TTS 客户端生成的回复文件 | 建议持久化；不用仓库内临时目录承载生产数据 |
| `/app/data/artifacts` | 随镜像发布的只读文物知识卡 | 不挂载运行时空目录覆盖 |
| `/app/data/museum` | 随镜像发布的只读馆级资料卡 | 不挂载运行时空目录覆盖 |

相机 `X-Request-ID` 幂等状态和 JPEG 分片元数据保存在 `/app/uploads/camera_idempotency.sqlite3`。WAI1 会话和分片索引保存在 `/app/uploads/ai_ws.sqlite3`，字节位于 `/app/uploads/ai_ws`，可跨 WebSocket 断线恢复。SQLite 事务可协调共享文件系统上的 finish claim，但活跃连接替换、主动推送、聊天记忆和现有 AI runtime task 仍是进程内状态，所以整体部署必须保持单进程、单副本；不要因为上传状态使用 SQLite 就增加 worker 数。进程重启后已落盘上传仍可查询和清理，但已发出的模型任务不会自动跨进程恢复。

## HTTP 路由

所有路径都位于容器端口 `8000`，没有额外前缀。

| 方法 | 路径 | 请求与用途 |
| --- | --- | --- |
| `GET` | `/health` | 存活/就绪检查 |
| `POST` | `/chat` | JSON：`message`、`device`；返回流式纯文本 |
| `GET` | `/sessions` | 会话摘要列表 |
| `GET` | `/sessions/{device_id}` | 单设备会话快照 |
| `POST` | `/sessions/{device_id}/clear` | 清空单设备会话 |
| `POST` | `/sessions/{device_id}/artifact-context` | JSON：`artifact_id`、可选 `vision_description`、`image_id` |
| `GET` | `/artifacts` | 文物知识卡摘要列表 |
| `GET` | `/artifacts/{artifact_id}` | 文物知识卡详情 |
| `POST` | `/camera/upload` | raw JPEG body；必须有 `Content-Length`；可选成对发送 `X-Request-ID`、`X-Content-SHA256`；query：`device`、可选 `artifact_id`、`vision_description`、`use_vision` |
| `POST` | `/camera/upload/chunk` | 最大 4096 字节 raw chunk；query：`device`、`request_id`、`offset`、`total`；完整图和分片 SHA 请求头必填 |
| `POST` | `/camera/upload/finish` | 空 body；query：`device`、`request_id`；同步校验、识别并返回最终业务结果 |
| `POST` | `/camera/upload/cancel` | 空 body；query：`device`、`request_id`；幂等取消未开始识别的上传 |
| `POST` | `/ai/start` | JSON：`device`、`language`、可选 `audio_format`；返回 session、确认格式和 `chunk_size=32768` |
| `POST` | `/ai/upload` | raw WAV 或 AOP1 chunk；query：`session`、`index`、`offset`、`total`、可选 `device` |
| `POST` | `/ai/finish` | query：`session`、可选 `device`；后台启动 ASR/LLM/TTS |
| `POST` | `/ai/result_info` | query：`session`、可选 `device`；轮询处理状态 |
| `POST` | `/ai/result_chunk` | query：`session`、`offset`、`len`、可选 `device`；返回 `audio/wav` |
| `POST` | `/ai/cancel` | query：`session`、可选 `device` |
| `POST` | `/ai/stop_audio` | query：`session`、可选 `device` |
| `WS` | `/ai/ws?device=...&protocol=wai1` | 独立 AI WebSocket；AOP1/JPEG 上传、状态推送、ROP1 下载与断线续传 |

请求参数、状态码和响应字段的完整示例见仓库根目录 `README.md`。部署层不得重写这些路径或改变请求体编码。

## 环境变量

真实密钥只通过部署环境或秘密管理注入，不写入镜像、Compose 文件或 Git。`.env.example` 是无秘密的完整配置模板。

| 分组 | 变量 | 默认值或要求 |
| --- | --- | --- |
| 日志 | `LOG_LEVEL` | `INFO` |
| 文本 LLM | `OPENAI_API_KEY` | 使用真实 LLM 时必填 |
| 文本 LLM | `OPENAI_MODEL` | `gpt-4o-mini`；示例配置使用 DeepSeek |
| 文本 LLM | `OPENAI_BASE_URL` | OpenAI-compatible endpoint，可选 |
| 百炼共享 | `DASHSCOPE_API_KEY` | 视觉、ASR、TTS 使用百炼时必填，可被各模块 key 覆盖 |
| 百炼共享 | `DASHSCOPE_BASE_URL` | OpenAI-compatible 百炼 endpoint，可选 |
| 视觉 | `VISION_PROVIDER`, `VISION_MODEL` | `dashscope`, `qwen3.6-flash-2026-04-16` |
| 视觉 | `VISION_API_KEY`, `VISION_BASE_URL` | 可选；默认复用百炼配置 |
| 视觉 | `VISION_ENABLE_THINKING` | `false`，降低延迟并稳定 JSON 输出 |
| 视觉 | `VISION_MIN_CONFIDENCE` | `0.60` |
| 视觉 | `VISION_TIMEOUT_SECONDS` | `120`；小于 120 的配置会提升到 120 秒 |
| 图片保存 | `MAX_SAVED_IMAGES_PER_DEVICE` | `10`，最小按 1 处理 |
| 图片上传 | `CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS` | `8`，连续无新字节时返回 HTTP 408 |
| 图片幂等 | `CAMERA_IDEMPOTENCY_DB_PATH` | 默认 `/app/uploads/camera_idempotency.sqlite3` |
| 图片幂等 | `CAMERA_IDEMPOTENCY_TTL_SECONDS`, `CAMERA_IDEMPOTENCY_MAX_RECORDS` | `1200`, `1000` |
| 图片幂等 | `CAMERA_IDEMPOTENCY_WAIT_TIMEOUT_SECONDS`, `CAMERA_IDEMPOTENCY_POLL_INTERVAL_SECONDS` | `180`, `0.1` |
| 图片分片 | `CAMERA_CHUNK_TEMP_DIR` | 默认 `/app/uploads/camera_chunks` |
| 图片分片 | `CAMERA_CHUNK_SESSION_TTL_SECONDS`, `CAMERA_CHUNK_COMPLETED_TTL_SECONDS` | `600`, `1200` |
| 图片分片 | `CAMERA_CHUNK_MAX_SESSIONS`, `CAMERA_CHUNK_MAX_TEMP_BYTES` | `100`, `67108864`（64 MiB） |
| 图片分片 | `CAMERA_CHUNK_CLEANUP_INTERVAL_SECONDS` | `60` 秒 |
| AI WebSocket | `AI_WS_DB_PATH`, `AI_WS_TEMP_DIR` | 默认 `/app/uploads/ai_ws.sqlite3`, `/app/uploads/ai_ws` |
| AI WebSocket | `AI_WS_SESSION_TTL_SECONDS` | `1200`，代码强制最小 600 秒 |
| AI WebSocket | `AI_WS_MAX_SESSIONS`, `AI_WS_MAX_SESSIONS_PER_DEVICE` | `200`, `4` |
| AI WebSocket | `AI_WS_MAX_TEMP_BYTES`, `AI_WS_CLEANUP_INTERVAL_SECONDS` | `67108864`, `60` |
| AI WebSocket | `AI_WS_SEND_QUEUE_SIZE` | `32` 条，满时关闭慢连接 |
| ASR | `ASR_PROVIDER`, `ASR_MODEL` | `dashscope`, `qwen3-asr-flash-2026-02-10` |
| ASR | `ASR_FALLBACK_MODEL` | `paraformer-realtime-v2`；留空禁用回退 |
| ASR | `ASR_API_KEY` | 可选；默认复用 `DASHSCOPE_API_KEY` |
| ASR | `ASR_LANGUAGE`, `ASR_ENABLE_ITN`, `ASR_TIMEOUT_SECONDS` | `zh`, `false`, `60` |
| ASR | `ASR_FRAME_BYTES`, `ASR_FRAME_SLEEP_SECONDS` | `3200`, `0` |
| ASR | `ASR_EXTRA_KWARGS` | 可选 JSON 对象 |
| ASR 纠错 | `ASR_ARTIFACT_PHONETIC_THRESHOLD` | `0.86` |
| TTS | `TTS_PROVIDER`, `TTS_API_STYLE`, `TTS_MODEL` | `dashscope`, 自动推断/示例为 `dashscope_qwen`, `qwen3-tts-flash-2025-11-27` |
| TTS | `TTS_API_KEY` | 可选；依次复用 `DASHSCOPE_API_KEY`、`OPENAI_API_KEY` |
| TTS | `TTS_BASE_URL`, `DASHSCOPE_TTS_BASE_URL`, `TTS_ENDPOINT` | 可选 endpoint 覆盖项 |
| TTS | `TTS_VOICE`, `TTS_LANGUAGE_TYPE`, `TTS_RESPONSE_FORMAT` | `Cherry`, `Chinese`, `wav` |
| TTS | `TTS_TIMEOUT_SECONDS` | `120` 秒 |
| TTS | `TTS_EXTRA_JSON` | 可选 JSON 对象 |
| 转码 | `FFMPEG_BIN` | 可选；镜像已安装 `ffmpeg` 并从 `PATH` 查找 |
| 离线测试 | `AI_MOCK_ASR_TEXT`, `AI_MOCK_LLM_TEXT` | 未设置；生产环境不要启用 |
| 离线测试 | `AI_ENABLE_MOCK_TTS`, `AI_MOCK_TTS_SECONDS` | `false`, `0.4`；生产环境不要启用 |
| 语音判定 | `AI_NO_SPEECH_TEXT`, `AI_ASR_ERROR_TEXT` | 见 `.env.example` 中的中文默认提示 |
| 语音判定 | `AI_SILENCE_RMS_THRESHOLD`, `AI_MIN_SPEECH_SECONDS` | `80`, `0.2` |

## 请求体、响应大小和超时

| 链路 | 应用限制 | 部署层要求 |
| --- | --- | --- |
| `/camera/upload` | JPEG 最小 128 字节，最大 8 MiB；必须校验 `Content-Length`；默认连续 8 秒无新字节才算上传空闲超时，没有固定总接收时长 | 反向代理请求体上限建议 9 MiB；请求体空闲超时约 15 秒；关闭请求缓冲，让应用立即消费 body |
| `/camera/upload/chunk` | 每片最多 4096 字节，非末片必须正好 4096 字节；完整图仍最大 8 MiB | 请求体上限建议 5 KiB；不要缓存或改写 raw body |
| `/camera/upload/finish` | 空 body；同步等待视觉模型并返回最终 JSON | 响应读取超时建议 300 秒 |
| `/ai/upload` | PCM WAV 最大 2,100,000 字节；AOP1 Opus 最大 262,144 字节；建议分片 32,768 字节 | 允许 `application/octet-stream` 和 `application/vnd.wkt.opus-packets`，不改写 query 或 raw body |
| `/ai/result_chunk` | 单次最多返回 32,768 字节 | 允许 `audio/wav` 二进制响应 |
| `/ai/ws` | WAI1 payload 最大 4096；30 秒无有效消息关闭；ROP1 最大 384 KiB | 必须转发 Upgrade；读写空闲超时建议 45 秒；关闭响应缓冲 |
| TTS 回复 WAV | 最大 4,000,000 字节 | `/app/uploads` 需要足够磁盘空间 |
| `/chat` | 流式纯文本响应 | 禁用代理响应缓冲，空闲/读取超时至少 120 秒 |
| `/camera/upload` 视觉识别 | 请求会等待外部视觉模型返回 | 上游超时至少 120 秒；网络较慢时建议 300 秒 |
| `/ai/finish` 后台处理 | 接口很快返回，客户端轮询 `/ai/result_info`；示例客户端总等待 300 秒 | 不要把长处理误判为 `/ai/finish` HTTP 超时 |

应用为视觉请求显式保留至少 `VISION_TIMEOUT_SECONDS=120`，TTS 默认 `TTS_TIMEOUT_SECONDS=120`。文本 LLM 和 ASR 还会受各 SDK、网络及模型服务端超时影响；生产代理应保留不低于上表的预算。

当前仓库的 Dockerfile 和 Compose 都直接启动一个 Uvicorn worker，未配置 Nginx 或 Traefik；正式部署仓库当前也是直接把宿主端口映射到容器端口。以后若在前面加入 Nginx，`/camera/upload` 建议单独配置：

```nginx
location = /camera/upload {
    client_max_body_size 9m;
    client_body_timeout 15s;
    proxy_request_buffering off;
    proxy_http_version 1.1;
    proxy_read_timeout 300s;
    proxy_send_timeout 15s;
    send_timeout 300s;
    proxy_pass http://wkt_ai_server;
}

location = /camera/upload/chunk {
    client_max_body_size 5k;
    client_body_timeout 15s;
    proxy_request_buffering off;
    proxy_http_version 1.1;
    proxy_send_timeout 15s;
    proxy_read_timeout 30s;
    proxy_pass http://wkt_ai_server;
}

location = /camera/upload/finish {
    client_max_body_size 1k;
    proxy_http_version 1.1;
    proxy_read_timeout 300s;
    send_timeout 300s;
    proxy_pass http://wkt_ai_server;
}

location = /camera/upload/cancel {
    client_max_body_size 1k;
    proxy_http_version 1.1;
    proxy_read_timeout 30s;
    proxy_pass http://wkt_ai_server;
}

location = /ai/ws {
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_buffering off;
    proxy_read_timeout 45s;
    proxy_send_timeout 45s;
    proxy_pass http://wkt_ai_server;
}
```

`client_body_timeout` 是客户端相邻两次发送之间的空闲超时，`proxy_send_timeout` 限制 Nginx 向应用写请求体时的连续无进展时间，两者都不是整个 JPEG 上传的固定总时长。`proxy_read_timeout` 则覆盖完整上传后的视觉识别等待。不要用视觉阶段的 300 秒去替代请求体空闲超时，也不要让代理先缓冲完整 body 后才转发给应用。

## Git 与秘密边界

- `.env`、`.env.*`（保留 `.env.example`）、`.venv`、`venv`、`uploads`、`outputs`、非版本化的 `data`/`samples` 内容、Python 缓存、私钥/证书密钥和 `secrets` 已被 Git 忽略。
- `data/artifacts` 和 `data/museum` 是运行必需、随源码版本化的静态知识卡，不是运行时数据。
- `samples/camera` 是既有离线测试夹具，不包含凭据，并已从 Docker 构建上下文排除。
- 运行时上传、模型输出和真实密钥不得加入 Git，也不得烘焙进镜像。

## CI 契约

`.github/workflows/ci.yml` 在 push 和 pull request 时执行 Python 编译、全部离线测试、Compose 配置校验和 `wkt-ai-server:<commit-sha>` 镜像构建。它不推送镜像、不启动远程容器，也不部署生产环境。
