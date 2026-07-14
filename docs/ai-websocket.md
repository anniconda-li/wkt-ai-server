# WAI1 独立 AI WebSocket 协议

WAI1 只承载 AI 语音、相机识别和 AI 回复音频，不承载对讲。设备连接：

```text
ws://<host>:18080/ai/ws?device=walkie-01&protocol=wai1
```

同一 `device` 在单服务进程中最多保留一个活跃连接；新连接以关闭码 `4001` 替换旧连接，但不会删除业务会话。会话身份只使用 `device + request_id + session`，不读取客户端 IP。服务端只有一个接收协程、一个有界发送队列和一个 writer 协程；模型任务在后台运行。

## 生命周期与 JSON 消息

服务端接受连接后首先发送：

```json
{"type":"hello","protocol":"wai1","max_payload":4096,"heartbeat_ms":10000,"reply_format":"rop1"}
```

心跳为 `{"type":"ping","seq":1}` 和 `{"type":"pong","seq":1}`。连续 30 秒未收到有效 JSON 或 WAI1 binary message，服务端关闭连接。发送队列默认最多 32 条，队列耗尽时以 `1013` 关闭慢连接。

### 语音上传

```json
{"type":"voice_start","request_id":"walkie-01-ai-123","language":"zh","input_format":"opus_packets_v1","total":4600,"sha256":"<64位小写十六进制>"}
```

```json
{"type":"voice_started","request_id":"walkie-01-ai-123","session":"<32位session>","stream_id":1,"next_offset":0,"max_payload":4096}
```

后续用 `packet_type=1` 的 WAI1 binary message 上传。每片成功落盘后返回：

```json
{"type":"upload_ack","kind":"voice","session":"<session>","stream_id":1,"next_offset":4096}
```

同一分片重传且内容一致时只重复 ACK，不重复写文件。上传结束：

```json
{"type":"voice_finish","session":"<session>","sha256":"<完整AOP1 SHA-256>"}
```

### 相机上传

```json
{"type":"camera_start","request_id":"walkie-01-camera-123","content_type":"image/jpeg","total":38686,"sha256":"<完整JPEG SHA-256>"}
```

```json
{"type":"camera_started","request_id":"walkie-01-camera-123","session":"<32位session>","stream_id":2,"next_offset":0,"max_payload":4096}
```

图片使用 `packet_type=2`，ACK 的 `kind` 为 `camera`、`stream_id` 为 `2`。上传结束：

```json
{"type":"camera_finish","session":"<session>","sha256":"<完整JPEG SHA-256>"}
```

最终相机消息使用 WAI1 外层处理状态 `text_ready`，并保留原相机业务字段；原 HTTP 响应中的 `status: ready` 显式放在 `camera_status`：

```json
{
  "type":"result",
  "session":"<session>",
  "kind":"camera",
  "status":"text_ready",
  "camera_status":"ready",
  "device_id":"walkie-01",
  "image":{},
  "recognition":{},
  "latest_artifact_id":null,
  "latest_image_id":"...",
  "latest_vision_description":"...",
  "upload_generation":1,
  "error":null
}
```

### 状态、恢复、取消

语音状态推送：

```json
{"type":"result","session":"<session>","kind":"voice","status":"asr_running","asr_text":"","answer_text":"","error":null}
```

状态为 `uploaded`、`asr_running`、`text_ready`、`tts_running`、`audio_ready`、`no_speech`、`failed` 或 `cancelled`。内部 `llm_running` 对设备仍显示为 `asr_running`；文字结果在 `text_ready` 立即推送。

断线后恢复当前状态：

```json
{"type":"session_resume","session":"<session>"}
```

已完成的语音会重新发送最终 `result` 和 `reply_ready`；已完成相机会重新发送完整业务结果。取消和停止音频：

```json
{"type":"cancel","session":"<session>"}
{"type":"cancelled","session":"<session>"}
{"type":"stop_audio","session":"<session>"}
{"type":"audio_stopped","session":"<session>"}
```

取消幂等。模型外部请求已经发出时只能尽力取消，但取消状态不会再被迟到的结果覆盖。`stop_audio` 停止当前连接继续下载，文字结果保留。

## WAI1 binary message

所有整数均为小端序，固定头为 32 字节：

| offset | size | field |
|---:|---:|---|
| 0 | 4 | `magic = "WAI1"` |
| 4 | 1 | `version = 1` |
| 5 | 1 | `packet_type`：1 voice、2 camera、3 reply |
| 6 | 2 | `flags` |
| 8 | 4 | `stream_id`：voice/reply=1，camera=2 |
| 12 | 4 | `sequence` |
| 16 | 4 | `offset` |
| 20 | 4 | `total` |
| 24 | 2 | `payload_len`，最大 4096 |
| 26 | 2 | `reserved = 0` |
| 28 | 4 | payload 的 IEEE CRC32 |

整个 WebSocket binary message 必须恰好为 `32 + payload_len`。服务端校验 magic、version、packet type、reserved、精确长度、CRC32、`offset + payload_len <= total`，并要求 stream 已由当前设备的 start/resume 消息绑定。

上传和回复下载第一版均采用 stop-and-wait。上传 offset 超前返回当前 `next_offset`；已确认 offset 内容一致则重复 ACK，内容冲突则拒绝。

统一错误示例：

```json
{"type":"error","code":"offset_mismatch","message":"chunk is ahead of server next_offset","session":"...","stream_id":1,"next_offset":4096,"retryable":true}
```

错误码包括 `invalid_message`、`unsupported_protocol`、`invalid_stream`、`offset_mismatch`、`payload_too_large`、`crc_mismatch`、`total_mismatch`、`sha256_mismatch`、`invalid_audio`、`invalid_jpeg`、`session_not_found`、`session_cancelled`、`server_busy` 和 `internal_error`。

## ROP1 回复音频

TTS 仍先生成 16 kHz、单声道、16-bit PCM WAV，再由 FFmpeg/libopus 编码为 20 ms、20 kbps、VOIP、VBR Opus。服务端解析 Ogg 中的 OpusHead、原始 Opus packets 和最终 granule position。

ROP1 固定头为 28 字节，所有整数均为小端序：

| offset | size | field |
|---:|---:|---|
| 0 | 4 | `magic = "ROP1"` |
| 4 | 1 | `version = 1` |
| 5 | 1 | `channels = 1` |
| 6 | 2 | `header_len = 28` |
| 8 | 4 | `sample_rate = 16000` |
| 12 | 2 | `frame_samples = 320` |
| 14 | 2 | `frame_ms = 20` |
| 16 | 4 | `frame_count` |
| 20 | 4 | `pcm_samples` |
| 24 | 2 | `pre_skip` |
| 26 | 2 | `end_trim` |

头后为重复的 `uint16 packet_len + packet bytes`，单包最大 1275 字节，整个 ROP1 最大 384 KiB。`pcm_samples`、`pre_skip`、`end_trim` 均使用设备 16 kHz 解码输出的采样单位；Ogg/Opus 的 48 kHz granule 和 OpusHead pre-skip 在写头时换算为 16 kHz 单位。这一点是设备端必须一致实现的明确单位约定。

生成完成：

```json
{"type":"reply_ready","session":"<session>","format":"rop1","total":24318,"sha256":"...","duration_ms":8420,"sample_rate":16000,"channels":1,"frame_ms":20,"bitrate":20000}
```

下载和 ACK：

```json
{"type":"reply_get","session":"<session>","offset":0}
{"type":"reply_ack","session":"<session>","next_offset":4096}
{"type":"reply_complete","session":"<session>","total":24318,"sha256":"..."}
```

服务端下行使用 `packet_type=3`、`stream_id=1`。设备重连后可从已经持久化的任意合法 offset 重新 `reply_get`；完整接收后必须校验 total 和 SHA-256 再播放。

## 状态和持久化

```text
uploading -> uploaded/processing -> asr_running -> text_ready -> tts_running
                                                    |              |
                                                    |              v
                                                    +--------> audio_ready

任意非终态 -> failed / cancelled
相机：uploading -> uploaded/processing -> text_ready
```

会话元数据和分片索引保存在 `uploads/ai_ws.sqlite3` 的 `ai_ws_sessions`、`ai_ws_parts` 表；字节保存在 `uploads/ai_ws/<device>/<session>/`。SQLite 使用 WAL 和 `BEGIN IMMEDIATE`，只有文件 `fsync` 和数据库提交完成后才 ACK。默认保留 20 分钟且不允许配置低于 10 分钟，默认全局最多 200 条记录、每设备最多 4 个处理中会话、临时空间预留最多 64 MiB；后台默认每 60 秒清理过期记录和孤儿文件。

当前整体服务仍按现有部署约束使用一个 Uvicorn worker 和单副本：SQLite 可以避免多 worker 重复 claim 同一 finish，但活跃连接替换、状态主动推送和现有 AI runtime task 属于进程内状态。不得仅依靠 SQLite 就把部署改成多 worker；若以后需要多 worker/多副本，需增加共享连接路由和任务通知机制。

环境变量：

| 变量 | 默认值 |
|---|---:|
| `AI_WS_DB_PATH` | `uploads/ai_ws.sqlite3` |
| `AI_WS_TEMP_DIR` | `uploads/ai_ws` |
| `AI_WS_SESSION_TTL_SECONDS` | `1200`，最小 600 |
| `AI_WS_MAX_SESSIONS` | `200` |
| `AI_WS_MAX_SESSIONS_PER_DEVICE` | `4` |
| `AI_WS_MAX_TEMP_BYTES` | `67108864` |
| `AI_WS_CLEANUP_INTERVAL_SECONDS` | `60` |
| `AI_WS_SEND_QUEUE_SIZE` | `32` |

## 反向代理

当前仓库直接运行 Uvicorn，不自带 Nginx/Traefik。若前置 Nginx，需要保留 WebSocket Upgrade，关闭响应缓冲，并让空闲超时大于应用 30 秒：

```nginx
location = /ai/ws {
    proxy_pass http://127.0.0.1:18080;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_buffering off;
    proxy_read_timeout 45s;
    proxy_send_timeout 45s;
}
```

旧 `/ai/*`、`/camera/upload*` HTTP 接口保持不变；对讲 `/intercom/ws` 和 OTA 不在本协议或本仓库修改范围内。
