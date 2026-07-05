from datetime import datetime, timezone


def get_device_status() -> dict[str, object]:
    return {
        "device_id": "esp32-demo-001",
        "device_type": "ESP32",
        "online": True,
        "wifi_rssi_dbm": -53,
        "uptime_seconds": 128734,
        "temperature_c": 26.8,
        "humidity_percent": 47.2,
        "free_heap_bytes": 186420,
        "led": "on",
        "relay": "off",
        "last_seen": datetime.now(timezone.utc).isoformat(),
    }
