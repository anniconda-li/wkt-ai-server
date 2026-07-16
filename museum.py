import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


MUSEUM_PROFILE_PATH = (
    Path(__file__).parent / "data" / "museum" / "pingdingshan_museum.json"
)

_MUSEUM_QUERY_PATTERNS = (
    "博物馆",
    "平博",
    "你们馆",
    "这个馆",
    "本馆",
    "馆里",
    "馆内",
    "镇馆之宝",
    "镇馆宝物",
    "代表性文物",
    "重点文物",
    "这里有什么",
    "有什么好看",
    "有什么展品",
    "有什么文物",
    "有哪些文物",
    "推荐看什么",
    "值得看什么",
    "馆藏",
    "藏品",
    "主要展览",
    "基本陈列",
    "展厅",
    "开馆",
    "闭馆",
    "开门",
    "关门",
    "开放时间",
    "营业时间",
    "几点开",
    "几点关",
    "周一开",
    "周一闭",
    "周一休",
    "节假日开",
    "节假日闭",
    "今天开",
    "今天闭",
    "今天营业",
    "地址",
    "怎么走",
    "怎么去",
    "在哪里",
    "在哪儿",
    "联系电话",
    "联系方式",
    "电话号码",
    "门票",
    "预约",
    "停车",
    "哪年建",
    "哪年开",
    "建筑面积",
    "占地面积",
    "建筑设计",
    "几A级",
    "几A景区",
)


@lru_cache(maxsize=1)
def load_museum_profile() -> dict[str, Any]:
    with MUSEUM_PROFILE_PATH.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    for field in ("id", "name", "visitor_information", "facts", "answer_policies"):
        if not profile.get(field):
            raise ValueError(f"Museum profile is missing required field: {field}")
    return profile


def is_museum_question(text: str) -> bool:
    normalized = re.sub(r"[\s，。！？、,.!?;；:：]", "", text).lower()
    profile = load_museum_profile()
    aliases = [profile["name"], *profile.get("aliases", [])]
    return any(term.lower() in normalized for term in aliases if term) or any(
        pattern.lower() in normalized for pattern in _MUSEUM_QUERY_PATTERNS
    )


def to_llm_context(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    source = profile or load_museum_profile()
    return {
        "name": source.get("name"),
        "aliases": source.get("aliases", []),
        "visitor_information": source.get("visitor_information", {}),
        "facts": source.get("facts", []),
        "featured_collections": source.get("featured_collections", {}),
        "answer_policies": source.get("answer_policies", []),
        "last_verified": source.get("last_verified"),
    }
