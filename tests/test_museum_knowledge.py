import json
import unittest

from museum import is_museum_question, load_museum_profile, to_llm_context
from router import build_messages, direct_response_for
from sessions import DeviceSession


class MuseumKnowledgeTests(unittest.TestCase):
    def test_profile_loads_official_visitor_information(self) -> None:
        profile = load_museum_profile()

        self.assertEqual(profile["name"], "平顶山博物馆")
        self.assertEqual(
            profile["visitor_information"]["regular_opening_hours"],
            "周二至周日9:00—17:00",
        )
        self.assertEqual(
            profile["featured_collections"]["official_treasure"]["name"],
            "应国玉鹰",
        )
        self.assertTrue(
            all(source["url"].startswith("https://") for source in profile["sources"])
        )

    def test_museum_question_intent_examples(self) -> None:
        questions = (
            "介绍一下平顶山博物馆",
            "你们馆几点关门？",
            "周一开馆吗",
            "镇馆之宝是什么",
            "这里有多少馆藏",
            "这里有什么好看的",
            "博物馆怎么走",
            "需要预约吗",
        )

        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_museum_question(question))

        self.assertFalse(is_museum_question("继续讲讲它的纹饰"))

    def test_museum_profile_sent_without_artifact_context(self) -> None:
        session = DeviceSession(
            device_id="museum-test",
            latest_artifact_id="denggong_gui",
            latest_vision_description="画面中是青铜簋。",
        )

        messages = build_messages("你们馆几点关门？", session)
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("周二至周日9:00—17:00", joined)
        self.assertIn("museum-level question", joined)
        self.assertNotIn("邓公簋是西周青铜器", joined)
        self.assertEqual(session.latest_artifact_id, "denggong_gui")

    def test_artifact_and_museum_context_can_be_combined(self) -> None:
        session = DeviceSession(device_id="museum-artifact-test")

        messages = build_messages("应国玉鹰为什么是镇馆之宝？", session)
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("Matched local artifact knowledge cards", joined)
        self.assertIn("Matched the local Pingdingshan Museum knowledge profile", joined)
        self.assertEqual(session.latest_artifact_id, "yingguo_jade_eagle")

    def test_today_question_includes_temporary_schedule_policy(self) -> None:
        session = DeviceSession(device_id="museum-today-test")

        messages = build_messages("今天开馆吗？", session)
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("可能临时调整", joined)
        self.assertIn("0375-2660518", joined)

    def test_museum_intro_is_not_blocked_by_vague_artifact_fallback(self) -> None:
        session = DeviceSession(device_id="museum-intro-test")

        self.assertIsNone(direct_response_for("介绍一下博物馆", session))
        self.assertIsNotNone(direct_response_for("介绍一下", session))

    def test_llm_context_excludes_source_urls(self) -> None:
        context = to_llm_context()
        serialized = json.dumps(context, ensure_ascii=False)

        self.assertIn("应国玉鹰", serialized)
        self.assertNotIn("https://", serialized)


if __name__ == "__main__":
    unittest.main()
