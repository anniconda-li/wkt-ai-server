import unittest

from vision_llm import build_recognition_prompt, build_vision_candidates


class VisionAnchorTests(unittest.TestCase):
    def test_every_candidate_has_decisive_anchors_and_conflicts(self) -> None:
        candidates = build_vision_candidates()

        self.assertEqual(len(candidates), 5)
        for candidate in candidates:
            with self.subTest(artifact_id=candidate["artifact_id"]):
                anchors = candidate["recognition_anchors"]
                self.assertGreaterEqual(len(anchors), 3)
                self.assertTrue(any(anchor["weight"] == 5 for anchor in anchors))
                self.assertTrue(candidate["recognition_conflicts"])

    def test_bronze_candidates_use_different_decisive_structures(self) -> None:
        candidates = {
            candidate["artifact_id"]: candidate for candidate in build_vision_candidates()
        }
        expected_decisive_terms = {
            "bronze_he_dragon_knob_lidded": ("管状流", "盘旋"),
            "denggong_gui": ("衔环", "没有长管状流"),
            "shuyao_chuilin_sheng_ding": ("高大立耳", "龙形怪兽"),
        }

        for artifact_id, terms in expected_decisive_terms.items():
            decisive_text = " ".join(
                anchor["feature"]
                for anchor in candidates[artifact_id]["recognition_anchors"]
                if anchor["weight"] == 5
            )
            with self.subTest(artifact_id=artifact_id):
                for term in terms:
                    self.assertIn(term, decisive_text)

    def test_prompt_tolerates_screen_capture_but_rejects_text_as_evidence(self) -> None:
        prompt = build_recognition_prompt(build_vision_candidates())

        self.assertIn("拍摄显示器", prompt)
        self.assertIn("不要把屏幕文字或标题当作识别证据", prompt)
        self.assertIn("器型和结构高于颜色", prompt)
        self.assertIn("5 是决定性独有锚点", prompt)
        self.assertIn("三个青铜器之间", prompt)
        self.assertIn("不能单独决定结果", prompt)
        self.assertIn("不得出现权重、锚点、候选", prompt)


if __name__ == "__main__":
    unittest.main()
