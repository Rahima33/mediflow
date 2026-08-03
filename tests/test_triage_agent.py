import importlib
import os
import sys
import unittest


class TriageAgentStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("GROQ_API_KEY", None)
        sys.modules.pop("agent.triage_agent", None)
        cls.module = importlib.import_module("agent.triage_agent")

    def test_import_without_groq_key(self):
        self.assertTrue(callable(self.module.build_triage_graph))

    def test_live_agent_uses_xrv_checkpoint_and_model(self):
        expected = os.path.normpath(os.path.join("models", "best_model_xrv_backbone.pth"))
        actual = os.path.normpath(self.module.CHECKPOINT_PATH)

        self.assertEqual(actual, expected)
        self.assertEqual(self.module._model.__class__.__name__, "XRVClassifier")
        self.assertIs(self.module._target_layer, self.module._model.features.norm5)

    def test_live_preprocess_is_single_channel_xrv(self):
        image_path = os.path.join(
            "sample_images_by_class",
            "PNEUMONIA",
            "person372_bacteria_1706.jpeg",
        )

        _, input_tensor = self.module.preprocess_image_xrv(image_path)

        self.assertEqual(tuple(input_tensor.shape), (1, 1, 224, 224))
        self.assertEqual(str(input_tensor.dtype), "torch.float32")


if __name__ == "__main__":
    unittest.main()