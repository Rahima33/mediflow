import importlib
import os
import sys
import unittest


class TriageAgentStartupTests(unittest.TestCase):
    def test_import_without_groq_key(self):
        os.environ.pop("GROQ_API_KEY", None)
        sys.modules.pop("agent.triage_agent", None)

        module = importlib.import_module("agent.triage_agent")

        self.assertTrue(callable(module.build_triage_graph))


if __name__ == "__main__":
    unittest.main()
