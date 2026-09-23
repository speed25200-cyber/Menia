import json
import unittest
from menia.core import Runtime
from menia.indicator_bridge import IndicatorAgentBridge
from menia.indicator_chat import IndicatorChatSession

ROOT = "artifacts/indicator-agent-v6"


class IndicatorChatTests(unittest.TestCase):
    def session(self, generate=None):
        bridge = IndicatorAgentBridge.from_artifacts(ROOT, 151, env_seed=930001000, agent_seed=151_000_000)
        runtime = Runtime(agent=bridge)
        self.addCleanup(runtime.memory.close)
        return IndicatorChatSession(runtime, generate), bridge

    def test_steps_journal_and_stop(self):
        session, bridge = self.session()
        self.assertEqual(bridge.version, 6)
        self.assertIn("pas encore agi", session.handle("/journal"))
        self.assertIn("Pas 1 : décision de l'agent", session.handle("/step"))
        self.assertEqual(len(session.handle("/run 3").splitlines()), 3)
        text = session.handle("/journal")
        self.assertTrue(text.startswith("Espace de travail de l'agent, pas 4."))
        self.assertIn("Décision de l'agent", text)
        self.assertEqual(session.handle("/why"), bridge.explain())
        self.assertIn("Mode structuré", session.handle("Où es-tu ?"))
        with self.assertRaises(ValueError):
            session.handle("/run 60")
        session.handle("/stop")
        with self.assertRaises(RuntimeError):
            session.handle("/step")
        session.handle("/resume")
        self.assertIn("Pas 5", session.handle("/step"))
        self.assertIn("terminée", session.handle("/run 48").splitlines()[-1])

    def test_the_language_model_receives_the_journal(self):
        def generate(messages):
            return json.loads(messages[-1]["content"])["application_state"]["agent"]["journal"]

        session, bridge = self.session(generate)
        session.handle("/run 6")
        self.assertEqual(session.handle("Que contient ton espace de travail ?"), bridge.context()["journal"])


if __name__ == "__main__":
    unittest.main()
