import json
import unittest
from pathlib import Path
import numpy as np
from menia.core import Runtime
from menia.conversation import answer_question
from menia.indicator_bridge import IndicatorAgentBridge

ROOT = "artifacts/indicator-agent-v5"


class IndicatorBridgeTests(unittest.TestCase):
    def test_menia_sees_the_workspace_and_a_faithful_decision(self):
        bridge = IndicatorAgentBridge.from_artifacts(ROOT, 113)
        runtime = Runtime(agent=bridge)
        self.addCleanup(runtime.memory.close)
        for _ in range(10):
            runtime.step(None)
        context = runtime.context("")["agent"]
        self.assertEqual(context["tick"], 10)
        self.assertEqual(context["workspace"]["position"]["square"], int(np.argmax(bridge.agent.W["pos"])))
        self.assertEqual(context["workspace"]["interoception"]["age"], bridge.agent.age["intero"])
        self.assertEqual(context["goal"], bridge.last["goal"])
        self.assertNotIn("schema_estimate", json.dumps(context))

        def generator(messages):
            return json.loads(messages[-1]["content"])["application_state"]["agent"]["explanation"]

        self.assertEqual(answer_question(runtime, "Pourquoi ce choix ?", generator), bridge.explain())
        self.assertIn("modèle de mes besoins", bridge.explain())

    def test_only_the_user_resumes(self):
        bridge = IndicatorAgentBridge.from_artifacts(ROOT, 113)
        runtime = Runtime(agent=bridge)
        self.addCleanup(runtime.memory.close)
        runtime.stop()
        with self.assertRaises(RuntimeError):
            runtime.step(None)
        with self.assertRaises(PermissionError):
            bridge.resume()
        runtime.resume(user_requested=True)
        self.assertEqual(runtime.step(None)["result"]["tick"], 1)

    def test_the_bridged_agent_lives_its_published_life(self):
        bridge = IndicatorAgentBridge.from_artifacts(ROOT, 113, env_seed=930001 * 1000, agent_seed=113 * 1_000_000)
        actions = [bridge.cycle()["result"]["action"] for _ in range(48)]
        record = json.loads(Path(ROOT, "lives-113-R.jsonl").read_text().splitlines()[0])
        self.assertEqual(actions, record["actions"])
        self.assertEqual(round(bridge.total_reward, 6), record["return"])
        self.assertEqual(bridge.cycle()["result"]["kind"], "done")

    def test_the_bridge_reads_the_version_of_a_run(self):
        self.assertEqual(IndicatorAgentBridge.from_artifacts(ROOT, 113).version, 5)
        bridge = IndicatorAgentBridge.from_artifacts(ROOT, 113, version=6)
        self.assertEqual(type(bridge.agent).__name__, "AgentV6")
        for _ in range(12):
            bridge.cycle()
        self.assertIn("version 6", bridge.context()["scope"])
        with self.assertRaises(ValueError):
            IndicatorAgentBridge.from_artifacts(ROOT, 113, version=4)


if __name__ == "__main__":
    unittest.main()
