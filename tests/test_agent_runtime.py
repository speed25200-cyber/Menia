import json
from pathlib import Path
import tempfile
import unittest
from menia.agent import SituatedAgent
from menia.action_model import ActionModel
from menia.core import Runtime, Capabilities
from menia.conversation import answer_question, messages_for_runtime
from menia.environments import VirtualRoom
from menia.run_agent import run
from menia.chat import ChatSession


class AgentRuntimeTests(unittest.TestCase):
    def runtime(self):
        agent = SituatedAgent()
        runtime = Runtime(agent=agent)
        self.addCleanup(agent.memory.close)
        self.addCleanup(runtime.memory.close)
        return runtime

    def test_same_agent_state_reaches_language_handoff(self):
        runtime = self.runtime()
        env = VirtualRoom()
        for _ in range(3): runtime.step(env)
        seen = []
        def generator(messages):
            seen.extend(messages)
            state = json.loads(messages[-1]['content'])['application_state']
            return state['agent']['explanation']
        response = answer_question(runtime, 'Pourquoi ce choix ?', generator)
        self.assertEqual(response, runtime.agent.explain())
        state = json.loads(seen[-1]['content'])['application_state']['agent']
        self.assertEqual(state['episode'], runtime.agent.episode)
        self.assertEqual(state['tick'], runtime.agent.tick)
        self.assertEqual([e['id'] for e in state['evidence']], runtime.agent.last_decision['evidence'])

    def test_runtime_stop_and_disabled_memory_cover_agent(self):
        runtime = self.runtime()
        env = VirtualRoom()
        runtime.stop()
        with self.assertRaises(RuntimeError): runtime.step(env)
        with self.assertRaises(RuntimeError): messages_for_runtime(runtime, 'Pourquoi ?')
        self.assertTrue(runtime.agent.stopped)
        runtime.resume(user_requested=True)
        runtime.step(env)
        runtime.capabilities = Capabilities(memory=False)
        self.assertNotIn('agent', runtime.context(''))
        with self.assertRaises(RuntimeError): runtime.step(env)

    def test_interactive_commands_use_same_runtime_and_provenance(self):
        runtime = self.runtime()
        env = VirtualRoom()
        chat = ChatSession(runtime, env)
        self.assertIn('Témoignage', chat.handle('/report target 99 99'))
        self.assertIn('sélectionne', chat.handle('/step'))
        self.assertEqual(runtime.agent.memory.latest(runtime.agent.episode, 'target')['payload']['value'], [4, 3])
        chat.handle('/run 80')
        self.assertEqual(env.position, env.target)
        self.assertEqual(chat.handle('/why'), runtime.agent.explain())
        chat.handle('/stop')
        with self.assertRaises(RuntimeError): chat.handle('/step')
        chat.handle('/resume')
        self.assertIn('termine', chat.handle('/step'))
        with self.assertRaises(ValueError): chat.handle('/run 0')

    def test_model_intervention_changes_decision_with_same_observations(self):
        runtime = self.runtime()
        env = VirtualRoom(target=(5, 0))
        runtime.agent.attend(env, 'target')
        runtime.agent.attend(env, 'position')
        model = ActionModel(env.actions)
        for action, effect in zip(env.actions, env.effects):
            for _ in range(8): model.learn(action, effect, model.forecast(action))
        runtime.agent.model = model
        first = runtime.agent.decide()
        self.assertEqual(first['action'], 'a')
        model.samples['a'], model.samples['b'] = model.samples['b'], model.samples['a']
        second = runtime.agent.decide()
        self.assertEqual(second['action'], 'b')
        self.assertEqual(first['evidence'], second['evidence'])
        self.assertNotEqual(runtime.agent.explain(first), runtime.agent.explain(second))

    def test_batch_entrypoint_exports_connected_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)/'new-run'
            summary = run(out, seed=31, steps=120, reverse_after=12)
            self.assertTrue(summary['goal_reached'])
            self.assertTrue(summary['control_reversal'])
            events = json.loads((out/'events.json').read_text(encoding='utf-8'))
            predictions = {e['id']: e for e in events if e['kind']=='prediction'}
            assessments = [e for e in events if e['kind']=='assessment']
            for event in assessments:
                self.assertIn(event['payload']['prediction_event'], predictions)
            context = json.loads((out/'context.json').read_text(encoding='utf-8'))
            messages = json.loads((out/'language-messages.json').read_text(encoding='utf-8'))
            passed = json.loads(messages[-1]['content'])['application_state']['agent']
            self.assertEqual(passed['explanation'], context['agent']['explanation'])
            with self.assertRaises(FileExistsError): run(out)


if __name__ == '__main__':
    unittest.main()
