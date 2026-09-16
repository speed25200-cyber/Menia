import json
from pathlib import Path
import tempfile
import unittest
from menia.action_model import ActionModel
from menia.agent import SituatedAgent
from menia.environments import VirtualRoom
from menia.episodic import EpisodeMemory


class IntegratedAgentTests(unittest.TestCase):
    def agent(self, **kwargs):
        agent = SituatedAgent(**kwargs)
        self.addCleanup(agent.memory.close)
        return agent

    def run_until(self, agent, env, predicate, limit=250):
        for _ in range(limit):
            result = agent.cycle(env)
            if predicate(result):
                return result
        self.fail("Bounded run did not meet condition")

    def test_learns_different_controls_and_reaches_target(self):
        for seed in (2, 11, 47, 91):
            with self.subTest(seed=seed):
                env = VirtualRoom(seed, target=(7, 5))
                agent = self.agent()
                # No knowledge of the environment's permutation at initialization.
                self.assertEqual(agent.model.samples['a'], agent.model.samples['d'])
                self.run_until(agent, env, lambda r: r['result']['kind'] == 'done')
                self.assertEqual(env.position, env.target)
                for action in env.actions:
                    self.assertEqual(tuple(agent.model.forecast(action)['effect']), env.mapping[action])

    def test_predictions_precede_actions_and_labels(self):
        agent = self.agent(episode='precommit')
        env = VirtualRoom()
        self.run_until(agent, env, lambda r: r['result']['kind'] == 'action')
        pending = json.loads(json.dumps(agent.pending))
        prediction = agent.memory.event(pending['prediction_event'])
        self.assertLess(pending['prediction_event'], pending['action_event'])
        self.assertEqual(agent.model.errors, [])
        self.assertTrue(all(not values for values in agent.model.samples.values()))
        agent.cycle(env)
        self.assertEqual(len(agent.model.errors), 1)
        self.assertIsNone(agent.pending)
        self.assertEqual(agent.memory.event(prediction['id']), prediction)
        assessment = [e for e in agent.memory.recent(agent.episode) if e['kind']=='assessment'][-1]
        self.assertEqual(assessment['payload']['prediction_event'], prediction['id'])
        self.assertGreater(assessment['id'], pending['action_event'])

    def test_forecasts_change_after_reversed_controls(self):
        agent = self.agent()
        env = VirtualRoom(target=(4, 3))
        self.run_until(agent, env, lambda r: len(agent.model.errors) == 12)
        old = {a: agent.model.forecast(a)['effect'] for a in env.actions}
        env.reverse_controls()
        self.run_until(agent, env, lambda r: r['result']['kind']=='done')
        self.assertEqual(env.position, env.target)
        self.assertTrue(any(e['regime_reset'] for e in agent.model.errors))
        self.assertTrue(any(agent.model.forecast(a)['effect'] != old[a] for a in env.actions))

    def test_report_is_not_perception_or_training_label(self):
        agent = self.agent()
        report = agent.report('target', [999, 999], source='other_agent')
        self.assertFalse(agent.context()['knowledge']['target']['observed_before'])
        self.assertEqual(agent.memory.event(report)['kind'], 'report')
        env = VirtualRoom()
        agent.cycle(env)
        observed = agent.memory.latest(agent.episode, 'target')
        self.assertEqual(observed['payload']['value'], [4, 3])
        agent.report('target', [-500, -500], source='user')
        self.assertEqual(agent.memory.latest(agent.episode, 'target'), observed)
        self.assertEqual(agent.model.errors, [])

    def test_unselected_fields_and_failed_receipts_are_not_seen(self):
        agent = self.agent()
        env = VirtualRoom(dropout=1)
        for _ in range(4):
            result = agent.cycle(env)
            self.assertFalse(result['result']['received'])
        state = agent.context()
        self.assertFalse(any(k['observed_before'] for k in state['knowledge'].values()))
        self.assertEqual(env.moves, [])
        self.assertLess(state['attention']['target']['estimated_receipt_probability'], .5)
        self.assertEqual(state['attention']['position']['attempts'], 0)
        env.dropout = 0
        agent.cycle(env)
        self.assertTrue(agent.context()['knowledge']['target']['observed_before'])
        self.assertFalse(agent.context()['knowledge']['landmark']['observed_before'])

    def test_evicted_working_memory_is_retrieved_from_own_history(self):
        agent = self.agent()
        env = VirtualRoom()
        agent.attend(env, 'target')
        agent.attend(env, 'position')
        agent.attend(env, 'landmark')
        self.assertFalse(agent.context()['knowledge']['target']['in_working_memory'])
        result = agent.cycle(env)
        self.assertEqual(result['decision']['kind'], 'recall')
        self.assertIn('histoire', result['explanation'])
        self.assertTrue(agent.context()['knowledge']['target']['in_working_memory'])
        self.run_until(agent, env, lambda r: r['result']['kind']=='action', limit=5)

    def test_explanations_reference_actual_predecision_evidence(self):
        agent = self.agent()
        env = VirtualRoom(target=(9, 9))
        result = self.run_until(agent, env, lambda r: r['decision']['reason']=='predicted_progress')
        d = result['decision']
        self.assertEqual(d['action'], min(d['expected_distances'], key=d['expected_distances'].get))
        self.assertIn(str(d['forecast']['effect']), result['explanation'])
        self.assertIn(d['action'], result['explanation'])
        for event_id in d['evidence']:
            event = agent.memory.event(event_id)
            self.assertEqual(event['kind'], 'observation')
            self.assertLess(event_id, d['event_id'])
        saved = agent.context()
        saved['action_model']['a']['effect'] = [999, 999]
        self.assertNotEqual(agent.context()['action_model']['a']['effect'], [999, 999])

    def test_foreign_episode_cannot_supply_current_position(self):
        agent = self.agent(episode='current')
        agent.memory.observe('foreign', 0, 'position', [99, 99], source='sensor')
        agent.memory.observe('foreign', 0, 'target', [99, 99], source='sensor')
        self.assertEqual(agent.decide()['kind'], 'attend')
        self.assertFalse(agent.context()['knowledge']['position']['observed_before'])

    def test_external_change_is_not_learned_as_single_action(self):
        agent = self.agent()
        env = VirtualRoom()
        self.run_until(agent, env, lambda r: r['result']['kind']=='action')
        env.relocate((100, 100))
        agent.cycle(env)
        self.assertEqual(agent.model.errors, [])
        assessment = [e for e in agent.memory.recent(agent.episode) if e['kind']=='assessment'][-1]
        self.assertFalse(assessment['payload']['attributable'])

    def test_save_resume_and_stop_preserve_pending_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'history.sqlite'
            memory = EpisodeMemory(path)
            agent = SituatedAgent(memory=memory)
            env = VirtualRoom()
            self.run_until(agent, env, lambda r: r['result']['kind']=='action')
            agent.stop()
            state = agent.state()
            memory.close()
            restored_memory = EpisodeMemory(path)
            self.addCleanup(restored_memory.close)
            restored = SituatedAgent.from_state(state, memory=restored_memory)
            with self.assertRaises(RuntimeError): restored.cycle(env)
            with self.assertRaises(PermissionError): restored.resume()
            restored.resume(user_requested=True)
            restored.cycle(env)
            self.assertIsNone(restored.pending)
            self.assertEqual(len(restored.model.errors), 1)
            restored_memory.close()

    def test_uncertain_execution_stops_and_requires_observation(self):
        class BrokenReceipt(VirtualRoom):
            def execute(self, action):
                super().execute(action)
                raise OSError('receipt lost after execution')
        agent = self.agent()
        env = BrokenReceipt()
        agent.cycle(env)
        agent.cycle(env)
        with self.assertRaises(OSError): agent.cycle(env)
        self.assertEqual(len(env.moves), 1)
        with self.assertRaises(RuntimeError): agent.cycle(env)
        agent.resume(user_requested=True)
        result = agent.cycle(env)
        self.assertEqual(result['decision']['reason'], 'reconcile_action')
        self.assertEqual(len(env.moves), 1)
        self.assertEqual(agent.model.errors, [])

    def test_model_round_trip_retains_learned_dynamics(self):
        model = ActionModel(['x', 'y'])
        for _ in range(8):
            model.learn('x', (0, 1), model.forecast('x'))
        restored = ActionModel.from_state(json.loads(json.dumps(model.state())))
        self.assertEqual(restored.forecast('x'), model.forecast('x'))
        self.assertEqual(restored.errors, model.errors)


if __name__ == '__main__':
    unittest.main()
