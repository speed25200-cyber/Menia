import unittest
from menia.agency_discovery import AttributionModel, ProbeAgent, ConfoundedChannels
from menia.episodic import EpisodeMemory
from research.evaluate_agency_discovery import counterexample


class AgencyDiscoveryTests(unittest.TestCase):
    def test_coupled_history_cannot_distinguish_action_and_cue(self):
        model = AttributionModel()
        for tick in range(24):
            cue = (-1, 1)[tick % 2]
            model.update(0, cue, cue, cue)
        self.assertEqual(model.posterior[0][0], model.posterior[0][2])
        self.assertEqual(model.assessment()['labels'][0], 'unknown')
        for cue in (-1, 1)*3:
            model.update(0, -cue, cue, -cue)
        self.assertEqual(model.assessment()['labels'][0], 'controlled')

    def test_missing_reading_never_updates_causal_posterior(self):
        model = AttributionModel()
        before = list(model.posterior[0])
        initial_gain = model.information_gain(0, 1, -1)
        for _ in range(12): model.update(0, 1, -1, None)
        self.assertEqual(before, model.posterior[0])
        self.assertEqual(model.received[0], 0)
        self.assertLess(model.information_gain(0, 1, -1), initial_gain)

    def test_probe_changes_with_inferred_cause_at_same_cue(self):
        agent = ProbeAgent()
        for tick in range(24):
            cue = (-1, 1)[tick % 2]
            # Alternate channel and vary cue within each channel.
            cue = (-1, 1)[(tick//2) % 2]
            decision = agent.choose(cue, warmup=True)
            agent.receive(cue)
        decision = agent.choose(1)
        self.assertEqual(decision['action'], -1)
        selected = decision['channel']
        agent.receive(-1)
        # Concentrate one channel on an already-resolved cause; attention goes elsewhere.
        agent.model.posterior[selected] = [1, 0, 0, 0, 0, 0]
        next_decision = agent.choose(1)
        self.assertNotEqual(next_decision['channel'], selected)

    def test_physical_label_of_perfect_copy_is_unidentifiable(self):
        first = ConfoundedChannels(7, condition='twins')
        second = ConfoundedChannels(7, condition='twins')
        second.selected_body = 1-first.selected_body
        for tick in range(40):
            self.assertEqual(first.cue(), second.cue())
            self.assertEqual(first.read((-1, 1)[tick % 2], tick % 2),
                             second.read((-1, 1)[tick % 2], tick % 2))

    def test_decision_journal_precedes_observation_and_no_double_receipt(self):
        memory = EpisodeMemory()
        self.addCleanup(memory.close)
        agent = ProbeAgent(memory=memory)
        decision = agent.choose(1)
        with self.assertRaises(RuntimeError): agent.choose(1)
        agent.receive(-1)
        events = memory.recent(agent.episode)
        self.assertEqual([e['kind'] for e in events], ['decision', 'observation', 'assessment'])
        self.assertEqual(events[0]['id'], decision['event_id'])
        self.assertEqual(events[-1]['payload']['decision']['event_id'], decision['event_id'])
        with self.assertRaises(RuntimeError): agent.receive(-1)

    def test_ablation_counterexample_and_true_intervention_diverge(self):
        worlds = counterexample()['worlds']
        self.assertEqual(worlds[0]['observations'], worlds[1]['observations'])
        self.assertNotEqual(worlds[0]['intervention_action_opposes_cue'], worlds[1]['intervention_action_opposes_cue'])
        for world in worlds:
            rows = world['observations']
            mse = sum((r['y']-r['ablated_prediction'])**2 for r in rows)/len(rows)
            self.assertEqual(mse, world['action_input_ablated_mse'])


if __name__ == '__main__':
    unittest.main()
