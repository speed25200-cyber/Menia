import unittest

from menia.replay_control import Policy, candidates, execute, improve, replay, unique


def worlds(signal=True):
    return [dict(id=i, features=dict(betaCell=.5, internal=float(i%2) if signal else .5),
                 outcomes=dict(direct=dict(pointLoss=float(i%2==0)), verify=dict(pointLoss=.2), abstain=dict(pointLoss=.35)))
            for i in range(40)]


class ReplayControllerTests(unittest.TestCase):
    def test_signal_can_improve_historical_routing_and_null_cannot(self):
        initial = Policy()
        selected, report = improve(worlds(), candidates(("betaCell","internal")), initial)
        self.assertAlmostEqual(report['incumbentLoss'], .2)
        self.assertAlmostEqual(report['selectedLoss'], .1)
        self.assertEqual(selected.source,'internal')
        selected, report = improve(worlds(False), candidates(("betaCell","internal")), initial)
        self.assertEqual(selected,initial)
        self.assertAlmostEqual(report['selectedLoss'], .2)

    def test_replay_does_not_guarantee_future_improvement(self):
        selected,_ = improve(worlds(), candidates(("betaCell","internal")), Policy())
        shifted = worlds()
        for row in shifted:
            row['outcomes']['direct']['pointLoss'] = 1-row['outcomes']['direct']['pointLoss']
        self.assertGreater(replay(selected,shifted)['meanPointLoss'], replay(Policy(),shifted)['meanPointLoss'])

    def test_missing_actions_and_repeated_rows_are_rejected(self):
        row=worlds()[0]; del row['outcomes']['verify']
        with self.assertRaises(ValueError): replay(Policy(),[row])
        with self.assertRaises(ValueError): replay(Policy(kind='direct'),[row,row])
        with self.assertRaises(ValueError): replay(Policy(),[])

    def test_policy_cannot_read_truth_and_constants_need_no_feature(self):
        policy = Policy(source='internal')
        features={'internal':.9}
        self.assertEqual(policy.choose(features),'direct')
        self.assertEqual(features,{'internal':.9})
        self.assertEqual(Policy(kind='abstain').choose({}),'abstain')
        with self.assertRaises(ValueError): policy.choose({'label':1})
        with self.assertRaises(ValueError): policy.choose({'internal':float('nan')})
        with self.assertRaises(ValueError): Policy.load(dict(policy.payload(), code='untrusted'))

    def test_primary_dispatch_only_invokes_selected_tool(self):
        calls=[]
        def verify(): calls.append(1); return '17'
        self.assertEqual(execute('direct','16',verify),'16')
        self.assertIsNone(execute('abstain','16',verify))
        self.assertEqual(calls,[])
        self.assertEqual(execute('verify','16',verify),'17')
        self.assertEqual(calls,[1])
        with self.assertRaises(ValueError): execute('retry','16',verify)
        with self.assertRaises(ValueError): execute('verify','16',lambda:None)

    def test_candidate_sets_and_serialization(self):
        self.assertEqual(len(unique(candidates())),99)
        self.assertEqual(len(unique(candidates(('betaCell','inputOnly')))),51)
        self.assertEqual(len(unique(candidates(('betaCell','inputOnly','internal')))),75)
        for p in unique(candidates()): self.assertEqual(p,Policy.load(p.payload()))


if __name__=='__main__': unittest.main()
