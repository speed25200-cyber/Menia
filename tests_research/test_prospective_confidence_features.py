import copy
import math
import unittest

import numpy as np

from research.prospective_confidence_features import features, NAMES, validate_capture


class ProspectiveFeatureTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(family='countA', level=8, question='Combien de lettres A contient cette chaîne : ABCDABCD ?')
        self.capture = dict(state={key: [float(i) for i in range(128)] for key in ('input', 'middle', 'final')},
                            preAnswer=dict(vocabularySize=4, topTokenId=0, maxProbability=.25,
                                           topTwoMargin=0., entropyNats=math.log(4), normalizedEntropy=1.))

    def test_nested_feature_sets_and_exact_dimensions(self):
        values = {name: features(self.task, self.capture, name) for name in NAMES}
        self.assertEqual({name: len(v) for name, v in values.items()},
                         dict(inputOnly=390, outputConfidence=9, inputConfidence=393, finalControl=521, internal=649))
        for short, long in (('inputOnly', 'inputConfidence'), ('inputConfidence', 'finalControl'), ('finalControl', 'internal')):
            np.testing.assert_array_equal(values[short], values[long][:len(values[short])])
        np.testing.assert_array_equal(values['internal'][-128:], self.capture['state']['middle'])
        np.testing.assert_array_equal(values['finalControl'][-128:], self.capture['state']['final'])

    def test_intermediate_changes_cannot_change_output_control(self):
        altered = copy.deepcopy(self.capture)
        altered['state']['middle'] = [-v for v in altered['state']['middle']]
        for name in NAMES[:-1]:
            np.testing.assert_array_equal(features(self.task, self.capture, name), features(self.task, altered, name))
        self.assertFalse(np.array_equal(features(self.task, self.capture, 'internal'), features(self.task, altered, 'internal')))
        # An argmax token is available before sampling but is intentionally not
        # a numeric feature: arbitrary vocabulary IDs must not be treated as magnitudes.
        altered = copy.deepcopy(self.capture)
        altered['preAnswer']['topTokenId'] = 3
        np.testing.assert_array_equal(features(self.task, self.capture, 'internal'), features(self.task, altered, 'internal'))

    def test_future_fields_and_inconsistent_confidence_are_rejected(self):
        for key in ('completion', 'answer', 'correct', 'predictions'):
            bad = dict(self.capture, **{key: {}})
            with self.assertRaisesRegex(ValueError, 'Only pre-answer'):
                features(self.task, bad, 'internal')
        for key, value in (('tokenIds', [0]), ('maxProbability', float('nan')), ('entropyNats', -1.),
                           ('normalizedEntropy', .5), ('vocabularySize', True), ('topTokenId', 4), ('topTwoMargin', .9)):
            bad = copy.deepcopy(self.capture)
            bad['preAnswer'][key] = value
            with self.assertRaises(ValueError):
                validate_capture(bad)


if __name__ == '__main__':
    unittest.main()
