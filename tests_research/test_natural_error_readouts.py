import copy
import math
import unittest
from unittest.mock import patch

from research import natural_error_readouts as study


def rows():
    result = []
    for rep in (0, 1):
        for split, count in (('train', 4), ('validation', 2), ('test', 2)):
            for family, level in study.CELLS:
                for i in range(count):
                    success = i % 2 == 0
                    letters = 'A' * level if family == 'countA' else ''
                    operands = [] if letters else [12]*level
                    correct = level if letters else 0
                    state = {key: [0.]*128 for key in ('input', 'middle', 'final')}
                    state['middle'] = [float(success)]*128
                    capture = dict(state=state, preAnswer=dict(vocabularySize=4, topTokenId=0, maxProbability=.25,
                                   topTwoMargin=0., entropyNats=math.log(4), normalizedEntropy=1.))
                    result.append(dict(task=dict(id=len(result), replication=rep, split=split, family=family, level=level,
                        letters=letters, operands=operands, question='a fixed question in this synthetic category'), capture=capture,
                        result=dict(status='ok', text=str(correct if success else correct+1))))
    return result


class NaturalErrorReadoutTests(unittest.TestCase):
    def test_future_rows_and_other_repetitions_cannot_change_fit(self):
        with patch.dict(study.COUNTS, train=4, validation=2):
            data = rows()
            actual = study.fit_bundle(data, 0)
            altered = copy.deepcopy(data)
            for row in altered:
                if row['task']['split'] == 'test' or row['task']['replication'] != 0:
                    row['result'] = dict(status='error', text='not a label')
                    row['capture'] = {'future': 'must not be read'}
            self.assertEqual(actual, study.fit_bundle(altered, 0))
            self.assertEqual((actual['trainCount'], actual['validationCount']), (24, 12))
            self.assertLess(min(x['validationBrier'] for x in actual['selection']['internal']), .001)
            self.assertEqual(min(x['validationBrier'] for x in actual['selection']['finalControl']), .25)
            self.assertTrue(all(len(c) == len(study.ALPHAS) for c in actual['selection'].values()))

    def test_predictions_use_current_middle_only_in_the_intended_arm(self):
        with patch.dict(study.COUNTS, train=4, validation=2):
            data = rows()
            fitted = study.fit_bundle(data, 0)
            row = next(r for r in data if r['task']['replication'] == 0 and r['task']['split'] == 'test')
            prediction = study.forecast(fitted, data, row['task'], row['capture'])
            changed = copy.deepcopy(row['capture'])
            changed['state']['middle'] = [0.]*128
            other = study.forecast(fitted, data, row['task'], changed)
            self.assertEqual(set(prediction), set(study.FORECASTS))
            self.assertGreater(prediction['internal'], .99)
            self.assertLess(other['internal'], .01)
            for name in set(study.FORECASTS)-{'internal'}:
                self.assertEqual(prediction[name], other[name])
            for donor, original in zip(fitted['shuffledTrainingDonorIds'], [r for r in data if r['task']['replication'] == 0 and r['task']['split'] == 'train']):
                donor_row = next(r for r in data if r['task']['id'] == donor)
                self.assertEqual(study.cell(donor_row), study.cell(original))

    def test_missing_duplicate_and_failed_fitting_rows_are_rejected(self):
        with patch.dict(study.COUNTS, train=4, validation=2):
            data = rows()
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                study.fit_bundle(data[1:], 0)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                study.fit_bundle(data+[data[0]], 0)
            data[0]['result']['status'] = 'error'
            with self.assertRaisesRegex(ValueError, 'Technical error'):
                study.fit_bundle(data, 0)


if __name__ == '__main__':
    unittest.main()
