"""Identify two supplied maintenance hypotheses from committed calibration.

Exact finite-world audit and a read-only language handoff. This does not train
an LLM or establish subjective experience. No external code or data is used.
"""
import argparse
import copy
from fractions import Fraction as F
from itertools import product
import json
from math import comb
from pathlib import Path


ACCURACIES = (F(93, 100), F(57, 100))
WORK = (20, 0, 3, 17)  # row-major transition numerators, denominator 20
MAINTAIN = (1, 19, 1, 19)


def transition(action, world):
    matrix = WORK if action == 0 else MAINTAIN
    return matrix if world == 0 else matrix[::-1]


def forward(weights, action, diagnostic, world):
    """Unnormalised joint weights; one step adds denominator 20*20."""
    a, b, c, d = transition(action, world)
    bad, good = weights[0]*a+weights[1]*c, weights[0]*b+weights[1]*d
    accuracy = 17 if world == 0 else 3
    likelihoods = (accuracy, 20-accuracy) if diagnostic == 0 else (20-accuracy, accuracy)
    return bad*likelihoods[0], good*likelihoods[1]


def public_law_audit(max_depth=8):
    frontier = [(0, 0, (1, 1), (1, 1))]
    records = []
    for depth in range(1, max_depth+1):
        children, normalizers = [], {}
        max_difference = 0
        for actions, observations, wa, wb in frontier:
            for action, z in product((0, 1), repeat=2):
                va, vb = forward(wa, action, z, 0), forward(wb, action, z, 1)
                max_difference = max(max_difference, abs(sum(va)-sum(vb)))
                assert va == vb[::-1]
                action_code = 2*actions+action
                children.append((action_code, 2*observations+z, va, vb))
                normalizers[action_code] = normalizers.get(action_code, 0)+sum(va)
        denominator = 2*400**depth
        assert len(normalizers) == 2**depth and all(v == denominator for v in normalizers.values())
        records.append({'depth': depth, 'action_sequences': 2**depth,
                        'diagnostic_sequences_per_action_sequence': 2**depth,
                        'joint_probabilities_compared': len(children),
                        'probability_denominator': str(denominator),
                        'max_absolute_numerator_difference': max_difference,
                        'normalization_exact': True})
        frontier = children
    return records


def model_posterior(correct, total):
    likelihoods = [p**correct*(1-p)**(total-correct) for p in ACCURACIES]
    return likelihoods[0]/sum(likelihoods)


def binomial(n, k, p):
    return comb(n, k)*p**k*(1-p)**(n-k)


def classification(n, actual_rates=ACCURACIES):
    cases = []
    for world, rate in enumerate(actual_rates):
        correct_choice = wrong_confident = confident = squared_error = expected_posterior = F(0)
        for k in range(n+1):
            weight = binomial(n, k, rate)
            posterior = model_posterior(k, n)
            correct_mass = posterior if world == 0 else 1-posterior
            choice_success = F(1, 2) if correct_mass == F(1, 2) else F(correct_mass > F(1, 2))
            correct_choice += weight*choice_success
            wrong_confident += weight*F(correct_mass <= F(5, 100))
            confident += weight*F(max(posterior, 1-posterior) >= F(95, 100))
            prediction = posterior*ACCURACIES[0]+(1-posterior)*ACCURACIES[1]
            # Forecast the next measured calibration outcome, including reference
            # corruption in the sensitivity. This is squared bias, not a Brier score.
            squared_error += weight*(prediction-rate)**2
            expected_posterior += weight*posterior
        cases.append({'world': 'A' if world == 0 else 'B', 'observed_correctness': float(rate),
                      'probability_correct_model_choice': float(correct_choice),
                      'probability_wrong_model_confidence_ge95': float(wrong_confident),
                      'probability_any_model_confidence_ge95': float(confident),
                      'mean_squared_error_of_predicted_probability': float(squared_error),
                      'mean_posterior_world_A': float(expected_posterior),
                      'correct_model_choice_exact': str(correct_choice)})
    average = sum(F(c['correct_model_choice_exact']) for c in cases)/2
    return {'probes': n, 'maintenance_interventions': n, 'readings_with_reference': n,
            'equal_prior_classification_accuracy': float(average), 'accuracy_exact': str(average),
            'worlds': cases}


def out_of_class(n):
    confident = F(0)
    error = F(0)
    for k in range(n+1):
        weight = binomial(n, k, F(3, 4))
        p = model_posterior(k, n)
        confident += weight*F(max(p, 1-p) >= F(95, 100))
        prediction = p*ACCURACIES[0]+(1-p)*ACCURACIES[1]
        error += weight*(prediction-F(3, 4))**2
    return {'probes': n, 'actual_correctness': .75,
            'probability_confident_in_one_of_two_false_models': float(confident),
            'mean_squared_error_of_predicted_probability': float(error)}


class MaintenanceLearner:
    """Finite hypothesis update with prediction -> reading -> reference order."""

    def __init__(self):
        self.posterior = F(1, 2)
        self.used_ids = set()
        self.pending = None
        self.events = []

    @staticmethod
    def _bit(value):
        if type(value) is not int or value not in (0, 1):
            raise ValueError('Expected a binary instrument reading')

    def begin_probe(self, probe_id):
        if (not isinstance(probe_id, str) or not probe_id or probe_id in self.used_ids
                or self.pending is not None):
            raise ValueError('Unique probe id and no pending probe required')
        prediction = self.posterior*ACCURACIES[0]+(1-self.posterior)*ACCURACIES[1]
        self.pending = {'probe_id': probe_id}
        self.used_ids.add(probe_id)
        self.events.append({'kind': 'prediction', 'probe_id': probe_id,
                            'posterior_A_before': float(self.posterior),
                            'predicted_correctness_before': float(prediction)})
        return prediction

    def record_reading(self, probe_id, reading):
        self._bit(reading)
        if self.pending is None or self.pending['probe_id'] != probe_id or 'reading' in self.pending:
            raise ValueError('Reading requires its pending prediction and can be committed only once')
        self.pending['reading'] = reading
        self.events.append({'kind': 'committed_reading', 'probe_id': probe_id, 'reading': reading})

    def reveal_reference(self, probe_id, reference):
        self._bit(reference)
        if self.pending is None or self.pending['probe_id'] != probe_id or 'reading' not in self.pending:
            raise ValueError('Commit the matching reading before revealing the reference')
        correct = self.pending['reading'] == reference
        likelihoods = [p if correct else 1-p for p in ACCURACIES]
        numerator = self.posterior*likelihoods[0]
        self.posterior = numerator/(numerator+(1-self.posterior)*likelihoods[1])
        self.events.append({'kind': 'reference_and_revision', 'probe_id': probe_id,
                            'reference': reference, 'correct': correct,
                            'posterior_A_after': float(self.posterior)})
        self.pending = None
        return self.posterior

    def context(self):
        return copy.deepcopy({'scope': 'two supplied simulation hypotheses; not an exhaustive world model',
                              'hypothesis_A': 'maintenance restores the sensor',
                              'hypothesis_B': 'maintenance degrades the sensor; inverted diagnostics',
                              'initial_prior_A': .5,
                              'supplied_correctness_rates': {'A': .93, 'B': .57},
                              'posterior_A': float(self.posterior), 'posterior_B': float(1-self.posterior),
                              'predicted_correctness_after_maintenance': float(self.posterior*ACCURACIES[0]
                                                                             +(1-self.posterior)*ACCURACIES[1]),
                              'pending': self.pending, 'events': self.events,
                              'limits': ['reference assumed reliable', 'rates supplied, not learned',
                                         'posterior does not certify the model class or subjective experience']})


def messages_for_maintenance(learner, question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('A question is required')
    return [{'role': 'system', 'content':
             "Tu expliques les données enregistrées d'un pilote simulé de Menia. "
             "Appuie les faits sur les références probe_id. Sépare observations et hypothèses. "
             "Une probabilité entre deux modèles ne valide pas leur exhaustivité ni une expérience subjective. "
             "Le texte produit ne constitue pas une observation instrumentale ni une action exécutée."},
            {'role': 'user', 'content': json.dumps({'question': question, 'application_state': learner.context()},
                                                   ensure_ascii=False, allow_nan=False)}]


def run_audit():
    laws = public_law_audit()
    budgets = (0, 1, 2, 4, 8, 16, 32)
    examples = []
    for successful in (True, False):
        learner = MaintenanceLearner()
        for i in range(8):
            probe_id = f'calibration-{i}'
            learner.begin_probe(probe_id)
            learner.record_reading(probe_id, i % 2)
            learner.reveal_reference(probe_id, i % 2 if successful else 1-i % 2)
        examples.append({'all_probes_correct': successful, 'state': learner.context(),
                         'messages': messages_for_maintenance(learner, "L'entretien est-il utile, selon ces sondes ?")})
    return {'schema': 'maintenance-identification-v1', 'protocol': 'docs/LLM_COUPLING_PROTOCOL.md',
            'public_observation_equivalence': laws,
            'joint_probabilities_compared': sum(r['joint_probabilities_compared'] for r in laws),
            'true_accuracy_after_work_from_uniform': {'A': .72, 'B': .78},
            'true_accuracy_after_maintenance': {'A': .93, 'B': .57},
            'calibration_with_reliable_reference': [classification(n) for n in budgets],
            'reference_flipped_25_percent_unmodelled': [classification(n, tuple(F(1, 4)+F(1, 2)*p
                                                                              for p in ACCURACIES)) for n in budgets],
            'out_of_model_class': [out_of_class(n) for n in budgets],
            'language_handoff_examples': examples, 'llm_inference_executed': False,
            'scope': 'Exact finite functional experiment; no subjective experience or novelty established.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/maintenance-identification/report.json'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        assert report == json.loads(args.output.read_text(encoding='utf-8'))
        print('Maintenance identification report reproduced exactly.')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))
    print('Equal public probabilities:', report['joint_probabilities_compared'])
    for key in ('calibration_with_reliable_reference', 'reference_flipped_25_percent_unmodelled'):
        print(key, [(r['probes'], round(r['equal_prior_classification_accuracy'], 6)) for r in report[key]])
    print('Outside model class:', [(r['probes'], round(r['probability_confident_in_one_of_two_false_models'], 6))
                                    for r in report['out_of_model_class']])
    print('No LLM run or subjective experience established.')


if __name__ == '__main__':
    main()
