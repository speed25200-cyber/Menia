"""Within-category paired learning; preserve every original training example.

Preparation for a new experiment, not a replacement for the frozen Colab23.
Only pairing uses correctness labels; neither labels nor partner texts enter
the model's context. All arms retain identical truthful pointwise targets.
"""
from collections import Counter
import copy
import random

from research.answer_confidence_plan import TRAINING as PARENT_TRAINING
from research.cross_model_prediction import CELLS, digest
from research.iphone_coupling_report import require

SEED = 202609240
ARMS = ('ce', 'rank', 'neutral')
CONFIG = dict(copy.deepcopy(PARENT_TRAINING), pairWeight=1., pairsPerBatch=4,
              loss='Mean code/EOS CE plus mean eligible logistic pair loss')


def rng_for(*parts):
    return random.Random(int(digest([SEED, *parts])[:16], 16))


def paired_batches(rows, replication):
    """Each of 576 examples appears once/epoch, even in single-class cells.

    Opposite-label pairing without replacement gives min(n0,n1) eligible
    pairs/cell/epoch. Remaining same-label pairs have zero auxiliary loss.
    No oversampling, cross-category pairing, or held-out examples are allowed.
    """
    require(type(replication) is int and 0 <= replication < 3, 'Replication')
    require(len(rows) == 1728 and len({r['sourceId'] for r in rows}) == 1728,
            'Complete distinct training examples required')
    require(all(r['split'] == 'train' and r['target'] in ('0', '1') for r in rows),
            'Only binary-labelled training data')
    counts = Counter((r['replication'], r['family'], r['level']) for r in rows)
    require(counts == Counter({(r, *c): 96 for r in range(3) for c in CELLS}),
            'Training strata')
    batches = []
    for epoch in range(CONFIG['epochs']):
        pairs = []
        for family, level in CELLS:
            group = sorted((r for r in rows if (r['replication'], r['family'], r['level']) ==
                            (replication, family, level)), key=lambda r: r['sourceId'])
            rng = rng_for('pairs', replication, epoch, family, level)
            good = [r for r in group if r['target'] == '1']
            bad = [r for r in group if r['target'] == '0']
            rng.shuffle(good); rng.shuffle(bad)
            n = min(len(good), len(bad))
            cell_pairs = list(zip(good[:n], bad[:n]))
            remaining = good[n:] + bad[n:]
            require(len(remaining) % 2 == 0, 'Odd unpaired remainder')
            cell_pairs.extend(zip(remaining[::2], remaining[1::2]))
            for left, right in cell_pairs:
                if rng.randrange(2):
                    left, right = right, left
                pairs.append(dict(left=left, right=right,
                                  eligible=left['target'] != right['target']))
        rng_for('batch-order', replication, epoch).shuffle(pairs)
        require(len(pairs) == 288, 'Epoch budget')
        batches.extend(pairs[start:start+4] for start in range(0, len(pairs), 4))
    require(len(batches) == CONFIG['updatesPerAdapter'], 'Update budget')
    return batches


def preparation_report(rows):
    """Exact schedule coverage, with rare/missing successes made explicit."""
    groups = {}; schedules = {}
    for rep in range(3):
        batches = paired_batches(rows, rep)
        schedules[str(rep)] = dict(hash=digest(batches), batches=len(batches),
                                  examplesPerEpoch=576, pairsPerEpoch=288)
        for family, level in CELLS:
            group = [r for r in rows if (r['replication'], r['family'], r['level']) ==
                     (rep, family, level)]
            n1 = sum(r['target'] == '1' for r in group); n0 = len(group)-n1
            groups[f'r{rep}/{family}/{level}'] = dict(
                correct=n1, incorrect=n0, eligiblePairsPerEpoch=min(n0, n1),
                eligibleExamplesPerEpoch=2*min(n0, n1),
                uniqueEligibleExamplesAcrossEpochs=len({
                    r['sourceId'] for batch in batches for pair in batch if pair['eligible']
                    for r in (pair['left'], pair['right'])
                    if (r['family'], r['level']) == (family, level)}),
                rareClass=min(n0, n1) <= 1)
    return dict(schema='menia-confidence-ranking-preparation-v1',
                status='Software and data preparation; no new learned result',
                trainingHash=digest(rows), config=CONFIG, arms=list(ARMS),
                groups=groups, schedules=schedules,
                uniqueTrainingExamples=len(rows), adapterUpdatesPlanned=1296,
                trainedAdaptersPlanned=9, initialAdaptersPlanned=3,
                scope='Rank 8 held fixed to isolate the loss. No capacity comparison, '
                      'privileged self-access, action benefit or consciousness conclusion.')
