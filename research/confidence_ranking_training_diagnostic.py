"""Describe pre-update training metrics without reading evaluation outcomes.

These are online observations at changing checkpoints and changing pairings,
not a frozen-checkpoint training-set score or a generalization estimate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from research.confidence_ranking import CONFIG, paired_batches
from research.confidence_ranking_journal import read_journal
from research.confidence_ranking_study import source_hash
from research.cross_model_prediction import CELLS
from research.iphone_coupling_report import require


def margin_metrics(margins):
    require(all(type(value) in (int, float) and math.isfinite(value) for value in margins), 'Finite signed margins')
    n = len(margins)
    wins = sum(value > 0 for value in margins); ties = sum(value == 0 for value in margins)
    logistic = lambda value: max(-value, 0.)+math.log1p(math.exp(-abs(value)))
    return dict(eligiblePairs=n, correctlyOrdered=wins, tied=ties, reversed=n-wins-ties,
                orderingCredit=(wins+.5*ties)/n if n else None,
                meanSignedMargin=math.fsum(margins)/n if n else None,
                meanAbsoluteMargin=math.fsum(abs(v) for v in margins)/n if n else None,
                meanLogisticRankingLoss=math.fsum(logistic(v) for v in margins)/n if n else None)


def analyze(path, freeze_path):
    path = Path(path); raw = path.read_bytes()
    freeze = json.loads(Path(freeze_path).read_text(encoding='utf-8'))
    require(hashlib.sha256(raw).hexdigest() == freeze['prefixSHA256'] and len(raw) == freeze['prefixBytes'], 'Frozen training-only prefix required')
    require(all(json.loads(line)['payload']['event'] in ('header', 'training_start', 'training_step', 'training_complete')
                for line in raw.splitlines()), 'Evaluation events forbidden')
    data = read_journal(path)
    require(not data['complete'] and not data['failed'] and data['recorded'] == 0 and data['pending'] is None, 'Training-only journal')
    require(data['activeTraining'] is None and len(data['checkpoints']) == 9 and len(data['steps']) == 1296, 'Complete training required')
    require(data['header']['sourceHash'] == freeze['sourceHash'] == source_hash(), 'Frozen science sources changed')
    rows = data['header']['trainingData']; units = {}; starts_equal = []
    for rep in range(3):
        batches = paired_batches(rows, rep); epochs = CONFIG['epochs']
        per_epoch = len(batches)//epochs
        require(epochs == 2 and per_epoch == 72, 'Declared epoch boundaries changed')
        first = []
        for arm in ('ce', 'rank', 'neutral'):
            key = f'r{rep}-{arm}'; steps = [step for step in data['steps'] if step['key'] == key]
            require([s['step'] for s in steps] == list(range(1, 145)), 'Complete ordered updates')
            first.append({field: steps[0][field] for field in ('losses', 'scoreDifferences', 'inputTokens')})
            tables = []
            for epoch in range(epochs):
                cells = {f'{family}/{level}': [] for family, level in CELLS}
                exposed = Counter(); point_losses = []; auxiliary = []; objectives = []; gradients = []
                start = epoch*per_epoch; stop = start+per_epoch
                for step, batch in zip(steps[start:stop], batches[start:stop]):
                    point_losses.extend(step['losses']); auxiliary.extend(step['pairLosses'])
                    objectives.append(step['objective']); gradients.append(step['gradientNorm'])
                    for pair, difference, eligible in zip(batch, step['scoreDifferences'], step['eligible']):
                        left, right = pair['left'], pair['right']
                        require((left['family'], left['level']) == (right['family'], right['level']), 'Cross-cell pair')
                        cell = f'{left["family"]}/{left["level"]}'; exposed[cell] += 2
                        if eligible:
                            cells[cell].append((int(left['target'])-int(right['target']))*difference)
                margins = [value for values in cells.values() for value in values]
                metrics = margin_metrics(margins)
                require(len(point_losses) == 576 and len(auxiliary) == 288 and set(exposed.values()) == {96}, 'Exposure budget')
                metrics.update(epoch=epoch+1, steps=per_epoch, examples=len(point_losses), allPairs=len(auxiliary),
                    meanPointwiseCodeEOSLoss=math.fsum(point_losses)/len(point_losses),
                    meanActualAuxiliaryLossAllPairs=math.fsum(auxiliary)/len(auxiliary),
                    meanRecordedObjective=math.fsum(objectives)/len(objectives),
                    meanPreclipGradientNorm=math.fsum(gradients)/len(gradients),
                    clippingSteps=sum(norm > CONFIG['clipNorm'] for norm in gradients),
                    withinCell={cell: dict(margin_metrics(values), examples=exposed[cell]) for cell, values in cells.items()})
                expected = metrics['meanPointwiseCodeEOSLoss']+CONFIG['pairWeight']*metrics['meanActualAuxiliaryLossAllPairs']
                require(math.isclose(metrics['meanRecordedObjective'], expected, rel_tol=1e-9, abs_tol=1e-9), 'Aggregate objective arithmetic')
                tables.append(metrics)
            units[key] = dict(replication=rep, arm=arm, epochs=tables,
                epoch2MinusEpoch1={field: tables[1][field]-tables[0][field] for field in (
                    'meanPointwiseCodeEOSLoss', 'meanSignedMargin', 'orderingCredit', 'meanLogisticRankingLoss')})
        starts_equal.append(first[0] == first[1] == first[2])
    require(all(starts_equal), 'Matched arms differ before first update')
    return dict(schema='menia-ranking-training-diagnostic-v1', origin='actual_training_preupdate_metrics',
        diagnosticOnly=True, journalSHA256=freeze['prefixSHA256'], scienceSourceHash=freeze['sourceHash'],
        planHash=freeze['planHash'], trainingSteps=1296, completedAdapters=9, recordedEvaluationCalls=0,
        matchedFirstForwardMetrics=starts_equal, units=units,
        diagnosticSourceSHA256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest(),
        scope='Online pre-update observations on old training labels. Checkpoints and pairings change; '
              'epoch differences are descriptive, not paired frozen-model gains, independent replications, '
              'convergence proof, evaluation performance, native action or consciousness evidence. '
              'The frozen Colab24 primary criterion is unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path); parser.add_argument('--freeze', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); report = analyze(args.journal, args.freeze)
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        require(args.output.read_text(encoding='utf-8') == content, 'Existing different report')
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream: stream.write(content)
    print(json.dumps(dict(origin=report['origin'], units={key: dict(
        orderingCredit=[e['orderingCredit'] for e in value['epochs']],
        rankingLoss=[e['meanLogisticRankingLoss'] for e in value['epochs']],
        pointwiseLoss=[e['meanPointwiseCodeEOSLoss'] for e in value['epochs']],
        clippingSteps=[e['clippingSteps'] for e in value['epochs']]) for key,value in report['units'].items()})))
