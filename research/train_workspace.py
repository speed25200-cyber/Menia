"""Train a recurrent shared workspace and test causal dependencies on held-out data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from .workspace import SharedWorkspace, DirectLookup, intermediate_labels, lookup_batch, lookup_test


def train(seed, steps=2000, rounds=4, variant='workspace'):
    torch.manual_seed(seed)
    model = DirectLookup() if variant == 'direct' else SharedWorkspace()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    history = []
    model.train()
    for step in range(steps):
        tables, query, target = lookup_batch(seed*100000+step)
        optimizer.zero_grad()
        if variant == 'direct':
            result = model(tables, query)
        else:
            result = model(tables, query, rounds=rounds,
                           intervention='no_broadcast' if variant == 'without_broadcast' else 'none')
        loss = F.cross_entropy(result['answer'], target)
        loss = loss + .25*F.cross_entropy(result['intermediate'], intermediate_labels(tables, query))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 0 or (step+1) % 250 == 0:
            entry = {'step': step+1, 'loss': float(loss.detach())}
            history.append(entry)
            print(json.dumps({'seed': seed, 'variant': variant, **entry}), flush=True)
    return model.eval(), history


@torch.no_grad()
def evaluate(model, *, rounds=4, trained_without_broadcast=False):
    tables, query, target = lookup_test()
    batch = len(target)
    result = {}
    conditions = ('none', 'no_broadcast', 'uniform_attention', 'reset_specialists',
                  'reset_workspace', 'no_table_a', 'no_table_b') if isinstance(model, SharedWorkspace) else ('none',)
    for condition in conditions:
        if isinstance(model, SharedWorkspace):
            actual_condition = 'no_broadcast' if trained_without_broadcast and condition == 'none' else condition
            output = model(tables, query, rounds=rounds, intervention=actual_condition)
            logits = output['answer']
        else:
            output = model(tables, query)
            logits = output['answer']
        correct = logits.argmax(-1) == target
        result[condition] = {'episodes': batch, 'correct': int(correct.sum()),
                             'accuracy': float(correct.float().mean()),
                             'all_four_queries_correct_pairs': int(correct.reshape(-1, 4).all(-1).sum()),
                             'intermediate_accuracy': float((output['intermediate'].argmax(-1) ==
                                 intermediate_labels(tables, query)).float().mean()),
                             'cross_entropy': float(F.cross_entropy(logits, target))}
    # Computed reference has access to the same observation tables.
    table_symbols = tables.reshape(batch, 2, 4, 4).argmax(-1)
    indices = torch.arange(batch)
    middle = table_symbols[indices, 0, query.argmax(-1)]
    oracle = table_symbols[indices, 1, middle]
    result['explicit_lookup_rule'] = {'accuracy': float((oracle == target).float().mean())}
    result['constant_zero'] = {'accuracy': float((target == 0).float().mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29, 43])
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError('Positive training steps required')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty destination')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    runs = []
    for seed in args.seeds:
        for variant in ('workspace', 'without_broadcast', 'direct'):
            model, history = train(seed, args.steps, variant=variant)
            checkpoint = out/f'{variant}-{seed}.pt'
            torch.save(model.state_dict(), checkpoint)
            # Explicit weights-only load; evaluate the exported model.
            reloaded = DirectLookup() if variant == 'direct' else SharedWorkspace()
            reloaded.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
            result = {'seed': seed, 'variant': variant, 'parameters': sum(p.numel() for p in model.parameters()),
                      'checkpoint': checkpoint.name, 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'history': history, 'evaluation': evaluate(reloaded.eval(),
                          trained_without_broadcast=variant == 'without_broadcast')}
            if variant == 'workspace':
                result['rounds_transfer'] = {str(r): evaluate(reloaded, rounds=r)['none']
                                             for r in (1, 2, 6, 8)}
            runs.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != 'history'}), flush=True)
    root = Path(__file__).resolve().parents[1]
    report = {'status': 'executed CPU experiment', 'python': platform.python_version(),
              'torch': torch.__version__, 'wall_seconds': time.perf_counter()-start,
              'steps': args.steps, 'train_batch': 128, 'train_rounds': 4,
              'test_split': 'all pair IDs divisible by 5; 116 table pairs, all 4 queries',
              'training_pairs': 460, 'test_pairs': 116,
              'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                  for name in ('research/workspace.py', 'research/train_workspace.py')},
              'limitations': ['symbolic composition only; not consciousness evidence',
                              'small finite permutation world; not broad generalization',
                              'four queries within a table pair are dependent; pair counts reported',
                              'all models receive the same auxiliary intermediate supervision',
                              'baseline parameter counts differ and are reported',
                              'inference ablations measure dependence, not architectural necessity',
                              'latent dimension is a bottleneck, not a bound on bits of information',
                              'not integrated with symbolic session, language model or iPhone'],
              'runs': runs}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
