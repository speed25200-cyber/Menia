"""Train independent small networks in a batched CPU operation; retain all seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .temporal_effects import FAMILIES, world_parameters, dataset, features, fit_ridge, metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('research/temporal_effects.py', 'research/train_temporal_effects.py',
           'docs/TEMPORAL_SELF_MODEL_PROTOCOL.md')


def train_networks(x, visible_y, mask, *, seed, hidden, updates=600):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(visible_y, dtype=torch.float32)
    observed = torch.tensor(mask, dtype=torch.float32)
    count, length, width = x.shape
    w1 = torch.nn.Parameter(torch.randn(count, width, hidden)/np.sqrt(width))
    b1 = torch.nn.Parameter(torch.zeros(count, 1, hidden))
    w2 = torch.nn.Parameter(torch.randn(count, hidden, 2)/np.sqrt(hidden))
    b2 = torch.nn.Parameter(torch.zeros(count, 1, 2))
    optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=.01)
    batch_id = torch.arange(count)[:, None]
    last_loss = None
    for _ in range(updates):
        indices = torch.randint(length, (count, 64))
        batch_x, batch_y, valid = x[batch_id, indices], y[batch_id, indices], observed[batch_id, indices]
        forecast = torch.bmm(torch.tanh(torch.bmm(batch_x, w1)+b1), w2)+b2
        loss = (((forecast-batch_y)**2)*valid).sum()/valid.sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    return [{name: parameter.detach().numpy()[i].reshape(shape) for name, parameter, shape in
             [('w1', w1, (width, hidden)), ('b1', b1, (hidden,)), ('w2', w2, (hidden, 2)), ('b2', b2, (2,))]}
            for i in range(count)], last_loss


def summarize(rows):
    summary = []
    for family in FAMILIES:
        for policy in ('coupled', 'intervened'):
            for model in ('temporal_mlp', 'instant_mlp', 'temporal_ridge'):
                selected = [r for r in rows if (r['world']['family'], r['policy'], r['model']) == (family, policy, model)]
                values = {}
                for key in ('prediction_mse', 'causal_mse_all', 'causal_mse_nonzero', 'spurious_effect_rms'):
                    available = [r['metrics'][key] for r in selected if r['metrics'][key] is not None]
                    values[key] = float(np.mean(available)) if available else None
                summary.append({'family': family, 'policy': policy, 'model': model, 'n': len(selected), **values})
    return summary


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'training', 'scope': 'learned finite-window effects; no consciousness or novelty established',
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in SOURCES},
              'versions': {'torch': torch.__version__, 'numpy': np.__version__}, 'rows': []}
    def save():
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save()
    configurations = [(world_parameters(seed, family), policy) for family in FAMILIES
                      for seed in range(300, 303) for policy in ('coupled', 'intervened')]
    training = [dataset(p, policy) for p, policy in configurations]
    mask = np.stack([data[2] for data in training])
    # Unobserved simulator targets are removed before reaching any optimizer.
    visible_y = np.where(mask, np.stack([data[1] for data in training]), 0.0)
    for name, kind, hidden in (('temporal_mlp', 'temporal', 32), ('instant_mlp', 'instant', 100)):
        x = np.stack([features(data[0], kind) for data in training])
        for initial_seed in (11, 23, 37):
            started = time.perf_counter()
            weights, loss = train_networks(x, visible_y, mask, seed=initial_seed, hidden=hidden)
            for i, ((parameters, policy), checkpoint) in enumerate(zip(configurations, weights)):
                filename = f'{name}-{initial_seed}-{i}.npz'
                np.savez_compressed(out/filename, **checkpoint)
                report['rows'].append({'model': name, 'features': kind, 'initial_seed': initial_seed,
                    'world': parameters, 'policy': policy, 'checkpoint': filename,
                    'checkpoint_sha256': hashlib.sha256((out/filename).read_bytes()).hexdigest(),
                    'metrics': metrics(checkpoint, parameters, policy, kind)})
            save()
            print(f'{name} seed={initial_seed}: {time.perf_counter()-started:.1f}s, last training loss={loss:.5f}', flush=True)
    for i, ((parameters, policy), (history, _, _)) in enumerate(zip(configurations, training)):
        checkpoint = fit_ridge(features(history), visible_y[i], mask[i])
        filename = f'temporal_ridge-{i}.npz'
        np.savez_compressed(out/filename, **checkpoint)
        report['rows'].append({'model': 'temporal_ridge', 'features': 'temporal', 'initial_seed': None,
            'world': parameters, 'policy': policy, 'checkpoint': filename,
            'checkpoint_sha256': hashlib.sha256((out/filename).read_bytes()).hexdigest(),
            'metrics': metrics(checkpoint, parameters, policy, 'temporal')})
    report.update(status='completed', summary=summarize(report['rows']))
    save()
    for row in report['summary']:
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
