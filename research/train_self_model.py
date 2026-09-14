"""Offline training and held-out exploratory evaluation; see SELF_MODEL_PILOT.md."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .recurrent import RecurrentMemory
from .self_model import SelfMonitor
from .evaluate_self_model import worlds, evaluate, FAMILIES


def fit(model, train, validation, field, seed, steps=1000):
    torch.manual_seed(seed)
    mean = train[field].mean(0)
    scale = np.maximum(train[field].std(0), .01)
    x = torch.tensor((train[field]-mean)/scale, dtype=torch.float32)
    y = torch.tensor(train['outcome'], dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1],32), torch.nn.ReLU(), torch.nn.Linear(32,1))
    optimizer = torch.optim.Adam(net.parameters(), lr=.003)
    rng = np.random.default_rng(seed)
    best, best_step, best_loss = None, None, float('inf')
    for step in range(steps):
        indices = rng.integers(len(x), size=256)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x[indices])[:,0], y[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0:
            monitor = SelfMonitor(model, mean, scale,
                net[0].weight.detach().numpy().T.copy(), net[0].bias.detach().numpy().copy(),
                net[2].weight.detach().numpy()[0].copy(), net[2].bias.detach().numpy()[0].copy())
            brier = float(np.mean((monitor.predict_features(validation[field])-validation['outcome'])**2))
            if brier < best_loss:
                best, best_step, best_loss = monitor, step+1, brier
    return best, {'selected_step':best_step, 'validation_brier':best_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/self-model')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'status':'executed exploratory pilot; not a consciousness test',
              'torch':torch.__version__, 'numpy':np.__version__,
              'training_episodes':12288, 'validation_episodes':2048,
              'evaluation_episodes_per_family':2048, 'steps':1000,
              'source_sha256':{}, 'runs':[]}
    for name in ('self_model.py','train_self_model.py','evaluate_self_model.py'):
        report['source_sha256']['research/'+name] = hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
    report['plan_sha256'] = hashlib.sha256(Path('docs/SELF_MODEL_PILOT.md').read_bytes()).hexdigest()
    for seed in (17,29,43):
        model = RecurrentMemory.load(f'artifacts/recurrent-memory/memory-seed-{seed}.json')
        seeds = {'train':seed*10000+1, 'validation':seed*10000+2, 'evaluation':seed*10000+3}
        train = worlds(model, seeds['train'], 12288)
        val = worlds(model, seeds['validation'], 2048)
        own, own_fit = fit(model,train,val,'own',seed)
        observer, observer_fit = fit(model,train,val,'observer',seed)
        record = {'seed':seed, 'data_seeds':seeds, 'own_fit':own_fit, 'observer_fit':observer_fit,
                  'model_sha256':own.model_sha256, 'families':{}, 'checkpoints':{}}
        for label, monitor in [('own',own),('observer',observer)]:
            name = f'{label}-{seed}.json'
            monitor.save(output/name, {'role':label,'scope':'synthetic recall', 'seed':seed})
            record['checkpoints'][label] = {'file':name,'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest()}
        for family in FAMILIES:
            data = worlds(model,seeds['evaluation'],2048,family)
            record['families'][family] = evaluate(own,observer,data)
        report['runs'].append(record)
        print(json.dumps({'seed':seed, 'families':record['families']}),flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
