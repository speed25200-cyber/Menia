"""Plot the fixed primary comparison, with descriptive per-adapter intervals."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('summary', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--synthetic', action='store_true')
args = parser.parse_args()
report = json.loads(args.summary.read_text(encoding='utf-8'))
assert report['complete'], 'Only plot a complete report.'
assert report['origin'] == ('synthetic_fixture' if args.synthetic else 'transformers_gpu'), 'Do not mislabel synthetic data.'
fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.4), sharey=True)
palette = {'base':'#626b75', 'strong':'#1767b0', 'shuffled':'#ce7922'}
labels = ['Base', 'Entraîné', 'Cibles mélangées']
for rep, ax in enumerate(axes):
    ax.axhline(.5, color='#8a919a', linestyle='--', linewidth=1)
    for i, arm in enumerate(('base','strong','shuffled')):
        family = 'strong' if arm=='base' else arm
        table = report['tables'][f'{rep}/test/{arm}/{family}/trained/canonical/primary']
        value = table['presenceAUROC']
        lo, hi = table['presenceAUROCInterval95']
        assert 0 <= lo <= hi <= 1 and 0 <= value <= 1
        # Draw endpoints directly; percentile intervals need not contain the point estimate.
        ax.vlines(i, lo, hi, color=palette[arm], linewidth=2)
        ax.hlines([lo, hi], i-.06, i+.06, color=palette[arm], linewidth=1.5)
        ax.plot(i, value, 'o', color=palette[arm], markersize=7)
        ax.annotate(f'{value:.3f}', (i,value), xytext=(0,10), textcoords='offset points', ha='center', fontsize=9)
    ax.set_title(f'Répétition {rep+1}', fontsize=11)
    ax.set_xticks(range(3), labels, fontsize=9)
    ax.set_xlim(-.5,2.5)
    ax.set_ylim(-.02,1.12)
    ax.set_yticks([0,.25,.5,.75,1])
    ax.grid(axis='y', alpha=.15)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('AUROC de présence')
prefix = 'DONNÉES SYNTHÉTIQUES — ' if args.synthetic else ''
fig.suptitle(prefix+'Détection de rotations internes : comparaison principale', fontsize=13, y=.97)
fig.text(.5,.035,'24 blocs de test par répétition · Intervalles descriptifs à 95 % par rééchantillonnage des blocs\n'
         'Condition entraînée, présentation canonique · Le trait pointillé indique une AUROC de 0,5',
         ha='center', fontsize=9, color='#4a535b')
fig.tight_layout(rect=(0,.13,1,.9))
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output, dpi=160, facecolor='white')
print(args.output.resolve())
