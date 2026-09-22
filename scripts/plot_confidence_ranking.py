"""Plot all frozen Colab24 primary contrasts after the numerical audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['svg.hashsalt'] = 'menia-confidence-ranking-primary-v1'
root = Path(__file__).resolve().parents[1]
folder = root/'artifacts/confidence-ranking-pilot'
summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
verification = json.loads((folder/'verification.json').read_text(encoding='utf-8'))
assert verification['verified'] and verification['comparisonsChecked'] == 18
assert len(summary['contrasts']) == 18 and summary['complete']
fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2), sharey=True)
labels = [f'R{rep+1} · {producer}' for rep in range(3) for producer in ('base', 'rank')]
for ax, comparator, title in zip(axes, ('ce', 'neutral', 'constant'), ('Face à CE seul', 'Face au témoin neutralisé', 'Face au score constant 0,5')):
    rows = [c for c in summary['contrasts'] if c['comparator'] == comparator]
    for i, c in enumerate(rows):
        low, high = c['familyInterval']; gain = c['withinAucGain']
        color = '#117A65' if c['passed'] else '#56667A'
        ax.plot([low, high], [i, i], color=color, linewidth=2)
        ax.scatter(gain, i, color=color, s=42, zorder=3)
    ax.axvline(0, color='#333333', linewidth=1)
    ax.axvline(rows[0]['minimumGain'], color='#A06113', linestyle=':', linewidth=1.5)
    ax.set_title(title, fontsize=11, pad=14)
    ax.set_xlim(-.20, .35); ax.set_xticks([-.1, 0, .1, .2, .3])
    ax.set_xlabel("Gain d'AUROC dans les catégories", fontsize=9)
    ax.set_yticks(range(6), labels)
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].invert_yaxis()
axes[0].set_ylabel('Répétition · producteur des réponses', fontsize=9)
fig.suptitle('Colab 24 — 4 contrastes réussis sur 18 ; critère global non atteint', fontsize=14, y=.98)
fig.text(.08, .04, 'Traits : intervalles bootstrap avec correction de Bonferroni pour 18 contrastes. Vert : contraste réussi.\n'
         'Pointillés : gain minimal exigé pour le point estimé. Couverture insuffisante dans les répétitions 1 et 3.', fontsize=9)
fig.subplots_adjust(left=.11, right=.98, top=.84, bottom=.20, wspace=.13)
for extension in ('png', 'svg'):
    path = folder/f'primary-contrasts.{extension}'
    fig.savefig(path, dpi=170, facecolor='white', metadata={'Date': None} if extension == 'svg' else None)
    if extension == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',
                        encoding='utf-8', newline='\n')
plt.close(fig)
print(json.dumps({'figure': str(folder/'primary-contrasts.png'), 'contrasts': 18, 'passed': sum(c['passed'] for c in summary['contrasts'])}))
