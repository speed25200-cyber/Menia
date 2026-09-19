"""Fixed presentation of every repetition, task control and primary contrast."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(summary, verification, destination):
    assert summary['complete'] and verification['verified']
    assert all(summary[k] == verification[k] for k in ('planHash', 'sourceHash'))
    assert len(summary['tables']) == verification['tables'] == 72
    assert len(summary['baseline']) == verification['baselineTables'] == 48
    assert len(summary['contrasts']) == verification['contrasts'] == 144
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(3, 3, figsize=(17, 11), gridspec_kw={'height_ratios': [1, 1, .7]})
    mechanisms = [('stateTransfer', 'Transfert d’état', '#117a8b'),
                  ('donorAnswer', 'Chiffre prévu du donneur', '#ba6b30'),
                  ('recipientUnchanged', 'Destinataire inchangé', '#8065a7')]
    for rep in range(3):
        for row, task in enumerate(('monitor', 'marker_first')):
            ax = axes[row, rep]
            for arm in ('prefix', 'base'):
                for offset, (metric, _, color) in enumerate(mechanisms):
                    x = [i + (offset - 1) * .085 + (.018 if arm == 'base' else -.018) for i in range(3)]
                    entries = [summary['tables'][f'{rep}/test/{arm}/{task}/{site}'][metric] for site in (17, 23, 35)]
                    values = [e['mean'] for e in entries]
                    ax.plot(x, values, color=color, ls='-' if arm == 'prefix' else '--',
                            marker='o' if arm == 'prefix' else 's', ms=4, lw=1.4,
                            alpha=1. if arm == 'prefix' else .55)
                    ax.vlines(x, [e['interval95'][0] for e in entries], [e['interval95'][1] for e in entries],
                              color=color, lw=1, alpha=1. if arm == 'prefix' else .4)
            ax.set(xticks=[0, 1, 2], xticklabels=['17 · chronologie', '23 · principal', '35 · copie'],
                   ylim=(-.035, 1.035), yticks=[0, .25, .5, .75, 1], yticklabels=['0', '25', '50', '75', '100'])
            ax.grid(axis='y', color='#ddd', lw=.6)
            numbers = {}
            for arm in ('prefix', 'base'):
                numbers[arm] = '/'.join(f"{100*summary['baseline'][f'{rep}/test/{arm}/{task}/{code}']['accuracy']:.1f}"
                                       for code in (0, 1))
            task_name = 'État caché' if task == 'monitor' else 'Marqueur public'
            ax.set_title(f"Répétition {rep+1} · {task_name}\nIntact N/I (%) : préfixe {numbers['prefix']} ; base {numbers['base']}", fontsize=10)
            if rep == 0:
                ax.set_ylabel('Accord avec la prédiction (%)')

        ax = axes[2, rep]
        tests = [c for c in summary['contrasts'] if c['primary'] and c['replication'] == rep]
        assert len(tests) == 2
        labels = []
        for i, c in enumerate(tests):
            ax.hlines(i, *c['interval95'], color='#117a8b', lw=2)
            ax.plot(c['difference'], i, 'o', color='#117a8b', ms=5)
            labels.append('État − ' + ('chiffre donneur' if c['comparison'].endswith('donorAnswer') else 'destinataire'))
        ax.axvline(0, color='#777', lw=.8)
        ax.set(xlim=(-1.03, 1.03), ylim=(-.7, 1.7), yticks=[0, 1], yticklabels=labels,
               xlabel='Différence d’accord appariée')
        ax.invert_yaxis()
        gate = 'satisfaits' if summary['prerequisites'][str(rep)]['allPassed'] else 'NON satisfaits'
        ax.set_title(f'Préfixe · état caché · couche 23\nPrérequis canoniques : {gate}', fontsize=10)

    handles = [Line2D([0], [0], color=color, lw=2, label=label) for _, label, color in mechanisms]
    handles += [Line2D([0], [0], color='#444', marker='o', ls='-', label='Préfixe entraîné'),
                Line2D([0], [0], color='#888', marker='s', ls='--', label='Base sans adaptateur')]
    fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(.5, .953), frameon=False)
    title = 'Colab 15 — quel contenu l’échange du vecteur transmet-il ?'
    if summary.get('origin') == 'synthetic_fixture':
        title = 'DONNÉES SYNTHÉTIQUES — test de présentation uniquement'
    fig.suptitle(title, fontsize=16, y=.992)
    fig.text(.5, .015, 'Jeu principal : 16 paires par répétition, 16 combinaisons par paire. N/I : codes normal/inversé. Formulation canonique uniquement.\n'
             'Intervalles individuels à 95 %, sans correction de multiplicité. Lexique réservé, pertes et copie effective dans les tables complètes.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(.01, .06, .995, .91), h_pad=2.1, w_pad=1.8)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('verification', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text(encoding='utf-8')),
         json.loads(args.verification.read_text(encoding='utf-8')), args.output)
