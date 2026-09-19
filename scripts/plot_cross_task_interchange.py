"""Fixed presentation of both cross-task directions and all 18 primary contrasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(summary, verification, destination):
    assert summary['schema'] == 'menia-cross-task-interchange-report-v1'
    assert summary['complete'] and verification['verified']
    assert all(summary[k] == verification[k] for k in ('planHash', 'sourceHash'))
    assert len(summary['tables']) == verification['tables'] == 144
    assert len(summary['baseline']) == verification['baselineTables'] == 48
    assert len(summary['contrasts']) == verification['contrasts'] == 432
    assert sum(c['primary'] for c in summary['contrasts']) == verification['primaryContrasts'] == 18
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(3, 3, figsize=(19, 13), gridspec_kw={'height_ratios': [1, 1, 1.1]})
    mechanisms = [('recipientTaskContent', 'Contenu pour la question destinataire', '#117a8b'),
                  ('donorBoolean', 'Valeur binaire de la question donneuse', '#bc8530'),
                  ('donorAnswer', 'Chiffre prévu du donneur', '#bc4258'),
                  ('recipientUnchanged', 'Destinataire inchangé', '#8065a7')]
    directions = [('monitor', 'marker_first', 'État caché → marqueur public', 'C→P'),
                  ('marker_first', 'monitor', 'Marqueur public → état caché', 'P→C')]
    rivals = {'donorBoolean': 'valeur donneuse', 'donorAnswer': 'chiffre donneur',
              'recipientUnchanged': 'destinataire'}
    for rep in range(3):
        for row, (dt, rt, label, _) in enumerate(directions):
            ax = axes[row, rep]
            for arm in ('prefix', 'base'):
                for offset, (metric, _, color) in enumerate(mechanisms):
                    x = [i + (offset - 1.5) * .09 + (.02 if arm == 'base' else -.02) for i in range(3)]
                    entries = [summary['tables'][f'{rep}/test/{arm}/{dt}-to-{rt}/{site}'][metric]
                               for site in (17, 23, 35)]
                    ax.plot(x, [e['mean'] for e in entries], color=color,
                            ls='-' if arm == 'prefix' else '--', marker='o' if arm == 'prefix' else 's',
                            ms=4, lw=1.3, alpha=1. if arm == 'prefix' else .5)
                    ax.vlines(x, [e['interval95'][0] for e in entries], [e['interval95'][1] for e in entries],
                              color=color, lw=1, alpha=1. if arm == 'prefix' else .4)
            ax.set(xticks=[0, 1, 2], xticklabels=['17 · chronologie', '23 · principal', '35 · copie'],
                   ylim=(-.035, 1.035), yticks=[0, .25, .5, .75, 1], yticklabels=['0', '25', '50', '75', '100'])
            ax.grid(axis='y', color='#ddd', lw=.6)
            numbers = {arm: '/'.join(f"{100*summary['baseline'][f'{rep}/test/{arm}/{rt}/{code}']['accuracy']:.1f}"
                                    for code in (0, 1)) for arm in ('prefix', 'base')}
            ax.set_title(f"Répétition {rep+1} · {label}\nDestinataire intact N/I (%) : préfixe {numbers['prefix']}\nBase N/I (%) : {numbers['base']}", fontsize=10)
            if rep == 0:
                ax.set_ylabel('Accord avec la prédiction (%)')

        ax = axes[2, rep]
        labels = []
        for dt, rt, _, short in directions:
            for rival in rivals:
                matches = [c for c in summary['contrasts'] if c['primary'] and c['replication'] == rep
                           and c['donorTask'] == dt and c['recipientTask'] == rt
                           and c['comparison'] == 'recipientTaskContent-' + rival]
                assert len(matches) == 1
                c = matches[0]
                i = len(labels)
                color = '#117a8b' if short == 'C→P' else '#435a92'
                ax.hlines(i, *c['interval95'], color=color, lw=2)
                ax.plot(c['difference'], i, 'o', color=color, ms=5)
                labels.append(short + ' : contenu − ' + rivals[rival])
        ax.axvline(0, color='#777', lw=.8)
        ax.axhline(2.5, color='#ddd', lw=.8)
        ax.set(xlim=(-1.03, 1.03), ylim=(-.6, 5.6), yticks=range(6), yticklabels=labels,
               xlabel='Différence d’accord appariée')
        ax.invert_yaxis()
        gate = 'satisfaits' if summary['prerequisites'][str(rep)]['allPassed'] else 'NON satisfaits'
        ax.set_title(f'Préfixe · couche 23 · six contrastes principaux\nPrérequis intacts et au sein des tâches : {gate}', fontsize=10)

    handles = [Line2D([0], [0], color=color, lw=2, label=label) for _, label, color in mechanisms]
    handles += [Line2D([0], [0], color='#444', marker='o', ls='-', label='Préfixe entraîné'),
                Line2D([0], [0], color='#888', marker='s', ls='--', label='Base sans adaptateur')]
    fig.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(.5, .956), frameon=False)
    title = 'Colab 16 — le contenu transféré reste-t-il utilisable quand la question change ?'
    if summary.get('origin') == 'synthetic_fixture':
        title = 'DONNÉES SYNTHÉTIQUES — test de présentation uniquement'
    fig.suptitle(title, fontsize=16, y=.993)
    fig.text(.5, .012, 'Jeu principal : 16 paires par répétition, 16 combinaisons par paire. N/I : codes normal/inversé. Formulation canonique uniquement.\n'
             'Intervalles individuels à 95 %, sans correction de multiplicité. Lexique réservé, transferts au sein des tâches, pertes et copie effective dans les tables complètes.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(.008, .055, .995, .9), h_pad=2.5, w_pad=1.8)
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
