"""Display every repetition and primary optimizer contrast, without selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, verification, destination):
    assert summary['complete'] and verification['verified']
    assert all(summary[k] == verification[k] for k in ('planHash', 'sourceHash'))
    assert len(summary['tables']) == verification['tables'] == 408
    assert len(summary['contrasts']) == verification['contrasts'] == 306
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), gridspec_kw={'width_ratios': [1.05, 1.1, .95, 1.2]})
    arms = ('prefix', 'carry', 'reset_m', 'zero_grad')
    names = ('Arrêt', 'Suite originale', 'Moment effacé', 'Gradients nuls')
    colors = ('#18768a', '#dd8837', '#8075b9', '#679642')
    tables = summary['tables']
    def table(rep, arm, split='test', form='trained', code=0):
        return tables[f'{rep}/{split}/{arm}/hidden/{form}/monitor/{code}']
    def heat(ax, values, labels, vmax=1, numbers='.2f'):
        values = np.array(values)
        ax.imshow(values, vmin=0, vmax=vmax, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(8), [name + (' · N' if code==0 else ' · I') for name in names for code in (0,1)], fontsize=8)
        for (i,j), value in np.ndenumerate(values):
            ax.text(j, i, format(value,numbers), ha='center', va='center', fontsize=8,
                    color='black' if value > .7*vmax else 'white')
    max_loss = max(table(r,a,code=c)['balancedCrossEntropy'] for r in range(3) for a in arms for c in (0,1))
    for rep, (acc, contrast, loss, transfer) in enumerate(axes):
        for offset, (arm, name, color) in enumerate(zip(arms, names, colors)):
            ts = [table(rep, arm, code=c) for c in (0,1)]
            x = np.arange(2)+(offset-1.5)*.19
            acc.bar(x, [t['nativeBalancedAccuracy'] for t in ts], width=.18, color=color, label=name)
            intervals=np.array([t['nativeInterval95'] for t in ts])
            acc.vlines(x, intervals[:,0], intervals[:,1], color='#222', lw=1)
        acc.set(ylim=(0,1.05), xticks=[0,1], xticklabels=['Normal', 'Inversé'])
        acc.axhline(.5,color='#999',lw=.6,ls='--')
        acc.set_ylabel(f'Répétition {rep+1}\nExactitude équilibrée')
        contrasts = [x for x in summary['contrasts'] if x['primary'] and x['replication']==rep]
        assert len(contrasts)==6
        labels=[]
        for i, c in enumerate(contrasts):
            contrast.hlines(i,*c['nativeInterval95'], color='#18768a',lw=2)
            contrast.plot(c['nativeDifference'],i,'o',color='#18768a',ms=4)
            labels.append({'carry-prefix':'Suite − arrêt','zero_grad-prefix':'Nuls − arrêt',
                           'reset_m-carry':'Effacé − suite'}[c['comparison']]+(' · N' if c['mapping']==0 else ' · I'))
        contrast.set(yticks=range(6),yticklabels=labels,xlim=(-1.03,1.03),xlabel='Différence appariée')
        contrast.invert_yaxis();contrast.axvline(0,color='#999',lw=.7)
        heat(loss,[[table(rep,arm,code=c)['balancedCrossEntropy']] for arm in arms for c in (0,1)],
             ['Perte du bon token'],vmax=max(max_loss,1))
        heat(transfer,[[table(rep,arm,split,form,c)['nativeBalancedAccuracy']
                       for split,form in (('test','paraphrase'),('lexical','trained'),('lexical','paraphrase'))]
                      for arm in arms for c in (0,1)],['Reform.\ntest','Canon.\nlexique','Reform.\nlexique'])
    for ax, title in zip(axes[0],('Décision native · canonique','18 contrastes primaires','Perte · canonique','Transfert de la décision')):
        ax.set_title(title,pad=12,fontsize=11)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.96),frameon=False)
    fig.suptitle('Colab 14 — détection cachée après les quatre suites d’entraînement',fontsize=16,y=.992)
    fig.text(.5,.017,'N/I : codes normal/inversé. Intervalles individuels à 95 %, sans correction de multiplicité.\n'
             'Lexique réservé à cet apprentissage ; formulations déjà étudiées. Tâches publiques et autres contrastes dans les tables complètes.',
             ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.92),w_pad=2.5,h_pad=2.2)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path);p.add_argument('verification',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    plot(json.loads(args.summary.read_text(encoding='utf-8')),
         json.loads(args.verification.read_text(encoding='utf-8')),args.output)
