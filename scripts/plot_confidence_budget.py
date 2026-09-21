"""Plot the three fixed descriptive checkpoints after local verification."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(summary_path,verification_path,prefix):
    summary=json.loads(Path(summary_path).read_text(encoding='utf-8'))
    verification=json.loads(Path(verification_path).read_text(encoding='utf-8'))
    assert verification['verified'] and summary['complete'] and summary['recordedCalls']==verification['recordedCalls']==8640
    assert all(summary[k]==verification[k] for k in ('planHash','sourceHash','chainEnd'))
    epochs=(2,4,8); units=summary['units']
    assert len(units)==18
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row')
    phases=(('train','Exemples appris','#116a7b','-'),('seen24base','Lot 24 déjà examiné','#c96a16','--'))
    upper=max(v['brier'] for v in units.values())*1.18
    for rep in range(3):
        for phase,label,color,style in phases:
            values=[units[f'r{rep}-e{epoch}/{phase}'] for epoch in epochs]
            assert len({v['n'] for v in values})==len({v['correct'] for v in values})==len({v['withinPairs'] for v in values})==1
            for row,metric in enumerate(('withinAuc','brier')):
                axes[row,rep].plot(epochs,[v[metric] for v in values],marker='o',color=color,linestyle=style,label=label,linewidth=2)
        axes[0,rep].set_title(f'Répétition {rep+1}')
        axes[0,rep].set_ylim(0,1)
        axes[0,rep].axhline(.5,color='#999999',linestyle=':',linewidth=1)
        axes[1,rep].set_ylim(0,max(.05,upper)); axes[1,rep].set_xlabel("Passages totaux d'apprentissage")
        for row in range(2):
            axes[row,rep].set_xticks(epochs); axes[row,rep].grid(axis='y',alpha=.2)
            axes[row,rep].spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('AUROC au sein des catégories ↑')
    axes[1,0].set_ylabel('Erreur de Brier ↓')
    fig.suptitle("Budget d'optimisation à capacité fixe — diagnostic exploratoire",fontsize=16)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False)
    fig.text(.5,.025,"Mêmes réponses à chaque point ; redémarrage initial d'AdamW. Aucun nouveau test réservé ni mesure de conscience.",
        ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.12,1,.94)); prefix=Path(prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg'): fig.savefig(prefix.with_suffix(suffix),dpi=160,facecolor='white')
    svg = prefix.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',
        encoding='utf-8',newline='\n')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary'); parser.add_argument('verification'); parser.add_argument('prefix')
    args=parser.parse_args(); plot(args.summary,args.verification,args.prefix)
