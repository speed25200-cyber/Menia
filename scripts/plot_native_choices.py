"""Fixed view of all public-risk groups, native conditions and12 primary contrasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def plot(report,verification,path):
    assert report['complete'] and verification['verified']
    assert all(report[k]==verification[k] for k in ('origin','planHash','sourceHash'))
    assert verification['nativeTables']==24 and verification['primaryContrasts']==12
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig=plt.figure(figsize=(18,12))
    gs=fig.add_gridspec(3,3,height_ratios=(.65,1.2,1.2))
    labels=['Form. 1\nnormal','Form. 1\ninversé','Form. 2\nnormal','Form. 2\ninversé']
    configs=[(w,m) for w in range(2) for m in range(2)]
    ax=fig.add_subplot(gs[0,:])
    values=[report['publicRisk'][f'w{w}/m{m}']['accuracy'] for w,m in configs]
    ax.bar(range(4),values,width=.45,color=['#117a8b' if v>=.9 else '#946342' for v in values])
    ax.axhline(.9,color='#555',ls=':',label='Seuil fixé : 90 % dans chaque groupe')
    for i,((w,m),v) in enumerate(zip(configs,values)):
        r=report['publicRisk'][f'w{w}/m{m}']
        ax.text(i,v+.025,f"{r['optimalChoices']}/{r['n']}",ha='center')
    ax.set(xticks=range(4),xticklabels=labels,ylim=(0,1.13),ylabel='Choix optimal',title='Risque public : probabilité connue, 45 décisions par formulation/code · seuil fixé : 90 %')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y',alpha=.15)
    comparisons=report['primaryContrasts']
    low=min(0,min(c['familyInterval'][0] for c in comparisons))
    high=max(.02,max(c['familyInterval'][1] for c in comparisons))
    pad=max(.03,(high-low)*.1)
    for column,cost in enumerate((.2,.5,.8)):
        ax=fig.add_subplot(gs[1,column])
        x=np.arange(4)
        for mode,offset,label,color in [('unassisted',-.18,'Sans mesure fournie','#40566f'),('history',.18,'Historique fourni','#117a8b')]:
            values=[report['native'][f'{mode}/{cost:.1f}/w{w}/m{m}']['meanActualLoss'] for w,m in configs]
            ax.bar(x+offset,values,width=.34,label=label,color=color)
            for position,value in zip(x+offset,values):ax.text(position,value+.015,f'{value:.3f}',ha='center',fontsize=9,rotation=90)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,1.18),ylabel='Perte moyenne en points',title=f'Coût de vérification : {cost:.1f}\n96 questions par condition · actions exécutées')
        ax.grid(axis='y',alpha=.15)
        if column==0:fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.958),ncol=2,frameon=False,fontsize=10)
        ax=fig.add_subplot(gs[2,column])
        for i,(w,m) in enumerate(configs):
            c=next(c for c in comparisons if (c['cost'],c['wording'],c['mapping'])==(cost,w,m))
            color='#117a8b' if c['passed'] else '#946342'
            ax.hlines(i,*c['familyInterval'],color=color,lw=1.5)
            ax.hlines(i,*c['interval95'],color=color,lw=5)
            ax.plot(c['unassistedMinusHistoryLoss'],i,'o',color=color,ms=7)
        ax.axvline(0,color='#666',lw=1)
        ax.axvline(.02,color='#117a8b',ls=':',lw=1)
        ax.set(yticks=range(4),yticklabels=[s.replace('\n',' · ') for s in labels],ylim=(-.6,3.6),
               xlim=(low-pad,high+pad),xlabel='Gain de l’historique · positif = perte réduite',title='Comparaisons appariées par question')
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.15)
    title='Colab 18 — les mesures antérieures améliorent-elles les décisions natives ?'
    if report['origin']=='synthetic_fixture':title='DONNÉES SYNTHÉTIQUES — vérification de la présentation'
    fig.suptitle(title,fontsize=16,y=.985)
    fig.text(.5,.035,'En bas : traits épais = intervalles individuels à 95 % ; traits fins = intervalles nominaux à 99,5833… % pour les douze comparaisons.\n'
        'Seuil de gain fixé : 0,02 point. Bootstrap de 96 questions stratifié par catégorie ; questions partagées entre conditions, pas 2 304 unités indépendantes.\n'
        'Les points sont une utilité attribuée, distincte des durées mesurées. L’historique peut modifier le choix et la réponse. Ce graphique ne mesure pas la conscience.',
        ha='center',fontsize=10)
    fig.tight_layout(rect=(.008,.11,.992,.955),h_pad=3,w_pad=2.5)
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path);parser.add_argument('verification',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    plot(json.loads(args.summary.read_text(encoding='utf-8')),json.loads(args.verification.read_text(encoding='utf-8')),args.output)
