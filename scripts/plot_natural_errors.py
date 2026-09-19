"""Predefined view of all8 readers and all9 primary prospective contrasts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS = [('betaCell','Fréquence Beta'), ('inputOnly','Entrée seule'),
          ('outputConfidence','Confiance de sortie'), ('inputConfidence','Entrée + confiance'),
          ('finalControl','Avec état final'), ('internal','Avec état intermédiaire'),
          ('shuffledMiddle','Intermédiaire mélangé'), ('donorMiddle','Intermédiaire d’un donneur')]
RIVALS = [('finalControl','État final'),('shuffledMiddle','État mélangé'),('donorMiddle','État donneur')]


def plot(report, verification, output):
    assert report['complete'] and verification['verified']
    assert all(report[k]==verification[k] for k in ('planHash','sourceHash'))
    assert verification['primaryContrasts']==9 and len(report['replications'])==3
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(17,10),gridspec_kw={'height_ratios':[1.5,1]})
    maximum=max(r['scores'][name]['brier'] for r in report['replications'].values() for name,_ in LABELS)
    primary=[c for c in report['contrasts'] if c['primary']]
    lower=min(c['primaryFamilyInterval'][0] for c in primary)
    upper=max(c['primaryFamilyInterval'][1] for c in primary)
    x0,x1=min(0,lower),max(.005,upper)
    pad=max(.005,(x1-x0)*.12)
    for rep in range(3):
        scores=report['replications'][str(rep)]['scores']
        ax=axes[0,rep]
        colors=['#b7bdc5' if name not in ('finalControl','internal') else '#40566f' if name=='finalControl' else '#117a8b' for name,_ in LABELS]
        values=[scores[name]['brier'] for name,_ in LABELS]
        ax.barh(range(len(LABELS)),values,color=colors,height=.68)
        for i,value in enumerate(values):ax.text(value+max(.0004,maximum*.012),i,f'{value:.4f}',va='center',fontsize=9)
        ax.set(yticks=range(len(LABELS)),yticklabels=[label for _,label in LABELS],xlim=(0,max(.03,maximum*1.28)),xlabel='Brier moyen · plus petit = meilleure prévision')
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.18)
        n,correct=scores['internal']['n'],scores['internal']['correct']
        ax.set_title(f'Répétition {rep+1}\n{n} tests : {correct} réussites / {n-correct} erreurs')
        ax=axes[1,rep]
        for i,(rival,_) in enumerate(RIVALS):
            c=next(c for c in primary if c['replication']==rep and c['comparison']==rival+'-internal')
            color='#117a8b' if c['primaryPassed'] else '#946342'
            ax.hlines(i,*c['primaryFamilyInterval'],color=color,lw=1.5)
            ax.hlines(i,*c['interval95'],color=color,lw=5)
            ax.plot(c['otherMinusInternalBrier'],i,'o',color=color,ms=7)
        ax.axvline(0,color='#666',lw=1)
        ax.axvline(.005,color='#117a8b',ls=':',lw=1)
        ax.set(yticks=range(3),yticklabels=[label+' − interne' for _,label in RIVALS],
               ylim=(-.55,2.55),xlim=(x0-pad,x1+pad),xlabel='Différence de Brier · positif = gain interne')
        ax.invert_yaxis()
        g=report['prerequisites'][str(rep)]
        ax.set_title('Neuf contrastes fixés, trois par répétition\nMinimum de classes : '+('satisfait' if g['enoughBothClasses'] else 'NON satisfait'))
        ax.grid(axis='x',alpha=.18)
    title='Colab 17 — les états intermédiaires améliorent-ils la prévision des erreurs ?'
    if report['origin']=='synthetic_fixture':title='DONNÉES SYNTHÉTIQUES — vérification de la présentation'
    fig.suptitle(title,fontsize=16,y=.984)
    fig.text(.5,.055,'En bas : traits épais = intervalles individuels à 95 % ; traits fins = intervalles nominaux à 99,444… % pour la famille de neuf contrastes.\n'
             'Ligne pointillée : gain minimal fixé de 0,005. Brun : contraste ne passant pas les deux conditions. Bootstrap conditionnel, pas garantie exacte.\n'
             'Lecteurs externes et mêmes poids Qwen dans les trois répétitions. Résultats par cellule, calibration et comparaisons secondaires dans les agrégats.',ha='center',fontsize=10)
    fig.tight_layout(rect=(.008,.13,.992,.945),h_pad=3,w_pad=2.2)
    fig.savefig(output,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path);parser.add_argument('verification',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    plot(json.loads(args.summary.read_text(encoding='utf-8')),json.loads(args.verification.read_text(encoding='utf-8')),args.output)
