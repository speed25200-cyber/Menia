"""Plot all fixed Colab20 transfer domains after the full weight/result audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/action-binding-pilot'


def main():
    report=json.loads((ART/'summary.json').read_text(encoding='utf-8'))
    verified=json.loads((ART/'verification.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['recorded']==6912
    assert verified['verified'] and verified['weightsChecked'] and len(verified['weights'])==9
    arms=('base','fixed','permuted')
    labels=('Modèle de base','Présentation fixe','Permutations')
    colors=('#667085','#2563EB','#00856A')
    domains=('trainedSurface','newWording','newSymbols','bothNew')
    ticks=('w0 / chiffres','w1–w2 / chiffres','w0 / lettres','w1–w2 / lettres')
    fig,axes=plt.subplots(1,2,figsize=(13.2,6.6),sharey=True)
    fig.patch.set_facecolor('white')
    for ax,info,title in zip(axes,('probability','expectedLoss'),('Probabilité donnée','Pertes moyennes données')):
        for a,(arm,color) in enumerate(zip(arms,colors)):
            means=[]
            for d,domain in enumerate(domains):
                rows=sorted((c for c in report['contrasts'] if c['information']==info and c['domain']==domain),key=lambda c:c['replication'])
                assert len(rows)==3 and [r['replication'] for r in rows]==[0,1,2]
                values=[100*r['accuracy'][arm] for r in rows]
                xs=[d+(a-1)*.19+(rep-1)*.04 for rep in range(3)]
                ax.scatter(xs,values,s=42,color=color,alpha=.82,zorder=3,edgecolors='white',linewidths=.6)
                mean=sum(values)/3;means.append(mean)
                ax.plot([d+(a-1)*.19-.055,d+(a-1)*.19+.055],[mean,mean],color=color,linewidth=2.5,zorder=4)
        ax.set_title(title,loc='left',fontsize=13,pad=15,fontweight='bold')
        ax.set_xticks(range(4),ticks,rotation=20,ha='right')
        ax.set_xlim(-.5,3.5);ax.set_ylim(-3,103)
        ax.set_yticks(range(0,101,20));ax.grid(axis='y',color='#E4E7EC')
        ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='both',labelsize=10)
    axes[0].set_ylabel('Choix optimaux (%)',fontsize=11)
    fig.suptitle('Menia — transfert après apprentissage des codes d’action',x=.06,ha='left',y=.97,fontsize=17,fontweight='bold')
    handles=[Line2D([0],[0],marker='o',color=c,linestyle='none',label=l,markersize=7) for c,l in zip(colors,labels)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.905),ncol=3,frameon=False,fontsize=11)
    fig.text(.06,.085,'Chaque point : une répétition ; trait : moyenne des trois. Même modèle de base, trois initialisations d’adaptateurs.',fontsize=10,color='#344054')
    fig.text(.06,.055,'16 cas test distincts. Les moyennes sur les présentations ne remplacent pas le critère fixé pour chacun des 432 groupes.',fontsize=10,color='#344054')
    fig.text(.06,.025,'6 912 décisions · Qwen3-4B · contrôle fonctionnel public · aucune mesure de conscience',fontsize=10,color='#344054')
    fig.subplots_adjust(left=.06,right=.985,bottom=.255,top=.77,wspace=.13)
    for suffix in ('png','pdf'):
        fig.savefig(ART/('transfer-domains.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)
    print(str(ART/'transfer-domains.png'))


if __name__=='__main__':main()
