"""Show global versus within-category ranking and external recalibration."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/confidence-calibration-diagnostic'


def main():
    report=json.loads((ART/'report.json').read_text(encoding='utf-8'))
    checked=json.loads((ART/'verification.json').read_text(encoding='utf-8'))
    assert report['exploratory'] and checked['verified'] and checked['calibrators']==27
    assert report['sourceHash']==checked['diagnosticSourceHash']
    rows=[report['results'][f'r{rep}/measured']['judges'] for rep in range(3)]
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.6))
    for i,row in enumerate(rows):
        score=row['measured'];part=score['aucPartition']
        axes[0].plot([i,i],[part['within']['auc'],part['auc']],color='#98A2B3',linewidth=2,zorder=1)
        axes[0].scatter(i,part['auc'],color='#344054',s=70,zorder=3)
        axes[0].scatter(i,part['within']['auc'],color='#2563EB',s=70,zorder=3)
        axes[0].scatter(i+.07,row['shuffled']['aucPartition']['within']['auc'],color='#B54708',marker='^',s=65,zorder=3)
        axes[1].plot([i,i],[score['raw']['brier'],score['calibrated']['brier']],color='#98A2B3',linewidth=2,zorder=1)
        axes[1].scatter(i,score['raw']['brier'],color='#344054',s=70,zorder=3)
        axes[1].scatter(i,score['calibrated']['brier'],color='#2563EB',s=70,zorder=3)
        axes[1].scatter(i+.07,row['shuffled']['calibrated']['brier'],color='#B54708',marker='^',s=65,zorder=3)
        axes[1].scatter(i-.07,row['betaCell']['raw']['brier'],color='#00856A',marker='s',s=55,zorder=3)
    axes[0].axhline(.5,color='#667085',linestyle='--',linewidth=1)
    axes[0].set_ylim(.35,1.);axes[1].set_ylim(.075,.165)
    axes[0].set_ylabel('AUROC — plus haut est meilleur');axes[1].set_ylabel('Brier — plus bas est meilleur')
    axes[0].set_title('Global et même catégorie : deux classements',loc='left',fontsize=12,fontweight='bold')
    axes[1].set_title('Recalibrage : gain surtout en répétition 2',loc='left',fontsize=12,fontweight='bold')
    legends=[ [('Global entraîné','#344054','o'),('Même catégorie, entraîné','#2563EB','o'),('Même catégorie, mélangé','#B54708','^')],
              [('Entraîné, brut','#344054','o'),('Entraîné, recalibré','#2563EB','o'),('Mélangé, recalibré','#B54708','^'),('Beta','#00856A','s')]]
    for ax,items in zip(axes,legends):
        ax.set_xticks(range(3),('Répétition 0','Répétition 1','Répétition 2'));ax.set_xlim(-.4,2.4)
        ax.set_axisbelow(True);ax.grid(axis='y',color='#E4E7EC');ax.spines[['top','right']].set_visible(False)
        ax.legend(handles=[Line2D([0],[0],marker=m,color=c,linestyle='none',label=l) for l,c,m in items],
                  fontsize=9,frameon=False,loc='upper left',bbox_to_anchor=(0.,-.12))
    fig.suptitle('Menia — deux limites différentes de la confiance',x=.075,ha='left',y=.975,fontsize=17,fontweight='bold')
    fig.text(.075,.905,'Sur les réponses du bras entraîné · 192 questions de test par répétition · analyse exploratoire',fontsize=11,color='#344054')
    fig.text(.075,.035,'Recalibrage externe sur calibration seulement ; classement inchangé. Aucun nouveau test confirmatoire ni gain natif.',fontsize=10,color='#344054')
    fig.subplots_adjust(left=.075,right=.98,top=.81,bottom=.28,wspace=.26)
    for suffix in ('png','pdf'):fig.savefig(ART/('calibration-versus-ranking.'+suffix),dpi=180,facecolor='white')
    plt.close(fig);print(ART/'calibration-versus-ranking.png')


if __name__=='__main__':main()
