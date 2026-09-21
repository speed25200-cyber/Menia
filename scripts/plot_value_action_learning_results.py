"""Plot the complete audited Colab22 domains, without selecting presentations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/value-action-learning-pilot'


def main():
    report=json.loads((ART/'summary.json').read_text(encoding='utf-8'))
    verified=json.loads((ART/'verification.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['recorded']==9216
    assert verified['verified'] and verified['weightsChecked'] and len(verified['weights'])==12
    arms=('base','choice','linked','shuffled')
    labels=('Base','Choix seuls','Associations correctes','Associations mélangées')
    colors=('#667085','#2563EB','#00856A','#B54708')
    domains=('trainedSurface','newWording','newSymbols','bothNew')
    ticks=('w1 / chiffres','w2–w3 / chiffres','w1 / lettres','w2–w3 / lettres')
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.7),gridspec_kw=dict(width_ratios=[2,1]),sharey=True)
    for arm_index,(arm,color) in enumerate(zip(arms,colors)):
        for domain_index,domain in enumerate(domains):
            rows=sorted((c for c in report['contrasts'] if c['domain']==domain),key=lambda c:c['replication'])
            assert [r['replication'] for r in rows]==[0,1,2]
            values=[100*r['accuracy'][arm] for r in rows]
            center=domain_index+(arm_index-1.5)*.19
            axes[0].scatter([center+(r-1)*.025 for r in range(3)],values,s=35,color=color,edgecolors='white',linewidths=.5,zorder=3)
            axes[0].plot([center-.05,center+.05],[sum(values)/3]*2,color=color,linewidth=2.5,zorder=4)
        values=[]
        for rep in range(3):
            rows=[v for k,v in report['groups'].items() if k.startswith(f'r{rep}/{arm}/lookup/')]
            assert len(rows)==8 and sum(v['n'] for v in rows)==192
            values.append(100*sum(v['correct'] for v in rows)/192)
        axes[1].scatter([arm_index+(r-1)*.06 for r in range(3)],values,s=42,color=color,edgecolors='white',linewidths=.5,zorder=3)
        axes[1].plot([arm_index-.10,arm_index+.10],[sum(values)/3]*2,color=color,linewidth=2.5,zorder=4)
    axes[0].set_title('Choisir l’action de perte minimale',loc='left',fontsize=13,fontweight='bold',pad=15)
    axes[1].set_title('Retrouver une association imposée',loc='left',fontsize=12,fontweight='bold',pad=15)
    axes[0].set_xticks(range(4),ticks,rotation=20,ha='right')
    axes[1].set_xticks(range(4),('Base','Choix','Correctes','Mélangées'),rotation=25,ha='right')
    for ax in axes:
        ax.set_ylim(-3,103);ax.set_yticks(range(0,101,20));ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E4E7EC');ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='both',labelsize=10)
    axes[0].set_xlim(-.5,3.5);axes[1].set_xlim(-.5,3.5)
    axes[0].set_ylabel('Réponses correctes (%)',fontsize=11)
    fig.suptitle('Menia — apprentissage des associations valeur-action',x=.065,ha='left',y=.975,fontsize=17,fontweight='bold')
    handles=[Line2D([0],[0],marker='o',color=c,linestyle='none',label=l,markersize=7) for c,l in zip(colors,labels)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.06,.915),ncol=4,frameon=False,fontsize=10)
    fig.text(.065,.095,'Points : trois initialisations ; traits : moyennes. 24 cas distincts, présentés plusieurs fois. Aucun intervalle de confiance.',fontsize=10,color='#344054')
    fig.text(.065,.065,'À droite : formulation w1 et chiffres seulement. Les moyennes ne remplacent pas les seuils fixés pour les 384 groupes.',fontsize=10,color='#344054')
    fig.text(.065,.035,'9 216 appels · Qwen3-4B · apprentissage supervisé de choix publics · aucune mesure de conscience',fontsize=10,color='#344054')
    fig.subplots_adjust(left=.065,right=.985,bottom=.27,top=.77,wspace=.14)
    for suffix in ('png','pdf'):fig.savefig(ART/('transfer-domains.'+suffix),dpi=180,facecolor='white')
    plt.close(fig);print(str(ART/'transfer-domains.png'))


if __name__=='__main__':main()
