"""Plot audited component scores and externally recombined choices separately."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/action-decomposition-pilot'


def main():
    report=json.loads((ART/'summary.json').read_text(encoding='utf-8'))
    audit=json.loads((ART/'verification.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['recorded']==2448 and report['replayMatched']==288 and audit['verified']
    arms=('base','fixed','permuted');colors=('#667085','#2563EB','#00856A')
    fig,axes=plt.subplots(1,2,figsize=(13.2,6.6),sharey=True)
    for panel,ax in enumerate(axes):
        for a,(arm,color) in enumerate(zip(arms,colors)):
            for d,stage in enumerate(('number','semantic','mapping') if panel==0 else ('original','semantic','number')):
                values=[]
                for rep in range(3):
                    prefix=f'r{rep}/{arm}/'
                    if panel==0:
                        rows=[v for k,v in report['groups'].items() if k.startswith(prefix+stage+'/')]
                        n=sum(r['n'] for r in rows);correct=sum(r['correct'] for r in rows)
                        assert n==(48 if stage=='mapping' else 96)
                    else:
                        use='semantic' if stage=='original' else stage
                        rows=[v for k,v in report['composed'].items() if k.startswith(prefix+use+'/')]
                        n=sum(r['n'] for r in rows);correct=sum(r['originalCorrect' if stage=='original' else 'composedCorrect'] for r in rows)
                        assert n==384
                    values.append(100*correct/n)
                xs=[d+(a-1)*.19+(r-1)*.04 for r in range(3)]
                ax.scatter(xs,values,s=42,color=color,alpha=.82,zorder=3,edgecolors='white',linewidths=.6)
                mean=sum(values)/3
                ax.plot([d+(a-1)*.19-.055,d+(a-1)*.19+.055],[mean,mean],color=color,linewidth=2.5,zorder=4)
        ax.set_title(('Opérations séparées','Choix initiaux et recombinaisons externes')[panel],loc='left',fontsize=13,pad=15,fontweight='bold')
        ticks=('Minimum\nnumérique','Choix par\nnom','Traduction\nimposée') if panel==0 else ('Choix natif\nColab 20','Nom puis code\n(programme)','Nombre puis code\n(programme)')
        ax.set_xticks(range(3),ticks);ax.set_xlim(-.5,2.5);ax.set_ylim(-3,103)
        ax.set_yticks(range(0,101,20));ax.grid(axis='y',color='#E4E7EC');ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='both',labelsize=10)
    axes[0].set_ylabel('Réponses correctes (%)',fontsize=11)
    fig.suptitle('Menia — où la décision se dégrade-t-elle ?',x=.06,ha='left',y=.97,fontsize=17,fontweight='bold')
    handles=[Line2D([0],[0],marker='o',color=c,linestyle='none',label=l,markersize=7) for c,l in zip(colors,('Modèle de base','Présentation fixe','Permutations'))]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.905),ncol=3,frameon=False,fontsize=11)
    fig.text(.06,.09,'Points : trois répétitions ; trait : leur moyenne. À gauche : tâches distinctes, 96 / 96 / 48 appels par bras et répétition.',fontsize=9.7,color='#344054')
    fig.text(.06,.06,'À droite : 384 présentations des mêmes 16 cas ; traductions réutilisées. La recombinaison ajoute des opérations externes.',fontsize=9.7,color='#344054')
    fig.text(.06,.03,'Pertes fournies · poids figés · diagnostic après lecture du lot 20 · aucune mesure de conscience',fontsize=10,color='#344054')
    fig.subplots_adjust(left=.06,right=.985,bottom=.235,top=.77,wspace=.13)
    for suffix in ('png','pdf'):fig.savefig(ART/('components.'+suffix),dpi=180,facecolor='white')
    plt.close(fig)
    print(ART/'components.png')


if __name__=='__main__':main()
