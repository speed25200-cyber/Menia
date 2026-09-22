"""Plot all preregistered contrasts only after complete local verification."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/native-answer-confidence-pilot'


def main():
    s=json.loads((ART/'summary.json').read_text(encoding='utf-8'))
    v=json.loads((ART/'verification.json').read_text(encoding='utf-8'))
    assert s['complete'] and s['recordedCalls']==10368 and len(s['contrasts'])==24
    assert v['verified'] and v['weightsChecked'] and v['comparisonsChecked']==24
    names={'base-measured':'Base','shuffled-measured':'Labels mélangés','betaCell-measured':'Fréquences Beta','outputConfidence-measured':'Confiance de sortie'}
    fig,axes=plt.subplots(1,2,figsize=(13.5,7.5),sharex=True,sharey=True)
    for ax,producer,title in zip(axes,('base','measured'),('Réponses du modèle de base','Réponses du modèle entraîné')):
        rows=[c for c in s['contrasts'] if c['producer']==producer]
        assert len(rows)==12
        for i,c in enumerate(rows):
            color='#00856A' if c['passed'] else '#B54708'
            ax.hlines(i,*c['familyInterval'],color=color,linewidth=2)
            ax.scatter(c['brierGain'],i,s=40,color=color,zorder=3)
        ax.set_yticks(range(12),[f"R{c['replication']} / {names[c['comparison']]}" for c in rows])
        ax.set_title(title,fontsize=13,fontweight='bold',loc='left',pad=14)
        ax.axvline(0,color='#667085',linewidth=1)
        ax.axvline(.005,color='#2563EB',linestyle='--',linewidth=1)
        ax.grid(axis='x',color='#E4E7EC');ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Brier du comparateur − Brier de Menia entraînée',fontsize=10)
        for y in (3.5,7.5):ax.axhline(y,color='#E4E7EC',linewidth=.8)
    axes[0].invert_yaxis();axes[1].tick_params(axis='y',labelleft=False)
    fig.suptitle('Menia — les 24 comparaisons de confiance après réponse',x=.19,ha='left',y=.97,fontsize=16,fontweight='bold')
    fig.text(.19,.915,'Gain positif : jugement entraîné meilleur. Pointillés bleus : gain minimal fixé à 0,005.',fontsize=10,color='#344054')
    fig.text(.19,.095,'Traits : intervalles bootstrap appariés, ajustement Bonferroni pour 24 comparaisons (approximation).',fontsize=10,color='#344054')
    fig.text(.19,.063,'Vert : gain et borne requis atteints ; orange : contraste non validé. Les autres critères restent séparés.',fontsize=10,color='#344054')
    fig.text(.19,.031,'192 questions de test par répétition · 3 répétitions · réponses identiques pour les cinq juges/comparateurs',fontsize=10,color='#344054')
    fig.subplots_adjust(left=.19,right=.985,bottom=.18,top=.84,wspace=.14)
    for suffix in ('png','pdf'):fig.savefig(ART/('primary-contrasts.'+suffix),dpi=180,facecolor='white')
    plt.close(fig);print(ART/'primary-contrasts.png')


if __name__=='__main__':main()
