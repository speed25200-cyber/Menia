"""Plot state/question composition and its public/visible controls from audited results."""
import argparse
import json
from pathlib import Path


def plot(summary,verification,output,allow_fixture=False):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    r=json.loads(Path(summary).read_text(encoding='utf-8'))
    audit=json.loads(Path(verification).read_text(encoding='utf-8'))
    if not r['complete'] or not audit['complete']:raise ValueError('Incomplete report')
    for key in ('origin','planHash','sourceHash','shamPairs','signalRuleMet','controlRuleMet','fixedReadingRuleMet'):
        if r[key]!=audit[key]:raise ValueError('Different audit: '+key)
    if r['recorded']!=audit['evaluations']:raise ValueError('Different evaluation count')
    fixture=r['origin']=='synthetic_fixture'
    if r['origin']!='transformers_gpu' and not (fixture and allow_fixture):raise ValueError('A real-model report is required')
    forms=('trained','paraphrase','new_digits');labels=('Consigne apprise','Reformulation réservée','Chiffres 2/3 réservés')
    colors=('#1766A3','#B33B75');fig,axes=plt.subplots(2,3,figsize=(13.5,7.4),sharex=True,sharey=True)
    counts=set()
    for rep in range(3):
        upper,lower=axes[:,rep]
        for y,form in enumerate(forms):
            for mapping in (0,1):
                t=r['tables'][f'{rep}/test/composed/hidden/{form}/monitor/{mapping}'];counts.add(t['blocks'])
                pos=y+(mapping-.5)*.20;value=t['presenceAUROC'];lo,hi=t['presenceAUROCInterval95']
                # Draw the interval independently: bootstrap intervals need not contain the point estimate.
                upper.plot([lo,hi],[pos,pos],color=colors[mapping],lw=1.5)
                upper.plot([lo,lo],[pos-.035,pos+.035],color=colors[mapping],lw=1)
                upper.plot([hi,hi],[pos-.035,pos+.035],color=colors[mapping],lw=1)
                upper.plot(value,pos,'o',color=colors[mapping],ms=5)
                controls=[r['tables'][f'{rep}/test/composed/{family}/{form}/{task}/{mapping}']['balancedAccuracy']
                          for family,task in [('hidden','marker_first'),('hidden','marker_second'),('visible','monitor'),('visible','marker_first'),('visible','marker_second')]]
                lower.plot(min(controls),pos,'o',color=colors[mapping],ms=5)
        signal=all(c['interval95'][0]>0 for c in r['contrasts'] if c['primary'] and c['replication']==rep)
        gates=[g for g in r['gates'] if g['table'].startswith(str(rep)+'/')]
        passed=all(g['accuracy'] and g['options'] for g in gates)
        upper.set_title(f"Répétition {rep+1} · critère signal : {'oui' if signal else 'non'}",fontsize=10)
        lower.set_title(f"Contrôles principaux : {'réussis' if passed else 'échoués'}",fontsize=10)
        for ax in (upper,lower):
            ax.set_xlim(-.025,1.035);ax.set_xticks([0,.25,.5,.75,1]);ax.grid(axis='x',alpha=.16)
            ax.axhspan(1.55,2.45,color='#eeeeee',zorder=-2)
            ax.spines[['top','right']].set_visible(False)
        upper.axvline(.5,color='#888888',ls='--',lw=.8)
        lower.axvline(.9,color='#888888',ls='--',lw=.8)
        upper.set_xlabel('AUROC de présence · IC descriptif à 95 %',fontsize=9)
        lower.set_xlabel('Minimum d’exactitude équilibrée\nsur les 5 tâches de contrôle',fontsize=9)
    axes[0,0].set_yticks(range(3),labels);axes[0,0].set_ylim(2.5,-.5)
    title='Menia : composer état interne et consigne'
    if fixture:title='EXEMPLE SYNTHÉTIQUE — '+title
    fig.suptitle(title,fontsize=15,y=.985)
    legend=[Line2D([0],[0],marker='o',color=colors[i],lw=0,label=('Code normal','Code inversé')[i]) for i in (0,1)]
    fig.legend(handles=legend,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.59,.105))
    note=(f"Bras composé · blocs par répétition : {', '.join(map(str,sorted(counts)))} · zone grise : transfert secondaire 2/3.\n"
          'Le critère signal inclut les témoins mélangé et consignes, absents de ce graphique : voir le rapport complet.\n'
          'Les contrôles incluent aussi le respect des codes de réponse. Ces scores ne mesurent pas la conscience.')
    fig.text(.59,.025,note,ha='center',fontsize=9,color='#444444',linespacing=1.6)
    fig.subplots_adjust(left=.19,right=.985,top=.88,bottom=.24,hspace=.60,wspace=.20)
    fig.savefig(output,dpi=160,facecolor='white');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('verification',type=Path);p.add_argument('output',type=Path);p.add_argument('--allow-fixture',action='store_true');a=p.parse_args()
    plot(a.summary,a.verification,a.output,a.allow_fixture)
