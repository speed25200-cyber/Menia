"""Plot paired raw logit changes from an audited question-specificity report."""
import argparse
import json
from pathlib import Path


def plot(summary,output):
    import matplotlib.pyplot as plt
    import numpy as np
    r=json.loads(Path(summary).read_text(encoding='utf-8'))
    if not r['complete'] or r['origin']!='transformers_gpu':raise ValueError('A complete real-model report is required')
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),sharex=True,sharey=True)
    labels=['Présence : 1 = présente','Absence : 1 = absente','Lecture : repère en 1','Lecture : repère en 2']
    questions=['presence','absence','marker_first','marker_second']
    colors={'base':'#697986','shuffled':'#D39832','strong':'#216AA6'}
    for rep,ax in enumerate(axes):
        for i,arm in enumerate(colors):
            values=[];lower=[];upper=[]
            for q in questions:
                t=r['tables'][f'{rep}/{arm}/hidden/{q}'];v=t['rawShift'];lo,hi=t['rawShiftInterval95']
                values.append(v);lower.append(max(0,v-lo));upper.append(max(0,hi-v))
            y=np.arange(4)+(i-1)*.17
            ax.errorbar(values,y,xerr=[lower,upper],fmt='o',ms=5,capsize=3,
                        color=colors[arm],label={'base':'Base','shuffled':'Cibles mélangées','strong':'Entraîné'}[arm])
        ax.axvline(0,color='#888888',lw=.8,ls='--');ax.set_title(f'Répétition {rep+1}',fontsize=11)
        ax.grid(axis='x',alpha=.15);ax.set_xlabel('Variation de logit(1) − logit(0)')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(range(4),labels);axes[0].invert_yaxis()
    handles,names=axes[-1].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.60,.06))
    fig.suptitle('Menia : le signal de perturbation suit-il la question ?',fontsize=15,y=.97)
    fig.text(.60,.88,'Rotation − absence, mêmes phrases · 24 blocs par répétition · IC descriptifs à 95 %',ha='center',fontsize=9,color='#555555')
    fig.text(.60,.02,'Adaptateurs figés du Colab 10 ; mesure fonctionnelle, sans conclusion sur la conscience.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.18,right=.985,top=.80,bottom=.26,wspace=.13)
    fig.savefig(output,dpi=160,facecolor='white');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('output',type=Path);a=p.parse_args();plot(a.summary,a.output)
