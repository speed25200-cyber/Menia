"""Plot all repetitions: public wording, native/external decisions, and score ranking."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(summary, verification, destination):
    assert summary['complete'] and verification['verified']
    assert summary['planHash']==verification['planHash'] and summary['sourceHash']==verification['sourceHash']
    tables=summary['tables']; reps=sorted({int(k.split('/')[0]) for k in tables})
    fig,axes=plt.subplots(len(reps),3,figsize=(17,3.6*len(reps)+1.1),squeeze=False,
                         gridspec_kw={'width_ratios':[1.15,1.2,1.]})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    arms=('base','parent','composed','grammar')
    names={'base':'Base','parent':'Parent','composed':'Composé','grammar':'Consignes'}
    forms=('trained','paraphrase')
    def heat(ax,values,xlabels,ylabels):
        values=np.asarray(values);im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(len(xlabels)),xlabels,fontsize=9)
        ax.set_yticks(range(len(ylabels)),ylabels,fontsize=9)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v=values[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=9,color='black' if v>.7 else 'white')
        return im
    for row,rep in enumerate(reps):
        ax,decision,auc=axes[row]
        values=[[tables[f'{rep}/{arm}/visible/{form}/{task}/{mapping}']['nativeBalancedAccuracy']
                 for task in ('marker_first','marker_second') for form in forms]
                for arm in arms for mapping in (0,1)]
        heat(ax,values,['P1\ncanon.','P1\nreform.','P2\ncanon.','P2\nreform.'],
             [names[a]+(' · N' if m==0 else ' · I') for a in arms for m in (0,1)])
        ax.set_ylabel(f'Répétition {rep+1}',fontsize=12)
        labels=['Canon. N','Canon. I','Reform. N','Reform. I']
        streams=[tables[f'{rep}/composed/hidden/{form}/monitor/{mapping}'] for form in forms for mapping in (0,1)]
        for offset,(metric,bounds,label,color) in enumerate((
                ('nativeBalancedAccuracy','nativeBalancedAccuracyInterval95','Réponse native','#167a92'),
                ('zeroThresholdBalancedAccuracy','zeroThresholdInterval95','Seuil zéro','#aab2bd'),
                ('calibratedBalancedAccuracy','calibratedInterval95','Seuil externe fixé','#d57925'))):
            y=np.array([t[metric] for t in streams]);bounds=np.array([t[bounds] for t in streams])
            # Bootstrap percentile intervals can exclude the point estimate; draw endpoints directly.
            xs=np.arange(4)+(offset-1)*.24
            decision.bar(xs,y,width=.23,label=label,color=color,zorder=2)
            decision.vlines(xs,bounds[:,0],bounds[:,1],color='#222',linewidth=1,zorder=3)
            decision.hlines(bounds[:,0],xs-.035,xs+.035,color='#222',linewidth=1,zorder=3)
            decision.hlines(bounds[:,1],xs-.035,xs+.035,color='#222',linewidth=1,zorder=3)
        decision.set_ylim(0,1.07);decision.set_xticks(range(4),labels,fontsize=9)
        decision.axhline(.5,linestyle='--',color='#777',linewidth=.8)
        decision.set_ylabel('Exactitude équilibrée');decision.grid(axis='y',alpha=.2,zorder=0)
        values=[[tables[f'{rep}/{arm}/hidden/{form}/monitor/{mapping}']['presenceAUROC']
                 for form in forms for mapping in (0,1)] for arm in arms]
        heat(auc,values,['Canon.\nN','Canon.\nI','Reform.\nN','Reform.\nI'],[names[a] for a in arms])
        if row==0:
            ax.set_title('Lecture publique : exactitude native',fontsize=12,pad=12)
            decision.set_title('Présence cachée : décisions du composé',fontsize=12,pad=12)
            auc.set_title('Présence cachée : AUROC du score',fontsize=12,pad=12)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    title='Colab 13 — consignes, décisions et classement'
    if summary['origin']=='synthetic_fixture':title='DONNÉES SYNTHÉTIQUES — '+title
    fig.suptitle(title,fontsize=16,y=.995)
    fig.text(.5,.012,'N : code normal · I : code inversé · IC 95 % par blocs. Seuils externes appris uniquement sur l’ancien apprentissage.\n'
             'Diagnostic prospectif choisi après le Colab 12 ; aucune conclusion sur la conscience.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.97),h_pad=2.0,w_pad=2.2)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path);p.add_argument('verification',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();plot(json.loads(a.summary.read_text(encoding='utf-8')),
                          json.loads(a.verification.read_text(encoding='utf-8')),a.output)
