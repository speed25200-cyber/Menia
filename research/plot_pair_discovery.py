"""Plot all reused episodes and keys in the single-pair V condition."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np


def render(journal, destination):
    events=[json.loads(line)['payload'] for line in Path(journal).read_text(encoding='utf-8').splitlines()]
    assert events[-1]['event']=='complete'
    states={e['case']['id']:e for e in events if e['event']=='state'}
    matrices=[np.full((24,8),np.nan) for _ in range(3)]; invalid=[[] for _ in range(3)]
    for e in events:
        if e['event'] not in ('task','forecast') or e['call']['condition']!='values_permuted': continue
        case=e['call']['case']; target=e['call']['target']; d=e['decoded']
        panel=0 if e['event']=='task' else 1+e['call']['mapping']
        if panel==0:
            expected=states[case]['case']['values'][target]
            value=float(d['validNativeResponse'] and d['decision']==('one' if expected else 'zero'))
        else: value=d['conditionalPositive']
        matrices[panel][case,target]=value
        if not d['validNativeResponse']: invalid[panel].append((target,case))
    assert all(int(np.isfinite(a).sum())==160 for a in matrices)
    cmap=LinearSegmentedColormap.from_list('pair_score',['#bd493d','#f4ecd9','#238575']); cmap.set_bad('#ededed')
    fig,axes=plt.subplots(1,3,figsize=(12.8,10.2),sharey=True)
    titles=['Rappel observé\n0 = faux ; 1 = correct',
            'Prévision : A = incorrect, B = correct\nScore conditionnel de réussite',
            'Prévision : B = incorrect, A = correct\nScore conditionnel de réussite']
    for panel,(ax,array,title) in enumerate(zip(axes,matrices,titles)):
        im=ax.imshow(np.ma.masked_invalid(array),cmap=cmap,vmin=0,vmax=1,aspect='auto',interpolation='none')
        ax.set_title(title,fontsize=10,pad=14)
        ax.set_xticks(range(8),['LUMA','NERI','VAKO','SUDI','PELA','TOVI','ZERA','MIKO'],rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(24),[f'{i:02d} · {states[i]["case"]["bindings"]} clés' for i in range(24)],fontsize=8)
        for case,state in states.items():
            for target in state['bindingMask']['selectedRows']:
                ax.add_patch(Rectangle((target-.48,case-.48),.96,.96,fill=False,edgecolor='#182b31',linewidth=1.15))
        for x,y in invalid[panel]: ax.text(x,y,'×',ha='center',va='center',color='#722f95',fontsize=14,fontweight='bold')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel('Épisode · taille du tableau',fontsize=10)
    fig.suptitle("Rappel et prévision après permutation d'une paire de valeurs",fontsize=16,y=.975,fontweight='bold')
    fig.subplots_adjust(left=.095,right=.94,bottom=.19,top=.88,wspace=.16)
    cax=fig.add_axes([.955,.27,.013,.49]); bar=fig.colorbar(im,cax=cax); bar.set_ticks([0,.5,1]); bar.ax.tick_params(labelsize=9)
    fig.text(.095,.065,'Contour sombre : clé dont la position du bit est permutée. Gris : clé absente. × : sortie au format invalide.\n'
        '24 épisodes déjà examinés, 18 tableaux distincts ; aucune confirmation indépendante. Prévisions enregistrées avant les rappels.\n'
        'Codage du verdict fourni « incorrect » : A = incorrect, 0/120 corrects ; B = incorrect, 120/120 corrects.\n'
        'Qwen3-4B, sans adaptation Menia. Le score conditionnel ne constitue pas une probabilité étalonnée de réussite.',fontsize=9,linespacing=1.55)
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg'):
        output=destination.with_suffix(suffix)
        if output.exists(): raise FileExistsError(output)
        fig.savefig(output,dpi=160,facecolor='white')
        if suffix=='.svg': output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); render(args.journal,args.output)
