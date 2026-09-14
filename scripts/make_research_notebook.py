import json
from pathlib import Path

def cell(kind,text):
    value={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
    if kind=='code': value.update({'execution_count':None,'outputs':[]})
    return value
cells=[
cell('markdown','''# Menia — mémoire récurrente et limites du rappel
Ce notebook entraîne un **petit module neuronal de mémoire**, pas une conscience
et pas le modèle de langage Qwen. L’expérience tourne sur CPU ; l’A100 n’est pas
nécessaire pour ce module de 1 540 paramètres.

Les poids d’un premier entraînement sont déjà dans le dépôt. Les cellules ci-dessous
permettent de les vérifier et d’en entraîner de nouveaux, puis de calibrer une
limite de rappel et de mesurer les erreurs sur des épisodes séparés.
'''),
cell('code','''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys,os,datetime,json
REPO=Path('/content/Menia-recurrent')
if not REPO.exists():
    subprocess.run(['git','clone','https://github.com/speed25200-cyber/Menia.git',str(REPO)],check=True)
os.chdir(REPO)
print(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
subprocess.run([sys.executable,'-m','pip','install','-r','requirements-research.txt'],check=True)
RUN_ROOT=Path('/content/drive/MyDrive/Menia/recurrent')
'''),
cell('code','''subprocess.run([sys.executable,'-m','unittest','discover','-s','tests_research','-v'],check=True)
subprocess.run([sys.executable,'scripts/check_recurrent_artifacts.py'],check=True)
subprocess.run([sys.executable,'scripts/check_reliability_artifacts.py'],check=True)
'''),
cell('markdown','''## Nouvel entraînement avec trois initialisations
300 updates par modèle, séquences de 12 pas. Test séparé : 256 épisodes de 32 pas.
La référence déterministe mémorise le dernier symbole ; elle permet de situer les
résultats sans attribuer de capacité spéciale au réseau.
'''),
cell('code','''OUT=RUN_ROOT/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
subprocess.run([sys.executable,'-m','research.train_recurrent','--out',str(OUT),'--steps','300','--seeds','17','29','43'],check=True)
report=json.loads((OUT/'report.json').read_text())
for run in report['runs']:
    print(run['seed'],json.dumps(run['evaluation'],indent=2))
'''),
cell('markdown','''## Mesurer les limites avant de faire confiance au rappel
La probabilité du symbole peut rester élevée alors que le souvenir est faux.
Une calibration séparée choisit un délai de rappel maximal pour chaque modèle.
Les tests utilisent de nouvelles histoires, jusqu’à 512 pas sans observation.
Le rapport expose aussi l’effacement et le mélange des états internes : un état
incorrect mais plausible peut tromper ce mécanisme.

Ce contrôle empirique est une première fonction de suivi des limites de mémoire,
pas une introspection générale ni la validation d’une théorie de la conscience.
Les bornes statistiques concernent chaque groupe de calibration séparément ;
elles ne garantissent pas une fiabilité universelle.
'''),
cell('code','''RELIABILITY=OUT/'reliability'
subprocess.run([sys.executable,'-m','research.evaluate_reliability','--models',str(OUT),'--out',str(RELIABILITY)],check=True)
reliability_report=json.loads((RELIABILITY/'report.json').read_text())
for run in reliability_report['runs']:
    print(run['checkpoint'], 'délai calibré :',run['calibrated_max_age'])
    for row in run['test']:
        if row['age'] in (1,8,32,128,512):
            print('délai',row['age'],'précision brute',round(row['raw_accuracy'],3),
                  'réponses acceptées',row['calibrated_policy']['answered'],
                  'précision acceptée',row['calibrated_policy']['accuracy_when_answered'])
'''),
cell('markdown','''## Session interactive symbolique
Fournir un symbole, puis ne plus l’observer. L’évaluation est calculée après la
prédiction. L’appareil ne perçoit pas le monde réel dans cette expérience.
'''),
cell('code','''from research.recurrent import RecurrentMemory
from research.session import CognitiveSession
from research.reliability import RecallPolicy
model=RecurrentMemory.load(OUT/'memory-seed-17.json')
policy=RecallPolicy.load(RELIABILITY/'memory-seed-17-policy.json',model)
session=CognitiveSession(model,policy=policy)
print('Sans observation :',session.observe())
print(session.observe(2))
print(session.observe(None))
print(session.assess(2))
for _ in range(max(0,policy.max_age)+1):
    event=session.observe(None)
print('Au-delà du délai calibré :',event)
print(session.context())
session.stop()
session.clear()
print('Session arrêtée et état effacé.')
'''),
cell('markdown','''## Interprétation
Comparer réseau complet, état effacé à chaque pas, observations masquées et règle
du dernier symbole. Réussir ce rappel démontre un mécanisme sur cette tâche ;
cela ne démontre ni conscience, ni ressenti, ni intelligence générale.
Une abstention a une couverture nulle et une précision indéfinie, jamais 100 %.
Le symbole vu est une observation ; le candidat brut du réseau reste un diagnostic.
Seul answer_symbol est la réponse de session, avec sa source et son éventuelle abstention.

Les poids JSON et le rapport sont sauvegardés dans OUT. Aucun upload automatique.
Le runner Swift est fourni dans MeniaKit ; il reste à compiler et à tester sur iPhone.
Le notebook 01 est un autre parcours : adaptation LoRA du modèle de langage sur A100.
''')]
for i,c in enumerate(cells): c['id']=f'recurrent-{i:02d}'
n={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
Path('notebooks/02_recurrent_research.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
