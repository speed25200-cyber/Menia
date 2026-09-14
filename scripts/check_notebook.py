import ast
import json
from pathlib import Path
p = Path('notebooks/01_colab_a100.ipynb')
n = json.loads(p.read_text())
assert n['nbformat'] == 4
assert len({c['id'] for c in n['cells']}) == len(n['cells'])
for i, c in enumerate(n['cells']):
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']), filename=f'{p}:cell-{i}')
        assert not c['outputs']
print('Notebook structure and Python syntax OK')
