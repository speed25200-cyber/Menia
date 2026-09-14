import ast
import json
from pathlib import Path
for p in sorted(Path('notebooks').glob('*.ipynb')):
    n = json.loads(p.read_text())
    assert n['nbformat'] == 4
    assert len({c['id'] for c in n['cells']}) == len(n['cells'])
    for i, c in enumerate(n['cells']):
        if c['cell_type'] == 'code':
            ast.parse(''.join(c['source']), filename=f'{p}:cell-{i}')
            assert not c['outputs']
    print(f'{p}: structure and Python syntax OK')
