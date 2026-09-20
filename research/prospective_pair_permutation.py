"""Select one opposite-bit pair using public rows, never observed outcomes."""
from research.prospective_binding_permutation import binding_swap
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def pair_swap(tokenizer, case, prompt_ids, prefix_length):
    located = binding_swap(tokenizer, case, prompt_ids, prefix_length)
    zeros = [i for i, row in enumerate(located['rows']) if row['bit'] == 0]
    ones = [i for i, row in enumerate(located['rows']) if row['bit'] == 1]
    selected = [zeros[case['id'] % len(zeros)], ones[(case['id']//len(zeros)) % len(ones)]]
    positions = [located['rows'][i]['position'] for i in selected]
    permutation = list(range(prefix_length))
    a, b = positions
    permutation[a], permutation[b] = b, a
    require(a != b and all(permutation[permutation[i]] == i for i in range(prefix_length)), 'One opposite-bit pair')
    return dict(schema='menia-pair-position-permutation-v1', case=case['id'],
        promptHash=located['promptHash'], prefixLength=prefix_length, rows=located['rows'],
        selectedRows=selected, selectedKeys=[located['rows'][i]['key'] for i in selected],
        permutation=permutation, permutationHash=digest(permutation),
        movedPositions=sorted(positions), unchangedPositions=prefix_length-2,
        scope='One public zero/one pair selected by fixed row-index arithmetic. No task outcome used.')
