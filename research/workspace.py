"""Small recurrent workspace candidate on two-hop symbolic lookup.

Architectural analogy only: no claim to instantiate every GWT indicator or
subjective consciousness. No language model, perception, or autonomous agent.
"""
from itertools import permutations
import torch
from torch import nn
from torch.nn import functional as F


PERMUTATIONS = torch.tensor(list(permutations(range(4))))
PAIR_IDS = torch.arange(24*24)


def encode_examples(pair_ids, queries):
    tables = torch.stack((PERMUTATIONS[pair_ids//24], PERMUTATIONS[pair_ids % 24]), 1)
    index = torch.arange(len(pair_ids))
    target = tables[index, 1, tables[index, 0, queries]]
    return F.one_hot(tables, 4).float().flatten(2), F.one_hot(queries, 4).float(), target


def lookup_batch(seed, batch=128):
    """Two private permutation tables; answer is B[A[query]].

    Table entries are legitimate observations; composed targets are never inputs.
    The tables are presented only at the first recurrent step.
    A table's majority class is uninformative because each output appears once.
    Hold out pair IDs divisible by five, for every query; never draw them here.
    """
    generator = torch.Generator().manual_seed(seed)
    training_pairs = PAIR_IDS[PAIR_IDS % 5 != 0]
    pairs = training_pairs[torch.randint(len(training_pairs), (batch,), generator=generator)]
    queries = torch.randint(4, (batch,), generator=generator)
    return encode_examples(pairs, queries)


def lookup_test():
    """Exhaustive 116 unseen table pairs x four queries, not 464 independent worlds."""
    held_out = PAIR_IDS[PAIR_IDS % 5 == 0]
    return encode_examples(held_out.repeat_interleave(4), torch.arange(4).repeat(len(held_out)))


class SharedWorkspace(nn.Module):
    def __init__(self, hidden=32, width=8):
        super().__init__()
        self.hidden, self.width = hidden, width
        # Each specialist sees only its own table and the previous broadcast.
        self.specialists = nn.ModuleList([nn.GRUCell(17+width, hidden) for _ in range(2)])
        self.keys = nn.ModuleList([nn.Linear(hidden, width) for _ in range(2)])
        self.values = nn.ModuleList([nn.Linear(hidden, width) for _ in range(2)])
        self.query = nn.Linear(width, width)
        self.initial = nn.Linear(4, width)
        self.update = nn.GRUCell(width, width)
        # Both readouts receive exactly the same workspace, no private-state skip.
        self.answer = nn.Linear(width, 4)
        self.intermediate = nn.Linear(width, 4)

    def forward(self, tables, query, *, rounds=4, intervention='none', return_trace=False):
        if tables.ndim != 3 or tables.shape[1:] != (2, 16) or query.shape != (len(tables), 4):
            raise ValueError('Expected [batch,2,16] tables and [batch,4] query')
        if type(rounds) is not int or not 1 <= rounds <= 32:
            raise ValueError('Invalid rounds')
        if intervention not in {'none', 'no_broadcast', 'uniform_attention', 'reset_specialists',
                                'reset_workspace', 'no_table_a', 'no_table_b'}:
            raise ValueError('Unknown intervention')
        batch = len(tables)
        state = [tables.new_zeros(batch, self.hidden) for _ in range(2)]
        workspace = torch.tanh(self.initial(query))
        trace = []
        for step in range(rounds):
            broadcast = torch.zeros_like(workspace) if intervention == 'no_broadcast' else workspace
            next_state = []
            for module in range(2):
                visible = step == 0 and intervention != ('no_table_a' if module == 0 else 'no_table_b')
                observation = tables[:, module] if visible else torch.zeros_like(tables[:, module])
                flag = tables.new_full((batch, 1), float(visible))
                previous = torch.zeros_like(state[module]) if intervention == 'reset_specialists' else state[module]
                next_state.append(self.specialists[module](torch.cat((observation, flag, broadcast), -1), previous))
            state = next_state
            keys = torch.stack([self.keys[i](state[i]) for i in range(2)], 1)
            values = torch.stack([self.values[i](state[i]) for i in range(2)], 1)
            scores = (keys*self.query(workspace)[:, None]).sum(-1)/(self.width**.5)
            attention = torch.softmax(scores, -1)
            if intervention == 'uniform_attention':
                attention = torch.full_like(attention, .5)
            selected = (values*attention[..., None]).sum(1)
            previous_workspace = torch.zeros_like(workspace) if intervention == 'reset_workspace' else workspace
            workspace = self.update(selected, previous_workspace)
            if return_trace:
                trace.append({'workspace': workspace, 'attention': attention,
                              'specialists': torch.stack(state, 1)})
        result = {'answer': self.answer(workspace), 'intermediate': self.intermediate(workspace)}
        return (result, trace) if return_trace else result


def intermediate_labels(tables, query):
    # Only an auxiliary TRAINING target, computed outside the model forward pass.
    return tables[:, 0].reshape(-1, 4, 4)[torch.arange(len(tables)), query.argmax(-1)].argmax(-1)


class DirectLookup(nn.Module):
    """Feedforward comparison: same observations, unrestricted joint access."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(36, 96), nn.Tanh(), nn.Linear(96, 96),
                                     nn.Tanh(), nn.Linear(96, 8))

    def forward(self, tables, query):
        output = self.network(torch.cat((tables.flatten(1), query), -1))
        return {'answer': output[:, :4], 'intermediate': output[:, 4:]}
