"""Crash-consistent snapshots of one virtual agent and its environment.

Two journal slots alternate; an atomic JSON pointer commits the matching agent,
environment and journal. Simulator internals are restored only to the simulator,
never supplied as observations or training labels to the agent.
"""
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
from .agent import SituatedAgent
from .core import Capabilities, Runtime
from .environments import VirtualRoom
from .episodic import EpisodeMemory


SLOTS = ('history-checkpoint-0.sqlite', 'history-checkpoint-1.sqlite')


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


def save_session(directory, runtime, environment):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    pointer = root/'session.json'
    previous = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else None
    generation = previous['generation']+1 if previous else 0
    slot = SLOTS[generation % 2]
    destination = sqlite3.connect(str(root/slot))
    try:
        runtime.agent.memory.db.backup(destination)
    finally:
        destination.close()
    world = {'mapping': environment.mapping, 'position': environment.position,
        'target': environment.target, 'dropout': environment.dropout, 'slip': environment.slip,
        'tick': environment.tick, 'rng': environment.rng.getstate(),
        'reads': environment.reads, 'moves': environment.moves}
    snapshot = {'version': 1, 'generation': generation, 'journal': slot,
        'journal_sha256': hashlib.sha256((root/slot).read_bytes()).hexdigest(),
        'working_memory_capacity': runtime.agent.memory.capacity,
        'capabilities': asdict(runtime.capabilities),
        'agent': runtime.agent.state(), 'environment': world}
    temporary = root/'session.pending.json'
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(snapshot, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, pointer)


def load_session(directory):
    root = Path(directory)
    snapshot = json.loads((root/'session.json').read_text(encoding='utf-8'))
    if snapshot['version'] != 1 or snapshot['journal'] not in SLOTS:
        raise ValueError('Unknown session format or journal name')
    path = root/snapshot['journal']
    if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot['journal_sha256']:
        raise ValueError('Session journal is incomplete or has changed')
    world = snapshot['environment']
    environment = VirtualRoom(position=world['position'], target=world['target'],
                              dropout=world['dropout'], slip=world['slip'])
    if set(world['mapping']) != set(environment.actions) or any(tuple(e) not in environment.effects for e in world['mapping'].values()):
        raise ValueError('Invalid simulator controls')
    environment.mapping = {a: tuple(e) for a, e in world['mapping'].items()}
    environment.tick = world['tick']
    environment.rng.setstate(_tuple_tree(world['rng']))
    environment.reads = [tuple(r) for r in world['reads']]
    environment.moves = list(world['moves'])
    memory = EpisodeMemory(root/'history.sqlite', capacity=snapshot['working_memory_capacity'])
    source = sqlite3.connect(str(path))
    try:
        source.backup(memory.db)
        agent = SituatedAgent.from_state(snapshot['agent'], memory=memory)
        if type(environment.tick) is not int or environment.tick < agent.tick:
            raise ValueError('Environment and agent clocks do not match')
        return Runtime(agent=agent, capabilities=Capabilities(**snapshot['capabilities'])), environment
    except Exception:
        memory.close()
        raise
    finally:
        source.close()


def open_session(directory, *, seed=17, target=(4, 3)):
    root = Path(directory)
    if (root/'session.json').exists():
        return load_session(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose a new empty session directory or an existing valid session')
    root.mkdir(parents=True, exist_ok=True)
    agent = SituatedAgent(memory=EpisodeMemory(root/'history.sqlite'))
    runtime = Runtime(agent=agent)
    environment = VirtualRoom(seed, target=target)
    save_session(root, runtime, environment)
    return runtime, environment
