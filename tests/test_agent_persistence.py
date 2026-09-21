import json
from pathlib import Path
import tempfile
import unittest
from menia.session_store import open_session, save_session, load_session
from menia.core import Capabilities


def close(runtime):
    runtime.memory.close()
    runtime.agent.memory.close()


class SessionPersistenceTests(unittest.TestCase):
    def test_restart_with_pending_prediction_and_stopped_state(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, env = open_session(root, seed=23)
            for _ in range(3): runtime.step(env)
            self.assertIsNotNone(runtime.agent.pending)
            runtime.stop()
            expected = runtime.agent.state()
            save_session(root, runtime, env)
            close(runtime)
            restored, world = load_session(root)
            try:
                self.assertEqual(restored.agent.state(), expected)
                self.assertEqual(world.position, env.position)
                self.assertEqual(world.mapping, env.mapping)
                with self.assertRaises(RuntimeError): restored.step(world)
                restored.resume(user_requested=True)
                restored.step(world)
                self.assertEqual(len(restored.agent.model.errors), 1)
                self.assertIsNone(restored.agent.pending)
            finally:
                close(restored)

    def test_uncommitted_changes_do_not_mix_new_history_with_old_state(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, env = open_session(root, seed=53)
            for _ in range(4): runtime.step(env)
            save_session(root, runtime, env)
            expected_state = runtime.agent.state()
            expected_events = runtime.agent.memory.recent(runtime.agent.episode)
            for _ in range(4): runtime.step(env)
            close(runtime)  # interrupted before the next checkpoint
            restored, world = load_session(root)
            try:
                self.assertEqual(restored.agent.state(), expected_state)
                self.assertEqual(restored.agent.memory.recent(restored.agent.episode), expected_events)
                self.assertEqual(world.tick, expected_state['tick'])
                save_session(root, restored, world)
            finally:
                close(restored)

    def test_snapshot_keeps_simulator_randomness_and_rejects_changed_journal(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, env = open_session(root)
            env.dropout = .5
            save_session(root, runtime, env)
            expected = [env.read('position') for _ in range(8)]
            close(runtime)
            restored, world = load_session(root)
            try:
                self.assertEqual([world.read('position') for _ in range(8)], expected)
            finally:
                close(restored)
            pointer = json.loads((Path(root)/'session.json').read_text(encoding='utf-8'))
            with (Path(root)/pointer['journal']).open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaises(ValueError): load_session(root)

    def test_non_session_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root)/'keep.txt'
            file.write_text('user data')
            with self.assertRaises(ValueError): open_session(root)
            self.assertEqual(file.read_text(), 'user data')

    def test_capability_settings_survive_restart(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, env = open_session(root)
            runtime.capabilities = Capabilities(memory=False)
            save_session(root, runtime, env)
            close(runtime)
            restored, world = load_session(root)
            try:
                self.assertFalse(restored.capabilities.memory)
                with self.assertRaises(RuntimeError): restored.step(world)
            finally:
                close(restored)


if __name__ == '__main__':
    unittest.main()
