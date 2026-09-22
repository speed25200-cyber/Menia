"""Provenance-preserving episode journal and limited working memory.

Observation provenance is a receipt from the application, not an inferred claim
that a reported event is true. Predictions never become observations in place.
"""
from collections import OrderedDict
import json
import sqlite3


class EpisodeMemory:
    KINDS = {"observation", "report", "prediction", "action", "decision",
             "assessment", "attention", "retrieval"}

    def __init__(self, path=":memory:", *, capacity=2):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Working memory capacity must be positive")
        self.capacity = capacity
        self.working = OrderedDict()
        self.receipts = {}
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, episode TEXT NOT NULL, tick INTEGER NOT NULL,
            kind TEXT NOT NULL, source TEXT NOT NULL, payload TEXT NOT NULL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS episode_events ON events(episode, id)")

    def append(self, episode, tick, kind, source, payload):
        if kind not in self.KINDS or not isinstance(source, str) or not source:
            raise ValueError("Invalid provenance")
        if type(tick) is not int or tick < 0 or not isinstance(episode, str) or not episode:
            raise ValueError("Invalid episode or tick")
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self.db:
            cursor = self.db.execute(
                "INSERT INTO events(episode,tick,kind,source,payload) VALUES(?,?,?,?,?)",
                (episode, tick, kind, source, encoded))
        return cursor.lastrowid

    def event(self, event_id):
        row = self.db.execute("SELECT id,episode,tick,kind,source,payload FROM events WHERE id=?",
                              (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return dict(zip(("id", "episode", "tick", "kind", "source", "payload"),
                        (*row[:-1], json.loads(row[-1]))))

    def recent(self, episode, limit=12):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Invalid event limit")
        ids = self.db.execute("SELECT id FROM events WHERE episode=? ORDER BY id DESC LIMIT ?",
                              (episode, limit)).fetchall()
        return [self.event(row[0]) for row in reversed(ids)]

    def observe(self, episode, tick, field, value, *, source):
        event_id = self.append(episode, tick, "observation", source,
                               {"field": field, "value": value})
        self.receipts[(episode, field)] = event_id
        self._cache(episode, field, self.event(event_id))
        return event_id

    def _cache(self, episode, field, event):
        key = (episode, field)
        self.working.pop(key, None)
        self.working[key] = event
        while len(self.working) > self.capacity:
            self.working.popitem(last=False)

    def latest(self, episode, field):
        """An explicit archive read; only application observations qualify."""
        for row in self.db.execute(
                "SELECT id,payload FROM events WHERE episode=? AND kind='observation' ORDER BY id DESC",
                (episode,)):
            if json.loads(row[1]).get("field") == field:
                return self.event(row[0])
        return None

    def recall(self, episode, tick, field):
        event = self.latest(episode, field)
        self.append(episode, tick, "retrieval", "episodic_memory",
                    {"field": field, "event_id": event["id"] if event else None})
        if event:
            self._cache(episode, field, event)
            self.receipts[(episode, field)] = event["id"]
        return event

    def knowledge(self, episode, field, tick):
        event = self.latest(episode, field)
        return {"observed_before": event is not None,
                "in_working_memory": (episode, field) in self.working,
                "archived_event_id": event["id"] if event else None,
                "age": tick-event["tick"] if event else None}

    def wipe(self):
        with self.db:
            self.db.execute("DELETE FROM events")
        self.working.clear()
        self.receipts.clear()

    def close(self):
        self.db.close()
