import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import codemagic_fetch  # noqa: E402


def zipped(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


class FakeClient:
    def __init__(self, builds, blobs):
        self.builds, self.blobs = builds, blobs

    def get_json(self, path):
        if path == "/apps":
            return {"applications": [{"_id": "app1", "appName": "menia", "repository": {"htmlUrl": "https://github.com/x/Menia"}}]}
        if path.startswith("/builds?"):
            return {"builds": self.builds}
        return {"build": next(b for b in self.builds if b["_id"] == path.rsplit("/", 1)[-1])}

    def request(self, url, data=None):
        return self.blobs[url]

    def download(self, url):
        return self.blobs[url]


class FetchTests(unittest.TestCase):
    def test_finished_build_is_unzipped_and_running_build_is_left(self):
        archive = zipped({"llm-lora/base/rows-R.jsonl": "{}\n", "llm-lora/VM/rows-R.jsonl": "{}\n"})
        builds = [
            {"_id": "b1", "tag": "lora-2", "status": "finished", "startedAt": "2026-09-22T17:00:00Z",
             "artefacts": [{"name": "llm-lora.zip", "url": "u1", "size": len(archive), "md5": hashlib.md5(archive).hexdigest()}],
             "buildActions": [{"name": "Publier", "status": "success", "logUrl": "l1"}]},
            {"_id": "b2", "tag": "latent-1", "status": "building", "startedAt": "2026-09-22T17:01:00Z"},
        ]
        client = FakeClient(builds, {"u1": archive, "l1": b"push to https://api.codemagic.io/hooks/abc-123 ok\n"})
        request = {"app": "Menia", "builds": [{"tag": "lora-2", "dest": "out/lora"}, {"tag": "latent-1", "dest": "out/latent"},
                                              {"tag": "lora-9", "dest": "out/none"}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = codemagic_fetch.fetch(client, request, root)
            self.assertTrue((root / "out/lora/base/rows-R.jsonl").exists())
            self.assertTrue((root / "out/lora/VM/rows-R.jsonl").exists())
            self.assertFalse((root / "out/latent").exists())
            log = (root / "out/lora/codemagic-logs/00-Publier.log").read_text()
            self.assertNotIn("abc-123", log)
            self.assertEqual(json.loads((root / "out/lora/codemagic-build.json").read_text())["status"], "finished")
        self.assertEqual(report["latent-1"], "en cours (building)")
        self.assertEqual(report["lora-9"], "introuvable")

    def test_same_names_do_not_overwrite(self):
        builds = [{"_id": "b1", "tag": "t", "status": "failed",
                   "artefacts": [{"name": "rows-R.jsonl", "url": "a"}, {"name": "rows-R.jsonl", "url": "b"}]}]
        with tempfile.TemporaryDirectory() as tmp:
            codemagic_fetch.fetch(FakeClient(builds, {"a": b"1", "b": b"2"}), {"builds": [{"tag": "t", "dest": "d"}]}, Path(tmp))
            self.assertEqual((Path(tmp) / "d/rows-R.jsonl").read_text(), "1")
            self.assertEqual((Path(tmp) / "d/rows-R.jsonl.1").read_text(), "2")


if __name__ == "__main__":
    unittest.main()
