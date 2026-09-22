"""Fetch the artefacts of finished Codemagic builds into the repository.

Run by .github/workflows/codemagic-fetch.yml, whose runner can reach
api.codemagic.io when the research session cannot. The request file names the
tags to fetch and where to put each build's artefacts:

    {"request": 1, "app": "Menia", "wait_minutes": 90,
     "builds": [{"tag": "latent-1", "dest": "artifacts/llm-latent-mac/run-1"},
                {"launch": {"workflow": "menia-lora-mac", "branch": "<branche>", "variables": {}},
                 "dest": "artifacts/llm-lora-mac/run-2"}]}

An entry with "launch" starts a new build of that codemagic.yaml workflow
through the API; the relay then waits up to wait_minutes for every build to
end. Nothing is written for a build that is not finished. Each fetched build leaves
codemagic-build.json (status, steps, artefact list with md5) and the tail of
every step's log, with webhook paths and credentials scrubbed. Standard library
only; the token comes from the CODEMAGIC_API_TOKEN environment variable.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.codemagic.io"
FINISHED = {"finished"}
ENDED = {"finished", "failed", "canceled", "timeout", "skipped"}
MAX_BYTES = 20 * 1024 * 1024
LOG_TAIL = 200
SECRETS = [re.compile(r"hooks/[A-Za-z0-9_-]+"), re.compile(r"(x-access-token|oauth2|token)[:=][^@\s]+@?", re.I),
           re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), re.compile(r"github_pat_[A-Za-z0-9_]{20,}")]


def scrub(text):
    for pattern in SECRETS:
        text = pattern.sub("[masqué]", text)
    return text


class Client:
    def __init__(self, token, base=API):
        self.token, self.base = token, base

    def request(self, url, data=None):
        if url.startswith("/"):
            url = self.base + url
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(url, data=body, headers={"x-auth-token": self.token, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.read()

    def get_json(self, path):
        return json.loads(self.request(path))

    def download(self, url):
        try:
            return self.request(url)
        except Exception:
            secure = url.split("/artifacts/", 1)[-1]
            public = json.loads(self.request(f"/artifacts/{secure}/public-url", {"expiresAt": 4102444800}))
            with urllib.request.urlopen(public["url"], timeout=300) as response:
                return response.read()


def find_app(client, name):
    apps = client.get_json("/apps").get("applications", [])
    wanted = name.lower()
    for app in apps:
        repo = json.dumps(app.get("repository", {})).lower()
        if wanted in (app.get("appName") or "").lower() or f"/{wanted}" in repo:
            return app
    raise SystemExit(f"application {name!r} absente parmi {[a.get('appName') for a in apps]}")


def builds_of(client, app_id):
    return client.get_json(f"/builds?appId={app_id}").get("builds", [])


def latest_for_tag(builds, tag):
    matching = [b for b in builds if b.get("tag") == tag]
    return max(matching, key=lambda b: b.get("startedAt") or b.get("createdAt") or "") if matching else None


def describe(build):
    return {"id": build.get("_id"), "tag": build.get("tag"), "branch": build.get("branch"),
            "workflow": build.get("fileWorkflowId") or build.get("workflowId"), "status": build.get("status"),
            "createdAt": build.get("createdAt"), "startedAt": build.get("startedAt"), "finishedAt": build.get("finishedAt"),
            "commit": (build.get("commit") or {}).get("hash"), "message": build.get("message"),
            "steps": [{"name": a.get("name"), "status": a.get("status"), "startedAt": a.get("startedAt"),
                       "finishedAt": a.get("finishedAt")} for a in build.get("buildActions") or []],
            "artefacts": [{"name": a.get("name"), "size": a.get("size"), "md5": a.get("md5"), "type": a.get("type")}
                          for a in build.get("artefacts") or []]}


def free_path(dest, name):
    path = dest / name
    k = 1
    while path.exists():
        path = dest / f"{name}.{k}"
        k += 1
    return path


def save_artefact(client, artefact, dest):
    """Unzip a zip artefact into dest, save any other file under its name; returns the saved paths."""
    name = Path(artefact.get("name") or "artefact").name
    if (artefact.get("size") or 0) > 20 * MAX_BYTES:
        return [f"ignoré (taille {artefact.get('size')}): {name}"]
    data = client.download(artefact["url"])
    if artefact.get("md5") and hashlib.md5(data).hexdigest() != artefact["md5"]:
        raise SystemExit(f"md5 différent pour {name}")
    saved = []
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                parts = [p for p in Path(info.filename).parts if p not in ("..", "/")]
                if parts and parts[0] in ("llm-latent", "llm-lora", "llm-atelier"):
                    parts = parts[1:]
                if not parts or info.file_size > MAX_BYTES:
                    saved.append(f"ignoré: {info.filename}")
                    continue
                target = dest.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
                saved.append(str(target))
    elif len(data) <= MAX_BYTES:
        target = free_path(dest, name)
        target.write_bytes(data)
        saved.append(str(target))
    else:
        saved.append(f"ignoré (taille {len(data)}): {name}")
    return saved


def save_logs(client, build, dest):
    logs = dest / "codemagic-logs"
    for n, action in enumerate(build.get("buildActions") or []):
        url = action.get("logUrl")
        if not url:
            continue
        try:
            text = client.request(url).decode("utf-8", "replace")
        except Exception as error:
            text = f"journal indisponible: {type(error).__name__}"
        logs.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", action.get("name") or "step").strip("-")[:60]
        (logs / f"{n:02d}-{name}.log").write_text(scrub("\n".join(text.splitlines()[-LOG_TAIL:])) + "\n")


def current_status(client, build_id):
    """The listing can lag behind a build; its own record is authoritative."""
    if build_id is None:
        return None
    return client.get_json(f"/builds/{build_id}").get("build", {}).get("status")


def launch(client, app_id, spec):
    """Start a build of a codemagic.yaml workflow on a branch; returns its id."""
    body = {"appId": app_id, "workflowId": spec["workflow"], "branch": spec["branch"]}
    if spec.get("variables"):
        body["environment"] = {"variables": spec["variables"]}
    return json.loads(client.request("/builds", body))["buildId"]


def label_of(wanted):
    return wanted.get("tag") or wanted["launch"]["workflow"]


def wait_for(client, ids, minutes, sleep=time.sleep, log=print):
    """Poll until every build id has ended, or the time runs out."""
    for minute in range(int(minutes) + 1):
        waiting = {label: current_status(client, i) for label, i in ids.items()}
        waiting = {label: s for label, s in waiting.items() if s is not None and s not in ENDED}
        if not waiting or minute == int(minutes):
            return
        log(f"minute {minute}: en attente de {waiting}", flush=True)
        sleep(60)


def fetch(client, request, root=Path("."), sleep=time.sleep):
    app = find_app(client, request.get("app", "Menia"))
    builds = builds_of(client, app["_id"])
    print("application:", app.get("appName"))
    for b in sorted(builds, key=lambda b: b.get("startedAt") or b.get("createdAt") or "", reverse=True)[:10]:
        print(f"  {b.get('tag') or b.get('branch')} {b.get('fileWorkflowId') or b.get('workflowId')} {b.get('status')} "
              f"{b.get('startedAt')} -> {b.get('finishedAt')}")
    ids = {}
    for wanted in request["builds"]:
        if "launch" in wanted:
            ids[label_of(wanted)] = launch(client, app["_id"], wanted["launch"])
            print(f"lancé {label_of(wanted)} sur {wanted['launch']['branch']} : build {ids[label_of(wanted)]}", flush=True)
        else:
            found = latest_for_tag(builds, wanted["tag"])
            ids[label_of(wanted)] = found["_id"] if found else None
    wait_for(client, ids, request.get("wait_minutes", 0), sleep)
    report = {}
    for wanted in request["builds"]:
        label = label_of(wanted)
        if ids[label] is None:
            report[label] = "introuvable"
            continue
        build = client.get_json(f"/builds/{ids[label]}").get("build", {"_id": ids[label]})
        status = build.get("status")
        if status not in ENDED:
            report[label] = f"en cours ({status})"
            continue
        dest = root / wanted["dest"]
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "codemagic-build.json").write_text(scrub(json.dumps(describe(build), indent=1, ensure_ascii=False)) + "\n")
        save_logs(client, build, dest)
        saved = []
        for artefact in build.get("artefacts") or []:
            saved += save_artefact(client, artefact, dest)
        report[label] = {"status": status, "dest": wanted["dest"], "files": len(saved),
                         "ignored": [s for s in saved if s.startswith("ignoré")]}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default=".github/codemagic-fetch.json")
    a = parser.parse_args(argv)
    token = os.environ.get("CODEMAGIC_API_TOKEN", "").strip()
    if not token:
        print("::error::Secret CODEMAGIC_API_TOKEN absent : l'ajouter dans Settings > Secrets and variables > Actions")
        sys.exit(1)
    fetch(Client(token), json.loads(Path(a.request).read_text()))


if __name__ == "__main__":
    main()
