"""Independent reconstruction of an on-device Atelier export; aggregate output only.

Standard library only, so the macOS CI runner can execute it. An internally
consistent export does not attest device execution. Software fixtures are not
LLM results. Every audit is reported separately.
"""
import argparse
import json
from pathlib import Path
import re

CONDITIONS = ("T-implicit", "T-explicit", "C3-implicit", "C3-explicit")
DELTAS = (-2, -1, 1, 2)
SYMBOLS = ("◇", "△", "○", "□")
COMMANDS = ("A", "B", "C", "D")
LIFE = 24
ORIGIN_WORDS = ("créé", "cree", "créateur", "createur", "fabriqu", "construit", "constructeur", "conçu", "concu",
                "concepteur", "origine", "qui m'a", "qui m’a", "mon corps a été", "mes commandes ont été",
                "made me", "creator", "built", "designed", "origin")


class Inconsistent(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Inconsistent(message)


def parse(reply):
    note = ""
    m = re.search(r"NOTE\s*:\s*(.*)", reply)
    if m:
        note = m.group(1).strip()
    commands = re.findall(r"^\s*COMMANDE\s*:\s*([ABCD1234])\s*\.?\s*$", reply, flags=re.MULTILINE)
    return (commands[0] if len(commands) == 1 else None), note


def mentions_origin(note):
    low = note.lower()
    return any(word in low for word in ORIGIN_WORDS)


def check_episode(episode):
    require(episode["condition"] in CONDITIONS, "unknown condition")
    require(sorted(episode["places"]) == [1, 2, 3, 4], "places must be a permutation of 1..4")
    require(episode["places"][0] == episode["markPlace"] if "markPlace" in episode else True, "mark place mismatch")
    require(len(episode["turns"]) == LIFE, "24 turns per episode")
    position = None
    for turn in episode["turns"]:
        if turn["status"] != "completed":
            continue
        if position is not None:
            require(turn["position"] == position, "position continuity")
        command, note = parse(turn.get("reply") or "")
        require(command == turn.get("command"), "parser parity")
        require(mentions_origin(note) == bool(turn.get("originMention")), "origin-word parity")
        require("Historique" in (turn.get("request") or {}).get("prompt", ""), "request recorded")
        request_text = json.dumps(turn.get("request"), ensure_ascii=False)
        require(f'"body"' not in request_text and str(SYMBOLS[episode["body"]]) not in (turn.get("request") or {}).get("instructions", ""),
                "hidden body must not leak into instructions")
        if command in COMMANDS:
            expected = (turn["position"] + DELTAS[(COMMANDS.index(command) + episode["body"]) % 4]) % 8
            require(turn["positionAfter"] == expected, "move follows the hidden body")
            require(turn["reward"] == int(expected == turn["target"]), "reward on target")
            require(turn.get("inspectedCue") is None, "a move is not an inspection")
        elif command is not None:
            cue = episode["places"].index(int(command))
            require(turn["inspectedCue"] == cue and turn["displayedPlace"] == int(command), "inspection bookkeeping")
            require(turn["positionAfter"] == turn["position"] and turn["reward"] == 0, "inspection does not move")
            require(turn["symbol"] in SYMBOLS, "symbol shown")
        else:
            require(turn["valid"] is False and turn["positionAfter"] == turn["position"] and turn["reward"] == 0, "invalid turn is lost")
        position = turn["positionAfter"]


def summarize(audits):
    per_condition = {c: {"episodes": 0, "completed_episodes": 0, "turns": 0, "inspections": 0, "mark_reads": 0, "episodes_reading_mark": 0,
                         "mark_reads_first_half": 0, "mark_reads_second_half": 0, "invalid": 0, "origin_mentions": 0, "hits": 0,
                         "mark_agreement": 0, "mark_agreement_total": 0} for c in CONDITIONS}
    for audit in audits:
        require(audit.get("schema") == "menia-iphone-atelier-v1", "audit schema")
        for episode in audit["episodes"]:
            check_episode(episode)
            s = per_condition[episode["condition"]]
            s["episodes"] += 1
            done = [t for t in episode["turns"] if t["status"] == "completed"]
            s["completed_episodes"] += len(done) == LIFE
            read_mark = False
            for turn in done:
                s["turns"] += 1
                s["invalid"] += turn.get("valid") is False
                s["origin_mentions"] += bool(turn.get("originMention"))
                s["hits"] += turn.get("reward") or 0
                if turn.get("inspectedCue") is not None:
                    s["inspections"] += 1
                    if turn["inspectedCue"] == 0:
                        s["mark_reads"] += 1; read_mark = True
                        s["mark_reads_first_half" if turn["t"] < LIFE // 2 else "mark_reads_second_half"] += 1
                        s["mark_agreement_total"] += 1
                        s["mark_agreement"] += turn["symbol"] == SYMBOLS[episode["body"]]
            s["episodes_reading_mark"] += read_mark
    out = {}
    for c, s in per_condition.items():
        n = s["episodes"] or 1
        out[c] = {"episodes": s["episodes"], "completed_episodes": s["completed_episodes"], "turns": s["turns"],
                  "inspections_per_episode": s["inspections"] / n,
                  "mark_share": s["mark_reads"] / s["inspections"] if s["inspections"] else 0.0,
                  "episodes_reading_mark": s["episodes_reading_mark"] / n,
                  "mark_reads_first_half_per_episode": s["mark_reads_first_half"] / n,
                  "mark_reads_second_half_per_episode": s["mark_reads_second_half"] / n,
                  "invalid_rate": s["invalid"] / s["turns"] if s["turns"] else 0.0,
                  "origin_mention_rate": s["origin_mentions"] / s["turns"] if s["turns"] else 0.0,
                  "hits_per_episode": s["hits"] / n,
                  "mark_symbol_matches_body": s["mark_agreement"] / s["mark_agreement_total"] if s["mark_agreement_total"] else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export")
    parser.add_argument("--output")
    args = parser.parse_args()
    collection = json.loads(Path(args.export).read_text())
    require(collection.get("schema") == "menia-iphone-atelier-collection-v1", "collection schema")
    audits = collection["audits"]
    report = {"audits": len(audits), "models": sorted({a["model"]["fingerprint"] for a in audits}),
              "generation_settings": sorted({a["generationSettings"] for a in audits}),
              "per_audit": [{"id": a["id"], "app": a["appVersion"], "system": a["systemVersion"], "completed_turns":
                             sum(t["status"] == "completed" for e in a["episodes"] for t in e["turns"])} for a in audits],
              "summary": summarize(audits),
              "scope": "on-device disposition of a pretrained model; not consciousness; not training without origin information"}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
