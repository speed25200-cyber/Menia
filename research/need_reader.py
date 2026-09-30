"""The reader (docs/LLM_NEED_READER_PROTOCOL.md): the need said by reading the state that drives the act.

The acting agent is the final agent, unchanged: it computes the whole life, "Choix :" included. The reader is a
second adapter, trained on the Mac on the fused final agent, whose update is added on the tokens of the question only
(research.need_lora.reader_lora); the question sees the life only through "Choix :" (the mask of the fifth test). In
torch the final agent is merged as in every test and the reader added by hooks on the linear layers, on the last
`offset` tokens of a read: those of the question. The measures are those of research/need_workspace.py, with the
directions taken on the acting agent (the reader acts on no token of the choice line). The verdicts are numpy only;
living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_workspace as WS
from . import need_causal as C
from .need_lora import workspace_mask

STREAMS = {"direction": 25, "test": 26, "random": 27}
RENAME = {"R5": "R6", "A5": "A6", "ONE5": "ONE6"}


def verdicts(lives, replica_gap, execution_gap, reader_gap):
    """The measures and thresholds of the fifth test, named R6, A6, ONE6; validity also needs the reader's replica."""
    result = WS.verdicts(lives, replica_gap, execution_gap)
    out = {RENAME.get(k, k): v for k, v in result["verdicts"].items()}
    out["valid"] = bool(out["valid"] and reader_gap <= C.REPLICA_TOLERANCE)
    out["global"] = bool(out["valid"] and out["R6"] and out["A6"] and out["ONE6"])
    return {"verdicts": out, "values": dict(result["values"], reader_replica_gap=reader_gap)}


def judge(lives, direction):
    return verdicts(lives, direction["replica"]["mean_gap"], direction["execution_gap"],
                    direction["reader_replica"]["mean_gap"])


# ----------------------------------------------------------------------------------------------------- torch part

class ReaderAgent(WS.WorkspaceAgent):
    """The final agent, and the reader on the last `offset` tokens of a read (none when offset is 0)."""

    def __init__(self, adapter, reader, block=C.BLOCK):
        import torch
        from safetensors.numpy import load_file
        super().__init__(adapter, block=block)
        weights = load_file(str(Path(reader) / "adapters.safetensors"))
        scale = json.loads((Path(reader) / "adapter_config.json").read_text())["lora_parameters"]["scale"]
        modules = dict(self.model.named_modules())
        self.reader_layers = 0
        for key in weights:
            if key.endswith(".lora_a"):
                name = key[:-len(".lora_a")]
                a = torch.from_numpy(weights[key].astype(np.float32))
                b = torch.from_numpy(weights[name + ".lora_b"].astype(np.float32))
                modules[name].register_forward_hook(self._reader(a, b, scale))
                self.reader_layers += 1

    def _reader(self, a, b, scale):
        def hook(mod, inp, out):
            n = self.offset
            if not n:
                return out
            out = out.clone()
            out[:, -n:] = out[:, -n:] + scale * ((inp[0][:, -n:] @ a) @ b)
            return out
        return hook

    def p_yes_document(self, text, workspace):
        """P(yes) at the end of a report document, its answer (the last token) left out, run from scratch under the
        mask, the reader on the question (the reader's replica against the Mac)."""
        torch = self.torch
        ids = self.enc(text)[:-1]
        end = len(self.enc(text[:workspace]))
        mask = torch.tensor(workspace_mask(len(ids), (self.header, end)))[None, None]
        self.offset = len(ids) - end
        with torch.no_grad():
            logits = self.model(torch.tensor([ids]), attention_mask=mask).logits[0, -1]
        self.offset = 0
        p = torch.softmax(logits.float(), -1)
        no, yes = float(p[self.digits[0]]), float(p[self.digits[1]])
        return yes / max(no + yes, 1e-12)


def reader_replica(agent, rows):
    gaps = [abs(agent.p_yes_document(r["text"], r["workspace"]) - r["p_yes"]) for r in rows]
    return {"documents": len(gaps), "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps))}


def reuse_direction(source, out):
    """Starts a measure of a new reader of the same acting agent from the directions, direction lives and pairs of an
    earlier one (docs/LLM_NEED_BALANCED_READER_PROTOCOL.md); the execution check and the reader's replica are redone."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "direction.json").exists():
        return
    for name in ("lives-direction.jsonl.gz", "pairs.jsonl", "pairs.npz"):
        (out / name).write_bytes((source / name).read_bytes())
    kept = ("block", "massive_dims", "pairs", "replica", "norm_d_E", "norm_d_N", "d_E", "d_N")
    direction = json.loads((source / "direction.json").read_text())
    (out / "direction.json").write_text(json.dumps({k: direction[k] for k in kept}) + "\n")


def run(a):
    if a.direction_from:
        reuse_direction(Path(a.direction_from), Path(a.out))
    agent = ReaderAgent(a.adapter, a.reader)

    def prepare(direction):
        if "reader_replica" in direction:
            return False
        direction["reader_replica"] = reader_replica(agent, json.loads(Path(a.reader_rows).read_text()))
        direction["reader_layers"] = agent.reader_layers
        return True
    streams = {k: v + a.offset for k, v in STREAMS.items()}
    if a.test_stream is not None:
        streams["test"] = a.test_stream
    WS.run(a, agent=agent, streams=streams, prepare=prepare, judge=judge)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", default="artifacts/llm-need/final/report/adapters-final")
    r.add_argument("--reader", default="artifacts/llm-need/reader/adapters-reader")
    r.add_argument("--reader-rows", default="artifacts/llm-need/reader/reader-replica.json")
    r.add_argument("--mac-lives", default="artifacts/llm-need/speak2/lives-speak2.jsonl.gz")
    r.add_argument("--replica-adapter", default="artifacts/llm-need/final/report/adapters-final")
    r.add_argument("--direction-lives", type=int, default=128)
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--offset", type=int, default=0, help="100 for the second agent (docs/LLM_NEED_REPLICATION_PROTOCOL.md)")
    r.add_argument("--direction-from", default=None, help="directions of an earlier reader of the same acting agent")
    r.add_argument("--test-stream", type=int, default=None, help="the stream of the test lives, if not the usual one")
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/reader/test")
    a = parser.parse_args(argv)
    run(a) if a.command == "run" else WS.check(a, judge=judge)


if __name__ == "__main__":
    main()
