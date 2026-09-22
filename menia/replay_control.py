"""Bounded, executable controllers selected by replay of supported outcomes.

Only the supplied feature packet reaches the policy. No generated Python is
executed; policies are versioned parameter records, with fixed action semantics.
"""
from dataclasses import asdict, dataclass
import math

SOURCES = ("betaCell", "inputOnly", "internal", "shuffledLabels")
ACTIONS = ("direct", "verify", "abstain")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Policy:
    kind: str = "threshold"
    source: str = "betaCell"
    threshold: float = .8
    below: str = "verify"

    def __post_init__(self):
        if self.kind not in ("threshold", *ACTIONS):
            raise ValueError("Unsupported policy kind")
        if self.source not in SOURCES or not number(self.threshold) or not 0 <= self.threshold <= 1.01:
            raise ValueError("Invalid policy feature or threshold")
        if self.below not in ("verify", "abstain"):
            raise ValueError("Invalid fallback action")
        if self.kind != "threshold" and (self.source != "betaCell" or self.threshold != .8 or self.below != "verify"):
            raise ValueError("Constant policies must have canonical fields")

    def payload(self):
        return asdict(self)

    @classmethod
    def load(cls, value):
        if not isinstance(value, dict) or set(value) != {"kind", "source", "threshold", "below"}:
            raise ValueError("Invalid policy schema")
        return cls(**value)

    def choose(self, features):
        if self.kind != "threshold":
            return self.kind
        p = features.get(self.source)
        if not number(p) or not 0 <= p <= 1:
            raise ValueError("Missing or invalid pre-answer feature: " + self.source)
        return "direct" if p >= self.threshold else self.below


def candidates(sources=SOURCES):
    if not sources or len(set(sources)) != len(sources) or any(s not in SOURCES for s in sources):
        raise ValueError("Invalid source set")
    # 99 policies for four sources; 51 for public-only; 75 for shuffled control.
    return [Policy(), *[Policy(kind=a) for a in ACTIONS],
            *[Policy(source=s, threshold=t, below=b) for s in sources
              for t in [i/10 for i in range(11)] + [1.01] for b in ("verify", "abstain")]]


def unique(policies):
    return list(dict.fromkeys(policies))


def replay(policy, episodes):
    """Refuse unsupported actions rather than inventing their consequences."""
    if not episodes:
        raise ValueError("Empty replay history")
    if len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("Repeated episode in replay history")
    losses, actions = [], []
    for episode in episodes:
        action = policy.choose(episode["features"])
        outcome = episode["outcomes"].get(action)
        if outcome is None:
            raise ValueError("Action not supported by replay: " + action)
        loss = outcome.get("pointLoss")
        if not number(loss) or loss < 0:
            raise ValueError("Invalid recorded outcome loss")
        losses.append(loss)
        actions.append(action)
    return dict(meanPointLoss=math.fsum(losses)/len(losses), losses=losses,
                actionCounts={a: actions.count(a) for a in ACTIONS})


def improve(episodes, policies, incumbent):
    """Incumbent wins exact ties; only a strictly smaller historical loss wins."""
    policies = unique([incumbent, *policies])
    scores = [dict(policy=p.payload(), **replay(p, episodes)) for p in policies]
    best = min(range(len(scores)), key=lambda i: scores[i]["meanPointLoss"])
    return policies[best], dict(episodes=len(episodes), candidates=len(policies),
                               incumbentLoss=scores[0]["meanPointLoss"],
                               selectedLoss=scores[best]["meanPointLoss"],
                               scores=[{k: v for k, v in s.items() if k != "losses"} for s in scores])


def execute(action, candidate, verify):
    """Run the selected action, with no evaluation label in the interface."""
    if action == "abstain":
        return None
    if action == "direct":
        if not isinstance(candidate, str):
            raise ValueError("Missing candidate answer")
        return candidate
    if action == "verify":
        text = verify()
        if not isinstance(text, str):
            raise ValueError("Verifier must return text")
        return text
    raise ValueError("Unsupported action")
