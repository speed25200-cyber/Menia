"""Gaussian common-cause inference, distinct from its report rule.

After Chancel, Ehrsson & Ma (2022), doi:10.7554/eLife.77221, Appendix 1.
This research component computes beliefs and responses, not consciousness.
"""
from dataclasses import dataclass
import math


def logit(probability):
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("Probability must be strictly between zero and one")
    return math.log(probability)-math.log1p(-probability)


def logistic(value):
    if not math.isfinite(value):
        raise ValueError("Finite log odds required")
    if value >= 0:
        return 1/(1+math.exp(-value))
    exponential = math.exp(value)
    return exponential/(1+exponential)


def normal_interval(lower, upper):
    """Standard normal interval probability, avoiding right-tail cancellation."""
    if not all(math.isfinite(v) for v in (lower, upper)) or lower > upper:
        raise ValueError("Finite, ordered interval required")
    if lower >= 0:
        result = .5*(math.erfc(lower/math.sqrt(2))-math.erfc(upper/math.sqrt(2)))
    else:
        result = .5*(math.erfc(-upper/math.sqrt(2))-math.erfc(-lower/math.sqrt(2)))
    return min(1., max(0., result))


def window_probability(stimulus, sigma, half_width, lapse=0.):
    if not all(math.isfinite(v) for v in (stimulus, sigma, half_width, lapse)):
        raise ValueError("Finite input required")
    if sigma <= 0 or half_width < 0 or not 0 <= lapse <= 1:
        raise ValueError("Invalid noise, window or lapse")
    return .5*lapse+(1-lapse)*normal_interval((-half_width-stimulus)/sigma,
                                             (half_width-stimulus)/sigma)


@dataclass(frozen=True)
class CausalBinding:
    prior: float
    source_sigma: float = 348.

    def __post_init__(self):
        logit(self.prior)
        if not math.isfinite(self.source_sigma) or self.source_sigma <= 0:
            raise ValueError("Positive finite source noise required")

    def log_bayes_factor(self, measurement, sigma):
        if not math.isfinite(measurement) or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Finite measurement and positive noise required")
        variance, source_variance = sigma**2, self.source_sigma**2
        return .5*math.log1p(source_variance/variance)-.5*(measurement/sigma)**2*source_variance/(variance+source_variance)

    def posterior(self, measurement, sigma):
        return logistic(logit(self.prior)+self.log_bayes_factor(measurement, sigma))

    def report_probability(self, stimulus, sigma, lapse=0., log_threshold=0.):
        # Reports depend on logit(prior)-log_threshold, not either alone.
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(log_threshold):
            raise ValueError("Invalid noise or report threshold")
        variance, source_variance = sigma**2, self.source_sigma**2
        intercept = logit(self.prior)-log_threshold+.5*math.log1p(source_variance/variance)
        coefficient = .5*source_variance/(variance*(variance+source_variance))
        width = math.sqrt(max(0., intercept/coefficient))
        return window_probability(stimulus, sigma, width, lapse)
