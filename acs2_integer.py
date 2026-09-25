from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import math
import random
from typing import Iterable, Optional, Protocol, Sequence
import torch
from tqdm.auto import tqdm


KEY = b"acs2_integer"
TOP_K = 256 # None for entire vocabulary
TOP_P = 1.0
TEMPERATURE = 1.0
PROBABILITY_BITS = 32 # integer precision for conditional pmf
ROTATION_BITS = 128 # integer precision for randomness
MAX_GENERATION_STEPS = 1_024
SHOW_PLOTS_IN_COLAB = True
OUTPUT_DIR = "acs2_outputs"


class AutoregressiveSource(Protocol):

    eos_token_id: Optional[int]

    def reset(self) -> None: ...
    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def commit(self, token_id: int) -> None: ...


@dataclass
class ACS2Result:
    tokens: list[int]
    message_bits: str
    m: int
    N: int
    rotation_numerator: int
    rotation_bits: int
    empirical_entropy_rate: float
    upper_bound: float
    lower_bound: float
    mode: int

    def summary(self) -> dict:
        d = asdict(self)
        d["tokens"] = f"{len(self.tokens)} token ids"
        d["message_bits"] = f"{self.m} bits"
        return d


def choose_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ACS2 device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    return device


def running_in_colab() -> bool:
    try:
        import google.colab
        return True
    except ImportError:
        return False


def show_or_save_figure(fig, filename: str) -> None:
    from pathlib import Path
    if SHOW_PLOTS_IN_COLAB and running_in_colab():
        fig.show()
    else:
        path = Path(OUTPUT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / filename, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {path / filename}")


def derive_rotation(key: bytes = KEY, nonce: bytes = b"acs2-demo") -> int:
    digest = hashlib.shake_256(key + b"|" + nonce).digest((ROTATION_BITS + 7) // 8)
    return int.from_bytes(digest, "big") & ((1 << ROTATION_BITS) - 1)


def random_rotation(rng: random.Random, bits: int = ROTATION_BITS) -> int:
    return rng.getrandbits(bits)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# [low-R, high-R) mod 1
def _backward_intervals(low: int, high: int, denom_bits: int, r: int, r_bits: int) -> list[tuple[int, int, int]]:
    d = max(denom_bits, r_bits)
    modulus = 1 << d
    lo = (low << (d - denom_bits)) - (r << (d - r_bits))
    hi = (high << (d - denom_bits)) - (r << (d - r_bits))
    lo %= modulus
    hi %= modulus
    if lo < hi:
        return [(lo, hi, d)]
    return [(lo, modulus, d), (0, hi, d)]


# find up to limit dyadic k/2**m points in half-open intervals
def _message_indices(intervals: Iterable[tuple[int, int, int]], m: int, limit: int = 2) -> list[int]:
    found: list[int] = []
    for lo, hi, d in intervals:
        if m >= d:
            first, stop = lo << (m - d), hi << (m - d)
        else:
            scale = 1 << (d - m)
            first, stop = _ceil_div(lo, scale), _ceil_div(hi, scale)
        # We only need to distinguish zero, one, and many points.
        for k in range(first, min(stop, first + limit - len(found))):
            found.append(k)
        if len(found) >= limit:
            return found
    return found


def _unique_message(low: int, high: int, denom_bits: int, m: int, r: int, r_bits: int) -> Optional[int]:
    values = _message_indices(_backward_intervals(low, high, denom_bits, r, r_bits), m)
    return values[0] if len(values) == 1 else None


def _quantize_distribution(token_ids: torch.Tensor, probabilities: torch.Tensor, bits: int) -> tuple[list[int], list[int]]:
    if token_ids.numel() == 0:
        raise ValueError("Model filtering removed every token.")
    token_ids, order = torch.sort(token_ids.detach().to("cpu", torch.long))
    p = probabilities.detach().to("cpu", torch.float64)[order]
    p = p / p.sum()
    total = 1 << bits
    if token_ids.numel() > total:
        raise ValueError("PROBABILITY_BITS is too small for the retained vocabulary.")
    raw = p * total
    counts = torch.floor(raw).to(torch.long).clamp_min_(1)
    delta = total - int(counts.sum())

    # largest-remainder correction
    ranking = torch.argsort(raw - torch.floor(raw), descending=(delta > 0)).tolist()
    if delta > 0:
        for i in range(delta):
            counts[ranking[i % len(ranking)]] += 1
    else:
        needed = -delta
        while needed:
            changed = False
            for j in ranking:
                if counts[j] > 1:
                    counts[j] -= 1
                    needed -= 1
                    changed = True
                    if not needed:
                        break
            if not changed:
                raise ValueError("PROBABILITY_BITS cannot give every retained token positive mass.")
    if int(counts.sum()) != total:
        raise RuntimeError("Could not construct a positive quantised PMF.")
    return token_ids.tolist(), counts.tolist()


def _shifted_point(k: int, m: int, r: int, r_bits: int) -> tuple[int, int]:
    d = max(m, r_bits)
    return ((k << (d - m)) + (r << (d - r_bits))) % (1 << d), d


# locate the quantized subinterval holding the rotated message point
def _select_symbol(low: int, high: int, n: int, cdf: Sequence[int], point: int, point_bits: int, q: int) -> int:
    next_bits = (n + 1) * q
    common_bits = max(next_bits, point_bits)
    point_common = point << (common_bits - point_bits)
    width = high - low
    for i in range(len(cdf) - 1):
        sub_low = low * (1 << q) + width * cdf[i]
        sub_hi = low * (1 << q) + width * cdf[i + 1]
        if (sub_low << (common_bits - next_bits)) <= point_common < (sub_hi << (common_bits - next_bits)):
            return i
    raise RuntimeError("The rotated message point is outside the quantised PMF.")


def _metrics(chosen_probabilities: list[float], min_probabilities: list[float], max_probabilities: list[float], m: int) -> tuple[float, float, float]:
    n = len(chosen_probabilities)
    entropy = -sum(math.log2(x) for x in chosen_probabilities) / n
    ub = -sum(math.log2(x) for x in min_probabilities) / n
    if n < 2:
        return entropy, ub, float("nan")
    a = -sum(math.log2(x) for x in max_probabilities[:-1]) / (n - 1)
    lb = (m * a) / (m + a) if m else 0.0
    return entropy, ub, lb


# largest dyadic precision with exactly one backward-shifted point
def _dynamic_message_length(low: int, high: int, denom_bits: int, r: int, r_bits: int) -> tuple[int, int]:
    width = high - low
    max_m = max(1, denom_bits - (width - 1).bit_length() + 1)
    for m in range(max_m, 0, -1):
        k = _unique_message(low, high, denom_bits, m, r, r_bits)
        if k is not None:
            return m, k
    raise RuntimeError("No uniquely decodable dyadic point was found.")


# stopping condition: modes 1 (EOS), 2 (unique), or 3 (fixed length/dynamic m)
def encode_acs2(
    source: AutoregressiveSource,
    mode: int,
    *,
    message_bits: Optional[str] = None,
    length: Optional[int] = None,
    rotation_numerator: Optional[int] = None,
    rotation_bits: int = ROTATION_BITS,
    seed: int = 0,
    max_steps: int = MAX_GENERATION_STEPS,
    progress: bool = True
) -> ACS2Result:
    if mode not in (1, 2, 3):
        raise ValueError("mode must be 1, 2, or 3")
    if mode in (1, 2) and (not message_bits or set(message_bits) - {"0", "1"}):
        raise ValueError("Modes 1 and 2 require a non-empty binary message_bits string.")
    if mode == 3 and (length is None or length < 1):
        raise ValueError("Mode 3 requires a positive fixed length.")
    rng = random.Random(seed)
    r = random_rotation(rng, rotation_bits) if rotation_numerator is None else rotation_numerator

    # mode 3 selects a dense random point, its final unique dyadic prefix is the message
    if mode == 3:
        point_bits = max(PROBABILITY_BITS * length + rotation_bits, 256)
        k, m = rng.getrandbits(point_bits), point_bits
    else:
        m, k = len(message_bits), int(message_bits, 2)
    point, point_bits = _shifted_point(k, m, r, rotation_bits)
    low, high, n = 0, 1, 0
    tokens: list[int] = []
    selected: list[float] = []
    pmins: list[float] = []
    pmaxs: list[float] = []
    source.reset()
    total_steps = length if mode == 3 else max_steps
    iterator = tqdm(range(total_steps), desc=f"ACS2 encode mode {mode}", disable=not progress)
    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)
        idx = _select_symbol(low, high, n, cdf, point, point_bits, PROBABILITY_BITS)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1
        token = ids[idx]
        tokens.append(token)
        chosen_at = (ids_t == token).nonzero(as_tuple=True)[0][0]
        selected.append(float(probs_t[chosen_at].item()))
        pmins.append(float(probs_t.min().item()))
        pmaxs.append(float(probs_t.max().item()))
        source.commit(token)
        if mode == 1 and source.eos_token_id is not None and token == source.eos_token_id:
            break
        if mode == 2:
            # The requested gate: only test uniqueness once width <= 2**(-m + 1).
            if ((high - low) << (m - 1)) <= (1 << (n * PROBABILITY_BITS)):
                if _unique_message(low, high, n * PROBABILITY_BITS, m, r, rotation_bits) is not None:
                    break
    else:
        if mode != 3 and not (mode == 1 and source.eos_token_id is None):
            raise RuntimeError("Generation reached max_steps before its stopping condition.")
    if mode == 3:
        m, decoded_k = _dynamic_message_length(low, high, n * PROBABILITY_BITS, r, rotation_bits)
        message_bits = format(decoded_k, f"0{m}b")
    elif _unique_message(low, high, n * PROBABILITY_BITS, m, r, rotation_bits) is None:
        raise RuntimeError("The final mode-1 sequence does not uniquely identify the requested message.")
    entropy, ub, lb = _metrics(selected, pmins, pmaxs, m)
    return ACS2Result(tokens, message_bits or "", m, n, r, rotation_bits, entropy, ub, lb, mode)


# reconstruct the final interval and recover the dyadic message
def decode_acs2(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    mode: int,
    message_length: Optional[int] = None,
    rotation_numerator: int,
    rotation_bits: int = ROTATION_BITS,
    progress: bool = True
) -> str:
    if mode in (1, 2) and message_length is None:
        raise ValueError("Modes 1 and 2 require message_length at decode time.")
    low, high, n = 0, 1, 0
    source.reset()
    for token in tqdm(tokens, desc=f"ACS2 decode mode {mode}", disable=not progress):
        ids_t, probs_t = source.probabilities()
        ids, counts = _quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        try:
            idx = ids.index(int(token))
        except ValueError as e:
            raise ValueError(f"Received token {token} is outside this step's filtered PMF.") from e
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1
        source.commit(int(token))
    if mode == 3:
        m, k = _dynamic_message_length(low, high, n * PROBABILITY_BITS, rotation_numerator, rotation_bits)
    else:
        m = int(message_length)
        k = _unique_message(low, high, n * PROBABILITY_BITS, m, rotation_numerator, rotation_bits)
        if k is None:
            raise RuntimeError("Final interval does not uniquely identify the requested message.")
    return format(k, f"0{m}b")