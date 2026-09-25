from __future__ import annotations
import heapq
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parent
OUTPUT_DIR = PROJECT_DIR / "imagegpt_celeba_outputs"
DATA_DIR = PROJECT_DIR / "celeba_data"

ACS2_FILE = PROJECT_DIR / "acs2_integer.py"

SEED = 666666

# Discop image configuration
MODEL_NAME = "openai/imagegpt-small"
IMAGE_SIZE = 32
VOCAB_IMAGE_SIZE = 512
CONTEXT_RATIO = 0.50

TEMPERATURE = 1.0
TOP_P = 1.0
APPLY_TOP_K = False
TOP_K = 256

# ACS defaults
PROBABILITY_BITS = 32
ROTATION_BITS = 128

# experiment 1
N_IMAGES_EXP1 = 100
MODE3 = 3
CELEBA_SPLIT = "test"
SAVE_ALL_EXP1_IMAGES = False

# experiment 2
EXP2_M_VALUES = list(range(1, 251))
TRIALS_PER_M = 10
EXP2_MAX_STEPS = None # None for all 512 generated image tokens

SHOW_PLOTS_IN_COLAB = True
SAVE_PLOTS_IN_COLAB = False
SAVE_TABLES = True
SAVE_JSON = True

SHOW_INNER_PROGRESS = False

LPIPS_NET = "alex"
METRIC_BATCH_SIZE = 16

HF_HOME = os.environ.get("HF_HOME")


def running_in_colab() -> bool:
    try:
        import google.colab
        return True
    except Exception:
        return False


def choose_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def show_or_save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if running_in_colab() and SHOW_PLOTS_IN_COLAB:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            fig.show()
        if SAVE_PLOTS_IN_COLAB:
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(f"Saved figure: {path}")
    else:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {path}")


def save_dataframe(df: pd.DataFrame, csv_path: Path, txt_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"Saved table: {csv_path}")
    print(f"Saved table: {txt_path}")
    print(df.to_string(index=False))


def random_bitstring(n_bits: int, rng: random.Random) -> str:
    return "".join("1" if rng.getrandbits(1) else "0" for _ in range(n_bits))


# def import_acs2_module():
#     if not ACS2_FILE.exists():
#         raise FileNotFoundError(f"Missing ACS2 implementation: {ACS2_FILE}")
#     spec = importlib.util.spec_from_file_location("acs2_integer_runner", ACS2_FILE)
#     if spec is None or spec.loader is None:
#         raise ImportError(f"Could not load {ACS2_FILE}")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module

def import_acs2_module():
    import importlib.util
    import sys

    path = PROJECT_DIR / "acs2_integer.py"

    spec = importlib.util.spec_from_file_location("acs2_integer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ACS2 module from {path}")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module

    spec.loader.exec_module(module)
    return module


ACS2_MOD = import_acs2_module()


class AutoregressiveSource(Protocol):

    eos_token_id: Optional[int]

    def reset(self) -> None: ...
    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def commit(self, token_id: int) -> None: ...


# integer ceiling division
def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _message_indices(
    intervals: Iterable[tuple[int, int, int]],
    m: int,
    limit: int = 2,
) -> list[int]:
    found: list[int] = []
    for lo, hi, d in intervals:
        if m >= d:
            first, stop = lo << (m - d), hi << (m - d)
        else:
            scale = 1 << (d - m)
            first, stop = _ceil_div(lo, scale), _ceil_div(hi, scale)
        for k in range(first, min(stop, first + limit - len(found))):
            found.append(k)
        if len(found) >= limit:
            return found
    return found


def acs1_unique_message(low: int, high: int, denom_bits: int, m: int) -> Optional[int]:
    values = _message_indices([(low, high, denom_bits)], m)
    return values[0] if len(values) == 1 else None


def _dynamic_message_length_acs1(low: int, high: int, denom_bits: int) -> tuple[int, int]:
    width = high - low
    if width <= 0:
        raise ValueError("Invalid interval.")
    max_m = max(1, denom_bits - (width - 1).bit_length() + 1)
    for m in range(max_m, 0, -1):
        k = acs1_unique_message(low, high, denom_bits, m)
        if k is not None:
            return m, k
    raise RuntimeError("No uniquely decodable dyadic point was found.")


def quantize_distribution(
    token_ids: torch.Tensor,
    probabilities: torch.Tensor,
    bits: int,
) -> tuple[list[int], list[int]]:
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
                    if needed == 0:
                        break
            if not changed:
                raise ValueError(
                    "PROBABILITY_BITS cannot give every retained token positive mass."
                )

    if int(counts.sum()) != total:
        raise RuntimeError("Could not construct a positive quantized PMF.")
    return token_ids.tolist(), counts.tolist()


def select_symbol_interval(
    low: int,
    high: int,
    n: int,
    cdf: Sequence[int],
    point: int,
    point_bits: int,
    q: int,
) -> int:
    next_bits = (n + 1) * q
    common_bits = max(next_bits, point_bits)
    point_common = point << (common_bits - point_bits)
    width = high - low
    for i in range(len(cdf) - 1):
        sub_low = low * (1 << q) + width * cdf[i]
        sub_hi = low * (1 << q) + width * cdf[i + 1]
        if (
            (sub_low << (common_bits - next_bits))
            <= point_common
            < (sub_hi << (common_bits - next_bits))
        ):
            return i
    raise RuntimeError("Message point fell outside the quantized PMF.")


def run_acs1(
    source: AutoregressiveSource,
    mode: int,
    *,
    message_bits: Optional[str] = None,
    length: Optional[int] = None,
    seed: int = 0,
    max_steps: Optional[int] = None,
    probability_bits: int = PROBABILITY_BITS,
    progress: bool = False,
) -> dict:
    if mode not in (2, 3):
        raise ValueError("This experiment runner uses ACS1 modes 2 and 3.")
    if mode == 2 and (not message_bits or set(message_bits) - {"0", "1"}):
        raise ValueError("ACS1 mode 2 requires a non-empty bit string.")
    if mode == 3 and (length is None or length < 1):
        raise ValueError("ACS1 mode 3 requires a positive fixed length.")

    rng = random.Random(seed)
    if mode == 3:
        point_bits = max(probability_bits * int(length) + 256, 256)
        point = rng.getrandbits(point_bits)
        requested_m = point_bits
    else:
        requested_m = len(message_bits or "")
        point = int(message_bits or "", 2)
        point_bits = requested_m

    low, high, n = 0, 1, 0
    tokens: list[int] = []
    selected_surprisal = 0.0

    source.reset()
    total_steps = int(length) if mode == 3 else int(max_steps or getattr(source, "total_tokens", 1024))
    iterator = tqdm(
        range(total_steps),
        desc=f"ACS1 mode {mode}",
        leave=False,
        disable=not progress,
    )

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = quantize_distribution(ids_t, probs_t, probability_bits)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)

        idx = select_symbol_interval(low, high, n, cdf, point, point_bits, probability_bits)
        old_low, width = low, high - low
        low = old_low * (1 << probability_bits) + width * cdf[idx]
        high = old_low * (1 << probability_bits) + width * cdf[idx + 1]
        n += 1

        token = ids[idx]
        tokens.append(token)
        p_lookup = (ids_t == token).nonzero(as_tuple=True)[0]
        if p_lookup.numel() == 0:
            raise RuntimeError("ACS1 selected token is absent from the unfiltered PMF.")
        selected_surprisal += -math.log2(max(float(probs_t[p_lookup[0]].item()), 1e-40))
        source.commit(token)

        if mode == 2 and acs1_unique_message(low, high, n * probability_bits, requested_m) is not None:
            break

    else:
        if mode == 2:
            raise RuntimeError("ACS1 mode 2 exhausted the sequence budget before decoding.")

    if mode == 3:
        m, decoded_k = _dynamic_message_length_acs1(low, high, n * probability_bits)
        decoded_bits = format(decoded_k, f"0{m}b")
    else:
        m = requested_m
        decoded_k = acs1_unique_message(low, high, n * probability_bits, m)
        if decoded_k is None:
            raise RuntimeError("ACS1 final interval is not uniquely decodable.")
        decoded_bits = format(decoded_k, f"0{m}b")

    return {
        "tokens": tokens,
        "message_bits": decoded_bits,
        "m": int(m),
        "N": int(n),
        "empirical_entropy_rate": selected_surprisal / n if n else float("nan"),
        "mode": mode,
    }


def decode_acs1_runner(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    mode: int,
    message_length: Optional[int] = None,
    probability_bits: int = PROBABILITY_BITS,
    progress: bool = False,
) -> str:
    if mode == 2 and message_length is None:
        raise ValueError("ACS1 mode 2 needs message_length at extraction time.")

    low, high, n = 0, 1, 0
    source.reset()
    iterator = tqdm(tokens, desc=f"ACS1 decode mode {mode}", leave=False, disable=not progress)
    for token in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = quantize_distribution(ids_t, probs_t, probability_bits)
        try:
            idx = ids.index(int(token))
        except ValueError as exc:
            raise ValueError(f"ACS1 token {token} is outside the current PMF.") from exc
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)
        old_low, width = low, high - low
        low = old_low * (1 << probability_bits) + width * cdf[idx]
        high = old_low * (1 << probability_bits) + width * cdf[idx + 1]
        n += 1
        source.commit(int(token))

    if mode == 3:
        m, k = _dynamic_message_length_acs1(low, high, n * probability_bits)
    else:
        m = int(message_length)
        k = acs1_unique_message(low, high, n * probability_bits, m)
        if k is None:
            raise RuntimeError("ACS1 extraction failed: final interval is not unique.")
    return format(k, f"0{m}b")


@dataclass
class HuffmanNode:

    prob: float
    index: int
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None
    search_path: int = 9

    @property
    def is_leaf(self) -> bool:
        return self.index != -1


def _contains_target(node: Optional[HuffmanNode]) -> bool:
    return bool(node is not None and node.search_path != 9)


def build_huffman_tree(
    indices: Sequence[int],
    probs: Sequence[float],
    search_for: Optional[int] = None,
) -> HuffmanNode:
    if len(indices) != len(probs):
        raise ValueError("indices and probs must have the same length")
    if not indices:
        raise ValueError("Empty vocabulary passed to Huffman tree builder")

    heap: list[tuple[float, int, HuffmanNode]] = []
    counter = 0
    for idx, prob in zip(indices, probs):
        node = HuffmanNode(
            prob=float(prob),
            index=int(idx),
            search_path=0 if search_for is not None and int(idx) == int(search_for) else 9,
        )
        heapq.heappush(heap, (node.prob, counter, node))
        counter += 1

    while len(heap) > 1:
        _, _, first = heapq.heappop(heap)
        _, _, second = heapq.heappop(heap)
        parent_search = 9
        if _contains_target(first):
            parent_search = -1
        elif _contains_target(second):
            parent_search = 1
        parent = HuffmanNode(
            prob=first.prob + second.prob,
            index=-1,
            left=first,
            right=second,
            search_path=parent_search,
        )
        heapq.heappush(heap, (parent.prob, counter, parent))
        counter += 1
    return heap[0][2]


def discop_encode_step(
    indices: Sequence[int],
    probs: Sequence[float],
    message_bits: str,
    rng: random.Random,
) -> tuple[int, int]:
    node = build_huffman_tree(indices, probs, search_for=None)
    n_bits = 0
    while not node.is_leaf:
        prob_sum = node.prob
        ptr = rng.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum

        partition = node.left.prob
        path0 = -1 if ptr_0 < partition else 1
        path1 = -1 if ptr_1 < partition else 1

        if n_bits >= len(message_bits):
            node = node.right if path0 == 1 else node.left
        else:
            bit = message_bits[n_bits]
            chosen_path = path0 if bit == "0" else path1
            node = node.right if chosen_path == 1 else node.left

        if path0 != path1:
            n_bits += 1
    return int(node.index), n_bits


def discop_decode_step(
    indices: Sequence[int],
    probs: Sequence[float],
    stego_t: int,
    rng: random.Random,
) -> str:
    node = build_huffman_tree(indices, probs, search_for=int(stego_t))
    decoded = ""
    while not node.is_leaf:
        prob_sum = node.prob
        ptr = rng.random()
        ptr_0 = ptr * prob_sum
        ptr_1 = (ptr + 0.5) * prob_sum
        if ptr_1 > prob_sum:
            ptr_1 -= prob_sum
        partition = node.left.prob
        path0 = -1 if ptr_0 < partition else 1
        path1 = -1 if ptr_1 < partition else 1

        if path0 != path1:
            if node.search_path == 9:
                raise RuntimeError("Discop extraction failed: ambiguous tree path.")
            if path0 == -1:
                path_table_swap = {-1: "0", 1: "1"}
            else:
                path_table_swap = {-1: "1", 1: "0"}
            decoded += path_table_swap[node.search_path]
            node = node.left if node.search_path == -1 else node.right
        else:
            node = node.left if path0 == -1 else node.right

    if node.search_path != 0:
        raise RuntimeError("Discop extraction failed: target leaf was not reached.")
    return decoded


def run_discop(
    source: AutoregressiveSource,
    mode: int,
    *,
    message_bits: Optional[str] = None,
    seed: int = 0,
    max_steps: int,
    progress: bool = False,
) -> dict:
    if mode not in (2, 3):
        raise ValueError("This experiment runner uses Discop modes 2 and 3.")

    rng = random.Random(seed)
    source.reset()

    if mode == 3:
        message_bits = random_bitstring(max(4096, max_steps * 32), random.Random(seed + 17))

    assert message_bits is not None
    remaining = message_bits
    tokens: list[int] = []
    embedded_bits = 0
    selected_surprisal = 0.0
    n_tokens = int(max_steps)

    iterator = tqdm(
        range(n_tokens),
        desc=f"Discop mode {mode}",
        leave=False,
        disable=not progress,
    )

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids = [int(x) for x in ids_t.detach().cpu().tolist()]
        probs = [float(x) for x in probs_t.detach().cpu().tolist()]
        token, used = discop_encode_step(ids, probs, remaining, rng)

        tokens.append(token)
        real_bits_before = len(remaining)
        used_from_message = min(int(used), real_bits_before)
        embedded_bits += used_from_message
        if used_from_message:
            remaining = remaining[used_from_message:]

        # empirical conditional surprisal of the actually sampled token
        token_pos = ids.index(token)
        selected_surprisal += -math.log2(max(probs[token_pos], 1e-40))
        source.commit(token)

        if mode == 2 and not remaining:
            break

    if mode == 2 and remaining:
        raise RuntimeError("Discop mode 2 exhausted the token budget before embedding m bits.")

    used_tokens = len(tokens)
    if mode == 2:
        embedded_bits = len(message_bits)

    return {
        "tokens": tokens,
        "message_bits": message_bits,
        "m": int(embedded_bits),
        "N": int(used_tokens),
        "empirical_entropy_rate": selected_surprisal / used_tokens if used_tokens else float("nan"),
        "mode": mode,
    }


# source with identical prefix/model settings
def make_fresh_source(source: "ImageGPTSource") -> "ImageGPTSource":
    return source.fresh_copy()


def decode_discop_runner(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    seed: int = 0,
    progress: bool = False,
) -> str:
    rng = random.Random(seed)
    source.reset()
    decoded: list[str] = []
    iterator = tqdm(tokens, desc="Discop decode", leave=False, disable=not progress)
    for token in iterator:
        ids_t, probs_t = source.probabilities()
        ids = [int(x) for x in ids_t.detach().cpu().tolist()]
        probs = [float(x) for x in probs_t.detach().cpu().tolist()]
        if int(token) not in ids:
            raise ValueError(f"Discop token {token} is outside the current PMF.")
        decoded.append(discop_decode_step(ids, probs, int(token), rng))
        source.commit(int(token))
    return "".join(decoded)


def load_imagegpt_classes():
    try:
        from transformers import ImageGPTImageProcessor, ImageGPTForCausalImageModeling
        return ImageGPTImageProcessor, ImageGPTForCausalImageModeling
    except ImportError:
        from transformers import ImageGPTFeatureExtractor, ImageGPTForCausalImageModeling
        return ImageGPTFeatureExtractor, ImageGPTForCausalImageModeling


def load_imagegpt(device: torch.device):
    ProcessorCls, ModelCls = load_imagegpt_classes()
    print(f"Loading ImageGPT: {MODEL_NAME}")
    processor = ProcessorCls.from_pretrained(MODEL_NAME)
    model = ModelCls.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    return processor, model


class ImageGPTSource:

    eos_token_id = None

    def __init__(
        self,
        model,
        processor,
        image_uint8: np.ndarray,
        device: torch.device,
        context_ratio: float,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        apply_top_k: bool = APPLY_TOP_K,
        top_k: int = TOP_K,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.context_ratio = float(context_ratio)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.apply_top_k = bool(apply_top_k)
        self.top_k = int(top_k)
        self.image_uint8 = np.asarray(image_uint8, dtype=np.uint8).copy()
        if self.image_uint8.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {self.image_uint8.shape}.")
        self.prefix_rows = int(round(IMAGE_SIZE * self.context_ratio))
        self.context_tokens = self._tokenize(self.image_uint8)[: self.prefix_rows * IMAGE_SIZE]
        self.total_tokens = IMAGE_SIZE * IMAGE_SIZE - len(self.context_tokens)
        self.reset()

    def _tokenize(self, image: np.ndarray) -> list[int]:
        pil = Image.fromarray(image, mode="RGB")
        ids = self.processor(pil, return_tensors="pt")["input_ids"][0]
        return [int(x) for x in ids.tolist()]

    def reset(self) -> None:
        sos = int(self.model.config.vocab_size - 1)
        context = torch.tensor([sos] + self.context_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        self._prev = context
        self._past = None
        self._last_probs: Optional[torch.Tensor] = None
        self._last_ids: Optional[torch.Tensor] = None
        self._step = 0

    @property
    def last_probs(self) -> Optional[torch.Tensor]:
        return self._last_probs

    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            output = self.model(
                self._prev,
                past_key_values=self._past,
                use_cache=True,
            )
        self._past = output.past_key_values
        logits = output.logits[0, -1, :].to(torch.float64)

        # sort logits, temperature, softmax, then optional top-p
        logits, ids = torch.sort(logits, descending=True)
        logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)

        if self.apply_top_k:
            k = min(self.top_k, int(probs.numel()))
            probs = probs[:k]
            ids = ids[:k]
            probs = probs / probs.sum()

        if self.top_p < 1.0:
            if not (0.0 < self.top_p <= 1.0):
                raise ValueError("TOP_P must lie in (0, 1].")
            cumulative = probs.cumsum(0)
            k = int((cumulative >= self.top_p).nonzero(as_tuple=True)[0][0].item()) + 1
            probs = probs[:k]
            ids = ids[:k]
            probs = probs / probs.sum()

        self._last_probs = probs
        self._last_ids = ids
        return ids, probs

    def commit(self, token_id: int) -> None:
        self._prev = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        self._step += 1

    def fresh_copy(self) -> "ImageGPTSource":
        return ImageGPTSource(
            self.model,
            self.processor,
            self.image_uint8,
            self.device,
            self.context_ratio,
            temperature=self.temperature,
            top_p=self.top_p,
            apply_top_k=self.apply_top_k,
            top_k=self.top_k,
        )


# 32 x 32
def resize_to_model(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def ids_to_image(processor, token_ids: Sequence[int]) -> np.ndarray:
    clusters = np.asarray(processor.clusters)
    arr = np.asarray(token_ids, dtype=np.int64)
    if arr.size != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Expected 1024 image tokens, got {arr.size}.")
    pixels = np.rint(127.5 * (clusters[arr] + 1.0)).astype(np.uint8)
    return pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 3)


def sample_completion(
    model,
    processor,
    image_uint8: np.ndarray,
    device: torch.device,
    context_ratio: float,
    seed: int,
) -> np.ndarray:
    torch_gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    torch_gen.manual_seed(seed)
    source = ImageGPTSource(
        model,
        processor,
        image_uint8,
        device,
        context_ratio,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        apply_top_k=APPLY_TOP_K,
        top_k=TOP_K,
    )
    generated = list(source.context_tokens)
    with torch.inference_mode():
        for _ in range(source.total_tokens):
            ids, probs = source.probabilities()
            idx = torch.multinomial(probs, 1, generator=torch_gen).item()
            token = int(ids[idx].item())
            generated.append(token)
            source.commit(token)
    return ids_to_image(processor, generated)


def image_from_generated_tokens(source: ImageGPTSource, tokens: Sequence[int]) -> np.ndarray:
    return ids_to_image(source.processor, list(source.context_tokens) + [int(x) for x in tokens])


# def load_celeba_subset(n_images: int, seed: int):
#     try:
#         from torchvision.datasets import CelebA
#     except ImportError as exc:
#         raise ImportError("torchvision is required for CelebA.") from exc

#     print(f"Loading CelebA split='{CELEBA_SPLIT}' from {DATA_DIR}")
#     dataset = CelebA(
#         root=str(DATA_DIR),
#         split=CELEBA_SPLIT,
#         target_type="attr",
#         download=True,
#     )
#     if len(dataset) < n_images:
#         raise ValueError(f"CelebA split has only {len(dataset)} images; need {n_images}.")

#     rng = random.Random(seed)
#     indices = sorted(rng.sample(range(len(dataset)), n_images))
#     print(f"Selected {n_images} deterministic CelebA images.")
#     return dataset, indices

def load_celeba_subset(n_images: int, seed: int):
    from datasets import load_dataset

    cache_dir = DATA_DIR / "selected_images"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_files = sorted(cache_dir.glob("*.png"))

    # Use existing cache if we already have enough images
    if len(cached_files) >= n_images:
        print(
            f"Found {len(cached_files)} cached CelebA images in "
            f"{cache_dir}"
        )

        selected_files = cached_files[:n_images]

        images = []
        for path in selected_files:
            images.append(Image.open(path).convert("RGB"))

        print(f"Using {n_images} cached CelebA images.")

        return images, list(range(n_images))

    # stream CelebA from Hugging Face
    print("No sufficient local CelebA cache found.")
    print("Streaming CelebA test split from Hugging Face...")

    dataset = load_dataset(
        "flwrlabs/celeba",
        split="test",
        streaming=True,
    )

    rng = random.Random(seed)

    # reservoir sampling
    reservoir = []

    for index, example in enumerate(dataset):

        image = example["image"].convert("RGB")

        if index < n_images:
            # initially fill the reservoir
            reservoir.append((index, image))

        else:
            # randomly decide whether this new image enters the reservoir
            j = rng.randint(0, index)

            if j < n_images:
                reservoir[j] = (index, image)

    if len(reservoir) < n_images:
        raise RuntimeError(
            f"Could only obtain {len(reservoir)} CelebA images "
            f"from the streaming dataset; needed {n_images}."
        )

    # shuffle the selected images so their order is randomized
    rng.shuffle(reservoir)

    # cache selected images.
    images = []

    for local_index, (original_index, image) in enumerate(reservoir):
        path = cache_dir / f"celeba_{local_index:04d}.png"

        image.save(path)

        images.append(image)

    print(
        f"Selected exactly {len(images)} CelebA images "
        f"from the streamed test split."
    )

    print(
        f"Cached images to:\n"
        f"  {cache_dir}"
    )

    return images, list(range(len(images)))


def get_selected_images(dataset, indices: Sequence[int]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for idx in indices:
        item = dataset[idx]
        if isinstance(item, tuple):
            pil = item[0]
        else:
            pil = item
        images.append(resize_to_model(pil))
    return images


def timed_encode_decode(
    algorithm: str,
    model,
    processor,
    image: np.ndarray,
    device: torch.device,
    *,
    mode: int,
    seed: int,
    message_bits: Optional[str] = None,
    context_ratio: float = CONTEXT_RATIO,
    max_steps: Optional[int] = None,
) -> tuple[np.ndarray, dict, float, float, str]:
    source = ImageGPTSource(
        model,
        processor,
        image,
        device,
        context_ratio,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        apply_top_k=APPLY_TOP_K,
        top_k=TOP_K,
    )
    n_steps = source.total_tokens if max_steps is None else int(max_steps)

    synchronize(device)
    start = time.perf_counter()

    if algorithm == "ACS1":
        result = run_acs1(
            source,
            mode,
            message_bits=message_bits,
            length=n_steps if mode == 3 else None,
            seed=seed,
            max_steps=n_steps,
            progress=SHOW_INNER_PROGRESS,
        )
    elif algorithm == "ACS2":
        old_prob_bits = getattr(ACS2_MOD, "PROBABILITY_BITS", PROBABILITY_BITS)
        old_rotation_bits = getattr(ACS2_MOD, "ROTATION_BITS", ROTATION_BITS)
        ACS2_MOD.PROBABILITY_BITS = PROBABILITY_BITS
        ACS2_MOD.ROTATION_BITS = ROTATION_BITS
        result = ACS2_MOD.encode_acs2(
            source,
            mode,
            message_bits=message_bits,
            length=n_steps if mode == 3 else None,
            seed=seed,
            max_steps=n_steps,
            progress=SHOW_INNER_PROGRESS,
        )
        ACS2_MOD.PROBABILITY_BITS = old_prob_bits
        ACS2_MOD.ROTATION_BITS = old_rotation_bits
    elif algorithm == "Discop":
        result = run_discop(
            source,
            mode,
            message_bits=message_bits,
            seed=seed,
            max_steps=n_steps,
            progress=SHOW_INNER_PROGRESS,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    synchronize(device)
    embedding_time = time.perf_counter() - start

    tokens = result["tokens"] if isinstance(result, dict) else result.tokens
    generated_image = image_from_generated_tokens(source, tokens)

    decode_source = ImageGPTSource(
        model,
        processor,
        image,
        device,
        context_ratio,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        apply_top_k=APPLY_TOP_K,
        top_k=TOP_K,
    )
    synchronize(device)
    start = time.perf_counter()

    if algorithm == "ACS1":
        decoded = decode_acs1_runner(
            decode_source,
            tokens,
            mode=mode,
            message_length=(int(result["m"]) if mode == 2 else None),
            probability_bits=PROBABILITY_BITS,
            progress=SHOW_INNER_PROGRESS,
        )
        expected = result["message_bits"]
    elif algorithm == "ACS2":
        decoded = ACS2_MOD.decode_acs2(
            decode_source,
            tokens,
            mode=mode,
            message_length=(int(result.m) if mode == 2 else None),
            rotation_numerator=int(result.rotation_numerator),
            rotation_bits=ROTATION_BITS,
            progress=SHOW_INNER_PROGRESS,
        )
        expected = result.message_bits
    else:
        decoded = decode_discop_runner(
            decode_source,
            tokens,
            seed=seed,
            progress=SHOW_INNER_PROGRESS,
        )

        # expected = result["message_bits"]
        # decoded = decoded[: len(expected)]

        expected = result["message_bits"][:result["m"]]
        decoded = decoded[:result["m"]]

    synchronize(device)
    extraction_time = time.perf_counter() - start

    if decoded != expected:
        raise RuntimeError(
            f"{algorithm} round-trip failed (mode={mode}, expected {len(expected)} bits, decoded {len(decoded)})."
        )

    if isinstance(result, dict):
        result = dict(result)
    else:
        result = {
            "tokens": result.tokens,
            "message_bits": result.message_bits,
            "m": result.m,
            "N": result.N,
            "empirical_entropy_rate": result.empirical_entropy_rate,
            "rotation_numerator": result.rotation_numerator,
            "mode": result.mode,
        }

    return generated_image, result, embedding_time, extraction_time, decoded


def compute_psnr(real_images: list[np.ndarray], fake_images: list[np.ndarray]) -> float:
    values = []
    for real, fake in zip(real_images, fake_images):
        real_f = real.astype(np.float32) / 255.0
        fake_f = fake.astype(np.float32) / 255.0
        mse = float(np.mean((real_f - fake_f) ** 2))
        values.append(float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse))
    return float(np.mean(values))


# HWC uint8 images -> NCHW float tensors in [0,1]
def _images_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)


# one FID over the 100-image sets and mean LPIPS over matched pairs
def compute_fid_lpips(
    real_images: list[np.ndarray],
    generated_by_method: dict[str, list[np.ndarray]],
    device: torch.device,
) -> dict[str, tuple[float, float]]:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise ImportError(
            "Install torchmetrics[image], torch-fidelity, and lpips before computing FID/LPIPS."
        ) from exc

    real_t = _images_tensor(real_images, device)

    real_lp = F.interpolate(real_t, size=(64, 64), mode="bilinear", align_corners=False)

    results: dict[str, tuple[float, float]] = {}
    for method, generated in generated_by_method.items():
        fake_t = _images_tensor(generated, device)

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        with torch.inference_mode():
            for i in range(0, len(real_t), METRIC_BATCH_SIZE):
                fid.update(real_t[i : i + METRIC_BATCH_SIZE], real=True)
                fid.update(fake_t[i : i + METRIC_BATCH_SIZE], real=False)
        fid_value = float(fid.compute().detach().cpu().item())

        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type=LPIPS_NET,
            reduction="mean",
            normalize=True,
        ).to(device)
        fake_lp = F.interpolate(fake_t, size=(64, 64), mode="bilinear", align_corners=False)
        with torch.inference_mode():
            for i in range(0, len(real_lp), METRIC_BATCH_SIZE):
                lpips_metric.update(real_lp[i : i + METRIC_BATCH_SIZE], fake_lp[i : i + METRIC_BATCH_SIZE])
        lpips_value = float(lpips_metric.compute().detach().cpu().item())
        results[method] = (fid_value, lpips_value)

    return results


# mode 3 completion on 100 fixed images
def run_experiment_1(
    model,
    processor,
    images: list[np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: MODE-3 CELEBA IMAGE COMPLETION")
    print("=" * 80)

    methods = ["ACS1", "Discop", "ACS2"]
    generated: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    baseline_images: list[np.ndarray] = []
    records: dict[str, list[dict]] = {m: [] for m in methods}
    demo_originals: list[np.ndarray] = []
    demo_baselines: list[np.ndarray] = []
    demo_outputs: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    total_steps = IMAGE_SIZE * IMAGE_SIZE - int(round(IMAGE_SIZE * CONTEXT_RATIO)) * IMAGE_SIZE
    print(f"Context ratio = {CONTEXT_RATIO:.2f}; generated tokens per image = {total_steps}")

    print("Generating pretrained ImageGPT baseline completions...")
    for i, image in enumerate(tqdm(images, desc="ImageGPT baseline", unit="img")):
        seed = SEED + 100_000 + i
        baseline = sample_completion(model, processor, image, device, CONTEXT_RATIO, seed)
        baseline_images.append(baseline)
        if i < 3:
            demo_originals.append(image)
            demo_baselines.append(baseline)

    for method in methods:
        print(f"\nRunning {method} mode 3 on {len(images)} images...")
        for i, image in enumerate(tqdm(images, desc=f"{method} mode 3", unit="img")):
            seed = SEED + 10_000 * (methods.index(method) + 1) + i
            output, result, embed_t, extract_t, _ = timed_encode_decode(
                method,
                model,
                processor,
                image,
                device,
                mode=MODE3,
                seed=seed,
                message_bits=None,
                context_ratio=CONTEXT_RATIO,
                max_steps=total_steps,
            )
            generated[method].append(output)

            if method == "ACS1":
                m = int(result["m"])
                n = int(result["N"])
                h = float(result["empirical_entropy_rate"])
            elif method == "Discop":
                m = int(result["m"])
                n = int(result["N"])
                h = float(result["empirical_entropy_rate"])
            else:
                m = int(result["m"])
                n = int(result["N"])
                h = float(result["empirical_entropy_rate"])

            capacity = m / n if n else float("nan")
            utilization = capacity / h if h > 0 else float("nan")
            records[method].append(
                {
                    "m_bits": m,
                    "N_tokens": n,
                    "capacity_bits_per_token": capacity,
                    "empirical_entropy_rate_bits_per_token": h,
                    "utilization": utilization,
                    "embedding_time_s": embed_t,
                    "extraction_time_s": extract_t,
                }
            )

            if i < 3:
                demo_outputs[method].append(output)

    # image quality metrics for the three schemes
    print("\nComputing FID and LPIPS...")
    quality_sets = {"ImageGPT baseline": baseline_images}
    quality_sets.update(generated)
    # quality_metrics = compute_fid_lpips(images, quality_sets, device)

    # compare all methods against the pretrained Image GPT completions
    quality_metrics = compute_fid_lpips(baseline_images, quality_sets, device)

    rows = []

    # # baseline_psnr = compute_psnr(images, baseline_images)
    # rows.append(
    #     {
    #         "Method": "ImageGPT baseline",
    #         "m_bits": 0,
    #         "N_tokens": total_steps,
    #         "capacity_bits_per_token": 0.0,
    #         "empirical_entropy_rate_bits_per_token": np.nan,
    #         "utilization": np.nan,
    #         "embedding_time_s": np.nan,
    #         "extraction_time_s": np.nan,
    #         "FID": quality_metrics["ImageGPT baseline"][0],
    #         # "PSNR_dB": baseline_psnr,
    #         "PSNR_dB": np.nan,
    #         "LPIPS": quality_metrics["ImageGPT baseline"][1],
    #     }
    # )

    for method in methods:
        df_method = pd.DataFrame(records[method])
        rows.append(
            {
                "Method": method,
                "m_bits": df_method["m_bits"].mean(),
                "N_tokens": df_method["N_tokens"].mean(),
                "capacity_bits_per_token": df_method["capacity_bits_per_token"].mean(),
                "empirical_entropy_rate_bits_per_token": df_method["empirical_entropy_rate_bits_per_token"].mean(),
                "utilization": df_method["utilization"].mean(),
                "embedding_time_s": df_method["embedding_time_s"].mean(),
                "extraction_time_s": df_method["extraction_time_s"].mean(),
                "FID": quality_metrics[method][0],
                # "PSNR_dB": compute_psnr(images, generated[method]),
                "PSNR_dB": compute_psnr(baseline_images, generated[method]),
                "LPIPS": quality_metrics[method][1],
            }
        )

    df = pd.DataFrame(rows)

    save_dataframe(
        df,
        OUTPUT_DIR / "experiment1_mode3_results.csv",
        OUTPUT_DIR / "experiment1_mode3_results.txt",
    )

    per_image_rows = []
    for method in methods:
        for i, rec in enumerate(records[method]):
            per_image_rows.append({"image_index": i, "Method": method, **rec})
    save_dataframe(
        pd.DataFrame(per_image_rows),
        OUTPUT_DIR / "experiment1_mode3_per_image.csv",
        OUTPUT_DIR / "experiment1_mode3_per_image.txt",
    )

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    titles = ["Original", "ImageGPT", "ACS1", "Discop", "ACS2"]
    panels = [demo_originals, demo_baselines, demo_outputs["ACS1"], demo_outputs["Discop"], demo_outputs["ACS2"]]
    for col, title in enumerate(titles):
        for row in range(3):
            axes[row, col].imshow(panels[col][row])
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title, fontsize=12)
    fig.suptitle("CelebA 32×32 Image Completion — Mode 3", fontsize=14)
    fig.tight_layout()
    show_or_save_figure(fig, OUTPUT_DIR / "experiment1_mode3_qualitative_5col.png")
    plt.close(fig)

    return df


def run_one_mode2_trial(
    algorithm: str,
    model,
    processor,
    image: np.ndarray,
    device: torch.device,
    m: int,
    trial_seed: int,
) -> tuple[float, float, float, int]:
    rng = random.Random(trial_seed)
    message = random_bitstring(m, rng)
    total_steps = IMAGE_SIZE * IMAGE_SIZE - int(round(IMAGE_SIZE * CONTEXT_RATIO)) * IMAGE_SIZE

    source = ImageGPTSource(
        model,
        processor,
        image,
        device,
        CONTEXT_RATIO,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        apply_top_k=APPLY_TOP_K,
        top_k=TOP_K,
    )

    if algorithm == "ACS1":
        result = run_acs1(
            source,
            2,
            message_bits=message,
            seed=trial_seed,
            max_steps=total_steps,
            probability_bits=PROBABILITY_BITS,
            progress=False,
        )
        bits = int(result["m"])
        n = int(result["N"])
        entropy_rate = float(result["empirical_entropy_rate"])

    elif algorithm == "ACS2":
        old_prob_bits = getattr(ACS2_MOD, "PROBABILITY_BITS", PROBABILITY_BITS)
        old_rotation_bits = getattr(ACS2_MOD, "ROTATION_BITS", ROTATION_BITS)
        ACS2_MOD.PROBABILITY_BITS = PROBABILITY_BITS
        ACS2_MOD.ROTATION_BITS = ROTATION_BITS
        result = ACS2_MOD.encode_acs2(
            source,
            2,
            message_bits=message,
            seed=trial_seed,
            max_steps=total_steps,
            progress=False,
        )
        ACS2_MOD.PROBABILITY_BITS = old_prob_bits
        ACS2_MOD.ROTATION_BITS = old_rotation_bits
        bits = int(result.m)
        n = int(result.N)
        entropy_rate = float(result.empirical_entropy_rate)

    elif algorithm == "Discop":
        result = run_discop(
            source,
            2,
            message_bits=message,
            seed=trial_seed,
            max_steps=total_steps,
            progress=False,
        )
        bits = int(result["m"])
        n = int(result["N"])
        entropy_rate = float(result["empirical_entropy_rate"])

    else:
        raise ValueError(algorithm)

    if bits != m:
        raise RuntimeError(f"{algorithm} embedded {bits} bits for a requested m={m} trial.")
    rate = bits / n
    utilization = rate / entropy_rate if entropy_rate > 0 else float("nan")
    return rate, utilization, entropy_rate, n


# mode 2 Monte Carlo on m
def run_experiment_2(
    model,
    processor,
    dataset,
    device: torch.device,
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MODE-2 EMPIRICAL RATE / UTILIZATION")
    print("=" * 80)

    methods = ["ACS1", "Discop", "ACS2"]
    rows = []
    outer = tqdm(EXP2_M_VALUES, desc="Experiment 2: m", unit="m")

    for m in outer:
        for method_idx, method in enumerate(methods):
            trial_rates: list[float] = []
            trial_utils: list[float] = []
            trial_entropy: list[float] = []
            trial_tokens: list[int] = []

            successful = 0
            trial_id = 0
            while successful < TRIALS_PER_M:
                trial_seed = SEED + 1_000_000 * (m + 1) + 10_000 * method_idx + trial_id
                image_rng = random.Random(trial_seed + 999)
                image_idx = image_rng.randrange(len(dataset))
                item = dataset[image_idx]
                pil_image = item[0] if isinstance(item, tuple) else item
                image = resize_to_model(pil_image)

                try:
                    rate, util, entropy_rate, n = run_one_mode2_trial(
                        method,
                        model,
                        processor,
                        image,
                        device,
                        m,
                        trial_seed,
                    )
                except RuntimeError as exc:
                    print(f"Retrying {method}, m={m}, trial={successful}: {exc}")
                    trial_id += 1
                    continue

                trial_rates.append(rate)
                trial_utils.append(util)
                trial_entropy.append(entropy_rate)
                trial_tokens.append(n)
                successful += 1
                trial_id += 1

            rows.append(
                {
                    "m_bits": m,
                    "Method": method,
                    "R_bits_per_token": float(np.mean(trial_rates)),
                    "utilization": float(np.mean(trial_utils)),
                    "empirical_entropy_rate_bits_per_token": float(np.mean(trial_entropy)),
                    "N_tokens_mean": float(np.mean(trial_tokens)),
                    "N_trials": successful,
                }
            )

    df = pd.DataFrame(rows)
    save_dataframe(
        df,
        OUTPUT_DIR / "experiment2_mode2_results.csv",
        OUTPUT_DIR / "experiment2_mode2_results.txt",
    )

    # R vs m
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        sub = df[df["Method"] == method].sort_values("m_bits")
        line = ax.plot(sub["m_bits"], sub["R_bits_per_token"], label=f"{method} R")[0]
        color = line.get_color()
        ax.plot(
            sub["m_bits"],
            sub["empirical_entropy_rate_bits_per_token"],
            linestyle="--",
            color=color,
            alpha=0.8,
            label=f"{method} entropy rate",
        )
    ax.set_xlabel("Message length m (bits)")
    ax.set_ylabel("Bits/token")
    ax.set_title("Experiment 2: Empirical embedding rate and entropy rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    show_or_save_figure(fig, OUTPUT_DIR / "experiment2_R_vs_m.png")
    plt.close(fig)

    # utilization vs m
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        sub = df[df["Method"] == method].sort_values("m_bits")
        ax.plot(sub["m_bits"], sub["utilization"], label=method)
    ax.set_xlabel("Message length m (bits)")
    ax.set_ylabel("Utilization")
    ax.set_title("Experiment 2: Utilization vs message length")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    show_or_save_figure(fig, OUTPUT_DIR / "experiment2_utilization_vs_m.png")
    plt.close(fig)

    return df


def save_configuration(device: torch.device) -> None:
    if not SAVE_JSON:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": SEED,
        "model_name": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "image_vocab": VOCAB_IMAGE_SIZE,
        "context_ratio": CONTEXT_RATIO,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "apply_top_k": APPLY_TOP_K,
        "top_k": TOP_K,
        "probability_bits": PROBABILITY_BITS,
        "rotation_bits": ROTATION_BITS,
        "n_images_exp1": N_IMAGES_EXP1,
        "celeba_split": CELEBA_SPLIT,
        "exp2_m_values": EXP2_M_VALUES,
        "trials_per_m": TRIALS_PER_M,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    with (OUTPUT_DIR / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration: {OUTPUT_DIR / 'experiment_config.json'}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_global_seed(SEED)
    device = choose_device()
    save_configuration(device)

    print("\nExperiment defaults:")
    print(f"  Model       = {MODEL_NAME}")
    print(f"  Image size  = {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Context     = {CONTEXT_RATIO:.2f}")
    print(f"  Temperature = {TEMPERATURE}")
    print(f"  Top-p       = {TOP_P}")
    print(f"  Top-k       = {TOP_K if APPLY_TOP_K else 'disabled'}")
    print(f"  P bits      = {PROBABILITY_BITS}")
    print(f"  R bits      = {ROTATION_BITS}")

    processor, model = load_imagegpt(device)

    model_vocab = int(model.config.vocab_size)
    processor_size_attr = getattr(processor, "size", IMAGE_SIZE)
    if isinstance(processor_size_attr, dict):
        processor_size = int(processor_size_attr.get("height", processor_size_attr.get("shortest_edge", IMAGE_SIZE)))
    else:
        processor_size = int(processor_size_attr)
    print(f"ImageGPT vocab_size={model_vocab}, processor.size={processor_size}")
    if processor_size != IMAGE_SIZE:
        raise RuntimeError(f"Expected ImageGPT size {IMAGE_SIZE}, got {processor_size}.")
    if model_vocab < VOCAB_IMAGE_SIZE + 1:
        raise RuntimeError(
            f"Expected at least {VOCAB_IMAGE_SIZE+1} ImageGPT tokens including SOS, got {model_vocab}."
        )

    dataset, indices = load_celeba_subset(N_IMAGES_EXP1, SEED)
    images = get_selected_images(dataset, indices)

    if SAVE_JSON:
        with (OUTPUT_DIR / "experiment1_selected_celeba_indices.json").open("w", encoding="utf-8") as f:
            json.dump({"split": CELEBA_SPLIT, "indices": indices}, f, indent=2)

    run_experiment_1(model, processor, images, device)
    run_experiment_2(model, processor, dataset, device)

    print("\nAll experiments completed.")
    print(f"Outputs are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()