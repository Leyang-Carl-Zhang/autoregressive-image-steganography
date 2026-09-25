from __future__ import annotations

try:
    import pip_system_certs
except ImportError:
    print(
        "Warning: pip-system-certs is not installed. "
        "HTTPS downloads of pretrained model weights may fail."
    )

import os

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    print(
        "Warning: certifi is not installed. "
        "HTTPS certificate verification may fail."
    )


import hashlib
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
PIXELCNN_DIR = PROJECT_DIR / "PixelCNN++"
ACS1_DIR = PROJECT_DIR / "ACS1Image"
DISCOP_DIR = PROJECT_DIR / "DiscopImage"
DATA_FILE = PROJECT_DIR / "cifar10_data" / "cifar-10-batches-py" / "test_batch"
CHECKPOINT = PIXELCNN_DIR / "checkpoints" / "pixelcnnpp_cifar10.pth"
OUTPUT_DIR = SCRIPT_DIR / "pixelcnnpp_cifar10_outputs"

DATA_DIR = PROJECT_DIR / "cifar10_data"

for _p in (PROJECT_DIR, PIXELCNN_DIR, ACS1_DIR, DISCOP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import PixelCNN
try:
    import acs2_integer as acs2_mod
except Exception:
    acs2_mod = None


SEED = 666666
CONTEXT_RATIO = 0.5
N_EVAL_IMAGES = 100
N_EXPERIMENT2_TRIALS = 10
M_MAX = 250
SHOW_PLOTS_IN_COLAB = True
SAVE_IMAGES = True
SAVE_TABLES = True
SAVE_PLOTS = True

MODE12_MESSAGE_BITS = 32
PROBABILITY_BITS = 32
ROTATION_BITS = 128
MAX_GENERATION_STEPS = 1_024
TOP_K = 256
TOP_P = 1.0
TEMPERATURE = 1.0

ACS2_KEY = getattr(acs2_mod, "KEY", b"pixelcnnpp_cifar10_image") if acs2_mod is not None else b"pixelcnnpp_cifar10_image"


class AutoregressiveSource(Protocol):
    eos_token_id: Optional[int]

    def reset(self) -> None: ...
    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def commit(self, token_id: int) -> None: ...


@dataclass
class SchemeRun:
    scheme: str
    mode: int
    image_index: int
    message_bits: str
    recovered_bits: str
    tokens: list[int]
    m: int
    N: int
    entropy_rate: float
    capacity: float
    utilization: float
    encode_seconds: float
    extract_seconds: float

    def summary(self) -> dict:
        d = asdict(self)
        d["tokens"] = f"{len(self.tokens)} token ids"
        d["message_bits"] = f"{self.m} bits"
        d["recovered_bits"] = f"{len(self.recovered_bits)} bits"
        return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")
    return device


def running_in_colab() -> bool:
    try:
        import google.colab
        return True
    except Exception:
        return False


def show_or_save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    if SAVE_PLOTS:
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {path}")
    if SHOW_PLOTS_IN_COLAB and running_in_colab():
        fig.show()


def save_table(df: pd.DataFrame, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    md_path = OUTPUT_DIR / f"{stem}.md"
    if SAVE_TABLES:
        df.to_csv(csv_path, index=False)
        md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
        print(f"Saved table: {csv_path}")
        print(f"Saved table: {md_path}")
    print(df.to_markdown(index=False))


def random_bitstring(n_bits: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join("1" if rng.getrandbits(1) else "0" for _ in range(n_bits))


def random_rotation(rng: random.Random, bits: int = ROTATION_BITS) -> int:
    return rng.getrandbits(bits)


def derive_rotation(key: bytes = ACS2_KEY, nonce: bytes = b"acs2-demo") -> int:
    digest = hashlib.shake_256(key + b"|" + nonce).digest((ROTATION_BITS + 7) // 8)
    return int.from_bytes(digest, "big") & ((1 << ROTATION_BITS) - 1)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint_obj


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    nr_filters = None
    for key in ("u_init.conv.weight_v", "u_init.conv.weight", "u_init.conv_v.weight"):
        if key in state_dict:
            nr_filters = int(state_dict[key].shape[0])
            break
    if nr_filters is None:
        for key, value in state_dict.items():
            if key.endswith("u_init.conv.weight_v") or key.endswith("u_init.conv.weight"):
                nr_filters = int(value.shape[0])
                break

    nr_resnet = None
    up_stage_indices = []
    for key in state_dict.keys():
        if key.startswith("up_layers.") and ".u_stream." in key:
            parts = key.split(".")
            try:
                up_stage_indices.append(int(parts[2]))
            except Exception:
                pass
    if up_stage_indices:
        nr_resnet = max(up_stage_indices) + 1

    nr_logistic_mix = None
    out_key = "nin_out.lin_a.weight_v"
    if out_key not in state_dict:
        out_key = "nin_out.lin_a.weight"
    if out_key in state_dict:
        out_ch = int(state_dict[out_key].shape[0])
        if out_ch % 10 == 0:
            nr_logistic_mix = out_ch // 10

    if nr_resnet is None:
        nr_resnet = 5
    if nr_filters is None:
        nr_filters = 160
    if nr_logistic_mix is None:
        nr_logistic_mix = 10
    return nr_resnet, nr_filters, nr_logistic_mix


def load_model(checkpoint_path: Path, device: torch.device) -> PixelCNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place pixelcnnpp_cifar10.pth in PixelCNN++/checkpoints/."
        )

    raw = torch.load(checkpoint_path, map_location="cpu")
    state = strip_module_prefix(extract_state_dict(raw))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a valid PyTorch state_dict.")

    nr_resnet, nr_filters, nr_logistic_mix = infer_model_config(state)
    print(
        f"Checkpoint architecture: nr_resnet={nr_resnet}, nr_filters={nr_filters}, "
        f"nr_logistic_mix={nr_logistic_mix}"
    )

    model = PixelCNN(
        nr_resnet=nr_resnet,
        nr_filters=nr_filters,
        nr_logistic_mix=nr_logistic_mix,
        input_channels=3,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys = {len(missing)}, unexpected keys = {len(unexpected)}")
        if missing:
            print("First missing keys:", missing[:5])
        if unexpected:
            print("First unexpected keys:", unexpected[:5])

    model.eval()
    return model


# def load_cifar10_test_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
#     if not path.exists():
#         raise FileNotFoundError(f"CIFAR-10 test batch not found: {path}")

#     with path.open("rb") as f:
#         obj = pickle.load(f, encoding="bytes")

#     data = obj[b"data"] if b"data" in obj else obj["data"]
#     labels = obj.get(b"labels", obj.get("labels", None))
#     images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
#     if labels is None:
#         labels = np.full((images.shape[0],), -1, dtype=np.int64)
#     else:
#         labels = np.asarray(labels, dtype=np.int64)
#     return images, labels

def load_cifar10_test_subset(
    n_images: int,
    seed: int,
):
    from datasets import load_dataset

    cache_dir = DATA_DIR / "selected_images"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_files = sorted(cache_dir.glob("cifar10_*.png"))

    # use cached images if enough are already available
    if len(cached_files) >= n_images:
        print(
            f"Found {len(cached_files)} cached CIFAR-10 images in "
            f"{cache_dir}"
        )

        selected_files = cached_files[:n_images]

        images = []
        labels = []

        for path in selected_files:
            image = Image.open(path).convert("RGB")
            images.append(np.asarray(image, dtype=np.uint8))

        # Labels
        metadata_path = cache_dir / "cifar10_labels.npy"

        if not metadata_path.exists():
            raise RuntimeError(
                "Cached CIFAR-10 images were found, but the label "
                f"file is missing: {metadata_path}"
            )

        all_cached_labels = np.load(metadata_path)
        labels = all_cached_labels[:n_images].astype(np.int64)

        images = np.stack(images, axis=0)

        print(
            f"Using {n_images} cached CIFAR-10 images."
        )

        return images, labels

    # stream CIFAR-10 from Hugging Face
    print("No sufficient local CIFAR-10 cache found.")
    print("Streaming CIFAR-10 test split from Hugging Face...")

    dataset = load_dataset(
        "cifar10",
        split="test",
        streaming=True,
    )

    rng = random.Random(seed)

    # reservoir sampling
    reservoir = []

    for index, example in enumerate(dataset):

        image = example["img"].convert("RGB")
        label = int(example["label"])

        if index < n_images:
            reservoir.append(
                (index, image, label)
            )

        else:
            j = rng.randint(0, index)

            if j < n_images:
                reservoir[j] = (
                    index,
                    image,
                    label,
                )

    if len(reservoir) < n_images:
        raise RuntimeError(
            f"Could only obtain {len(reservoir)} CIFAR-10 images "
            f"from the streaming dataset; needed {n_images}."
        )

    # randomize final ordering
    rng.shuffle(reservoir)

    # cache the selected images
    images = []
    labels = []

    for local_index, (_, image, label) in enumerate(reservoir):

        path = cache_dir / f"cifar10_{local_index:04d}.png"

        image.save(path)

        images.append(
            np.asarray(image, dtype=np.uint8)
        )
        labels.append(label)

    images = np.stack(images, axis=0)
    labels = np.asarray(labels, dtype=np.int64)

    np.save(
        cache_dir / "cifar10_labels.npy",
        labels,
    )

    print(
        f"Selected exactly {len(images)} CIFAR-10 test images."
    )

    print(
        f"Cached images to:\n"
        f"  {cache_dir}"
    )

    return images, labels


def resize_image_if_needed(image_uint8: np.ndarray, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    if image_uint8.shape[:2] == size:
        return np.asarray(image_uint8, dtype=np.uint8)
    pil = Image.fromarray(np.asarray(image_uint8, dtype=np.uint8))
    pil = pil.resize(size, resample=Image.BICUBIC)
    return np.asarray(pil, dtype=np.uint8)


def to_tensor(image_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    if image_uint8.shape != (32, 32, 3):
        raise ValueError(f"Expected a 32x32 RGB image, got {image_uint8.shape}.")
    x = torch.from_numpy(image_uint8.transpose(2, 0, 1)).float() / 127.5 - 1.0
    return x.unsqueeze(0).to(device)


def to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    arr = ((tensor[0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5)
    return np.clip(arr, 0, 255).astype(np.uint8)


def sample_from_probs(ids: torch.Tensor, probs: torch.Tensor, rng: Optional[torch.Generator] = None) -> int:
    if rng is None:
        idx = torch.multinomial(probs, 1).item()
    else:
        idx = torch.multinomial(probs, 1, generator=rng).item()
    return int(ids[idx].item())


def _logistic_bin_mass(x: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(torch.clamp(log_scale, min=-7.0))
    centered_x = x - mean
    inv_scale = 1.0 / scale
    plus_in = inv_scale * (centered_x + 1.0 / 255.0)
    min_in = inv_scale * (centered_x - 1.0 / 255.0)

    cdf_plus = torch.sigmoid(plus_in)
    cdf_min = torch.sigmoid(min_in)

    probs = cdf_plus - cdf_min
    probs = torch.where(x < -0.999, cdf_plus, probs)
    probs = torch.where(x > 0.999, 1.0 - cdf_min, probs)
    return probs.clamp_min(1e-40)


def _split_pixelcnnpp_params(params_1d: torch.Tensor, nr_mix: int):
    logit_probs = params_1d[:nr_mix]
    rest = params_1d[nr_mix:].contiguous().view(3, 3, nr_mix)
    means = rest[:, 0, :]
    log_scales = torch.clamp(rest[:, 1, :], min=-7.0)
    coeffs = torch.tanh(rest[:, 2, :])
    return logit_probs, means, log_scales, coeffs


def _component_bin_masses(candidate_values: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    return _logistic_bin_mass(candidate_values, means.to(torch.float64), log_scales.to(torch.float64))


def _rgb_channel_probs_from_model(
    params_1d: torch.Tensor,
    channel_index: int,
    current_pixel_norm: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    nr_mix = params_1d.numel() // 10
    logit_probs, means, log_scales, coeffs = _split_pixelcnnpp_params(params_1d, nr_mix)
    mix_weights = torch.softmax(logit_probs.to(torch.float64), dim=0)

    candidate_values = torch.linspace(-1.0, 1.0, 256, device=device, dtype=torch.float64).view(256, 1)

    red_means = means[0].to(torch.float64).view(1, -1)
    red_log_scales = log_scales[0].to(torch.float64).view(1, -1)
    red_component_probs = _component_bin_masses(candidate_values, red_means, red_log_scales)

    if channel_index == 0:
        probs = (red_component_probs * mix_weights.view(1, -1)).sum(dim=1)

    elif channel_index == 1:
        x_r = current_pixel_norm[0].to(torch.float64)
        green_means = (means[1].to(torch.float64) + coeffs[0].to(torch.float64) * x_r).view(1, -1)
        green_log_scales = log_scales[1].to(torch.float64).view(1, -1)
        green_component_probs = _component_bin_masses(candidate_values, green_means, green_log_scales)

        r_value = int(torch.clamp(torch.round((x_r + 1.0) * 127.5), 0, 255).item())
        red_mass = red_component_probs[r_value, :]
        posterior_weights = mix_weights * red_mass
        posterior_weights = posterior_weights / posterior_weights.sum()
        probs = (green_component_probs * posterior_weights.view(1, -1)).sum(dim=1)

    elif channel_index == 2:
        x_r = current_pixel_norm[0].to(torch.float64)
        x_g = current_pixel_norm[1].to(torch.float64)

        green_means = (means[1].to(torch.float64) + coeffs[0].to(torch.float64) * x_r).view(1, -1)
        green_log_scales = log_scales[1].to(torch.float64).view(1, -1)
        green_component_probs = _component_bin_masses(candidate_values, green_means, green_log_scales)

        blue_means = (
            means[2].to(torch.float64)
            + coeffs[1].to(torch.float64) * x_r
            + coeffs[2].to(torch.float64) * x_g
        ).view(1, -1)
        blue_log_scales = log_scales[2].to(torch.float64).view(1, -1)
        blue_component_probs = _component_bin_masses(candidate_values, blue_means, blue_log_scales)

        r_value = int(torch.clamp(torch.round((x_r + 1.0) * 127.5), 0, 255).item())
        g_value = int(torch.clamp(torch.round((x_g + 1.0) * 127.5), 0, 255).item())
        red_mass = red_component_probs[r_value, :]
        green_mass = green_component_probs[g_value, :]

        posterior_weights = mix_weights * red_mass * green_mass
        posterior_weights = posterior_weights / posterior_weights.sum()
        probs = (blue_component_probs * posterior_weights.view(1, -1)).sum(dim=1)

    else:
        raise ValueError("channel_index must be 0, 1, or 2")

    probs = probs.clamp_min(1e-40)
    probs = probs / probs.sum()
    ids = torch.arange(256, device=device, dtype=torch.long)
    return ids, probs


class PixelCNNPPChannelSource:

    eos_token_id = None

    def __init__(self, model: PixelCNN, prefix_image_uint8: np.ndarray, context_ratio: float):
        self.model = model
        self.device = next(model.parameters()).device
        self.context_ratio = float(context_ratio)
        if not (0.0 <= self.context_ratio <= 1.0):
            raise ValueError("context_ratio must lie in [0, 1].")
        self.prefix_image_uint8 = np.asarray(prefix_image_uint8, dtype=np.uint8)
        if self.prefix_image_uint8.shape != (32, 32, 3):
            raise ValueError(f"Expected a 32x32 RGB image, got {self.prefix_image_uint8.shape}.")
        self.prefix_rows = int(round(32 * self.context_ratio))
        self.reset()

    @property
    def total_tokens(self) -> int:
        return (32 - self.prefix_rows) * 32 * 3

    def reset(self) -> None:
        self.image = to_tensor(self.prefix_image_uint8, self.device)
        if self.prefix_rows < 32:
            self.image[:, :, self.prefix_rows:, :] = 0.0
        self._next_token_index = self.prefix_rows * 32 * 3

    def _position(self) -> tuple[int, int, int]:
        pixel_index, channel_index = divmod(self._next_token_index, 3)
        row, col = divmod(pixel_index, 32)
        return row, col, channel_index

    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        row, col, channel_index = self._position()
        with torch.inference_mode():
            out = self.model(self.image, sample=True)
        params = out[0, :, row, col]
        current_pixel = self.image[0, :, row, col]
        return _rgb_channel_probs_from_model(params, channel_index, current_pixel, self.device)

    def commit(self, token_id: int) -> None:
        row, col, channel_index = self._position()
        value = float(token_id) / 127.5 - 1.0
        self.image[0, channel_index, row, col] = value
        self._next_token_index += 1


def _acs1_message_indices(intervals: Iterable[tuple[int, int, int]], m: int, limit: int = 2) -> list[int]:
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


def _acs1_unique_message(low: int, high: int, denom_bits: int, m: int) -> Optional[int]:
    values = _acs1_message_indices([(low, high, denom_bits)], m)
    return values[0] if len(values) == 1 else None


def _acs1_dynamic_message_length(low: int, high: int, denom_bits: int) -> tuple[int, int]:
    width = high - low
    if width <= 0:
        raise ValueError("Invalid interval.")
    max_m = max(1, denom_bits - (width - 1).bit_length() + 1)
    for m in range(max_m, 0, -1):
        k = _acs1_unique_message(low, high, denom_bits, m)
        if k is not None:
            return m, k
    raise RuntimeError("No uniquely decodable dyadic point was found.")


def _acs1_quantize_distribution(token_ids: torch.Tensor, probabilities: torch.Tensor, bits: int) -> tuple[list[int], list[int]]:
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
                    if not needed:
                        break
            if not changed:
                raise ValueError("PROBABILITY_BITS cannot give every retained token positive mass.")

    if int(counts.sum()) != total:
        raise RuntimeError("Could not construct a positive quantised PMF.")
    return token_ids.tolist(), counts.tolist()


def _acs1_select_symbol(low: int, high: int, n: int, cdf: Sequence[int], point: int, point_bits: int, q: int) -> int:
    next_bits = (n + 1) * q
    common_bits = max(next_bits, point_bits)
    point_common = point << (common_bits - point_bits)
    width = high - low
    for i in range(len(cdf) - 1):
        sub_low = low * (1 << q) + width * cdf[i]
        sub_hi = low * (1 << q) + width * cdf[i + 1]
        if (sub_low << (common_bits - next_bits)) <= point_common < (sub_hi << (common_bits - next_bits)):
            return i
    raise RuntimeError("Message point fell outside the quantised PMF.")


# use a random point and recover the longest uniquely decodable message
def acs1_encode_mode3(
    source: AutoregressiveSource,
    *,
    length: int,
    seed: int,
    progress: bool = True,
) -> tuple[list[int], str, int, int, float]:
    rng = random.Random(seed)
    point_bits = max(PROBABILITY_BITS * int(length) + 256, 256)
    k = rng.getrandbits(point_bits)
    m = point_bits

    low, high, n = 0, 1, 0
    tokens: list[int] = []
    chosen_probs: list[float] = []

    source.reset()
    iterator = range(int(length))
    if progress:
        iterator = tqdm(iterator, desc="ACS1 encode mode 3", leave=False)

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _acs1_quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)

        idx = _acs1_select_symbol(low, high, n, cdf, k, m, PROBABILITY_BITS)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1

        token = ids[idx]
        tokens.append(token)
        chosen_probs.append(float(probs_t[(ids_t == token).nonzero(as_tuple=True)[0][0]].item()))
        source.commit(token)

    m, decoded_k = _acs1_dynamic_message_length(low, high, n * PROBABILITY_BITS)
    message_bits = format(decoded_k, f"0{m}b")
    entropy_rate = float(np.mean([-math.log2(max(p, 1e-40)) for p in chosen_probs])) if chosen_probs else float("nan")
    return tokens, message_bits, m, n, entropy_rate


def acs1_encode_mode2(
    source: AutoregressiveSource,
    *,
    message_bits: str,
    seed: int,
    max_steps: Optional[int] = None,
    progress: bool = True,
) -> tuple[list[int], str, int, int, float]:
    if not message_bits or set(message_bits) - {"0", "1"}:
        raise ValueError("Mode 2 requires a non-empty binary message_bits string.")

    m = len(message_bits)
    k = int(message_bits, 2)
    low, high, n = 0, 1, 0
    tokens: list[int] = []
    chosen_probs: list[float] = []

    source.reset()
    total_steps = int(max_steps) if max_steps is not None else getattr(source, "total_tokens", MAX_GENERATION_STEPS)
    iterator = range(total_steps)
    if progress:
        iterator = tqdm(iterator, desc="ACS1 encode mode 2", leave=False)

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _acs1_quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)

        idx = _acs1_select_symbol(low, high, n, cdf, k, m, PROBABILITY_BITS)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1

        token = ids[idx]
        tokens.append(token)
        chosen_probs.append(float(probs_t[(ids_t == token).nonzero(as_tuple=True)[0][0]].item()))
        source.commit(token)

        if _acs1_unique_message(low, high, n * PROBABILITY_BITS, m) is not None:
            break
    else:
        raise RuntimeError("Generation reached max_steps before the message became uniquely decodable.")

    entropy_rate = float(np.mean([-math.log2(max(p, 1e-40)) for p in chosen_probs])) if chosen_probs else float("nan")
    return tokens, message_bits, m, n, entropy_rate


def acs1_decode_tokens(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    mode: int,
    message_length: Optional[int] = None,
    progress: bool = False,
) -> str:
    if mode in (1, 2) and message_length is None:
        raise ValueError("Modes 1 and 2 require message_length at decode time.")

    low, high, n = 0, 1, 0
    source.reset()
    iterator = tokens
    if progress:
        iterator = tqdm(tokens, desc=f"ACS1 decode mode {mode}", leave=False)

    for token in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _acs1_quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
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
        m, k = _acs1_dynamic_message_length(low, high, n * PROBABILITY_BITS)
    else:
        m = int(message_length)
        k = _acs1_unique_message(low, high, n * PROBABILITY_BITS, m)
        if k is None:
            raise RuntimeError("Final interval does not uniquely identify the requested message.")
    return format(k, f"0{m}b")


def _acs2_message_indices(intervals: Iterable[tuple[int, int, int]], m: int, limit: int = 2) -> list[int]:
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


def _acs2_backward_intervals(low: int, high: int, denom_bits: int, r: int, r_bits: int) -> list[tuple[int, int, int]]:
    d = max(denom_bits, r_bits)
    modulus = 1 << d
    lo = (low << (d - denom_bits)) - (r << (d - r_bits))
    hi = (high << (d - denom_bits)) - (r << (d - r_bits))
    lo %= modulus
    hi %= modulus
    if lo < hi:
        return [(lo, hi, d)]
    return [(lo, modulus, d), (0, hi, d)]


def _acs2_unique_message(low: int, high: int, denom_bits: int, m: int, r: int, r_bits: int) -> Optional[int]:
    values = _acs2_message_indices(_acs2_backward_intervals(low, high, denom_bits, r, r_bits), m)
    return values[0] if len(values) == 1 else None


def _acs2_shifted_point(k: int, m: int, r: int, r_bits: int) -> tuple[int, int]:
    d = max(m, r_bits)
    return ((k << (d - m)) + (r << (d - r_bits))) % (1 << d), d


def _acs2_quantize_distribution(token_ids: torch.Tensor, probabilities: torch.Tensor, bits: int) -> tuple[list[int], list[int]]:
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
                    if not needed:
                        break
            if not changed:
                raise ValueError("PROBABILITY_BITS cannot give every retained token positive mass.")

    if int(counts.sum()) != total:
        raise RuntimeError("Could not construct a positive quantised PMF.")
    return token_ids.tolist(), counts.tolist()


def _acs2_select_symbol(low: int, high: int, n: int, cdf: Sequence[int], point: int, point_bits: int, q: int) -> int:
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


def acs2_dynamic_message_length(low: int, high: int, denom_bits: int, r: int, r_bits: int) -> tuple[int, int]:
    width = high - low
    max_m = max(1, denom_bits - (width - 1).bit_length() + 1)
    for m in range(max_m, 0, -1):
        k = _acs2_unique_message(low, high, denom_bits, m, r, r_bits)
        if k is not None:
            return m, k
    raise RuntimeError("No uniquely decodable dyadic point was found.")


def acs2_encode_mode3(
    source: AutoregressiveSource,
    *,
    length: int,
    seed: int,
    rotation_numerator: Optional[int] = None,
    rotation_bits: int = ROTATION_BITS,
    progress: bool = True,
) -> tuple[list[int], str, int, int, float, int, int]:
    rng = random.Random(seed)
    r = random_rotation(rng, rotation_bits) if rotation_numerator is None else int(rotation_numerator)
    point_bits = max(PROBABILITY_BITS * int(length) + rotation_bits, 256)
    k, m = rng.getrandbits(point_bits), point_bits
    point, point_bits = _acs2_shifted_point(k, m, r, rotation_bits)

    low, high, n = 0, 1, 0
    tokens: list[int] = []
    chosen_probs: list[float] = []

    source.reset()
    iterator = range(int(length))
    if progress:
        iterator = tqdm(iterator, desc="ACS2 encode mode 3", leave=False)

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _acs2_quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)

        idx = _acs2_select_symbol(low, high, n, cdf, point, point_bits, PROBABILITY_BITS)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1

        token = ids[idx]
        tokens.append(token)
        chosen_probs.append(float(probs_t[(ids_t == token).nonzero(as_tuple=True)[0][0]].item()))
        source.commit(token)

    m, decoded_k = acs2_dynamic_message_length(low, high, n * PROBABILITY_BITS, r, rotation_bits)
    message_bits = format(decoded_k, f"0{m}b")
    entropy_rate = float(np.mean([-math.log2(max(p, 1e-40)) for p in chosen_probs])) if chosen_probs else float("nan")
    return tokens, message_bits, m, n, entropy_rate, r, rotation_bits


def acs2_encode_mode2(
    source: AutoregressiveSource,
    *,
    message_bits: str,
    seed: int,
    rotation_numerator: Optional[int] = None,
    rotation_bits: int = ROTATION_BITS,
    max_steps: Optional[int] = None,
    progress: bool = True,
) -> tuple[list[int], str, int, int, float, int, int]:
    if not message_bits or set(message_bits) - {"0", "1"}:
        raise ValueError("Mode 2 requires a non-empty binary message_bits string.")

    rng = random.Random(seed)
    r = random_rotation(rng, rotation_bits) if rotation_numerator is None else int(rotation_numerator)
    m = len(message_bits)
    k = int(message_bits, 2)
    point, point_bits = _acs2_shifted_point(k, m, r, rotation_bits)

    low, high, n = 0, 1, 0
    tokens: list[int] = []
    chosen_probs: list[float] = []

    source.reset()
    total_steps = int(max_steps) if max_steps is not None else getattr(source, "total_tokens", MAX_GENERATION_STEPS)
    iterator = range(total_steps)
    if progress:
        iterator = tqdm(iterator, desc="ACS2 encode mode 2", leave=False)

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _acs2_quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)

        idx = _acs2_select_symbol(low, high, n, cdf, point, point_bits, PROBABILITY_BITS)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1

        token = ids[idx]
        tokens.append(token)
        chosen_probs.append(float(probs_t[(ids_t == token).nonzero(as_tuple=True)[0][0]].item()))
        source.commit(token)

        if _acs2_unique_message(low, high, n * PROBABILITY_BITS, m, r, rotation_bits) is not None:
            break
    else:
        raise RuntimeError("Generation reached max_steps before the message became uniquely decodable.")

    entropy_rate = float(np.mean([-math.log2(max(p, 1e-40)) for p in chosen_probs])) if chosen_probs else float("nan")
    return tokens, message_bits, m, n, entropy_rate, r, rotation_bits


def acs2_decode_tokens(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    mode: int,
    message_length: Optional[int] = None,
    rotation_numerator: int,
    rotation_bits: int = ROTATION_BITS,
    progress: bool = False,
) -> str:
    if mode in (1, 2) and message_length is None:
        raise ValueError("Modes 1 and 2 require message_length at decode time.")

    low, high, n = 0, 1, 0
    source.reset()
    iterator = tokens
    if progress:
        iterator = tqdm(tokens, desc=f"ACS2 decode mode {mode}", leave=False)

    for token in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _acs2_quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
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
        m, k = acs2_dynamic_message_length(low, high, n * PROBABILITY_BITS, rotation_numerator, rotation_bits)
    else:
        m = int(message_length)
        k = _acs2_unique_message(low, high, n * PROBABILITY_BITS, m, rotation_numerator, rotation_bits)
        if k is None:
            raise RuntimeError("Final interval does not uniquely identify the requested message.")
    return format(k, f"0{m}b")


@dataclass
class HuffmanNode:
    prob: float
    index: int
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None
    search_path: int = 9  # 0 = leaf, -1 = target in left, 1 = target in right, 9 = unknown

    @property
    def is_leaf(self) -> bool:
        return self.index != -1


def _contains_target(node: Optional[HuffmanNode]) -> bool:
    return bool(node is not None and node.search_path != 9)


def build_huffman_tree(indices: Sequence[int], probs: Sequence[float], search_for: Optional[int] = None) -> HuffmanNode:
    if len(indices) != len(probs):
        raise ValueError("indices and probs must have the same length")
    if not indices:
        raise ValueError("Empty vocabulary passed to Huffman tree builder")

    heap: list[tuple[float, int, HuffmanNode]] = []
    counter = 0
    for idx, prob in zip(indices, probs):
        sp = 0 if (search_for is not None and int(idx) == int(search_for)) else 9
        node = HuffmanNode(prob=float(prob), index=int(idx), search_path=sp)
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
    message_decoded_t = ""

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
                raise RuntimeError("Fail to decode: the token is not uniquely decodable.")
            if path0 == -1:
                path_table_swap = {-1: "0", 1: "1"}
            else:
                path_table_swap = {-1: "1", 1: "0"}
            message_decoded_t += path_table_swap[node.search_path]
            node = node.left if node.search_path == -1 else node.right
        else:
            node = node.left if path0 == -1 else node.right

    if node.search_path != 0:
        raise RuntimeError("Fail to decode: the tree did not end at the target leaf.")
    return message_decoded_t


def discop_encode_mode3(
    source: AutoregressiveSource,
    *,
    seed: int,
    max_steps: int,
    progress: bool = True,
) -> tuple[list[int], str, str, int, int, float]:
    rng = random.Random(seed)
    long_budget = max(4096, int(max_steps) * 32)
    message_bits = random_bitstring(long_budget, seed=seed + 17)

    source.reset()
    tokens: list[int] = []
    embedded_bits = 0
    embedding_tokens = 0
    chosen_probs: list[float] = []

    iterator = range(int(max_steps))
    if progress:
        iterator = tqdm(iterator, desc="Discop encode mode 3", leave=False)

    remaining = message_bits
    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids = [int(x) for x in ids_t.tolist()]
        probs = [float(x) for x in probs_t.tolist()]

        real_bits_before = len(remaining)
        token, used = discop_encode_step(ids, probs, remaining, rng)
        tokens.append(token)
        chosen_probs.append(float(probs[ids.index(token)]))
        if real_bits_before > 0 and used > 0:
            embedding_tokens += 1
        if used:
            embedded_bits += min(used, real_bits_before)
            remaining = remaining[min(used, real_bits_before):]
        source.commit(token)

    entropy_rate = float(np.mean([-math.log2(max(p, 1e-40)) for p in chosen_probs])) if chosen_probs else float("nan")
    return tokens, message_bits, "", embedded_bits, embedding_tokens, entropy_rate


def discop_encode_mode2(
    source: AutoregressiveSource,
    *,
    message_bits: str,
    seed: int,
    max_steps: Optional[int] = None,
    progress: bool = True,
) -> tuple[list[int], str, int, int, float]:
    if not message_bits or set(message_bits) - {"0", "1"}:
        raise ValueError("Mode 2 requires a non-empty binary message_bits string.")

    rng = random.Random(seed)
    source.reset()

    if max_steps is None:
        if not hasattr(source, "total_tokens"):
            raise ValueError("max_steps must be provided for a source without total_tokens.")
        max_steps = int(getattr(source, "total_tokens"))

    original_message_bits = message_bits
    tokens: list[int] = []
    embedded_bits = 0
    embedding_tokens = 0
    chosen_probs: list[float] = []

    iterator = range(int(max_steps))
    if progress:
        iterator = tqdm(iterator, desc="Discop encode mode 2", leave=False)

    remaining = message_bits
    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids = [int(x) for x in ids_t.tolist()]
        probs = [float(x) for x in probs_t.tolist()]

        real_bits_before = len(remaining)
        token, used = discop_encode_step(ids, probs, remaining, rng)
        tokens.append(token)
        chosen_probs.append(float(probs[ids.index(token)]))
        if real_bits_before > 0 and used > 0:
            embedding_tokens += 1
        if used:
            embedded_bits += min(used, real_bits_before)
            remaining = remaining[min(used, real_bits_before):]
        source.commit(token)

        candidate = discop_decode_tokens(source, tokens, seed=seed, progress=False, message_length=len(original_message_bits))
        if candidate == original_message_bits:
            break

    if remaining:
        raise RuntimeError("Message was not fully embedded before the sequence budget was exhausted.")

    entropy_rate = float(np.mean([-math.log2(max(p, 1e-40)) for p in chosen_probs])) if chosen_probs else float("nan")
    return tokens, original_message_bits, embedded_bits, embedding_tokens, entropy_rate


def discop_decode_tokens(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    seed: int = 0,
    progress: bool = True,
    message_length: Optional[int] = None,
) -> str:
    rng = random.Random(seed)
    source.reset()

    message_decoded = ""
    iterator = tokens
    if progress:
        iterator = tqdm(tokens, desc="Discop decode", leave=False)

    for token in iterator:
        ids_t, probs_t = source.probabilities()
        ids = [int(x) for x in ids_t.tolist()]
        probs = [float(x) for x in probs_t.tolist()]

        try:
            _ = ids.index(int(token))
        except ValueError as e:
            raise ValueError(f"Token {token} is outside the current PMF.") from e

        message_decoded += discop_decode_step(ids, probs, int(token), rng)
        source.commit(int(token))

    if message_length is not None:
        message_decoded = message_decoded[:message_length]
    return message_decoded


def complete_image_without_message(model: PixelCNN, image_uint8: np.ndarray, context_ratio: float) -> np.ndarray:
    source = PixelCNNPPChannelSource(model, image_uint8, context_ratio)
    total_steps = source.total_tokens
    if total_steps == 0:
        return np.asarray(image_uint8, dtype=np.uint8)

    pbar = tqdm(total=total_steps, desc="PixelCNN++ completion", leave=False)
    with torch.inference_mode():
        for _ in range(total_steps):
            ids_t, probs_t = source.probabilities()
            token = sample_from_probs(ids_t, probs_t)
            source.commit(token)
            pbar.update(1)
    pbar.close()
    return to_uint8_image(source.image)


def reconstruct_image_from_tokens(model: PixelCNN, image_uint8: np.ndarray, context_ratio: float, tokens: Sequence[int]) -> np.ndarray:
    source = PixelCNNPPChannelSource(model, image_uint8, context_ratio)
    source.reset()
    for token in tokens:
        source.commit(int(token))
    return to_uint8_image(source.image)


def batchify_images(images: Sequence[np.ndarray], batch_size: int = 32) -> Iterable[np.ndarray]:
    for i in range(0, len(images), batch_size):
        yield np.stack(images[i : i + batch_size], axis=0)


def numpy_images_to_tensor(images: np.ndarray, device: torch.device, *, uint8: bool = True) -> torch.Tensor:
    arr = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(device)
    if uint8:
        return arr.to(torch.uint8)
    return arr.float()


def compute_psnr_avg(real_images: Sequence[np.ndarray], fake_images: Sequence[np.ndarray]) -> float:
    values: list[float] = []
    for r, f in zip(real_images, fake_images):
        rt = torch.from_numpy(r.transpose(2, 0, 1)).float() / 255.0
        ft = torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
        mse = torch.mean((rt - ft) ** 2).item()
        values.append(float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse))
    return float(np.mean(values)) if values else float("nan")


def build_lpips_metric(device: torch.device):
    try:
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    except Exception as e:
        raise RuntimeError(
            "Could not construct LPIPS. The pretrained AlexNet weights were not available. "
            "Run once with internet access or pre-cache the model weights."
        ) from e
    metric.eval()
    return metric


def compute_lpips_avg(real_images: Sequence[np.ndarray], fake_images: Sequence[np.ndarray], device: torch.device) -> float:
    metric = build_lpips_metric(device)
    vals: list[float] = []
    with torch.inference_mode():
        for r, f in zip(real_images, fake_images):
            rt = torch.from_numpy(r.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 127.5 - 1.0
            ft = torch.from_numpy(f.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 127.5 - 1.0
            vals.append(float(metric(rt, ft).item()))
    return float(np.mean(vals)) if vals else float("nan")


def compute_fid(real_images: Sequence[np.ndarray], fake_images: Sequence[np.ndarray], device: torch.device) -> float:
    try:
        metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    except Exception as e:
        raise RuntimeError(
            "Could not construct FID. The pretrained Inception weights were not available. "
            "Run once with internet access or pre-cache the model weights."
        ) from e
    metric.eval()
    with torch.inference_mode():
        for batch in batchify_images(real_images, batch_size=16):
            metric.update(numpy_images_to_tensor(batch, device, uint8=True), real=True)
        for batch in batchify_images(fake_images, batch_size=16):
            metric.update(numpy_images_to_tensor(batch, device, uint8=True), real=False)
        value = metric.compute()
    return float(value.detach().cpu().item())


# compute fid once on the full 100-image set
def compute_metric_summary(
    real_images: Sequence[np.ndarray],
    fake_images: Sequence[np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    fid = compute_fid(real_images, fake_images, device=device)
    psnr = compute_psnr_avg(real_images, fake_images)
    lpips = compute_lpips_avg(real_images, fake_images, device=device)
    return {"FID": fid, "PSNR": psnr, "LPIPS": lpips}


# mode 3 comparison on 100 images
def run_experiment_1(model: PixelCNN, images: np.ndarray, device: torch.device) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]]]:
    print("\n=== Experiment 1: mode 3 comparison on 100 CIFAR-10 images ===")
    eval_indices = np.linspace(0, len(images) - 1, N_EVAL_IMAGES, dtype=int)
    eval_images = [resize_image_if_needed(images[i]) for i in eval_indices]
    example_indices = eval_indices[:3]

    generated: dict[str, list[np.ndarray]] = {"Baseline": [], "ACS1": [], "Discop": [], "ACS2": []}

    # per-image run records for the three schemes
    rows: list[dict] = []

    for idx_i, img_idx in enumerate(tqdm(eval_indices, desc="Experiment 1 images")):
        image = resize_image_if_needed(images[int(img_idx)])
        source = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
        total_tokens = source.total_tokens

        # baseline PixelCNN++ completion
        t0 = time.perf_counter()
        baseline = complete_image_without_message(model, image, CONTEXT_RATIO)
        baseline_time = time.perf_counter() - t0
        generated["Baseline"].append(baseline)

        # ACS1 mode 3
        source_acs1 = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
        t0 = time.perf_counter()
        img_seed = SEED + int(img_idx)
        acs1_tokens, acs1_message, acs1_m, acs1_N, acs1_entropy = acs1_encode_mode3(
            source_acs1, length=total_tokens, seed=img_seed, progress=False
        )
        acs1_encode_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        acs1_recovered = acs1_decode_tokens(
            PixelCNNPPChannelSource(model, image, CONTEXT_RATIO),
            acs1_tokens,
            mode=3,
            progress=False,
        )
        acs1_extract_time = time.perf_counter() - t0
        acs1_image = reconstruct_image_from_tokens(model, image, CONTEXT_RATIO, acs1_tokens)
        generated["ACS1"].append(acs1_image)

        # Discop mode 3
        source_discop = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
        t0 = time.perf_counter()
        discop_tokens, discop_message, _, discop_embedded_bits, discop_embedding_tokens, discop_entropy = discop_encode_mode3(
            source_discop, seed=img_seed, max_steps=source_discop.total_tokens, progress=False
        )
        discop_encode_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        discop_recovered = discop_decode_tokens(
            PixelCNNPPChannelSource(model, image, CONTEXT_RATIO),
            discop_tokens,
            seed=img_seed,
            progress=False,
            message_length=len(discop_message),
        )
        discop_extract_time = time.perf_counter() - t0
        discop_image = reconstruct_image_from_tokens(model, image, CONTEXT_RATIO, discop_tokens)
        generated["Discop"].append(discop_image)

        # ACS2 mode 3
        source_acs2 = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
        rotation_numerator = derive_rotation(ACS2_KEY, nonce=f"exp1-{int(img_idx)}".encode("utf-8"))
        t0 = time.perf_counter()
        acs2_tokens, acs2_message, acs2_m, acs2_N, acs2_entropy, acs2_r, acs2_rbits = acs2_encode_mode3(
            source_acs2,
            length=source_acs2.total_tokens,
            seed=img_seed,
            rotation_numerator=rotation_numerator,
            rotation_bits=ROTATION_BITS,
            progress=False,
        )
        acs2_encode_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        acs2_recovered = acs2_decode_tokens(
            PixelCNNPPChannelSource(model, image, CONTEXT_RATIO),
            acs2_tokens,
            mode=3,
            rotation_numerator=acs2_r,
            rotation_bits=acs2_rbits,
            progress=False,
        )
        acs2_extract_time = time.perf_counter() - t0
        acs2_image = reconstruct_image_from_tokens(model, image, CONTEXT_RATIO, acs2_tokens)
        generated["ACS2"].append(acs2_image)

        # bits/token and normalized by entropy rate
        def cap_u(bits: int, n_tokens: int, entropy_rate: float) -> tuple[float, float]:
            cap = bits / n_tokens if n_tokens else float("nan")
            util = cap / entropy_rate if entropy_rate and not math.isnan(entropy_rate) else float("nan")
            return cap, util

        acs1_capacity, acs1_util = cap_u(len(acs1_recovered), acs1_N, acs1_entropy)
        discop_capacity, discop_util = cap_u(len(discop_recovered), len(discop_tokens), discop_entropy)
        acs2_capacity, acs2_util = cap_u(len(acs2_recovered), acs2_N, acs2_entropy)

        rows.append(
            {
                "image_index": int(img_idx),
                "scheme": "Baseline",
                "FID": np.nan,
                "PSNR": np.nan,
                "LPIPS": np.nan,
                "capacity_bits_per_token": np.nan,
                "utilization": np.nan,
                "embedding_seconds": baseline_time,
                "extraction_seconds": np.nan,
            }
        )
        rows.append(
            {
                "image_index": int(img_idx),
                "scheme": "ACS1",
                "FID": np.nan,
                "PSNR": np.nan,
                "LPIPS": np.nan,
                "capacity_bits_per_token": acs1_capacity,
                "utilization": acs1_util,
                "embedding_seconds": acs1_encode_time,
                "extraction_seconds": acs1_extract_time,
            }
        )
        rows.append(
            {
                "image_index": int(img_idx),
                "scheme": "Discop",
                "FID": np.nan,
                "PSNR": np.nan,
                "LPIPS": np.nan,
                "capacity_bits_per_token": discop_capacity,
                "utilization": discop_util,
                "embedding_seconds": discop_encode_time,
                "extraction_seconds": discop_extract_time,
            }
        )
        rows.append(
            {
                "image_index": int(img_idx),
                "scheme": "ACS2",
                "FID": np.nan,
                "PSNR": np.nan,
                "LPIPS": np.nan,
                "capacity_bits_per_token": acs2_capacity,
                "utilization": acs2_util,
                "embedding_seconds": acs2_encode_time,
                "extraction_seconds": acs2_extract_time,
            }
        )

        if idx_i == 0:
            print("  ACS1 round-trip OK:", acs1_recovered == acs1_message)
            print("  Discop round-trip OK:", discop_recovered[: len(discop_message)] == discop_message[: len(discop_recovered)])
            print("  ACS2 round-trip OK:", acs2_recovered == acs2_message)

    # # image-level metrics for the generated sets
    # method_metrics = {}
    # for scheme in ("Baseline", "ACS1", "Discop", "ACS2"):
    #     print(f"Computing FID / PSNR / LPIPS for {scheme}...")
    #     method_metrics[scheme] = compute_metric_summary(eval_images, generated[scheme], device=device)

    # pretrained-model output (Baseline) is the reference
    reference_images = generated["Baseline"]

    print("\n=== Metric input diagnostics ===")

    for scheme in ("Baseline", "ACS1", "Discop", "ACS2"):
        imgs = generated[scheme]

        print(f"\n{scheme}:")
        print(f"  Number of images: {len(imgs)}")

        if len(imgs) > 0:
            print(f"  Shape[0]: {imgs[0].shape}")
            print(f"  Dtype[0]: {imgs[0].dtype}")
            print(f"  Min[0]: {np.min(imgs[0])}")
            print(f"  Max[0]: {np.max(imgs[0])}")
            print(f"  Mean[0]: {np.mean(imgs[0])}")
            print(f"  NaN present: {np.isnan(imgs[0]).any()}")
            print(f"  Inf present: {np.isinf(imgs[0]).any()}")

            print(
                f"  All finite: "
                f"{all(np.isfinite(img).all() for img in imgs)}"
            )

    print("\nReference vs schemes:")
    for scheme in ("ACS1", "Discop", "ACS2"):
        print(
            f"  Baseline vs {scheme}: "
            f"{len(reference_images)} vs {len(generated[scheme])}"
        )

    # pretrained-model output (Baseline) is the reference
    method_metrics = {}
    reference_images = generated["Baseline"]
    for scheme in ("ACS1", "Discop", "ACS2"):
        print(f"Computing FID / PSNR / LPIPS for {scheme} vs. pretrained PixelCNN++...")
        method_metrics[scheme] = compute_metric_summary(
            reference_images,
            generated[scheme],
            device=device,
        )

    # # global image-set metrics
    # summary_rows: list[dict] = []
    # for scheme in ("Baseline", "ACS1", "Discop", "ACS2"):
    #     subset = pd.DataFrame(rows)[lambda d: d["scheme"] == scheme]
    #     summary_rows.append(
    #         {
    #             "scheme": scheme,
    #             "FID": method_metrics[scheme]["FID"],
    #             "PSNR": method_metrics[scheme]["PSNR"],
    #             "LPIPS": method_metrics[scheme]["LPIPS"],
    #             "capacity_bits_per_token": float(subset["capacity_bits_per_token"].mean(skipna=True)),
    #             "utilization": float(subset["utilization"].mean(skipna=True)),
    #             "embedding_seconds": float(subset["embedding_seconds"].mean(skipna=True)),
    #             "extraction_seconds": float(subset["extraction_seconds"].mean(skipna=True)),
    #         }
    #     )

    summary_rows: list[dict] = []
    for scheme in ("ACS1", "Discop", "ACS2"):
        subset = pd.DataFrame(rows)[lambda d: d["scheme"] == scheme]
        summary_rows.append(
            {
                "scheme": scheme,
                "FID": method_metrics[scheme]["FID"],
                "PSNR": method_metrics[scheme]["PSNR"],
                "LPIPS": method_metrics[scheme]["LPIPS"],
                "capacity_bits_per_token": float(
                    subset["capacity_bits_per_token"].mean(skipna=True)
                ),
                "utilization": float(
                    subset["utilization"].mean(skipna=True)
                ),
                "embedding_seconds": float(
                    subset["embedding_seconds"].mean(skipna=True)
                ),
                "extraction_seconds": float(
                    subset["extraction_seconds"].mean(skipna=True)
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(rows)
    save_table(summary_df, "experiment1_mode3_summary")
    save_table(detail_df, "experiment1_mode3_per_image_details")

    # exmaple images
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    column_titles = ["Original", "PixelCNN++", "ACS1", "Discop", "ACS2"]
    for j, title in enumerate(column_titles):
        axes[0, j].set_title(title)
    for row_i, img_idx in enumerate(example_indices):
        axes[row_i, 0].imshow(eval_images[row_i])
        axes[row_i, 1].imshow(generated["Baseline"][row_i])
        axes[row_i, 2].imshow(generated["ACS1"][row_i])
        axes[row_i, 3].imshow(generated["Discop"][row_i])
        axes[row_i, 4].imshow(generated["ACS2"][row_i])
        for col_i in range(5):
            axes[row_i, col_i].axis("off")
        axes[row_i, 0].set_ylabel(f"idx {int(img_idx)}")
    fig.suptitle("PixelCNN++ CIFAR-10 completion with mode 3", y=1.02)
    fig.tight_layout()
    show_or_save_figure(fig, "experiment1_mode3_examples.png")
    plt.close(fig)

    return summary_df, generated


# mode 2 for m = 1, ... , 100
def run_experiment_2(model: PixelCNN, images: np.ndarray, device: torch.device) -> pd.DataFrame:
    print("\n=== Experiment 2: mode 2 sweep on m = 1..100 ===")
    rng = np.random.default_rng(SEED + 999)
    trial_records: list[dict] = []

    m_values = list(range(1, M_MAX + 1))
    outer = tqdm(m_values, desc="Experiment 2 message lengths")

    for m in outer:
        acc = {
            "ACS1": {"R": [], "u": [], "entropy": []},
            "Discop": {"R": [], "u": [], "entropy": []},
            "ACS2": {"R": [], "u": [], "entropy": []},
        }

        for trial in range(N_EXPERIMENT2_TRIALS):
            img_idx = int(rng.integers(0, len(images)))
            image = resize_image_if_needed(images[img_idx])
            trial_seed = SEED + 10_000 + 100 * m + trial
            message_bits = random_bitstring(m, seed=trial_seed)

            # each trial shares the same image/message across schemes for fairness
            # ACS1
            s1 = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
            tokens1, msg1, m1, N1, entropy1 = acs1_encode_mode2(
                s1,
                message_bits=message_bits,
                seed=trial_seed,
                max_steps=s1.total_tokens,
                progress=False,
            )
            R1 = m1 / len(tokens1)
            u1 = R1 / entropy1 if entropy1 and not math.isnan(entropy1) else float("nan")
            acc["ACS1"]["R"].append(R1)
            acc["ACS1"]["u"].append(u1)
            acc["ACS1"]["entropy"].append(entropy1)

            # Discop
            s2 = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
            tokens2, msg2, embedded2, embtok2, entropy2 = discop_encode_mode2(
                s2,
                message_bits=message_bits,
                seed=trial_seed,
                max_steps=s2.total_tokens,
                progress=False,
            )
            R2 = len(message_bits) / len(tokens2)
            u2 = R2 / entropy2 if entropy2 and not math.isnan(entropy2) else float("nan")
            acc["Discop"]["R"].append(R2)
            acc["Discop"]["u"].append(u2)
            acc["Discop"]["entropy"].append(entropy2)

            # ACS2
            s3 = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
            rot = derive_rotation(ACS2_KEY, nonce=f"exp2-{m}-{trial}".encode("utf-8"))
            tokens3, msg3, m3, N3, entropy3, r3, rbits3 = acs2_encode_mode2(
                s3,
                message_bits=message_bits,
                seed=trial_seed,
                rotation_numerator=rot,
                rotation_bits=ROTATION_BITS,
                max_steps=s3.total_tokens,
                progress=False,
            )
            R3 = m3 / len(tokens3)
            u3 = R3 / entropy3 if entropy3 and not math.isnan(entropy3) else float("nan")
            acc["ACS2"]["R"].append(R3)
            acc["ACS2"]["u"].append(u3)
            acc["ACS2"]["entropy"].append(entropy3)

        trial_records.append(
            {
                "m": m,
                "ACS1_R": float(np.mean(acc["ACS1"]["R"])),
                "ACS1_entropy": float(np.mean(acc["ACS1"]["entropy"])),
                "ACS1_u": float(np.mean(acc["ACS1"]["u"])),
                "Discop_R": float(np.mean(acc["Discop"]["R"])),
                "Discop_entropy": float(np.mean(acc["Discop"]["entropy"])),
                "Discop_u": float(np.mean(acc["Discop"]["u"])),
                "ACS2_R": float(np.mean(acc["ACS2"]["R"])),
                "ACS2_entropy": float(np.mean(acc["ACS2"]["entropy"])),
                "ACS2_u": float(np.mean(acc["ACS2"]["u"])),
            }
        )

    summary_df = pd.DataFrame(trial_records)
    save_table(summary_df, "experiment2_mode2_summary")

    # R vs m
    # fig1, ax1 = plt.subplots(figsize=(9, 6))
    # ax1.plot(summary_df["m"], summary_df["ACS1_R"], label="ACS1 R", linewidth=2)
    # ax1.plot(summary_df["m"], summary_df["Discop_R"], label="Discop R", linewidth=2)
    # ax1.plot(summary_df["m"], summary_df["ACS2_R"], label="ACS2 R", linewidth=2)
    # ax1.plot(summary_df["m"], summary_df["ACS1_entropy"], label="ACS1 entropy rate", linestyle="--")
    # ax1.plot(summary_df["m"], summary_df["Discop_entropy"], label="Discop entropy rate", linestyle="--")
    # ax1.plot(summary_df["m"], summary_df["ACS2_entropy"], label="ACS2 entropy rate", linestyle="--")
    # ax1.set_xlabel("Message length m (bits)")
    # ax1.set_ylabel("Bits per token")
    # ax1.set_title("Empirical embedding rate and entropy rate vs. message length")
    # ax1.legend(ncol=2, fontsize=9)
    # ax1.grid(True, alpha=0.25)
    # fig1.tight_layout()
    # show_or_save_figure(fig1, "experiment2_R_vs_m.png")
    # plt.close(fig1)

    fig1, ax1 = plt.subplots(figsize=(9, 6))

    # empirical embedding rate R
    line_acs1, = ax1.plot(
        summary_df["m"], summary_df["ACS1_R"],
        label="ACS1 R", linewidth=2
    )
    line_discop, = ax1.plot(
        summary_df["m"], summary_df["Discop_R"],
        label="Discop R", linewidth=2
    )
    line_acs2, = ax1.plot(
        summary_df["m"], summary_df["ACS2_R"],
        label="ACS2 R", linewidth=2
    )

    # entropy rate
    ax1.plot(
        summary_df["m"], summary_df["ACS1_entropy"],
        label="ACS1 entropy rate",
        linestyle="--",
        color=line_acs1.get_color()
    )
    ax1.plot(
        summary_df["m"], summary_df["Discop_entropy"],
        label="Discop entropy rate",
        linestyle="--",
        color=line_discop.get_color()
    )
    ax1.plot(
        summary_df["m"], summary_df["ACS2_entropy"],
        label="ACS2 entropy rate",
        linestyle="--",
        color=line_acs2.get_color()
    )

    ax1.set_xlabel("Message length m (bits)")
    ax1.set_ylabel("Bits per token")
    ax1.set_title("Empirical embedding rate and entropy rate vs. message length")
    ax1.legend(ncol=2, fontsize=9)
    ax1.grid(True, alpha=0.25)

    fig1.tight_layout()
    show_or_save_figure(fig1, "experiment2_R_vs_m.png")
    plt.close(fig1)

    # utilization vs m
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(summary_df["m"], summary_df["ACS1_u"], label="ACS1 u", linewidth=2)
    ax2.plot(summary_df["m"], summary_df["Discop_u"], label="Discop u", linewidth=2)
    ax2.plot(summary_df["m"], summary_df["ACS2_u"], label="ACS2 u", linewidth=2)
    ax2.set_xlabel("Message length m (bits)")
    ax2.set_ylabel("Utilization u = R / entropy rate")
    ax2.set_title("Utilization vs. message length")
    ax2.legend()
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    show_or_save_figure(fig2, "experiment2_u_vs_m.png")
    plt.close(fig2)

    return summary_df


def main() -> None:
    set_seed(SEED)
    device = choose_device()
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)

    # print(f"Loading CIFAR-10 test batch: {DATA_FILE}")
    # images, labels = load_cifar10_test_batch(DATA_FILE)
    # print(f"Loaded {len(images)} test images.")

    print(f"Streaming {N_EVAL_IMAGES} CIFAR-10 test images...")
    images, labels = load_cifar10_test_subset(
        n_images=N_EVAL_IMAGES,
        seed=SEED,
    )
    print(f"Loaded {len(images)} test images.")

    print(f"Context ratio = {CONTEXT_RATIO:.2f}")
    print(f"Available channel tokens per image = {(32 - int(round(32 * CONTEXT_RATIO))) * 32 * 3}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    eval_indices = rng.choice(len(images), size=N_EVAL_IMAGES, replace=False)
    eval_images = images[eval_indices]

    # experiment 1: mode 3 across all methods
    try:
        run_experiment_1(model, eval_images, device)
    except Exception as e:
        print("Experiment 1 failed:", e)
        raise

    # experiment 2: mode 2 sweep across message lengths
    try:
        run_experiment_2(model, images, device)
    except Exception as e:
        print("Experiment 2 failed:", e)
        raise

    print(f"\nAll outputs were written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()