from __future__ import annotations
from dataclasses import asdict, dataclass
import pickle
import random
import sys
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ACS2_ROOT = SCRIPT_DIR.parent
PIXELCNN_DIR = ACS2_ROOT / "PixelCNN++"
DATA_FILE = ACS2_ROOT / "cifar10_data" / "cifar-10-batches-py" / "test_batch"
CHECKPOINT = PIXELCNN_DIR / "checkpoints" / "pixelcnnpp_cifar10.pth"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_FIGURE = OUTPUT_DIR / "acs1_pixelcnnpp_comparison.png"

if str(ACS2_ROOT) not in sys.path:
    sys.path.insert(0, str(ACS2_ROOT))
if str(PIXELCNN_DIR) not in sys.path:
    sys.path.insert(0, str(PIXELCNN_DIR))

from model import PixelCNN


SEED = 666666
CONTEXT_RATIO = 0.5
IMAGE_INDEX = 23
MODE12_MESSAGE_BITS = 32
MODE3_TOKEN_BUDGET = None # defaults to all remaining image tokens
PROBABILITY_BITS = 32
MAX_GENERATION_STEPS = 1_024
SHOW_PLOTS_IN_COLAB = True


class AutoregressiveSource(Protocol):
    eos_token_id: Optional[int]

    def reset(self) -> None: ...
    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def commit(self, token_id: int) -> None: ...


@dataclass
class ACS1Result:
    tokens: list[int]
    message_bits: str
    m: int
    N: int
    mode: int
    embedding_rate: float

    def summary(self) -> dict:
        d = asdict(self)
        d["tokens"] = f"{len(self.tokens)} token ids"
        d["message_bits"] = f"{self.m} bits"
        return d


def choose_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"ACS1 device: {device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    )
    return device


def running_in_colab() -> bool:
    try:
        import google.colab
        return True
    except Exception:
        return False


def show_or_save_figure(fig, filename: Path) -> None:
    if SHOW_PLOTS_IN_COLAB and running_in_colab():
        fig.show()
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {filename}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_bitstring(n_bits: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join("1" if rng.getrandbits(1) else "0" for _ in range(n_bits))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# find up to limit dyadic k/2**m points in half-open intervals
def _message_indices(intervals: Iterable[tuple[int, int, int]], m: int, limit: int = 2) -> list[int]:
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


def _unique_message(low: int, high: int, denom_bits: int, m: int) -> Optional[int]:
    values = _message_indices([(low, high, denom_bits)], m)
    return values[0] if len(values) == 1 else None


# largest m for which exactly one dyadic point exists in [low, high)
def _dynamic_message_length(low: int, high: int, denom_bits: int) -> tuple[int, int]:
    width = high - low
    if width <= 0:
        raise ValueError("Invalid interval.")
    # no finer grid than this can have only one point.
    max_m = max(1, denom_bits - (width - 1).bit_length() + 1)
    for m in range(max_m, 0, -1):
        k = _unique_message(low, high, denom_bits, m)
        if k is not None:
            return m, k
    raise RuntimeError("No uniquely decodable dyadic point was found.")


def _quantize_distribution(
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
                    if not needed:
                        break
            if not changed:
                raise ValueError("PROBABILITY_BITS cannot give every retained token positive mass.")

    if int(counts.sum()) != total:
        raise RuntimeError("Could not construct a positive quantized PMF.")
    return token_ids.tolist(), counts.tolist()


def _select_symbol(
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
        if (sub_low << (common_bits - next_bits)) <= point_common < (sub_hi << (common_bits - next_bits)):
            return i
    raise RuntimeError("Message point fell outside the quantized PMF.")


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
        f"Checkpoint architecture: nr_resnet={nr_resnet}, "
        f"nr_filters={nr_filters}, nr_logistic_mix={nr_logistic_mix}"
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


def load_cifar10_test_image(path: Path, index: int = 0) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"CIFAR-10 test batch not found: {path}")

    with path.open("rb") as f:
        obj = pickle.load(f, encoding="bytes")

    data = obj[b"data"] if b"data" in obj else obj["data"]
    image = np.asarray(data[index], dtype=np.uint8).reshape(3, 32, 32).transpose(1, 2, 0)
    return image


def to_tensor(image_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    if image_uint8.shape != (32, 32, 3):
        raise ValueError(f"Expected a 32x32 RGB image, got {image_uint8.shape}.")
    x = torch.from_numpy(image_uint8.transpose(2, 0, 1)).float() / 127.5 - 1.0
    return x.unsqueeze(0).to(device)


def to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    arr = ((tensor[0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5)
    return np.clip(arr, 0, 255).astype(np.uint8)


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


def _component_bin_masses(
    candidate_values: torch.Tensor,
    means: torch.Tensor,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    return _logistic_bin_mass(
        candidate_values,
        means.to(torch.float64),
        log_scales.to(torch.float64),
    )


# discrete PixelCNN++ pmf for one rgb channel
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


def sample_from_probs(ids: torch.Tensor, probs: torch.Tensor) -> int:
    idx = torch.multinomial(probs, 1).item()
    return int(ids[idx].item())


def complete_image_without_message(
    model: PixelCNN,
    image_uint8: np.ndarray,
    context_ratio: float,
    device: torch.device,
) -> np.ndarray:
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
    return


def encode_acs1(
    source: AutoregressiveSource,
    mode: int,
    *,
    message_bits: Optional[str] = None,
    length: Optional[int] = None,
    seed: int = 0,
    max_steps: Optional[int] = None,
    progress: bool = True,
) -> ACS1Result:
    if mode not in (1, 2, 3):
        raise ValueError("mode must be 1, 2, or 3")

    if mode in (1, 2):
        if not message_bits or set(message_bits) - {"0", "1"}:
            raise ValueError("Modes 1 and 2 require a non-empty binary message_bits string.")
    if mode == 3 and (length is None or length < 1):
        raise ValueError("Mode 3 requires a positive fixed token budget.")

    rng = random.Random(seed)

    if mode == 3:
        point_bits = max(PROBABILITY_BITS * int(length) + 256, 256)
        k = rng.getrandbits(point_bits)
        m = point_bits
    else:
        m = len(message_bits)
        k = int(message_bits, 2)

    low, high, n = 0, 1, 0
    tokens: list[int] = []

    source.reset()
    total_steps = int(length) if mode == 3 else (max_steps if max_steps is not None else getattr(source, "total_tokens", MAX_GENERATION_STEPS))
    iterator = tqdm(range(total_steps), desc=f"ACS1 encode mode {mode}", disable=not progress)

    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids, counts = _quantize_distribution(ids_t, probs_t, PROBABILITY_BITS)
        cdf = [0]
        for c in counts:
            cdf.append(cdf[-1] + c)

        idx = _select_symbol(low, high, n, cdf, k, m, PROBABILITY_BITS)
        old_low, width = low, high - low
        low = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx]
        high = old_low * (1 << PROBABILITY_BITS) + width * cdf[idx + 1]
        n += 1

        token = ids[idx]
        tokens.append(token)
        source.commit(token)

        if mode == 2:
            if _unique_message(low, high, n * PROBABILITY_BITS, m) is not None:
                break

        if mode == 1 and source.eos_token_id is not None and token == source.eos_token_id:
            break

    else:
        if mode == 2:
            raise RuntimeError("Generation reached max_steps before the message became uniquely decodable.")

    if mode == 3:
        m, decoded_k = _dynamic_message_length(low, high, n * PROBABILITY_BITS)
        message_bits = format(decoded_k, f"0{m}b")
    else:
        decoded_k = _unique_message(low, high, n * PROBABILITY_BITS, m)
        if decoded_k is None:
            raise RuntimeError("Final interval does not uniquely identify the requested message.")

    embedding_rate = (m / n) if n else 0.0
    return ACS1Result(tokens=tokens, message_bits=message_bits or "", m=m, N=n, mode=mode, embedding_rate=embedding_rate)


def decode_acs1(
    source: AutoregressiveSource,
    tokens: Sequence[int],
    *,
    mode: int,
    message_length: Optional[int] = None,
    progress: bool = True,
) -> str:
    if mode in (1, 2) and message_length is None:
        raise ValueError("Modes 1 and 2 require message_length at decode time.")

    low, high, n = 0, 1, 0
    source.reset()

    iterator = tqdm(tokens, desc=f"ACS1 decode mode {mode}", disable=not progress)
    for token in iterator:
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
        m, k = _dynamic_message_length(low, high, n * PROBABILITY_BITS)
    else:
        m = int(message_length)
        k = _unique_message(low, high, n * PROBABILITY_BITS, m)
        if k is None:
            raise RuntimeError("Final interval does not uniquely identify the requested message.")

    return format(k, f"0{m}b")


def complete_image_with_acs1(
    model: PixelCNN,
    image_uint8: np.ndarray,
    context_ratio: float,
    mode: int,
    *,
    message_bits: str | None = None,
    length: int | None = None,
    seed: int = SEED,
    progress: bool = True,
) -> tuple[np.ndarray, str, int, int]:
    source = PixelCNNPPChannelSource(model, image_uint8, context_ratio)
    max_steps = source.total_tokens

    result = encode_acs1(
        source,
        mode,
        message_bits=message_bits,
        length=length,
        seed=seed,
        max_steps=max_steps,
        progress=progress,
    )

    # reconstruct the full image
    source.reset()
    with torch.inference_mode():
        for token in result.tokens:
            ids_t, probs_t = source.probabilities()
            legal = (ids_t == int(token)).nonzero(as_tuple=True)[0]
            if legal.numel() == 0:
                raise RuntimeError(f"ACS1 produced an invalid token {token} at the current step.")
            source.commit(int(token))

    image = to_uint8_image(source.image)
    recovered = decode_acs1(
        PixelCNNPPChannelSource(model, image_uint8, context_ratio),
        result.tokens,
        mode=mode,
        message_length=(result.m if mode in (1, 2) else None),
        progress=progress,
    )

    return image, result.message_bits, result.N, len(recovered)


def make_four_panel_figure(
    original: np.ndarray,
    completed: np.ndarray,
    mode1_image: np.ndarray,
    mode3_image: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    panels = [
        (original, "Original"),
        (completed, "PixelCNN++ completion"),
        (mode1_image, "ACS1 mode 1"),
        (mode3_image, "ACS1 mode 3"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def main() -> None:
    set_seed(SEED)
    device = choose_device()

    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)

    print(f"Loading CIFAR-10 image: {DATA_FILE} (index={IMAGE_INDEX})")
    image = load_cifar10_test_image(DATA_FILE, index=IMAGE_INDEX)

    prefix_rows = int(round(32 * CONTEXT_RATIO))
    total_available_tokens = (32 - prefix_rows) * 32 * 3
    print(f"Context ratio = {CONTEXT_RATIO:.2f} -> fixed prefix rows = {prefix_rows}")
    print(f"Available channel tokens to generate: {total_available_tokens}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # no message
    print("Running PixelCNN++ completion without message...")
    completed = complete_image_without_message(
        model=model,
        image_uint8=image,
        context_ratio=CONTEXT_RATIO,
        device=device,
    )
    Image.fromarray(completed).save(OUTPUT_DIR / "pixelcnnpp_completion.png")
    print(f"Saved: {OUTPUT_DIR / 'pixelcnnpp_completion.png'}")

    message_bits = random_bitstring(MODE12_MESSAGE_BITS, seed=SEED + 1)

    print(f"Running ACS1 mode 1 with a {MODE12_MESSAGE_BITS}-bit message...")
    mode1_image, recovered1, n1, _ = complete_image_with_acs1(
        model=model,
        image_uint8=image,
        context_ratio=CONTEXT_RATIO,
        mode=1,
        message_bits=message_bits,
        seed=SEED,
        progress=True,
    )
    assert recovered1 == message_bits, "ACS1 mode 1 round-trip failed."
    Image.fromarray(mode1_image).save(OUTPUT_DIR / "acs1_mode1.png")
    print(f"Saved: {OUTPUT_DIR / 'acs1_mode1.png'}")
    print(f"Mode 1 tokens generated: {n1}")

    print(f"Running ACS1 mode 2 with the same {MODE12_MESSAGE_BITS}-bit message...")
    mode2_image, recovered2, n2, _ = complete_image_with_acs1(
        model=model,
        image_uint8=image,
        context_ratio=CONTEXT_RATIO,
        mode=2,
        message_bits=message_bits,
        seed=SEED,
        progress=True,
    )
    assert recovered2 == message_bits, "ACS1 mode 2 round-trip failed."
    Image.fromarray(mode2_image).save(OUTPUT_DIR / "acs1_mode2.png")
    print(f"Saved: {OUTPUT_DIR / 'acs1_mode2.png'}")
    print(f"Mode 2 tokens generated: {n2}")

    mode3_length = MODE3_TOKEN_BUDGET if MODE3_TOKEN_BUDGET is not None else total_available_tokens
    print(f"Running ACS1 mode 3 with token budget = {mode3_length}...")
    mode3_image, recovered3, n3, recovered3_len = complete_image_with_acs1(
        model=model,
        image_uint8=image,
        context_ratio=CONTEXT_RATIO,
        mode=3,
        length=mode3_length,
        seed=SEED,
        progress=True,
    )
    Image.fromarray(mode3_image).save(OUTPUT_DIR / "acs1_mode3.png")
    print(f"Saved: {OUTPUT_DIR / 'acs1_mode3.png'}")
    print(f"Mode 3 tokens generated: {n3}")
    print(f"Mode 3 recovered message length: {len(recovered3)} bits")
    print(f"Mode 3 decoded length reported by decoder: {recovered3_len} bits")

    fig = make_four_panel_figure(image, completed, mode1_image, mode3_image)
    show_or_save_figure(fig, OUTPUT_FIGURE)
    print(f"Saved comparison figure: {OUTPUT_FIGURE}")

    if not running_in_colab():
        plt.close(fig)


if __name__ == "__main__":
    main()