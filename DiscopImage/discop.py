from __future__ import annotations
import heapq
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PIXELCNN_DIR = PROJECT_DIR / "PixelCNN++"
for _p in (PROJECT_DIR, PIXELCNN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model import PixelCNN


ROOT_DIR = SCRIPT_DIR
DATA_FILE = PROJECT_DIR / "cifar10_data" / "cifar-10-batches-py" / "test_batch"
CHECKPOINT = PROJECT_DIR / "PixelCNN++" / "checkpoints" / "pixelcnnpp_cifar10.pth"
OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_FIGURE = OUTPUT_DIR / "discop_pixelcnnpp_test.png"

CONTEXT_RATIO = 0.50
IMAGE_INDEX = 233

MODE12_MESSAGE_BITS = 32
MODE3_BIT_BUDGET_MULTIPLIER = 32

SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoregressiveSource(Protocol):

    eos_token_id: Optional[int]

    def reset(self) -> None: ...
    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def commit(self, token_id: int) -> None: ...


def choose_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    )
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def random_bitstring(n_bits: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join("1" if rng.getrandbits(1) else "0" for _ in range(n_bits))


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


def sample_from_probs(ids: torch.Tensor, probs: torch.Tensor, rng: Optional[torch.Generator] = None) -> int:
    if rng is None:
        idx = torch.multinomial(probs, 1).item()
    else:
        idx = torch.multinomial(probs, 1, generator=rng).item()
    return int(ids[idx].item())


# one-channel-at-a-time image completion
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

    candidate_values = torch.linspace(
        -1.0,
        1.0,
        256,
        device=device,
        dtype=torch.float64,
    ).view(256, 1)

    red_means = means[0].to(torch.float64).view(1, -1)
    red_log_scales = log_scales[0].to(torch.float64).view(1, -1)
    red_component_probs = _component_bin_masses(candidate_values, red_means, red_log_scales)

    if channel_index == 0:
        probs = (red_component_probs * mix_weights.view(1, -1)).sum(dim=1)

    elif channel_index == 1:
        x_r = current_pixel_norm[0].to(torch.float64)
        green_means = (
            means[1].to(torch.float64) + coeffs[0].to(torch.float64) * x_r
        ).view(1, -1)
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

        green_means = (
            means[1].to(torch.float64) + coeffs[0].to(torch.float64) * x_r
        ).view(1, -1)
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


# huffman tree
@dataclass
class HuffmanNode:
    prob: float
    index: int
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None
    search_path: int = 9 # 0 = leaf, -1 = target in left, 1 = target in right, 9 = unknown

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


# encode one token and return
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
            # if no user message left, sample naturally from the unrotated copy
            node = node.right if path0 == 1 else node.left
        else:
            bit = message_bits[n_bits]
            chosen_path = path0 if bit == "0" else path1
            node = node.right if chosen_path == 1 else node.left

        if path0 != path1:
            n_bits += 1

    return int(node.index), n_bits


# recover the embedded bit chunk for one stego token
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


@dataclass
class DiscopResult:
    tokens: list[int]
    message_bits: str
    recovered_bits: str
    embedded_bits: int
    embedding_tokens: int
    embedding_rate: float
    mode: int


def encode_discop(
    source: AutoregressiveSource,
    mode: int,
    *,
    message_bits: Optional[str] = None,
    seed: int = 0,
    max_steps: Optional[int] = None,
    progress: bool = True,
) -> DiscopResult:

    if mode not in (1, 2, 3):
        raise ValueError("mode must be 1, 2, or 3")
    if mode in (1, 2) and (not message_bits or set(message_bits) - {"0", "1"}):
        raise ValueError("Modes 1 and 2 require a non-empty binary message_bits string.")

    rng = random.Random(seed)
    source.reset()

    if max_steps is None:
        if not hasattr(source, "total_tokens"):
            raise ValueError("max_steps must be provided for a source without total_tokens.")
        max_steps = int(getattr(source, "total_tokens"))

    if mode == 3:
        long_budget = max(4096, int(max_steps) * MODE3_BIT_BUDGET_MULTIPLIER)
        message_bits = random_bitstring(long_budget, seed=seed + 17)

    assert message_bits is not None
    original_message_bits = message_bits

    tokens: list[int] = []
    embedded_bits = 0
    embedding_tokens = 0

    iterator = range(int(max_steps))
    if progress:
        iterator = tqdm(iterator, desc=f"Discop encode mode {mode}", leave=False)

    remaining = message_bits
    for _ in iterator:
        ids_t, probs_t = source.probabilities()
        ids = [int(x) for x in ids_t.tolist()]
        probs = [float(x) for x in probs_t.tolist()]

        real_bits_before = len(remaining)
        token, used = discop_encode_step(ids, probs, remaining, rng)
        tokens.append(token)
        if real_bits_before > 0 and used > 0:
            embedding_tokens += 1
        if used:
            embedded_bits += min(used, real_bits_before)
            remaining = remaining[min(used, real_bits_before):]
        source.commit(token)

        # if mode == 2 and not remaining:
        #     break

        if mode == 2:
            candidate = decode_discop(
                source=source,
                tokens=tokens,
                seed=seed,
                progress=False,
                message_length=len(original_message_bits),
            )
            if candidate == original_message_bits:
                break

    if mode in (1, 2) and remaining:
        raise RuntimeError("Message was not fully embedded before the sequence budget was exhausted.")

    # recovered_bits = decode_discop(
    #     source=source,
    #     tokens=tokens,
    #     seed=seed,
    #     progress=progress,
    #     message_length=(len(original_message_bits) if mode == 1 else None),
    # )

    recovered_bits = decode_discop(
        source=source,
        tokens=tokens,
        seed=seed,
        progress=progress,
        message_length=len(original_message_bits),
    )

    # if mode == 1:
    #     recovered_bits = recovered_bits[: len(original_message_bits)]

    embedding_rate = embedded_bits / embedding_tokens if embedding_tokens > 0 else 0.0

    return DiscopResult(
        tokens=tokens,
        message_bits=original_message_bits,
        recovered_bits=recovered_bits,
        embedded_bits=embedded_bits if mode != 1 else len(original_message_bits),
        embedding_tokens=embedding_tokens,
        embedding_rate=embedding_rate,
        mode=mode,
    )


def decode_discop(
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


def complete_image_without_message(
    model: PixelCNN,
    image_uint8: np.ndarray,
    context_ratio: float,
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
    return to_uint8_image(source.image)


def complete_image_with_discop(
    model: PixelCNN,
    image_uint8: np.ndarray,
    context_ratio: float,
    mode: int,
    *,
    message_bits: Optional[str] = None,
    seed: int = SEED,
    progress: bool = True,
) -> tuple[np.ndarray, DiscopResult, PixelCNNPPChannelSource]:
    source = PixelCNNPPChannelSource(model, image_uint8, context_ratio)
    max_steps = source.total_tokens
    result = encode_discop(
        source,
        mode,
        message_bits=message_bits,
        seed=seed,
        max_steps=max_steps,
        progress=progress,
    )

    # reconstruct the image from the returned tokens
    source.reset()
    decode_iterator = result.tokens
    if progress:
        decode_iterator = tqdm(result.tokens, desc=f"Reconstruct mode {mode}", leave=False)
    for token in decode_iterator:
        source.commit(int(token))

    image = to_uint8_image(source.image)
    return image, result, source


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
        (mode1_image, "Discop mode 1"),
        (mode3_image, "Discop mode 3"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"Saved figure: {path}")


def main() -> None:
    set_seed(SEED)

    device = choose_device()
    print(f"Loading checkpoint: {CHECKPOINT}")
    model = load_model(CHECKPOINT, device)

    print(f"Loading CIFAR-10 image: {DATA_FILE} (index={IMAGE_INDEX})")
    image = load_cifar10_test_image(DATA_FILE, index=IMAGE_INDEX)

    prefix_rows = int(round(32 * CONTEXT_RATIO))
    total_available_tokens = (32 - prefix_rows) * 32 * 3
    print(f"Context ratio = {CONTEXT_RATIO:.2f} -> prefix rows = {prefix_rows}")
    print(f"Available channel tokens to generate: {total_available_tokens}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # baseline
    print("Running PixelCNN++ completion without message...")
    completed = complete_image_without_message(model=model, image_uint8=image, context_ratio=CONTEXT_RATIO)
    Image.fromarray(completed).save(OUTPUT_DIR / "pixelcnnpp_completion.png")
    print(f"Saved: {OUTPUT_DIR / 'pixelcnnpp_completion.png'}")

    # mode 1: embed a user specified message, then complete the image
    message_bits = random_bitstring(MODE12_MESSAGE_BITS, seed=SEED + 1)
    print(f"Running Discop mode 1 with a {MODE12_MESSAGE_BITS}-bit message...")
    mode1_image, mode1_result, _ = complete_image_with_discop(
        model=model,
        image_uint8=image,
        context_ratio=CONTEXT_RATIO,
        mode=1,
        message_bits=message_bits,
        seed=SEED,
        progress=True,
    )
    assert mode1_result.recovered_bits == message_bits, "Discop mode 1 round-trip failed."
    Image.fromarray(mode1_image).save(OUTPUT_DIR / "discop_mode1.png")
    print(f"Saved: {OUTPUT_DIR / 'discop_mode1.png'}")
    print(f"Mode 1 embedded bits: {mode1_result.embedded_bits}")
    print(f"Mode 1 embedding rate: {mode1_result.embedding_rate:.4f} bits/token")

    # mode 2: stop as soon as the message is decodable
    print(f"Running Discop mode 2 with the same {MODE12_MESSAGE_BITS}-bit message...")
    mode2_source = PixelCNNPPChannelSource(model, image, CONTEXT_RATIO)
    mode2_result = encode_discop(
        mode2_source,
        2,
        message_bits=message_bits,
        seed=SEED,
        max_steps=mode2_source.total_tokens,
        progress=True,
    )
    assert mode2_result.recovered_bits == message_bits[: len(mode2_result.recovered_bits)], (
        "Discop mode 2 round-trip failed."
    )
    print(f"Mode 2 tokens generated: {len(mode2_result.tokens)}")
    print(f"Mode 2 recovered bits: {len(mode2_result.recovered_bits)}")
    print(f"Mode 2 embedding rate: {mode2_result.embedding_rate:.4f} bits/token")

    # mode 3: saturate the raster with a long bitstream.
    print("Running Discop mode 3 with a long deterministic bitstream...")
    mode3_image, mode3_result, _ = complete_image_with_discop(
        model=model,
        image_uint8=image,
        context_ratio=CONTEXT_RATIO,
        mode=3,
        seed=SEED,
        progress=True,
    )
    Image.fromarray(mode3_image).save(OUTPUT_DIR / "discop_mode3.png")
    print(f"Saved: {OUTPUT_DIR / 'discop_mode3.png'}")
    print(f"Mode 3 embedded bits: {mode3_result.embedded_bits}")
    print(f"Mode 3 recovered bits: {len(mode3_result.recovered_bits)}")
    print(f"Mode 3 embedding rate: {mode3_result.embedding_rate:.4f} bits/token")
    print(
        "Mode 3 round-trip: ",
        mode3_result.recovered_bits == mode3_result.message_bits[: len(mode3_result.recovered_bits)],
    )

    fig = make_four_panel_figure(image, completed, mode1_image, mode3_image)
    save_figure(fig, OUTPUT_FIGURE)
    plt.close(fig)


if __name__ == "__main__":
    main()