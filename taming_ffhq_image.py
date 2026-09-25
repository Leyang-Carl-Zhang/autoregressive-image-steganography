from __future__ import annotations
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import hashlib
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from tqdm.auto import tqdm


SEED = 666666

RUN_EXPERIMENT_1 = True
RUN_EXPERIMENT_2 = True

NUM_IMAGES = 100
NUM_DISPLAY_IMAGES = 3
NUM_TRIALS_PER_M = 10
M_VALUES = list(range(1, 251))

# for Taming Transformer
IMAGE_SIZE = 256
LATENT_H = 16
LATENT_W = 16
CONTEXT_RATIO = 0.5
TOP_K = 256
TOP_P = 1.0
TEMPERATURE = 1.0
PROBABILITY_BITS = 32
ROTATION_BITS = 128 # ACS2 rotation precision.

PROJECT_DIR = Path(__file__).resolve().parent
TAMING_DIR = PROJECT_DIR / "taming_ffhq_cache" / "taming-transformers"
TAMING_CHECKPOINT = PROJECT_DIR / "taming_ffhq_cache" / "ffhq_transformer.ckpt"
TAMING_PROJECT_CONFIG = PROJECT_DIR / "taming_ffhq_cache" / "ffhq_transformer_project.yaml"

LOCAL_FFHQ_DIR: Optional[Path] = None

# HF_DATASET_ID = "kristinealli/ffhq-dataset"
# HF_DATASET_FALLBACK_ID = "gaunernst/ffhq-1024-wds"
# HF_DATASET_SPLIT = "train"

# FFHQ streaming dataset
HF_DATASET_ID = "gaunernst/ffhq-1024-wds"
HF_DATASET_SPLIT = "train"

DATA_CACHE_DIR = PROJECT_DIR / "taming_ffhq_cache" / "ffhq_100_cache"

OUTPUT_DIR = PROJECT_DIR / "taming_ffhq_outputs"

SHOW_PLOTS_IN_COLAB = True
SAVE_PLOTS_IN_COLAB = False

NATURAL_COMPLETION_SAMPLE = True

METRIC_BATCH_SIZE = 16


def running_in_colab() -> bool:
    try:
        import google.colab
        return True
    except Exception:
        return False


def show_or_save_figure(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    in_colab = running_in_colab()
    if (in_colab and SAVE_PLOTS_IN_COLAB) or not in_colab:
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {path}")
    if in_colab and SHOW_PLOTS_IN_COLAB:
        fig.show()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device}")
    return device


def seed_from_key(key: bytes) -> int:
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big")


# random binary message
def random_bitstring(rng: random.Random, n_bits: int) -> str:
    return "".join("1" if rng.getrandbits(1) else "0" for _ in range(n_bits))


def add_taming_repo_to_path() -> None:
    if not TAMING_DIR.exists():
        raise FileNotFoundError(
            f"Taming Transformer repo not found at {TAMING_DIR}. "
            "Expected the cloned repo under taming_ffhq_cache/taming-transformers."
        )
    if str(TAMING_DIR) not in sys.path:
        sys.path.insert(0, str(TAMING_DIR))


def load_taming_model(device: torch.device):

    add_taming_repo_to_path()
    try:
        from omegaconf import OmegaConf
        from main import instantiate_from_config
    except Exception as exc:
        raise RuntimeError(
            "Could not import the Taming Transformer dependencies. "
            "Install requirements.txt and make sure the cloned repo is present."
        ) from exc

    if not TAMING_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing Taming checkpoint: {TAMING_CHECKPOINT}")
    if not TAMING_PROJECT_CONFIG.exists():
        raise FileNotFoundError(f"Missing Taming project config: {TAMING_PROJECT_CONFIG}")

    print(f"Loading Taming config: {TAMING_PROJECT_CONFIG}")
    config = OmegaConf.load(str(TAMING_PROJECT_CONFIG))
    model_config = config.model

    params = model_config.params
    if "ckpt_path" in params:
        params.ckpt_path = None
    if "first_stage_config" in params and "params" in params.first_stage_config:
        if "ckpt_path" in params.first_stage_config.params:
            params.first_stage_config.params.ckpt_path = None
    if "cond_stage_config" in params and isinstance(params.cond_stage_config, dict):
        cparams = params.cond_stage_config.get("params")
        if cparams is not None and "ckpt_path" in cparams:
            cparams.ckpt_path = None

    print("Instantiating Taming Transformer...")
    model = instantiate_from_config(model_config)

    # checkpoint = torch.load(str(TAMING_CHECKPOINT), map_location="cpu")

    checkpoint = torch.load(
        str(TAMING_CHECKPOINT),
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Checkpoint missing keys: {len(missing)}")
    if unexpected:
        print(f"Checkpoint unexpected keys: {len(unexpected)}")
    if missing:
        print("First missing keys:", missing[:5])
    if unexpected:
        print("First unexpected keys:", unexpected[:5])

    model = model.to(device).eval()
    return model


def _pil_from_row_value(value: Any) -> Optional[Image.Image]:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            from io import BytesIO
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        from io import BytesIO
        return Image.open(BytesIO(value)).convert("RGB")
    return None


def load_one_ffhq_image_from_hf_row(row: dict[str, Any]) -> Image.Image:
    for key in ("image", "png", "jpg", "jpeg", "webp"):
        if key in row:
            image = _pil_from_row_value(row[key])
            if image is not None:
                return image
    for key, value in row.items():
        if key.startswith("__"):
            continue
        image = _pil_from_row_value(value)
        if image is not None:
            return image
    raise RuntimeError(f"Could not identify an image field in streamed row keys={list(row)}")

# 256 x 256 rgb
def resize_to_model(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def cache_local_images(image_paths: Sequence[Path]) -> list[np.ndarray]:
    paths = sorted([p for p in image_paths if p.is_file()])
    if len(paths) < NUM_IMAGES:
        raise RuntimeError(f"Found only {len(paths)} image files under {LOCAL_FFHQ_DIR}; need {NUM_IMAGES}.")
    images: list[np.ndarray] = []
    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(tqdm(paths[:NUM_IMAGES], desc="Loading FFHQ local images")):
        arr = resize_to_model(Image.open(path))
        images.append(arr)
        Image.fromarray(arr).save(DATA_CACHE_DIR / f"{i:04d}.png")
    return images


# def stream_and_cache_ffhq() -> list[np.ndarray]:
#     try:
#         from datasets import load_dataset
#     except Exception as exc:
#         raise RuntimeError(
#             "The FFHQ dataset is not local and the Hugging Face 'datasets' package "
#             "is unavailable. Install requirements.txt or set LOCAL_FFHQ_DIR."
#         ) from exc

#     print(
#         f"Streaming {NUM_IMAGES} FFHQ images from {HF_DATASET_ID} "
#         f"(no full-dataset download)..."
#     )
#     try:
#         ds = load_dataset(HF_DATASET_ID, split=HF_DATASET_SPLIT, streaming=True)
#         active_dataset_id = HF_DATASET_ID
#     except Exception as first_exc:
#         print(f"Primary streaming dataset failed: {first_exc}")
#         print(f"Trying fallback streaming dataset: {HF_DATASET_FALLBACK_ID}")
#         ds = load_dataset(HF_DATASET_FALLBACK_ID, split=HF_DATASET_SPLIT, streaming=True)
#         active_dataset_id = HF_DATASET_FALLBACK_ID
#     print(f"Active streaming dataset: {active_dataset_id}")
#     DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

#     images: list[np.ndarray] = []
#     for row in tqdm(ds, total=NUM_IMAGES, desc="Streaming FFHQ"):
#         image = load_one_ffhq_image_from_hf_row(row)
#         arr = resize_to_model(image)
#         images.append(arr)
#         Image.fromarray(arr).save(DATA_CACHE_DIR / f"{len(images)-1:04d}.png")
#         if len(images) >= NUM_IMAGES:
#             break

#     if len(images) != NUM_IMAGES:
#         raise RuntimeError(f"Stream ended after {len(images)} images; expected {NUM_IMAGES}.")
#     return images

def stream_and_cache_ffhq() -> list[np.ndarray]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The FFHQ dataset is not local and the Hugging Face 'datasets' package "
            "is unavailable. Install requirements.txt or set LOCAL_FFHQ_DIR."
        ) from exc

    print(
        f"Streaming {NUM_IMAGES} FFHQ images from {HF_DATASET_ID} "
        f"(no full-dataset download)..."
    )

    DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading streaming dataset: {HF_DATASET_ID}")

    ds = load_dataset(
        HF_DATASET_ID,
        split=HF_DATASET_SPLIT,
        streaming=True,
    )

    print(f"Active streaming dataset: {HF_DATASET_ID}")

    images: list[np.ndarray] = []

    for row in tqdm(ds, total=NUM_IMAGES, desc="Streaming FFHQ"):
        image = load_one_ffhq_image_from_hf_row(row)

        arr = resize_to_model(image)

        images.append(arr)

        Image.fromarray(arr).save(
            DATA_CACHE_DIR / f"{len(images)-1:04d}.png"
        )

        if len(images) >= NUM_IMAGES:
            break

    if len(images) != NUM_IMAGES:
        raise RuntimeError(
            f"Stream ended after {len(images)} images; "
            f"expected {NUM_IMAGES}."
        )

    print(f"Cached {len(images)} FFHQ images to {DATA_CACHE_DIR}")

    return images


def load_ffhq_sample_set() -> list[np.ndarray]:
    cached = sorted(DATA_CACHE_DIR.glob("*.png"))
    if len(cached) >= NUM_IMAGES:
        print(f"Using cached FFHQ sample set: {DATA_CACHE_DIR}")
        return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in cached[:NUM_IMAGES]]

    if LOCAL_FFHQ_DIR is not None:
        if not LOCAL_FFHQ_DIR.exists():
            raise FileNotFoundError(f"LOCAL_FFHQ_DIR does not exist: {LOCAL_FFHQ_DIR}")
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            files.extend(LOCAL_FFHQ_DIR.rglob(ext))
        return cache_local_images(files)

    return stream_and_cache_ffhq()


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    if not (0.0 < top_p <= 1.0):
        raise ValueError("TOP_P must be in (0, 1].")

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    filtered = probs.clone()
    filtered[sorted_idx[remove]] = 0.0
    filtered = filtered / filtered.sum()
    return filtered


def filter_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits / TEMPERATURE
    if TOP_K is not None:
        k = min(int(TOP_K), logits.shape[-1])
        values, _ = torch.topk(logits, k=k)
        threshold = values[..., -1]
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    probs = F.softmax(logits, dim=-1)
    probs = top_p_filter(probs, TOP_P)
    return probs


# 16 x 16 VQGAN latent
class TamingLatentSource:

    eos_token_id = None

    def __init__(self, model: Any, image_uint8: np.ndarray, context_ratio: float):
        self.model = model
        self.device = next(model.parameters()).device
        self.image_uint8 = np.asarray(image_uint8, dtype=np.uint8)
        if self.image_uint8.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}, got {self.image_uint8.shape}.")
        self.context_ratio = float(context_ratio)
        self.quant_z, self.all_indices = self._encode_image(self.image_uint8)
        self.zshape = self.quant_z.shape
        self.total_latent_tokens = int(self.all_indices.shape[1])
        self.context_tokens = int(round(self.total_latent_tokens * self.context_ratio))
        if self.context_tokens < 0 or self.context_tokens >= self.total_latent_tokens:
            raise ValueError("CONTEXT_RATIO must leave a non-empty prefix and completion region.")
        self.prefix = self.all_indices[:, :self.context_tokens].detach().clone()
        self.reset()

    @staticmethod
    def _to_model_tensor(image_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
        x = torch.from_numpy(image_uint8).to(device=device)
        x = x.permute(2, 0, 1).float() / 127.5 - 1.0
        return x.unsqueeze(0)

    def _encode_image(self, image_uint8: np.ndarray):
        x = self._to_model_tensor(image_uint8, self.device)
        with torch.inference_mode():
            quant_z, indices = self.model.encode_to_z(x)
        return quant_z, indices

    @property
    def total_tokens(self) -> int:
        return self.total_latent_tokens - self.context_tokens

    @property
    def full_sequence(self) -> torch.Tensor:
        if self.generated.numel() == 0:
            return self.prefix.clone()
        return torch.cat((self.prefix, self.generated), dim=1)

    # reset the generated suffix
    def reset(self) -> None:
        self.generated = torch.empty((1, 0), dtype=torch.long, device=self.device)

    # def _next_logits(self) -> torch.Tensor:
    #     seq = self.full_sequence
    #     if seq.shape[1] >= self.model.transformer.get_block_size() + 1:
    #         raise RuntimeError("Latent context exceeded the transformer block size.")
    #     # Following the official completion implementation: use all context and
    #     # generated tokens except the final one, then read the final logit.
    #     transformer_input = seq[:, :-1]
    #     logits, _ = self.model.transformer(transformer_input)
    #     return logits[:, -1, :].squeeze(0)

    def _next_logits(self) -> torch.Tensor:

        seq = self.full_sequence

        # unconditional generation: no image context
        if self.context_tokens == 0:
            sos = torch.tensor(
                [[self.model.sos_token]],
                device=self.device,
                dtype=torch.long,
            )

            transformer_input = torch.cat((sos, seq), dim=1)

        # conditional image completion: use the image prefix
        else:
            transformer_input = seq

        if transformer_input.shape[1] > self.model.transformer.get_block_size():
            raise RuntimeError(
                "Latent context exceeded the transformer block size."
            )

        logits, _ = self.model.transformer(transformer_input)

        return logits[:, -1, :].squeeze(0)

    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            logits = self._next_logits()
            probs = filter_logits(logits)
        ids = torch.arange(probs.numel(), device=self.device, dtype=torch.long)
        keep = probs > 0
        return ids[keep], probs[keep]

    # append one latent token
    def commit(self, token_id: int) -> None:
        token = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        self.generated = torch.cat((self.generated, token), dim=1)

    # vqgan decode
    def decode_image(self) -> np.ndarray:
        if self.generated.shape[1] != self.total_tokens:
            raise RuntimeError(
                f"Cannot decode incomplete image: have {self.generated.shape[1]} generated tokens, "
                f"need {self.total_tokens}."
            )
        full = torch.cat((self.prefix, self.generated), dim=1)
        with torch.inference_mode():
            decoded = self.model.decode_to_img(full, self.zshape)
        image = ((decoded[0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5)
        return np.clip(image, 0, 255).astype(np.uint8)


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    mse = float(np.mean((aa - bb) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


class LPIPSMetric:

    def __init__(self, device: torch.device):
        try:
            import lpips
        except Exception as exc:
            raise RuntimeError("Install the 'lpips' package to compute LPIPS.") from exc
        self.model = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @staticmethod
    def _batch(images: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
        x = torch.from_numpy(np.stack(images)).to(device=device, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2) / 127.5 - 1.0
        return x

    def batch_scores(self, outputs: Sequence[np.ndarray], refs: Sequence[np.ndarray], batch_size: int) -> list[float]:
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(outputs), batch_size):
                a = self._batch(outputs[start:start + batch_size], self.device)
                b = self._batch(refs[start:start + batch_size], self.device)
                values = self.model(a, b).reshape(-1).detach().cpu().numpy().tolist()
                scores.extend(float(v) for v in values)
        return scores


class FIDMetric:

    def __init__(self, device: torch.device):
        try:
            from torchvision.models import Inception_V3_Weights, inception_v3
        except Exception as exc:
            raise RuntimeError("A compatible torchvision installation is required for FID.") from exc

        self.device = device
        self.weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=self.weights, aux_logits=True).to(device).eval()
        self.model.fc = torch.nn.Identity()
        self.transform = self.weights.transforms()

    def _features(self, images: Sequence[np.ndarray]) -> np.ndarray:
        features: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), METRIC_BATCH_SIZE):
                batch = torch.from_numpy(np.stack(images[start:start + METRIC_BATCH_SIZE])).to(
                    self.device, dtype=torch.float32
                )
                # Convert to NCHW [0,1] and use torchvision's model transform.
                batch = batch.permute(0, 3, 1, 2) / 255.0
                batch = self.transform(batch)
                out = self.model(batch)
                if hasattr(out, "logits"):
                    out = out.logits
                features.append(out.detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    @staticmethod
    def _stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def score(self, generated: Sequence[np.ndarray], reference: Sequence[np.ndarray]) -> float:
        g = self._features(generated)
        r = self._features(reference)
        mu_g, cov_g = self._stats(g)
        mu_r, cov_r = self._stats(r)
        diff = mu_g - mu_r
        covmean, _ = linalg.sqrtm(cov_g.dot(cov_r), disp=False)
        if not np.isfinite(covmean).all():
            eps = 1e-6
            covmean, _ = linalg.sqrtm((cov_g + np.eye(cov_g.shape[0]) * eps).dot(cov_r + np.eye(cov_r.shape[0]) * eps), disp=False)
        covmean = np.real(covmean)
        fid = diff.dot(diff) + np.trace(cov_g + cov_r - 2.0 * covmean)
        return float(np.real(fid))


@dataclass
class RunResult:

    method: str
    image_index: int
    mode: int
    image: np.ndarray
    tokens: list[int]
    m: int
    N: int
    entropy_rate: float
    embedding_time_s: float
    extraction_time_s: float
    message_bits: str
    recovered_bits: str

    @property
    def capacity(self) -> float:
        return self.m / self.N if self.N else 0.0

    @property
    def utilization(self) -> float:
        # capacity / empirical entropy rate
        return self.capacity / self.entropy_rate if self.entropy_rate > 0 else float("nan")



def entropy_from_step(probabilities: torch.Tensor, token_id: int) -> float:
    pos = (probabilities == token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError(f"Generated token {token_id} not found in the current PMF.")
    p = float(probabilities[pos[0]].item())
    return -math.log2(max(p, 1e-40))


def replay_tokens_and_entropy(source: TamingLatentSource, tokens: Sequence[int], progress: bool = False) -> float:
    total_surprisal = 0.0
    source.reset()
    iterator = tqdm(tokens, desc="Entropy replay", leave=False, disable=not progress)
    for token in iterator:
        ids, probs = source.probabilities()
        pos = (ids == int(token)).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            raise RuntimeError(f"Token {token} is illegal under the replay distribution.")
        total_surprisal += -math.log2(max(float(probs[pos[0]].item()), 1e-40))
        source.commit(int(token))
    return total_surprisal / len(tokens) if tokens else float("nan")


def configure_supplied_modules() -> tuple[Any, Any, Any]:
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    import acs2_integer as acs2
    from ACS1Image import acs1
    from DiscopImage import discop

    acs1.PROBABILITY_BITS = PROBABILITY_BITS
    acs2.PROBABILITY_BITS = PROBABILITY_BITS
    acs2.ROTATION_BITS = ROTATION_BITS
    acs2.TOP_K = TOP_K
    acs2.TOP_P = TOP_P
    acs2.TEMPERATURE = TEMPERATURE
    discop.MODE3_BIT_BUDGET_MULTIPLIER = 32
    return acs1, discop, acs2


def run_mode3_acs1(model: Any, image: np.ndarray, image_index: int, seed: int) -> RunResult:
    acs1, _, _ = configure_supplied_modules()
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    t0 = time.perf_counter()
    result = acs1.encode_acs1(
        source,
        3,
        length=source.total_tokens,
        seed=seed,
        max_steps=source.total_tokens,
        progress=False,
    )
    embedding_time = time.perf_counter() - t0
    out_image = source.decode_image()

    extract_source = TamingLatentSource(model, image, CONTEXT_RATIO)
    t1 = time.perf_counter()
    recovered = acs1.decode_acs1(extract_source, result.tokens, mode=3, progress=False)
    extraction_time = time.perf_counter() - t1

    if recovered != result.message_bits:
        raise RuntimeError("ACS1 mode-3 extraction mismatch.")
    entropy = replay_tokens_and_entropy(TamingLatentSource(model, image, CONTEXT_RATIO), result.tokens)
    return RunResult(
        "ACS1", image_index, 3, out_image, result.tokens, result.m, result.N,
        entropy, embedding_time, extraction_time, result.message_bits, recovered,
    )


def run_mode3_discop(model: Any, image: np.ndarray, image_index: int, seed: int) -> RunResult:
    _, discop, _ = configure_supplied_modules()
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    t0 = time.perf_counter()
    result = discop.encode_discop(
        source,
        3,
        seed=seed,
        max_steps=source.total_tokens,
        progress=False,
    )
    embedding_time = time.perf_counter() - t0
    out_image = source.decode_image()

    extract_source = TamingLatentSource(model, image, CONTEXT_RATIO)
    t1 = time.perf_counter()
    recovered = discop.decode_discop(
        extract_source,
        result.tokens,
        seed=seed,
        progress=False,
        message_length=result.embedded_bits,
    )
    extraction_time = time.perf_counter() - t1

    expected = result.message_bits[:result.embedded_bits]
    if recovered != expected:
        raise RuntimeError("Discop mode-3 extraction mismatch.")

    entropy = replay_tokens_and_entropy(TamingLatentSource(model, image, CONTEXT_RATIO), result.tokens)
    return RunResult(
        "Discop", image_index, 3, out_image, result.tokens, result.embedded_bits, len(result.tokens),
        entropy, embedding_time, extraction_time, expected, recovered,
    )


def run_mode3_acs2(model: Any, image: np.ndarray, image_index: int, seed: int) -> RunResult:
    _, _, acs2 = configure_supplied_modules()
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    t0 = time.perf_counter()
    result = acs2.encode_acs2(
        source,
        3,
        length=source.total_tokens,
        rotation_bits=ROTATION_BITS,
        seed=seed,
        max_steps=source.total_tokens,
        progress=False,
    )
    embedding_time = time.perf_counter() - t0
    out_image = source.decode_image()

    extract_source = TamingLatentSource(model, image, CONTEXT_RATIO)
    t1 = time.perf_counter()
    recovered = acs2.decode_acs2(
        extract_source,
        result.tokens,
        mode=3,
        rotation_numerator=result.rotation_numerator,
        rotation_bits=ROTATION_BITS,
        progress=False,
    )
    extraction_time = time.perf_counter() - t1

    if recovered != result.message_bits:
        raise RuntimeError("ACS2 mode-3 extraction mismatch.")
    entropy = replay_tokens_and_entropy(TamingLatentSource(model, image, CONTEXT_RATIO), result.tokens)
    return RunResult(
        "ACS2", image_index, 3, out_image, result.tokens, result.m, result.N,
        entropy, embedding_time, extraction_time, result.message_bits, recovered,
    )


# baseline completion
def natural_completion(model: Any, image: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    generator = torch.Generator(device=source.device)
    generator.manual_seed(seed)
    source.reset()
    t0 = time.perf_counter()
    for _ in range(source.total_tokens):
        ids, probs = source.probabilities()
        idx = torch.multinomial(probs, 1, generator=generator)
        source.commit(int(ids[idx].item()))
    elapsed = time.perf_counter() - t0
    return source.decode_image(), elapsed


def make_exp1_table(
    references: Sequence[np.ndarray],
    baseline: Sequence[np.ndarray],
    runs: dict[str, list[RunResult]],
    device: torch.device,
) -> pd.DataFrame:
    print("Computing LPIPS/FID metrics for Experiment 1...")
    lpips_metric = LPIPSMetric(device)
    fid_metric = FIDMetric(device)

    rows: list[dict[str, Any]] = []

    # baseline_psnr = float(np.mean([psnr_uint8(a, b) for a, b in zip(baseline, references)]))
    # baseline_lpips = float(np.mean(lpips_metric.batch_scores(baseline, references, METRIC_BATCH_SIZE)))
    # baseline_fid = fid_metric.score(baseline, references)
    rows.append({
        "Method": "Pretrained completion",
        # "FID": baseline_fid,
        # "PSNR (dB)": baseline_psnr,
        # "LPIPS": baseline_lpips,
        "FID": np.nan,
        "PSNR (dB)": np.nan,
        "LPIPS": np.nan,
        "m (bits)": np.nan,
        "N (tokens)": np.nan,
        "Capacity m/N": np.nan,
        "Entropy rate": np.nan,
        "Utilization": np.nan,
        "Embedding time (s/image)": np.nan,
        "Extraction time (s/image)": np.nan,
    })

    for method, method_runs in runs.items():
        outputs = [r.image for r in method_runs]
        # fid = fid_metric.score(outputs, references)
        # psnr = float(np.mean([psnr_uint8(r.image, references[i]) for i, r in enumerate(method_runs)]))
        # lpips = float(np.mean(lpips_metric.batch_scores(outputs, references, METRIC_BATCH_SIZE)))
        fid = fid_metric.score(outputs, baseline)
        psnr = float(np.mean([psnr_uint8(r.image, baseline[i]) for i, r in enumerate(method_runs)]))
        lpips = float(np.mean(lpips_metric.batch_scores(outputs, baseline, METRIC_BATCH_SIZE)))
        mean_m = float(np.mean([r.m for r in method_runs]))
        mean_n = float(np.mean([r.N for r in method_runs]))
        mean_capacity = float(np.mean([r.capacity for r in method_runs]))
        mean_entropy = float(np.mean([r.entropy_rate for r in method_runs]))
        mean_utilization = float(np.mean([r.utilization for r in method_runs]))
        mean_embed_time = float(np.mean([r.embedding_time_s for r in method_runs]))
        mean_extract_time = float(np.mean([r.extraction_time_s for r in method_runs]))
        rows.append({
            "Method": method,
            "FID": fid,
            "PSNR (dB)": psnr,
            "LPIPS": lpips,
            "m (bits)": mean_m,
            "N (tokens)": mean_n,
            "Capacity m/N": mean_capacity,
            "Entropy rate": mean_entropy,
            "Utilization": mean_utilization,
            "Embedding time (s/image)": mean_embed_time,
            "Extraction time (s/image)": mean_extract_time,
        })
    return pd.DataFrame(rows)


def make_exp1_visualization(
    references: Sequence[np.ndarray],
    baseline: Sequence[np.ndarray],
    runs: dict[str, list[RunResult]],
    image_indices: Sequence[int],
) -> plt.Figure:
    fig, axes = plt.subplots(NUM_DISPLAY_IMAGES, 5, figsize=(15, 9))
    if NUM_DISPLAY_IMAGES == 1:
        axes = np.expand_dims(axes, axis=0)

    columns = ["Original", "Pretrained completion", "ACS1", "Discop", "ACS2"]
    for col, title in enumerate(columns):
        axes[0, col].set_title(title, fontsize=11)

    for row, idx in enumerate(image_indices):
        panels = [
            references[idx],
            baseline[idx],
            runs["ACS1"][idx].image,
            runs["Discop"][idx].image,
            runs["ACS2"][idx].image,
        ]
        for col, image in enumerate(panels):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"Image {idx}", rotation=90, fontsize=10, labelpad=8)
    fig.suptitle("FFHQ f=16 image completion: mode 3", fontsize=14)
    fig.tight_layout()
    return fig


def run_experiment_1(model: Any, images: Sequence[np.ndarray], device: torch.device) -> None:
    print("\n================ Experiment 1: mode 3 ================")
    references = list(images)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_outputs: list[np.ndarray] = []
    baseline_times: list[float] = []
    for i, image in enumerate(tqdm(references, desc="Baseline completions")):
        out, elapsed = natural_completion(model, image, SEED + i * 31)
        baseline_outputs.append(out)
        baseline_times.append(elapsed)

    runs: dict[str, list[RunResult]] = {"ACS1": [], "Discop": [], "ACS2": []}
    runners = {
        "ACS1": run_mode3_acs1,
        "Discop": run_mode3_discop,
        "ACS2": run_mode3_acs2,
    }
    for method, runner in runners.items():
        print(f"Running {method} mode 3 on {len(references)} images...")
        for i, image in enumerate(tqdm(references, desc=f"{method} mode 3")):
            run = runner(model, image, i, SEED + 10000 + i * 31)
            runs[method].append(run)

    raw_rows: list[dict[str, Any]] = []
    for method, method_runs in runs.items():
        for r in method_runs:
            raw_rows.append({
                "method": method,
                "image_index": r.image_index,
                "m_bits": r.m,
                "N_tokens": r.N,
                "capacity_bits_per_token": r.capacity,
                "entropy_rate_bits_per_token": r.entropy_rate,
                "utilization": r.utilization,
                "embedding_time_s": r.embedding_time_s,
                "extraction_time_s": r.extraction_time_s,
                "message_length": len(r.message_bits),
                "round_trip_ok": r.message_bits == r.recovered_bits,
            })
    pd.DataFrame(raw_rows).to_csv(OUTPUT_DIR / "experiment1_per_image_metrics.csv", index=False)

    table = make_exp1_table(references, baseline_outputs, runs, device)
    table.to_csv(OUTPUT_DIR / "experiment1_summary.csv", index=False)
    formatted = table.to_string(index=False, float_format=lambda x: f"{x:.6f}")
    print("\nExperiment 1 results:\n")
    print(formatted)
    (OUTPUT_DIR / "experiment1_summary.txt").write_text(formatted + "\n", encoding="utf-8")

    chosen_indices = [0, 1, 2]
    if len(references) < 3:
        chosen_indices = list(range(len(references)))
    fig = make_exp1_visualization(references, baseline_outputs, runs, chosen_indices)
    show_or_save_figure(fig, "experiment1_five_column_comparison.png")
    if not running_in_colab():
        plt.close(fig)

    print(f"Mean baseline completion time: {np.mean(baseline_times):.4f} s/image")


def derive_trial_key_and_message(master_rng: random.Random, m: int) -> tuple[bytes, str, int]:
    key = master_rng.getrandbits(256).to_bytes(32, "big")
    key_seed = seed_from_key(key)
    message_rng = random.Random(key_seed ^ (m << 17))
    message = random_bitstring(message_rng, m)
    return key, message, key_seed


def run_mode2_trial_acs1(model: Any, image: np.ndarray, m: int, message: str, trial_seed: int) -> tuple[int, float]:
    acs1, _, _ = configure_supplied_modules()
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    result = acs1.encode_acs1(
        source,
        2,
        message_bits=message,
        seed=trial_seed,
        max_steps=source.total_tokens,
        progress=False,
    )
    recovered = acs1.decode_acs1(
        TamingLatentSource(model, image, CONTEXT_RATIO),
        result.tokens,
        mode=2,
        message_length=m,
        progress=False,
    )
    if recovered != message:
        raise RuntimeError("ACS1 mode-2 trial failed round-trip verification.")
    entropy = replay_tokens_and_entropy(TamingLatentSource(model, image, CONTEXT_RATIO), result.tokens)
    return result.N, entropy


def run_mode2_trial_discop(model: Any, image: np.ndarray, m: int, message: str, trial_seed: int) -> tuple[int, float]:
    _, discop, _ = configure_supplied_modules()
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    result = discop.encode_discop(
        source,
        2,
        message_bits=message,
        seed=trial_seed,
        max_steps=source.total_tokens,
        progress=False,
    )
    recovered = discop.decode_discop(
        TamingLatentSource(model, image, CONTEXT_RATIO),
        result.tokens,
        seed=trial_seed,
        progress=False,
        message_length=m,
    )
    if recovered != message:
        raise RuntimeError("Discop mode-2 trial failed round-trip verification.")
    entropy = replay_tokens_and_entropy(TamingLatentSource(model, image, CONTEXT_RATIO), result.tokens)
    return len(result.tokens), entropy


def run_mode2_trial_acs2(model: Any, image: np.ndarray, m: int, message: str, key: bytes, trial_seed: int) -> tuple[int, float]:
    _, _, acs2 = configure_supplied_modules()
    rotation = acs2.derive_rotation(key=key, nonce=b"taming-ffhq-exp2")
    source = TamingLatentSource(model, image, CONTEXT_RATIO)
    result = acs2.encode_acs2(
        source,
        2,
        message_bits=message,
        rotation_numerator=rotation,
        rotation_bits=ROTATION_BITS,
        seed=trial_seed,
        max_steps=source.total_tokens,
        progress=False,
    )
    recovered = acs2.decode_acs2(
        TamingLatentSource(model, image, CONTEXT_RATIO),
        result.tokens,
        mode=2,
        message_length=m,
        rotation_numerator=rotation,
        rotation_bits=ROTATION_BITS,
        progress=False,
    )
    if recovered != message:
        raise RuntimeError("ACS2 mode-2 trial failed round-trip verification.")
    entropy = replay_tokens_and_entropy(TamingLatentSource(model, image, CONTEXT_RATIO), result.tokens)
    return result.N, entropy


# m = 1, ... , 100 Monte Carlo
def run_experiment_2(model: Any, images: Sequence[np.ndarray]) -> None:
    print("\n================ Experiment 2: mode 2 ================")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    methods = ("ACS1", "Discop", "ACS2")
    records: list[dict[str, Any]] = []
    master_rng = random.Random(SEED + 500000)

    for m in tqdm(M_VALUES, desc="Experiment 2 bit-length sweep"):
        aggregate = {method: {"rates": [], "entropies": [], "Ns": []} for method in methods}
        for trial in range(NUM_TRIALS_PER_M):
            image_index = master_rng.randrange(len(images))
            image = images[image_index]
            key, message, trial_seed = derive_trial_key_and_message(master_rng, m)
            trial_seed ^= trial + 7919 * m

            for method in methods:
                if method == "ACS1":
                    n, entropy = run_mode2_trial_acs1(model, image, m, message, trial_seed)
                elif method == "Discop":
                    n, entropy = run_mode2_trial_discop(model, image, m, message, trial_seed)
                else:
                    n, entropy = run_mode2_trial_acs2(model, image, m, message, key, trial_seed)
                if n <= 0:
                    raise RuntimeError(f"{method} produced an invalid N={n} for m={m}.")
                rate = m / n
                aggregate[method]["rates"].append(rate)
                aggregate[method]["entropies"].append(entropy)
                aggregate[method]["Ns"].append(n)

        for method in methods:
            rates = np.asarray(aggregate[method]["rates"], dtype=np.float64)
            entropies = np.asarray(aggregate[method]["entropies"], dtype=np.float64)
            Ns = np.asarray(aggregate[method]["Ns"], dtype=np.float64)
            total_bits = float(m * len(Ns))
            total_tokens = float(np.sum(Ns))
            pooled_rate = total_bits / total_tokens
            pooled_entropy = float(np.sum(entropies * Ns) / total_tokens)
            pooled_utilization = pooled_rate / pooled_entropy if pooled_entropy > 0 else float("nan")
            records.append({
                "method": method,
                "m_bits": m,
                "R_bits_per_token": pooled_rate,
                "utilization": pooled_utilization,
                "entropy_rate_bits_per_token": pooled_entropy,
                "mean_N_tokens": float(np.mean(Ns)),
                "std_R": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                "trials": len(rates),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "experiment2_results.csv", index=False)
    table = df.pivot(index="m_bits", columns="method", values=["R_bits_per_token", "utilization", "entropy_rate_bits_per_token", "mean_N_tokens"])
    formatted = table.to_string(float_format=lambda x: f"{x:.6f}")
    print("\nExperiment 2 results:\n")
    print(formatted)
    (OUTPUT_DIR / "experiment2_results.txt").write_text(formatted + "\n", encoding="utf-8")

    # R vs m
    fig_r, ax_r = plt.subplots(figsize=(9, 6))
    for method in methods:
        sub = df[df["method"] == method].sort_values("m_bits")
        line = ax_r.plot(sub["m_bits"], sub["R_bits_per_token"], label=f"{method} R")[0]
        ax_r.plot(
            sub["m_bits"], sub["entropy_rate_bits_per_token"],
            linestyle="--", color=line.get_color(), label=f"{method} entropy rate",
        )
    ax_r.set_xlabel("Message length m (bits)")
    ax_r.set_ylabel("Bits/token")
    ax_r.set_title("Experiment 2: empirical embedding rate and entropy rate")
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(ncol=2)
    fig_r.tight_layout()
    show_or_save_figure(fig_r, "experiment2_R_vs_m.png")
    if not running_in_colab():
        plt.close(fig_r)

    # u vs m
    fig_u, ax_u = plt.subplots(figsize=(9, 6))
    for method in methods:
        sub = df[df["method"] == method].sort_values("m_bits")
        ax_u.plot(sub["m_bits"], sub["utilization"], label=method)
    ax_u.set_xlabel("Message length m (bits)")
    ax_u.set_ylabel("Utilization u")
    ax_u.set_ylim(0.0, 1.1)
    ax_u.set_title("Experiment 2: utilization")
    ax_u.grid(True, alpha=0.25)
    ax_u.legend()
    fig_u.tight_layout()
    show_or_save_figure(fig_u, "experiment2_utilization_vs_m.png")
    if not running_in_colab():
        plt.close(fig_u)


def main() -> None:
    set_all_seeds(SEED)
    print("=== Taming Transformer FFHQ / ACS1 / Discop / ACS2 experiments ===")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Taming repo:       {TAMING_DIR}")
    print(f"Checkpoint:        {TAMING_CHECKPOINT}")
    print(f"Project config:    {TAMING_PROJECT_CONFIG}")
    print(f"Context ratio:     {CONTEXT_RATIO}")
    print(f"Image resolution:  {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Latent grid:       {LATENT_H}x{LATENT_W} ({LATENT_H * LATENT_W} tokens)")
    print(f"Completion tokens: {int(LATENT_H * LATENT_W * (1.0 - CONTEXT_RATIO))}")
    print(f"TOP_K={TOP_K}, TOP_P={TOP_P}, TEMPERATURE={TEMPERATURE}")

    device = choose_device()
    images = load_ffhq_sample_set()
    if len(images) != NUM_IMAGES:
        raise RuntimeError(f"Expected {NUM_IMAGES} reference images, got {len(images)}.")

    configure_supplied_modules()

    model = load_taming_model(device)

    if RUN_EXPERIMENT_1:
        run_experiment_1(model, images, device)
    if RUN_EXPERIMENT_2:
        run_experiment_2(model, images)

    print("\nAll experiments finished.")
    print(f"Results directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()