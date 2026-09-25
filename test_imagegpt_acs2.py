from __future__ import annotations
from pathlib import Path
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import ImageGPTForCausalImageModeling, ImageGPTImageProcessor

from acs2_integer import (
    MAX_GENERATION_STEPS, OUTPUT_DIR, TOP_K, TOP_P, TEMPERATURE, choose_device,
    decode_acs2, encode_acs2, show_or_save_figure,
)

IMAGE_SIZE = 32
CONTEXT_RATIO = 0.75
REFERENCE_IMAGE_URL = "https://raw.githubusercontent.com/comydream/Discop/main/temp/small.png"


class ImageGPTSource:
    eos_token_id = None # ImageGPT has no EOS: mode 1 uses its explicit 1024-pixel cap

    def __init__(self, model, prefix_ids: list[int], *, top_k=TOP_K, top_p=TOP_P, temperature=TEMPERATURE):
        self.model, self.top_k, self.top_p, self.temperature = model, top_k, top_p, temperature
        self.device = next(model.parameters()).device
        self.prefix_ids = prefix_ids
        self.reset()

    def reset(self):
        self.past = None
        self.pending = torch.tensor([[self.model.config.vocab_size - 1, *self.prefix_ids]], device=self.device)

    def probabilities(self):
        with torch.inference_mode():
            out = self.model(input_ids=self.pending, past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values

        logits = out.logits[0, -1, :-1].float() / self.temperature
        if self.top_k is not None:
            values, ids = torch.topk(logits, min(self.top_k, logits.numel()))
        else:
            ids = torch.arange(logits.numel(), device=logits.device)
            values = logits
        if self.top_p < 1.0:
            order = torch.argsort(values, descending=True)
            sorted_probs = torch.softmax(values[order], dim=0)
            keep = torch.cumsum(sorted_probs, dim=0) <= self.top_p
            keep[0] = True
            ids, values = ids[order][keep], values[order][keep]
        return ids, torch.softmax(values, dim=0)

    def commit(self, token_id: int):
        self.pending = torch.tensor([[token_id]], device=self.device)


def indices_to_image(indices: list[int], processor: ImageGPTImageProcessor) -> Image.Image:
    clusters = processor.clusters
    pixels = torch.as_tensor(clusters[indices]).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    data = torch.round(127.5 * (pixels + 1)).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(data)


def save_image(image: Image.Image, filename: str) -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    image.save(Path(OUTPUT_DIR) / filename)
    print(f"Saved image: {Path(OUTPUT_DIR) / filename}")


def one_mode(model, processor, prefix_ids: list[int], original_ids: list[int], mode: int, message: str, *, seed: int = 666666):
    source = ImageGPTSource(model, prefix_ids)
    
    pixels_to_generate = IMAGE_SIZE * IMAGE_SIZE - len(prefix_ids)
    result = encode_acs2(source, mode, message_bits=message if mode < 3 else None,
                         length=pixels_to_generate if mode == 3 else None,
                         max_steps=pixels_to_generate, seed=seed)
    
    raster_ids = prefix_ids + result.tokens + original_ids[len(prefix_ids) + result.N:]
    image = indices_to_image(raster_ids, processor)
    save_image(image, f"acs2_mode_{mode}_watermarked.png")
    recovered = decode_acs2(ImageGPTSource(model, prefix_ids), result.tokens, mode=mode,
                            message_length=result.m if mode < 3 else None,
                            rotation_numerator=result.rotation_numerator,
                            rotation_bits=result.rotation_bits)
    assert recovered == result.message_bits, "ACS2 round-trip failed"
    print("Encoder return:", result.summary())
    print(f"Mode {mode}: round-trip passed ({result.m} bits in {result.N} pixels).")
    return image


def load_reference_image() -> Image.Image:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(OUTPUT_DIR) / "discop_small_before_watermark.png"
    if not path.exists():
        print("Downloading Discop's image-completion input...")
        urlretrieve(REFERENCE_IMAGE_URL, path)
    image = Image.open(path).convert("RGB")
    return image.resize((IMAGE_SIZE, IMAGE_SIZE))


def main():
    device = choose_device()
    print("Loading the Discop reference image model/dataset interface: openai/imagegpt-small")
    model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small").to(device).eval()
    processor = ImageGPTImageProcessor.from_pretrained("openai/imagegpt-small")
    cover = load_reference_image()
    save_image(cover, "cover_before_watermark.png")
    input_ids = processor(cover, return_tensors="pt")["input_ids"][0].tolist()
    prefix_ids = input_ids[:round(IMAGE_SIZE * CONTEXT_RATIO) * IMAGE_SIZE]
    print(f"Using Discop's {CONTEXT_RATIO:.0%} image-completion context ({len(prefix_ids)} pixels).")
    message = "101100111000101011001011"
    images = [one_mode(model, processor, prefix_ids, input_ids, mode, message) for mode in (1, 2, 3)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, mode in zip(axes, images, (1, 2, 3)):
        ax.imshow(image); ax.set_title(f"ACS2 mode {mode}"); ax.axis("off")
    show_or_save_figure(fig, "acs2_modes.png")


if __name__ == "__main__":
    main()