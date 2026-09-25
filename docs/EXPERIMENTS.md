# Experiment Notes

The repository evaluates autoregressive image steganography on three model/dataset pairs.

## 1. PixelCNN++ / CIFAR-10

Entry point:

```bash
python pixelcnnpp_cifar10_image.py
```

The script compares ACS, Discop, and ACS 2.0, records image-quality metrics, embedding/extraction time, embedding rate, and entropy utilization.

Saved portfolio artifacts are under:

```text
results/pixelcnn_cifar10/
```

A pretrained PixelCNN++ checkpoint is required under `PixelCNN++/checkpoints/`.

## 2. ImageGPT / CelebA

Entry point:

```bash
python imagegpt_celeba_image.py
```

The script uses `openai/imagegpt-small` and evaluates the same three steganography schemes on image completions.

Saved portfolio artifacts are under:

```text
results/imagegpt_celeba/
```

The original experiment configuration is included as `experiment_config.json`.

## 3. Taming Transformer / FFHQ

Entry point:

```bash
python taming_ffhq_image.py
```

This experiment requires a local clone of the CompVis Taming Transformers repository plus the FFHQ transformer checkpoint/configuration described in the main README.

Saved portfolio artifacts are under:

```text
results/taming_ffhq/
```

## Evaluation metrics

- **Embedding rate (bits/token):** hidden message bits divided by generated tokens.
- **Entropy utilization:** embedding rate divided by the empirical source entropy rate.
- **FID:** distribution-level image quality metric.
- **PSNR:** pixel-space similarity metric.
- **LPIPS:** perceptual similarity metric.
- **Embedding / extraction time:** measured wall-clock runtime for encode/decode.

The committed results are preserved from the completed experiments; this portfolio cleanup does not regenerate or alter the reported measurements.
