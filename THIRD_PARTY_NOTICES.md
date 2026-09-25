# Third-Party Notices

This repository uses published steganography methods and pretrained generative models for research and evaluation.

## PixelCNN++

The files under `PixelCNN++/` are derived from the PyTorch implementation at:

- https://github.com/pclucas14/pixel-cnn-pp

The upstream license text is preserved as `PixelCNN++/license.md`.

## Taming Transformers

The Taming Transformer source code is **not vendored** in this repository. The experiment script expects a local clone of:

- https://github.com/CompVis/taming-transformers

The upstream project is distributed under the MIT License.

## Discop

The Discop implementation in this repository is a research reimplementation of the method described in:

J. Ding, K. Chen, Y. Wang, N. Zhao, W. Zhang, and N. Yu,  
“Discop: Provably Secure Steganography in Practice Based on Distribution Copies,”  
IEEE Symposium on Security and Privacy, 2023.

Reference implementation:

- https://github.com/comydream/Discop

## Other pretrained models and datasets

The experiments rely on external pretrained models and datasets, including ImageGPT, CIFAR-10, CelebA, and FFHQ. These assets are not redistributed here; users should obtain them from their original providers and follow the corresponding licenses/terms.

No project-wide license is granted for third-party components beyond the rights provided by their original authors.
