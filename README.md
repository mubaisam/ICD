
# 📸 ICD: Low-Light Image Enhancement

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.02627-b31b1b.svg)](https://arxiv.org/abs/2605.02627)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/mubaisam/ICD?style=social)](https://github.com/mubaisam/ICD)

<h2>Rethinking Low-Light Image Enhancement: <br>
A Log-Domain Intensity--Chromaticity Decoupling Perspective</h2>

<p>
  <a href="#-results">📊 Results</a> •
  <a href="#-paper">📄 Paper</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-usage">📖 Usage</a> •
  <a href="#-citation">📝 Citation</a>
</p>

</div>

---

## 🧭 Overview

Low-light image enhancement is commonly formulated as a direct RGB-to-RGB restoration problem, where brightness recovery, chromatic correction, and noise suppression are jointly modeled in a coupled color space.

This repository provides the official implementation of **ICD**, a log-domain intensity--chromaticity decoupling framework for low-light image enhancement. Instead of directly learning coupled RGB enhancement, ICD decomposes the image into an **intensity envelope** and a **log-domain chromaticity component**, enabling separate modeling of exposure recovery and chromatic correction.

---

## 📊 Results

<div align="center">

[![Quantitative Results](./assets/Quantitative%20results.png)](./assets/Quantitative%20results.png)

</div>

### Quantitative Results

| Methods              | LOLv1<br />(PSNR / SSIM) | LOLv2-Real<br />(PSNR / SSIM) | LOLv2-Syn<br />(PSNR / SSIM) | MIT-Adobe FiveK<br />(PSNR / SSIM) | LSRW-Huawei<br />(PSNR / SSIM) | LSRW-Nikon<br />(PSNR / SSIM) |
| :------------------- | :----------------------: | :---------------------------: | :--------------------------: | :--------------------------------: | :----------------------------: | :---------------------------: |
| **Ours (ICD)** |     26.1978 / 0.8536     |  **29.7100 / 0.8895**  |       25.2272 / 0.9078       |     **25.4061 / 0.9184**     |   **21.2399 / 0.6330**   |  **17.8946 / 0.5190**  |

### Evaluation Protocol

All quantitative results reported for ICD are evaluated **without using the `gtmean` adjustment**.
For fair comparison, users are encouraged to follow the same evaluation protocol when reproducing or comparing results.

---

## 🌟 Highlights

- **Log-domain intensity--chromaticity decoupling**Reformulates low-light enhancement by separating intensity recovery from chromaticity correction.
- **Constrained RGB reconstruction**Uses representation-derived constraints to reduce abnormal channel amplification and chromatic noise.
- **Dual-branch interaction network**Separately models intensity and chromaticity while allowing cross-branch information exchange.
- **Chromaticity confidence gate**Regulates unstable chromaticity updates in severely underexposed regions.
- **Competitive benchmark performance**
  Achieves strong quantitative and visual results on multiple low-light enhancement datasets.

---

## 📄 Paper

**Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective**

| Item   | Information                                                     |
| :----- | :-------------------------------------------------------------- |
| arXiv  | [2605.02627](https://arxiv.org/abs/2605.02627)                     |
| Status | Under review                                                    |
| Year   | 2026                                                            |
| Code   | [https://github.com/mubaisam/ICD](https://github.com/mubaisam/ICD) |

```bibtex
@article{ICD2026,
  title={Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective},
  author={Author Name},
  journal={arXiv preprint arXiv:2605.02627},
  year={2026}
}
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mubaisam/ICD.git
cd ICD
pip install -r requirements.txt
```

### Requirements

| Package      | Version                 |
| :----------- | :---------------------- |
| Python       | >= 3.8                  |
| PyTorch      | >= 1.9                  |
| torchvision  | required                |
| numpy        | required                |
| pillow       | required                |
| lpips        | required for evaluation |
| scikit-image | required for evaluation |

---

## 📖 Usage

### 1. Direct Inference

Use `test.py` for direct image enhancement without reference images.

```bash
python test.py --input_dir ./data/Test_Data --save_dir ./data/Results
```

Use a specific GPU:

```bash
python test.py --gpu_id 0 --input_dir ./data/Test_Data --save_dir ./data/Results
```

Use a custom checkpoint:

```bash
python test.py \
  --ckpt ./weights/custom_model.pth \
  --input_dir ./data/Test_Data \
  --save_dir ./data/Results
```

Disable the chromaticity gate:

```bash
python test.py \
  --input_dir ./data/Test_Data \
  --save_dir ./data/Results \
  --disable_gate
```

#### Arguments for `test.py`

| Argument           | Description                                 | Default                                     |
| :----------------- | :------------------------------------------ | :------------------------------------------ |
| `--gpu_id`       | GPU device ID                               | `0`                                       |
| `--input_dir`    | Directory containing low-light input images | `./data/Test_Data`                        |
| `--save_dir`     | Directory for saving enhanced images        | `./data/Results`                          |
| `--ckpt`         | Path to model checkpoint                    | `./weights/EXPANDLOL_REAL_best_model.pth` |
| `--disable_gate` | Disable the chromaticity confidence gate    | `False`                                   |

---

### 2. Inference and Evaluation

Use `eval.py` for image enhancement with full-reference quantitative evaluation.

```bash
python eval.py --base_dir ./data --dataset LOL_real
```

Use a specific GPU and checkpoint:

```bash
python eval.py \
  --gpu_id 0 \
  --base_dir ./data \
  --dataset LOL_real \
  --ckpt ./weights/LOL_REAL_best_model.pth
```

Evaluate another dataset:

```bash
python eval.py --base_dir ./data --dataset LOL_syn
```

#### Arguments for `eval.py`

| Argument       | Description                                                 | Default                               |
| :------------- | :---------------------------------------------------------- | :------------------------------------ |
| `--gpu_id`   | GPU device ID                                               | `0`                                 |
| `--base_dir` | Base directory containing `Test_Data/` and `Reference/` | `./data`                            |
| `--dataset`  | Dataset name                                                | `LOL_real`                          |
| `--ckpt`     | Path to model checkpoint                                    | `./weights/LOL_REAL_best_model.pth` |

#### Expected Data Structure

```text
data/
├── Test_Data/
│   ├── LOL_real/
│   └── LOL_syn/
├── Reference/
│   ├── LOL_real/
│   └── LOL_syn/
└── Results/
    ├── LOL_real/
    └── LOL_syn/
```

The enhanced images will be saved to:

```text
./data/Results/{dataset}/
```

The evaluation script reports:

- PSNR
- SSIM
- LPIPS

---

## 🏗️ Architecture

The ICD framework decomposes an RGB image into an intensity component and a log-domain chromaticity component.

```text
Input RGB
    │
    ▼
[Log-Domain Intensity--Chromaticity Decoupling]
    ├── Intensity Component
    └── Chromaticity Component
    │
    ▼
[Dual-Branch Interaction Network]
    ├── Intensity Branch
    └── Chromaticity Branch
    │
    ▼
[Constrained RGB Reconstruction]
    │
    ▼
Enhanced RGB Output
```

### Main Components

| Component                          | Function                                                                      |
| :--------------------------------- | :---------------------------------------------------------------------------- |
| Intensity branch                   | Performs exposure compensation.                                               |
| Chromaticity branch                | Corrects relative inter-channel color ratios.                                 |
| Cross-branch interaction           | Exchanges information between intensity and chromaticity streams.             |
| Chromaticity confidence gate       | Regulates unstable chromaticity updates in dark regions.                      |
| Constrained inverse reconstruction | Reconstructs valid RGB outputs and suppresses abnormal channel amplification. |

---

## 📁 Repository Structure

```text
ICD/
├── data/
│   ├── Test_Data/
│   ├── Reference/
│   └── Results/
├── weights/
│   └── *.pth
├── assets/
│   └── Quantitative results.png
├── test.py
├── eval.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{ICD2026,
  title={Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective},
  author={Author Name},
  journal={arXiv preprint arXiv:2605.02627},
  year={2026}
}
```

---

## 📄 License

This project is released under the MIT License.
See the `LICENSE` file for details.

---

## 🙋 Contact

For questions, bug reports, or reproduction issues, please open an issue in this repository.

You may also contact the author by email:

📧 **mubai@mail.ustc.edu.cn**

---

<div align="center">

**If this repository is helpful, please consider giving it a star.**

</div>
