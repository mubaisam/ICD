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

Low-light image enhancement is commonly formulated as a direct RGB-to-RGB restoration problem. ICD decomposes the image into an **intensity envelope** and a **log-domain chromaticity component**, enabling separate modeling of exposure recovery and chromatic correction.

---

## 📊 Results

<div align="center">

[![Quantitative Results](./assets/Quantitative%20results.png)](./assets/Quantitative%20results.png)

</div>

| Methods | LOLv1<br />(PSNR / SSIM) | LOLv2-Real<br />(PSNR / SSIM) | LOLv2-Syn<br />(PSNR / SSIM) | MIT-Adobe FiveK<br />(PSNR / SSIM) | LSRW-Huawei<br />(PSNR / SSIM) | LSRW-Nikon<br />(PSNR / SSIM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ours (ICD)** | 26.1978 / 0.8536 | **29.7100 / 0.8895** | 25.2272 / 0.9078 | **25.4061 / 0.9184** | **21.2399 / 0.6330** | **17.8946 / 0.5190** |

**Evaluation Protocol:** All results are evaluated **without using the `gtmean` adjustment**.

---

## 🌟 Highlights

- **Log-domain intensity--chromaticity decoupling**: Separates intensity and chromaticity for better enhancement
- **Constrained RGB reconstruction**: Reduces abnormal channel amplification and chromatic noise
- **Dual-branch interaction network**: Separate intensity and chromaticity modeling with cross-branch interaction
- **Chromaticity confidence gate**: Regulates unstable chromaticity updates in dark regions
- **Competitive benchmark performance**: Strong quantitative and visual results across multiple datasets

---

## 📄 Paper

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

```bash
git clone https://github.com/mubaisam/ICD.git
cd ICD
pip install -r requirements.txt
```

---

## 📖 Usage

### 1. Direct Inference (`test.py`)

```bash
python test.py --input_dir ./data/Test_Data --save_dir ./data/Results --gpu_id 0 --ckpt ./weights/EXPANDLOL_REAL_best_model.pth
```

Disable the chromaticity gate:

```bash
python test.py --input_dir ./data/Test_Data --save_dir ./data/Results --disable_gate --gpu_id 0 --ckpt ./weights/EXPANDLOL_REAL_best_model.pth
```

**Arguments**

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input_dir` | Directory with low-light images | `./data/Test_Data` |
| `--save_dir` | Directory for saving outputs | `./data/Results` |
| `--gpu_id` | GPU device ID | `0` |
| `--ckpt` | Path to model checkpoint | `./weights/EXPANDLOL_REAL_best_model.pth` |
| `--disable_gate` | Disable chromaticity gate | False |

### 2. Inference + Evaluation (`eval.py`):contentReference[oaicite:2]{index=2}

```bash
python eval.py --base_dir ./data --dataset LOL_real --gpu_id 0 --ckpt ./weights/LOL_REAL_best_model.pth
```

Supports other datasets by changing `--dataset` argument (`LOL_syn` etc).

**Expected Data Structure**

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

---

## 🏗️ Architecture

- Input RGB → Log-Domain Decoupling → Dual-Branch Interaction → Constrained RGB Reconstruction → Output
- **Intensity branch**: Exposure compensation  
- **Chromaticity branch**: Inter-channel color correction  
- **Cross-branch interaction**: Bidirectional feature exchange  
- **Chromaticity confidence gate**: Stabilizes color updates  
- **Constrained reconstruction**: Ensures valid RGB outputs

---

## 🙏 Acknowledgement

This work was partially inspired by the excellent HVI-CIDNet project.  
- HVI-CIDNet: [https://github.com/Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet)

---

## 📄 License

MIT License

---

## 🙋 Contact

Email: **mubai@mail.ustc.edu.cn**  
For issues, please open a GitHub issue.

---

**If this repository is helpful, please consider giving it a star.**