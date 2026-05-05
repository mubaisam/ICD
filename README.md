# 📸 ICD: Low-Light Image Enhancement

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.02627-b31b1b.svg)](https://arxiv.org/abs/2605.02627)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/mubaisam/ICD?style=social)](https://github.com/mubaisam/ICD)

**Rethinking Low-Light Image Enhancement: A Log-Domain Intensity-Chromaticity Decoupling Perspective**

[📄 Paper](#paper) • [🚀 Quick Start](#quick-start) • [📊 Results](#results) • [📖 Documentation](#documentation)

### 📊 Results

<div align="center">

[![Quantitative Results](./assets/Quantitative%20results.png)](./assets/Quantitative%20results.png)


| Methods              | LOLv1 `<br>`(PSNR / SSIM) |   LOLv2_Real `<br>`(PSNR / SSIM)   | LOLv2_Syn `<br>`(PSNR / SSIM) |      MIT `<br>`(PSNR / SSIM)      |  LSRW_Huawei `<br>`(PSNR / SSIM)  |   LSRW_Nikon `<br>`(PSNR / SSIM)   |
| :------------------- | :-------------------------: | :----------------------------------: | :-----------------------------: | :----------------------------------: | :----------------------------------: | :----------------------------------: |
| **Ours (ICD)** |       26.1978/0.8536       | **29.7100** / **0.8895** |      *25.2272* / 0.9078      | **25.4061** / **0.9184** | **21.2399** / **0.6330** | **17.8946** / **0.5190** |

---

## 🌟 Highlights

- **Novel Decoupling Framework**: Separates intensity and chromaticity in log-domain for superior enhancement
- **Color Fidelity**: Preserves natural colors while recovering brightness details
- **State-of-the-art Performance**: Outperforms existing methods on standard benchmarks
- **Efficient & Practical**: Fast inference suitable for real-world applications
- **Easy to Use**: Simple API for integration into existing pipelines

---

## 📄 Paper

**Rethinking Low-Light Image Enhancement: A Log-Domain Intensity-Chromaticity Decoupling Perspective**

- **arXiv**: [2605.02627](https://arxiv.org/abs/2605.02627)
- **Conference**: [Pending/Your Conference]
- **Year**: 2026

```bibtex
@article{ICD2026,
  title={Rethinking Low-Light Image Enhancement: A Log-Domain Intensity--Chromaticity Decoupling Perspective},
  author={Author Name},
  journal={arXiv preprint arXiv:2605.02627},
  year={2026}
}
```

### 💡 Evaluation Note & A Small Trick

> **Notice:** All quantitative evaluations of our method are strictly conducted **WITHOUT** using the `gtmean` (ground truth mean) trick.

Here is a small observation/trick regarding the LOL datasets for fair comparison:
It is known that the **LOLv1** dataset has some inherent issues (e.g., imperfect exposure alignment), and **LOLv2_Real** was proposed as a corrected and refined version of LOLv1. Therefore, under strictly fair settings (i.e., *without* using `gtmean`), a normal method's PSNR on LOLv2_Real should logically be **higher** than its PSNR on LOLv1.

⚠️ **Red Flag:** If you observe a method reporting a significantly higher PSNR on LOLv1 compared to LOLv2_Real, it is highly likely that their evaluation data was adjusted or the `gtmean` trick was covertly applied. Please be aware of this fairness issue when comparing methods!

---

**中文说明**：

> **声明**：本文方法的所有指标评测均严格在**不使用 `gtmean`** 的前提下进行。

分享一个关于 LOL 数据集评测的避坑小技巧：
熟悉该领域的学者可能知道，LOLv1 数据集本身存在一些瑕疵，而 LOLv2_Real 是对 LOLv1 的修正版本。因此，在不使用 `gtmean` 的正常公平评测下，任何方法在 LOLv2_Real 上的 PSNR 理应**高于**它在 LOLv1 上的分数。
如果你发现某个方法的 LOLv1 的 PSNR 数值异常偏高（甚至反超了它的 LOLv2_Real 分数），那么大概率它的数据是调整过的，或者暗中使用了 `gtmean` 技巧。在做同行对比时，请大家注意甄别。
