# Independent external validation of virtual multiplex immunofluorescence from H&E

Code, derived results and figure sources for the manuscript:

> **Independent external validation of virtual multiplex immunofluorescence from H&E reveals that predicted
> channels encode tissue composition rather than cell identity**
> Liangliang Chen, Leiling Wang, Zhanbo Yi
> Ningbo No. 2 Hospital, Ningbo, Zhejiang, China

---

## What this repository contains

An independent external validation of [GigaTIME](https://huggingface.co/prov-gigatime/GigaTIME) and
[GigaTIME-Flash](https://huggingface.co/prov-gigatime/gigatime-flash), which predict 21 channels of virtual
multiplex immunofluorescence (mIF) from routine H&E slides.

We first reproduced the authors' own released evaluation protocol on their published sample tiles with their
official U-Net++ weights, and then tested both models against four independently generated real-measurement
datasets spanning pixel, spatial-transcriptomic, patient and single-cell resolution.

## Headline findings

| Finding | Value |
|---|---|
| Reproduced official-protocol Dice (original U-Net++; authors' stored value 0.3357) | **0.3637** |
| Dice of the authors' declared *background* channel TRITC, which outranks 15 of 21 markers | **0.434** |
| Virtual channels with **negative** specificity (prediction matches another marker's mask better) | **7 of 21** |
| DAPI specificity, the only clearly positive channel | **+0.299** |
| Marker pairs passing specificity in L1 / L2 / L3 / L4 (3, 11, 3, 13 pairs tested) | **1 / 3 / 0 / 2** |
| Position of each channel's cognate protein within its own 175-protein correlation distribution | **34th-54th pct** |
| Patient-level recovery of measured epithelial/immune composition (ORION-CRC, 41 patients) | **rho = +0.711** |
| Pre-specified composition score containing no fitted parameters | **rho = +0.631** |

**Conclusion:** virtual mIF from H&E supports tissue-composition characterisation but not quantitative marker
readout; evaluation should report specificity against negative controls rather than paired correlation alone.

## Definition of specificity

Throughout, *specificity* of a nominal pair (marker *m*, virtual channel *c*) is

    specificity(c, m) = |rho(c, m)| - max over competing markers m' of |rho(c, m')|

A pair is called specific only when this quantity is positive. In L3 the competing set is 175 RPPA proteins
(Bonferroni threshold |rho| > 0.205 across 23 channels); in the other layers it is the marker set measured in
that dataset.

---

## Validation datasets

| Layer | Dataset | Real measurement | Granularity | Scale |
|---|---|---|---|---|
| L1 | [HEMIT](https://data.mendeley.com/datasets/3gx53zm49d) | mIHC, 3 channels | Pixel | 945 patches |
| L2 | [GSE230424](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE230424) | Visium spatial transcriptomics | Spot | 15,489 spots / 4 specimens |
| L3 | [TCGA-THCA](https://portal.gdc.cancer.gov/) + [RPPA](https://www.cbioportal.org/) | Bulk protein, 175 proteins | Patient | 222 patients / 227 slides |
| L4 | [ORION-CRC](https://doi.org/10.1038/s43018-023-00576-1) | Single-cell protein, 19 markers | Cell | 41 patients / 994,729 cells |
| Control | GigaTIME released sample tiles | mIHC, 23 channels | Pixel | 50 patches |

All datasets are public. None was generated with reference to GigaTIME.

## Repository layout

    analysis/            analysis scripts, numbered in the order reported in the manuscript
      01..16_*.py        supplementary experiments (sampling depth, normalisation, scale, specificity, ...)
      l1..l4_*.py        the four validation layers
      models/            GigaTIME and GigaTIME-Flash inference modules; the original U-Net++ was reconstructed
                         verbatim from the authors' released archs.py
      figures/           figure-generating scripts
    results/             all machine-readable results (JSON), including the full 23x23 Dice matrix and the
                         23x175 cross-reactivity matrix
    figures/             figures as published (PDF vector + 300 dpi PNG)
    data/README.md       data dictionary and the directory layout the scripts expect

## Reproducing the analysis

### Environment

Python 3.12 with torch 2.6.0+cu124, scikit-image 0.26, opencv-python, scipy, numpy, pandas, tiffslide,
scanpy, anndata, matplotlib and python-docx. See requirements.txt. A single NVIDIA RTX 3080 Ti (12.9 GB) was
used for all inference.

### Model weights

Download from Hugging Face (both repositories are gated; accept the licence first):

    prov-gigatime/GigaTIME          original U-Net++, 9.16 M parameters
    prov-gigatime/gigatime-flash    distilled, 23.8 M parameters

### Data layout

The scripts use absolute paths from the authors' working environment. Adjust the ROOT constant at the top of
each script, or create a matching directory tree, as described in data/README.md.

## Notes on reproducibility

* The official reference masks released with the sample tiles are **bit-packed**: three 8-bit channels encode
  23 binary channels and must be unpacked with np.unpackbits before use. Reading them as three channels yields
  a meaningless array.
* The official reference is a **cell-level convex-hull mask**, not the thresholded mIF image. Reconstruction
  details are in analysis/10_official_mask_reconstruction.py.
* HEMIT releases no resolution metadata. Its scale was calibrated to 0.284 um/px from DAPI nuclear size
  (analysis/12_hemit_scale_calibration.py), and a four-fold range was swept to confirm that the conclusions do
  not depend on that calibration (analysis/13_hemit_scale_sweep.py).
* Stain normalisation uses Reinhard colour transfer in CIELAB space, targeting the official GigaTIME H&E
  statistics: mean [87.82, 6.07, -4.99], SD [8.41, 7.57, 4.28].

## Citation

If you use this code or these results, please cite the manuscript (see CITATION.cff). The archived release is
available at Zenodo: DOI [to be inserted after the first release].

## Licence

MIT (see LICENSE). The GigaTIME model weights are released by their authors under Apache-2.0 and are not
redistributed here; the validation datasets retain their original licences.

## Contact

Zhanbo Yi (corresponding author), Ningbo No. 2 Hospital - 15376775258@163.com
