# Data dictionary

All datasets are public. This repository does **not** redistribute them. Download instructions and the
directory layout the scripts expect are below.

## Expected directory layout

The scripts use absolute paths from the authors' environment (typically W:\<project>\...). To reproduce,
either edit the ROOT / D constants at the top of each script or create the following tree.

    <ROOT>/
      data/
        sample_test_data/sample_test_data/data/      GigaTIME released sample tiles (50 tiles)
            <stem>_he.png                            556 x 556 H&E
            <stem>_comet.npy                         (556, 556, 25) uint8, 23 mIF channels + 2
            <stem>_comet_binarized.json              23 flags, which channels are evaluable
            <stem>_comet_binary_thres_labels.pkl.gz  BIT-PACKED masks + cell labels + thresholds
            <stem>_dice_metric.json                  authors' stored Dice (scalar per tile)
        HEMIT/                                       colorectal mIHC + H&E, independent institution
            test/input/*.tif                         945 patches, 1024 x 1024 RGB H&E
            test/label/*.tif                         945 patches, 1024 x 1024 RGB mIHC
                                                     channel order: 0 = pan-cytokeratin, 1 = CD3, 2 = DAPI
        TCGA_THCA/                                   TCGA-THCA diagnostic slides (GDC, ~256 GB)
            *.svs
        ORION_CRC/                                   ORION-CRC (s3://lin-2023-orion-crc, anonymous)
            <pat>_HE.ome.tif                         H&E, ~0.6-1.1 GB, mpp 0.3250
            <pat>_mask.ome.tif                       cell segmentation mask
            <pat>_cells.csv                          per-cell 19 marker intensities + X_centroid/Y_centroid
      models/
        GigaTIME/model.pth                           original U-Net++, 9.16 M parameters
        gigatime-flash/model.pth                     distilled, 23.8 M parameters
      analysis/                                      results are written here

## Key metadata

| Dataset | Access | Notes |
|---|---|---|
| GigaTIME sample tiles | prov-gigatime/GigaTIME on Hugging Face (gated) | 50 tiles, 556 x 556 px, 0.2302 um/px |
| GigaTIME-Flash weights | prov-gigatime/gigatime-flash on Hugging Face (gated) | |
| HEMIT | Mendeley Data, doi:10.17632/3gx53zm49d | no resolution metadata; calibrated to 0.284 um/px |
| GSE230424 | NCBI GEO | 4 thyroid specimens, 15,489 in-tissue Visium spots |
| TCGA-THCA | GDC portal | 227 diagnostic slides; RPPA via cBioPortal, 175 proteins, 222 patients |
| ORION-CRC | s3://lin-2023-orion-crc (anonymous range requests) | 41 patients; 19-marker single-cell CSV |

## Important details

* **ORION-CRC marker columns** (19): Hoechst, AF1, CD31, CD45, CD68, Argo550, CD4, FOXP3, CD8a, CD45RO,
  CD20, PD-L1, CD3e, CD163, E-cadherin, PD-1, Ki67, Pan-CK, SMA.
* **Mapping to the 23 GigaTIME channels**: Hoechst to DAPI, CD31 to CD34, CD68 to CD68_1:100, CD4 to CD4,
  CD8a to CD8, CD20 to CD20, PD-L1 to PD-L1, CD3e to CD3_1:1000, PD-1 to PD-1_1:200, Ki67 to Ki67_1:150,
  Pan-CK and E-cadherin to CK_1:150, SMA to Actin-D. AF1, Argo550, CD45, FOXP3, CD45RO and CD163 have **no**
  corresponding virtual channel and are used as negative controls.
* **HEMIT label channel order** is 0 = pan-cytokeratin, 1 = CD3, 2 = DAPI. Ground truth was obtained by Otsu
  thresholding of the corresponding channel.
* **GigaTIME training scale** is 0.2302 um/px (128 um mapped to 556 px), i.e. a 256-pixel model window covers
  58.9 um. Every input is resampled to this domain.
* **RPPA proteins** used from the TCGA-THCA panel are listed in results/L3/thca_rppa.txt (175 proteins after
  removing entries with missing values).
