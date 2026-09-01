# `data/1_data/` — imaging inputs for the analysis scripts

Every neural analysis in this repository reads its imaging input from this
folder: region-of-interest (ROI) multivoxel-pattern matrices, whole-brain
parcel timecourses, an unfiltered-preprocessing variant, and the audio
amplitude envelopes. All extractions start from the same preprocessed
functional series (fMRIPrep output in MNI152 space, temporal high-pass /
linear detrend, 3 mm isotropic voxels), and each subfolder name is the
`processing_level` token that `scripts/helper/data_structure.py::find_file`
resolves.

The folder has two kinds of content:

1. **Shipped with the GitHub repository** — the small `audenv/` workbooks
   (Section 1 below).
2. **Not on GitHub because of size (9.1 GB)** — the four imaging-matrix
   subfolders (Section 2 below). **To run the analysis scripts, download
   them from the Zenodo archive and place them here**, as follows: get
   `1_data.zip` from the Zenodo record
   https://doi.org/10.5281/zenodo.22260086 (also listed in the paper's Data
   and materials availability statement, reference 34) and
   unzip it inside the repository's `data/` directory — the zip unpacks as
   `1_data/…`, so its contents merge into this folder and produce exactly
   the tree shown in Section 2. (The same Zenodo record also archives the repository itself as
   `mental_continuity-1.0.0.zip`, so the record alone reconstructs the
   full project: unzip the repository zip, then unzip the data zip inside
   its `data/` directory.) Until then, any script needing these
   inputs stops with a `FileNotFoundError` naming this folder.

---

## Section 1 — shipped with the GitHub repository

### `audenv/` — audio amplitude envelopes (8 files)

```
audenv/
└── audenv_{task}_{condition}.xlsx        2 tasks x 4 conditions = 8 files
```

One workbook per run (`audenv_{task}_{condition}.xlsx`): the amplitude
envelope of that run's audio on the 1.5 s TR grid. The interruption-epoch
timing table (`data/stimuli/interruption_epochs.csv` /
`data_structure.py::INTERRUPTION_PARAMS`) is aligned to these envelopes.
Read by the Figure 1 paradigm panels
(`scripts/figures/figure1/figure1_entire-demo.py`) to draw the story
soundwave, and shipped as the shareable audio-timing record because the
narrated audio itself is not redistributed (see `data/stimuli/README.md`).


---

## Section 2 — not on GitHub (size); download from the Zenodo archive

After unzipping `1_data.zip` inside `data/`, this
folder must contain the following four subfolders (file counts included so
you can verify the unzip was complete):

```
data/1_data/
├── mvp_zscore-entire/                             72 files
│   └── {task}_{condition}_{ROI}_shape_nsub_ntr_nvox_{n}_{t}_{v}.npy
│                                                    2 tasks x 4 conditions x 9 ROIs
│                                                    (A1+, AG, PCC, PMC, dlPFC,
│                                                     dmPFC, hipp, mSTG, vmPFC)
├── mvp_raw/                                       12 files
│   ├── {task}_{condition}_PMC_shape_*.npy           8 files (2 tasks x 4 conditions)
│   └── combined_4rois/
│       └── carver_{condition}_joint4roi_shape_*.npy 4 files (4 conditions)
├── fmriprep_no-filter_resampled-3mm-space_smooth-4mm/
│   └── sub-XXX_carver_{ROI}-3mm_mvp.csv           176 files
│                                                    88 participants x 2 ROIs (A1+, PMC)
└── voi/
    └── base-adj_story-8tr_shift0tr/
        └── carver_{condition}_nsubj-nroi-ntr_{n}-400-1026.csv   4 files (4 conditions)
```

If a subfolder is absent or a count does not match, the archive has not been
(fully) unzipped into place.

### `mvp_zscore-entire/` — the canonical ROI multivoxel patterns (72 files)

One NumPy array per task x condition x ROI,
`{task}_{condition}_{ROI}_shape_nsub_ntr_nvox_{n}_{t}_{v}.npy`, of shape
(n_subjects, n_TRs, n_voxels). Each array holds the preprocessed BOLD series
of every voxel inside that ROI's mask (`data/roi_masks/`), with each voxel's
timecourse z-scored across the entire run (`zscore_entire` in
`scripts/helper/vendor/01_preproc_zscore_methods.py`). The whole-run
z-scoring places every voxel on a common scale, so inter-subject pattern
correlations reflect the spatial patterning of activity rather than
voxel-wise differences in temporal mean or variance — the normalization the
Supplementary Methods describe as the default for all pattern analyses.

Used by every ROI pattern analysis: the main-text PMC tests
(`Result2_1`–`Result2_3`, `Result3_1`–`Result3_3`), the hippocampal onset
response (`Result1_3`), the Result 4 derive recipes
(`scripts/helper/derive_*.py`), and supplement sections S3–S5, S7–S9, S13
(filtered arm), S16, and S17. Filenames use the paper ROI names
throughout.

### `mvp_raw/` — un-normalized ROI extractions (12 files)

The same per-ROI extraction *before* any per-voxel normalization, kept for
the analyses that must apply their own normalization:

- `{task}_{condition}_PMC_*.npy` (both narratives, all four conditions) —
  read by `S11_invert-control-2_separate-zscore.py`, which re-standardizes
  the story and interruption phases separately to test whether the
  story-to-interruption inversion depends on whole-run z-scoring.
- `combined_4rois/{carver}_{condition}_joint4roi_*.npy` — the joint
  AG + PCC + dmPFC + vmPFC voxel set in one matrix, read by
  `scripts/helper/derive_carver_neural-realign_combo-4DMN.py`, which
  z-scores it per voxel across the run and derives the Result 4
  default-mode-network realignment measure.

### `fmriprep_no-filter_resampled-3mm-space_smooth-4mm/` — the no-high-pass variant (176 files)

Per-subject CSV matrices (`sub-XXX_{task}_{ROI}-3mm_mvp.csv`, TR x voxel) for
A1+ and PMC, extracted from the fMRIPrep output *without* the temporal
high-pass / detrend step, resampled to 3 mm space with 4 mm spatial
smoothing. These exist to answer one supplement question: is the
story-to-interruption pattern inversion an artifact of temporal high-pass
filtering? `S12_invert-control-3_highpass-filter-off.py` re-runs the
inversion tests on them, and `S13_unfiltered-sustained-pattern.py` re-derives
the Figure 3 similarity time course from them.

### `voi/base-adj_story-8tr_shift0tr/` — parcel-mean timecourses for global ISC (4 files)

One CSV per condition (`{task}_{condition}_nsubj-nroi-ntr_{n}-400-{t}.csv`):
each subject's mean timecourse in every Schaefer-400 parcel. The folder-name
tokens describe how the timecourses were prepared:

- `base-adj` — baseline-adjusted to the story phase. All four conditions
  share the story phase, so the story-phase signal serves as the common
  baseline: shifting every timecourse to that baseline makes the
  inter-subject correlation (ISC) values comparable across conditions.
- `story-8tr` — the story-phase definition drops the first 8 TRs after each
  return to the story (8 TRs x 17 epochs in the main narrative), so signal
  carrying over from the preceding interruption does not contaminate the
  story-phase baseline or the ISC window.
- `shift0tr` — no additional hemodynamic TR shift is applied.

Read by `Result1_2_global-ISC.py` (the whole-brain story-listening ISC map,
manuscript Fig. 1G) and `S1_global-ISC.py` (the four-condition extension and
per-network table, supplement Section S1), through
`vendor/global_isc.py::load_voi`.
