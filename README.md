# mental_continuity

Code and data associated with the manuscript **"Cortical and Hippocampal Pathways to Mental Continuity"**

**Authors**: Xian Li, Hongmi Lee, and Christopher J. Honey

**Archive**: A frozen version of this entire project — this repository plus the large `data/1_data/` imaging inputs that GitHub does not carry — is archived as a single Zenodo record: https://doi.org/10.5281/zenodo.22260086 (also listed in the paper's Data and materials availability statement).

---

## Overview

This repository contains the analysis code, analysis-ready data, and generated
outputs for a functional magnetic resonance imaging (fMRI) study of how
listeners maintain narrative context across interruptions. Participants
listened to a narrated story either continuously (CT) or with interruptions
inserted — silent pauses in the intact story (IP), theory-of-mind question
periods in the intact story (IT), or silent pauses in a scrambled story (SP).
The analyses trace how the posterior medial cortex (PMC) carries the evolving
narrative context through the interruptions in a transformed format, and how
hippocampal activity at interruption onset and the persistent PMC trace relate
to neural resumption of the story and later recall.

Every statistic and figure reported in the manuscript and Supplementary
Materials can be regenerated from this repository; each analysis script writes
an HTML report, statistics tables, and figures into its own output folder.

## Repository structure

```
mental_continuity/
├── README.md                   # this file — repository guide
├── requirements.txt            # pinned Python dependencies
├── CITATION.cff  LICENSE       # citation metadata (MIT license)
│
├── scripts/                    # analysis scripts
│   ├── README.md               # script-by-script documentation
│   ├── Result{N}_{sub}_{slug}.py   # main paper analyses, numbered by paper sequence
│   ├── supplement/             # supplement analyses (S{N}_{slug}.py, sections S1–S17)
│   ├── figures/                # figure-assembly scripts (figure{1-4}/…, FigS{1-13}_…)
│   └── helper/                 # shared helpers + derive_*.py provenance recipes
│
├── data/                       # every input the scripts read
│   ├── 1_data/                 # imaging inputs (large; shipped via the Zenodo archive)
│   │   └── README.md           # what each extraction is and which analyses read it
│   ├── stimuli/                # narrative transcripts + interruption-epoch timing
│   │   └── README.md           # stimulus documentation
│   ├── beh/                    # behavioral tallies and scored probe responses
│   ├── cohort/                 # subject→condition assignment + scan-QC exclusions
│   ├── derived/                # per-subject derived inputs and whole-brain digests
│   ├── masks/  roi_masks/  schaefer_surf/   # atlases, ROI masks, surface annotations
│   ├── figure_assets/          # schematic art for the figure-assembly scripts
│   └── supplement/             # misc supplement inputs, incl. the per-participant
│                               #   condition-assignment + exclusion record
│                               #   (participant_conditions_and_exclusions.csv)
│
└── output/                     # all generated outputs, one folder per script
    ├── Result{N}_{sub}_{slug}/ # main-text analysis reports, tables, figures
    ├── supplement/             # supplement analysis + figure outputs
    └── figures/figure{1-4}/    # the four main-text composite figures
```

## Key components

### 1. `scripts/` — analysis code

Main paper analyses (`Result{N}_{sub}_{slug}.py`, numbered by the sequence they
appear in the paper), supplement analyses (`supplement/S{N}_{slug}.py`, one per
supplementary section S1–S17), figure assembly (`figures/`), and shared
helpers plus the provenance recipes that rebuild the Result 4 regression
inputs (`helper/`).

**See**: [`scripts/README.md`](scripts/README.md) for the script-by-script
tables, ROI naming, terminology, and the default analysis parameters;
[`scripts/figures/README.md`](scripts/figures/README.md) for the figure
scripts and the shared figure style guide.

### 2. `data/` — analysis inputs

All inputs the scripts read, organized by kind: the imaging extractions
(`1_data/` — region-of-interest (ROI) multivoxel patterns, parcel
timecourses, audio envelopes, and an unfiltered-preprocessing variant),
stimulus transcripts and interruption timing (`stimuli/`), behavioral data
(`beh/`), the cohort manifests that define the 16/19/19/19 analysis sample
(`cohort/`), derived per-subject inputs and whole-brain digests (`derived/`),
and atlases/masks/surface annotations. The narrated audio recordings are not
redistributed for copyright reasons; their amplitude envelopes ship in
`data/1_data/audenv/`.

**See**: [`data/1_data/README.md`](data/1_data/README.md) and
[`data/stimuli/README.md`](data/stimuli/README.md).

### 3. `output/` — generated results

`output/` mirrors `scripts/` one-to-one: each analysis script owns the folder
named by its script stem, holding its HTML report, statistics tables, and
figures. Every number cited in the manuscript or Supplementary Materials
appears in one of these reports or tables. The four main-text composite
figures land in `output/figures/figure{1-4}/`.

**See**: the "Output conventions" section of
[`scripts/README.md`](scripts/README.md).

## Quick start

1. **Install dependencies** (pinned to the versions that generated the shipped
   outputs; Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```
   The first run of a surface-rendering script downloads nilearn's fsaverage
   surfaces into `~/nilearn_data/`, so that first run needs an internet
   connection; afterwards everything runs offline.

2. **Get the imaging inputs.** The GitHub repository tracks everything except
   `data/1_data/` (9.1 GB — over GitHub's per-file size limit). Download the
   data archive from the Zenodo record
   (https://doi.org/10.5281/zenodo.22260086) and unzip it so it forms
   `data/1_data/`; the scripts pick it up with no code change.

3. **Run any analysis script** from its own folder:
   ```bash
   cd scripts
   python Result2_1_PMC-reliable.py
   cd supplement
   python S1_global-ISC.py
   ```
   Each script regenerates its HTML report, tables, and figures under
   `output/`.

4. **Whole-brain scripts (S2, S6, S15) run in digest mode** out of the box:
   the 35 GB Schaefer-400 voxel slabs are not part of the bundle, and the
   scripts rebuild their reports from the shipped per-parcel digest CSVs (set
   `MENTAL_CONTINUITY_WB_DATA_ROOT`, or place the slabs in
   `data/1_data/mvp_raw/n400_net17/`, to force a full recompute).

## Data availability

- **This repository (GitHub)**: all code, outputs, and small data (~0.4 GB) —
  everything except the large imaging inputs in `data/1_data/`.
- **Zenodo (frozen archive)**: this repository is archived as a frozen
  version in a single Zenodo record, https://doi.org/10.5281/zenodo.22260086,
  holding two files: `mental_continuity-1.0.0.zip` (this repository,
  exactly as on GitHub) and `1_data.zip` (the
  large `data/1_data/` imaging inputs that GitHub does not carry). To
  reconstruct the full ready-to-run project from the record alone: unzip the
  repository zip, then unzip the data zip inside its `data/` directory — see
  `data/1_data/README.md`. GitHub cloners need only the data zip.
- **Raw and preprocessed fMRI data**: deposited on OpenNeuro (accession in the
  paper's availability statement).

## Code availability

All analysis and figure-generation code is in `scripts/`. Analyses use fixed
random seeds where applicable, and scripts record per-item failures in their
HTML reports rather than skipping silently.

## Citation

If you use this code or data, please cite the accompanying paper (see
[`CITATION.cff`](CITATION.cff)):

> Li, X., Lee, H., & Honey, C. J. Cortical and Hippocampal Pathways to Mental
> Continuity.

## Contact

For questions about the code, data, or analyses, please open an issue on
GitHub or contact xianl.cogneuro@gmail.com.

## License

This project is licensed under the MIT License.
