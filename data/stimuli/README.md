# `data/stimuli/` — stimulus transcripts and interruption-epoch timing

The narrated audio recordings are professionally narrated performances of
published works (Raymond Carver, "So Much Water So Close to Home"; Andy
Christie, "Not the Fall That Gets You") and are not redistributed, for
copyright reasons. This folder provides the audio-related information the
analyses rely on:

- `transcript_main-narrative_18segments.txt` — full transcript of the main
  narrative, divided into its 18 story segments (`1:` … `18:`). The segment
  boundaries are the positions at which interruption epochs were inserted
  (17 insertion points).
- `transcript_second-narrative_12segments.txt` — full transcript of the
  second narrative, divided into its 12 story segments (11 insertion points).
- `interruption_epochs.csv` — the interruption-epoch onset and offset
  timestamps used by every analysis, one row per task x condition x epoch:
  0-indexed TR indices into the run's volume grid (onset inclusive, offset
  exclusive; TR = 1.5 s) plus the equivalent times in seconds. These are the
  exact values returned by `scripts/helper/data_structure.py::
  get_interruption_epochs`, which all analysis scripts call. The `continuous`
  rows mark the segment-boundary positions in the uninterrupted condition
  (zero-length: the continuous narrative has no interruption epochs).

Related stimulus information elsewhere in the bundle:

- `data/supplement/tom_probes_table.csv` — the theory-of-mind vignette and
  question texts of the IT condition, with answer keys (Table S1).
- `data/aud-info-main.xlsx` and `data/supplement/carver_scramble_map.csv` —
  the sub-section reordering that builds the scrambled (SP) narrative.
- `data/1_data/audenv/` — the amplitude envelope of each run's audio (one
  `values` time series per task x condition), the acoustic input the figure
  scripts draw.
