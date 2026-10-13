# What Pooled Frame-Level AUC Measures in Weakly-Supervised Video Anomaly Detection

Code and archived results for the measurement paper. The paper reports no new
method; it decomposes a standard reporting metric and tests the decomposition
with two label-free interventions.

**Target journal:** Image and Vision Computing (Elsevier)

**Repository:** https://github.com/SVIL2024/Pooled-AUC-VAD

---

## What is here

```
src/           scripts that produce every number and figure in the paper
results/       archived result files written by those scripts -- the evidence
data/          the two ShanghaiTech score archives, so the third benchmark can
               be checked without rerunning inference
exploration/   method lines that were tried and falsified, kept for transparency
docs/          what was deliberately left out of the release, and why
```

Nothing in `results/` was typed by hand. Each file is written by the script named
in the table below.

## Reproducing the paper's numbers

The pipeline reads archived per-clip scores and writes one JSON per stage. The
score archives for UCF-Crime and XD-Violence are regenerable from the released
checkpoints; the ShanghaiTech ones ship in `data/`.

| Paper item | Script | Output |
|---|---|---|
| Table 1, main measurements | `src/s2_table.py` | `results/s2_table.json` |
| Intervention A and B | `src/s3_deoffset.py` | `results/s3_deoffset.json` |
| Retained shares | `src/s4_g5redo.py` | `results/s4_g5redo.json` |
| Bootstrap intervals | `src/s5_ci.py` | `results/s5_ci.json` |
| Sufficiency and controls | `src/s6_sufficiency.py` | `results/s6_sufficiency.json` |
| ShanghaiTech scoring | `src/s2b_score_sht.py` | score archives |
| ShanghaiTech order check | `src/s2v_align2.py` | `results/s2v_align2.json` |
| Within-video pair weight | `src/audit_within_w.py` | `results/audit_within_w.json` |
| Figures 1 to 4 | `src/make_figures.py` | `fig*.pdf` |
| Graphical abstract | `src/make_graphical_abstract.py` | `graphical_abstract.pdf` |

Every stage takes its paths as arguments. A typical sequence:

```bash
python src/s2b_score_sht.py --out data/cdv_scores_ucf2sht.npz   # needs the checkpoints
python src/s2_table.py      --p3 <scores> --lens <lengths> --sht <sht scores> --out results/s2_table.json
python src/s3_deoffset.py   --p3 <scores> --lens <lengths> --sht <sht scores> --out results/s3_deoffset.json
python src/s4_g5redo.py     --p3 <scores> --lens <lengths> --sht <sht scores> --out results/s4_g5redo.json
python src/s5_ci.py         --p3 <scores> --lens <lengths> --sht <sht scores> --boot 200 --out results/s5_ci.json
python src/s6_sufficiency.py --p3 <scores> --lens <lengths> --sht <sht scores> --out results/s6_sufficiency.json
python src/make_figures.py
```

`--p3`, `--lens` and `--sht` are the score and length archives the original run
consumed; consult each script's `--help`.

## Checking the paper against the archives

This is the part worth running, because it is the check that keeps the paper
honest.

```bash
python src/audit_tex_numbers.py "/path/to/Pooled Frame-Level AUC Measures.tex"
```

The script extracts every decimal in the manuscript and requires each one to be
traceable to a value in `results/`. Anything it cannot source is reported by
name, and it exits non-zero while such numbers remain.

Four numbers in the current manuscript are deliberately untraceable and will be
listed by the audit: they are figures quoted from Song and Lee
(arXiv:2608.21854) inside Related Work, not measurements made here. The audit
cannot yet distinguish a cited number from a measured one, which is a known
limitation of the script rather than a gap in the evidence.

```bash
python src/recompute_sht_reference.py          # reproduces the ShanghaiTech within-video AUC
python src/record_sht_split.py                # per-video anomalous fractions, both resolutions
python src/check_journal_compliance.py imavis  # word count, highlights, abstract
python src/scan_for_secrets.py .              # run before publishing anything derived from this
```

`recompute_sht_reference.py` is worth reading for one reason. Within-video AUC is
a **mean of per-video AUCs**, with single-class and constant-score videos
skipped. A reimplementation that pools within-video pairs across videos computes
a different quantity and will disagree with the paper on identical scores. Two
earlier drafts of that script did exactly that and produced a spurious sign
reversal. The estimator here is the reference one from `src/s2_table.py`.

## Environment

```bash
pip install -r requirements.txt
```

`src/s2b_score_sht.py` additionally needs the two released checkpoints and their
source repositories. Point at them with environment variables:

| Variable | Points at |
|---|---|
| `VADCLIP_ROOT` | the VadCLIP source tree (default `VadCLIP`) |
| `CDV_ROOT` | the cross-domain source tree (default `crossdomain-vad`) |
| `VAD_ROOT` | the run root, where `runs/` and `results/` live (default `.`) |
| `SHT_FRAME_MASKS` | the ShanghaiTech `test_frame_mask` directory |

## Data sources

UCF-Crime, XD-Violence and ShanghaiTech, all cited in the paper. The ShanghaiTech
frame masks are redistributed by the dataset itself and are not mirrored here;
`src/record_sht_split.py` reads them once `SHT_FRAME_MASKS` points at them.

No new data were collected. No human subjects and no personally identifying data.

## A note on `exploration/`

These scripts are not part of the paper's evidence. They record method lines that
were implemented and then falsified, several more than once. They are included
because a negative result described but not shipped is not checkable, and because
the paper's framing depends on having tried them. They will not run without the
checkpoints and may expect paths that no longer exist.

## Citation

If this code is used, please cite the paper. This repository is the artifact
referenced in the manuscript's Data Availability statement; every number in the
paper is reproducible from `results/` by the scripts in `src/`. The decomposition of pooled
frame-level AUC into within-video and cross-video comparisons is not claimed as
new; it was carried out by Song and Lee, *Look Inside Each Video: Rethinking How
Video Anomaly Detection Is Evaluated*, arXiv:2608.21854 (2026). What this
repository implements on top of that decomposition is stated in the paper's
Introduction.

## License

MIT. See `LICENSE`.