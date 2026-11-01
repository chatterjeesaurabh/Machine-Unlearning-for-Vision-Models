# Machine Unlearning for Vision Models (ResNet-18 / CIFAR-10)

CS6886 - Systems Engineering for Deep Learning, IIT Madras - course project by
**Saurabh Chatterjee (NS26Z334)** and **Reva Bharara (NS26Z212)**.

Repository: <https://github.com/chatterjeesaurabh/Machine-Unlearning-for-Vision-Models>

We remove the influence of a *forget set* from a trained ResNet-18 without retraining from scratch, compare five unlearning methods
(Gradient Ascent, Gradient Ascent + Descent, Teacher-Student, Random Relabeling, Boundary Relabeling), each with and without
Gradient based Weight Saliency, against a retrained-from-scratch reference, and evaluate them with accuracy, a membership inference
attack, embedding entanglement, linear probing, perturbed inputs, post-unlearning quantization and cost.

- Short report (2 pages): [`report/short_report.pdf`](report/short_report.pdf)
- Full report (20 pages): [`report/report.pdf`](report/report.pdf)
- All experiment records (1,845 runs, tables, figures, method pages): [`records/`](records/) - start at [`records/README.md`](records/README.md)

## Headline results (3 seeds, mean; avg_gap = mean |difference to Retrain| over forget / retain / test accuracy and MIA, lower is better)

| Scenario | Doing nothing | Best method (configuration) | avg_gap | Time vs. retraining |
|---|---|---|---|---|
| S1: forget class *airplane* | 29.05 | Boundary Relabeling, our C1-mix (logit-2 labels + retain pass + mixed BN batches) | **0.16** (paper setting: 6.92) | 28 s vs 402 s |
| S2-10: forget 10% of cat/dog/truck | 4.47 | GA + Descent (beta 0.999) | 3.52 | 24 s vs 479 s |
| S2-30 | 4.75 | Random Relabeling + Weight Saliency | 3.10 | 32 s vs 445 s |
| S2-50 | 5.26 | Random Relabeling + Weight Saliency | 2.58 | 32 s vs 368 s |

## Repository layout

```
src/mu/                 library
  data.py               CIFAR-10 kept on the GPU (uint8) + GPU-side augmentation
  splits.py             45k/5k/10k split and forgetting scenarios (S1c0, S2k359p{10,30,50}), entanglement partitions
  models/resnet.py      ResNet-18 (CIFAR stem)
  unlearn/methods.py    the five unlearning methods (+ Teacher-Student joint variant)
  unlearn/engine.py     shared training loop: early stopping, BN options, mixed BN batches, masked updates
  saliency.py           Gradient based Weight Saliency mask
  eval/                 accuracy, loss-based MIA, entanglement, linear probe / NCC, embeddings
  best_configs.py       best configuration of every method per scenario
  records.py            run records (records/runs/<run_id>/)
scripts/                entry points (training, unlearning, grids, evaluations, analysis) - see below
configs/methods.yaml    default hyperparameters of every method (paper settings)
configs/final_hparams.yaml  hyperparameters selected from the pilot grid (main grid)
data/splits/            the exact index splits used for all results (created automatically if missing)
records/                every run (config, per-epoch history, metrics) + results tables, figures, method pages, journal
report/                 LaTeX sources and PDFs of the reports (bash report/build.sh)
env.sh                  sets MU_ROOT and PYTHONPATH
```

## 1. Check out and set up

```bash
git clone https://github.com/chatterjeesaurabh/Machine-Unlearning-for-Vision-Models.git
cd Machine-Unlearning-for-Vision-Models

# Python 3.11 environment with CUDA (results were produced with the versions in requirements.txt)
python -m venv .venv && source .venv/bin/activate          # or a conda environment
pip install -r requirements.txt

# CIFAR-10 (python version) -> data/CIFAR10/cifar-10-batches-py/
mkdir -p data/CIFAR10
wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C data/CIFAR10

source env.sh        # sets MU_ROOT=<repo root> and PYTHONPATH=src (run in every new shell)
```

Hardware used: one NVIDIA A100-40GB (a second one for parallel batches). Every job needs < 4 GB of GPU memory, so a smaller GPU
works. Scripts take `--gpu <id>` (default 1 - use `--gpu 0` on a single-GPU machine) and `--mem_frac` (cap on the fraction of GPU memory).
A project-wide lock (`/tmp/mu_cs6886_gpu.lock`) runs one GPU job at a time unless a batch runner holds it (see section 4).

## 2. Quick start (about 20 minutes)

```bash
# Original model on the 45k training pool (182 epochs, ~7 min on an A100) -> outputs/checkpoints/orig__original__s0.pt
python scripts/train.py --scenario orig --seed 0 --gpu 0

# Retrain-from-scratch reference for scenario S1c0 (forget class 0), seed 0 (~7 min)
python scripts/train.py --scenario S1c0 --seed 0 --gpu 0

# Unlearn with the best S1 configuration of Boundary Relabeling (C1-mix), ~30 s
python scripts/unlearn.py --scenario S1c0 --seed 0 --method RLbnd --gpu 0 \
    --hp nb_search=logit2 lr=0.001 use_retain=1 bn_mix_retain=1 --tag imp5_bndlogit2mix

# the same with the Gradient based Weight Saliency mask (50% of the weights updated)
python scripts/unlearn.py --scenario S1c0 --seed 0 --method RLbnd --mask saliency --mask_ratio 0.5 --gpu 0 \
    --hp nb_search=logit2 lr=0.001 use_retain=1 bn_mix_retain=1 --tag imp5_bndlogit2mix
```

Each run prints a `FINAL` line and writes `records/runs/<run_id>/` (`config.json` with the exact command and hyperparameters,
`history.csv`, `metrics.json` with forget/retain/test accuracy, MIA and avg_gap to the Retrain reference, `per_sample.npz`,
`summary.md`). Collect all runs into one table with `python scripts/aggregate.py` (-> `records/results/all_runs.csv`) and browse
them in `records/index.html` (`python scripts/build_index.py`).

**Scenarios** (`--scenario`): `S1c0` = forget all 4,500 airplane images; `S2k359p10|30|50` = forget a random 10/30/50% of classes
3, 5, 9 (cat, dog, truck). **Methods** (`--method`): `GA`, `GAGD`, `TS`, `RLrand`, `RLbnd` (and `none` = evaluate the Original).
Default hyperparameters come from `configs/methods.yaml`; `--hp key=value ...` overrides them, e.g. `bn_mix_retain=1` (mixed
BatchNorm batches), `stop_every_iters=2 stop_forget_acc=retrain` (iteration-level early stop at the Retrain forget accuracy),
`beta` (GA + Descent), `alpha gamma kd_T msteps forget_batch_size` (Teacher-Student), `nb_search=fgsm|pgd|logit2 eps use_retain`
(Boundary Relabeling). Configuration names (G3, D1, T1, R1, C1-mix, ...) are defined in `src/mu/best_configs.py` and the method pages
`records/methods/*.md`.

## 3. Reproduce the headline results (best configurations, all scenarios, 3 seeds)

```bash
python scripts/reproduce_best.py > outputs/reproduce_batch.sh      # add --scenarios S1c0 --seeds 0 for a subset
bash scripts/gpu_batch.sh outputs/reproduce_batch.sh              # 13 trainings (~1.5 h) + 120 unlearning runs (~1 h) on one A100
python scripts/aggregate.py && python scripts/build_eval_tables.py
```

`reproduce_best.py` trains the Original and the 12 Retrain references (4 scenarios x 3 seeds), evaluates the Original on every
scenario and runs the best configuration of every method with and without Weight Saliency, using the exact command stored in each
run record (runs that already exist are skipped; `--force` re-runs them). Results: `records/results/all_runs.csv`,
`records/results/method_comparison.md`. Seeds fix the S2 forget-set sampling and all randomness of training / unlearning; small
run-to-run differences (< 0.5 points) remain because of non-deterministic cuDNN kernels.

## 4. Full experimental pipeline (how every table of the report was produced)

Each `*_grid.py` script prints a bash batch of `train.py` / `unlearn.py` commands and skips runs that already exist. Run a batch
sequentially with `bash scripts/gpu_batch.sh <batch.sh>`, or in parallel (one command per line, alternating two GPUs) with
`bash scripts/run_parallel.sh <jobs.txt> <n_parallel> <log_dir>`.

| Stage | Commands | Output |
|---|---|---|
| References | `bash scripts/queue_refs.sh 0` (Original + 12 Retrains) | `outputs/checkpoints/`, `records/runs/` |
| Pilot grid (seed 0, S1 + S2-30) | `python scripts/pilot_grid.py > b.sh; bash scripts/gpu_batch.sh b.sh`; `python scripts/select_hparams.py` | `configs/final_hparams.yaml` (selection by validation-only `sel_gap`) |
| Main grid (5 methods x +/-WS x 4 scenarios x 3 seeds) | `python scripts/main_grid.py > b.sh; bash scripts/gpu_batch.sh b.sh`; `python scripts/analyze_main.py` | `records/results/main_table.md` |
| Improvements (BN fixes, neighbour search, Teacher-Student schedules) and 3-seed confirmation | `scripts/improve_grid.py`, `bnfix_grid.py`, `ts_grid.py`, `improve_confirm.py` (same pattern) | `records/results/method_comparison.md`, `runs_<method>.md` (`scripts/method_runs_table.py`) |
| Weight Saliency ablation (mask ratio, random mask) | `scripts/ablation_grid.py` | `records/methods/M4_weight_saliency.md` |
| Linear probe / NCC / t-SNE (S1) | `python scripts/probe_best_models.py` | `records/results/probe_best.md`, `records/figures/fig_embed_best_S1c0.png` |
| Perturbed forget samples | `python scripts/eval_perturb.py --pattern 'S2k359p30__*'`; `python scripts/analyze_robustness.py` | `records/results/perturb_table.csv` |
| Post-unlearning quantization (best configs) | `python scripts/quant_best_grid.py > b.sh; bash scripts/gpu_batch.sh b.sh`; `python scripts/analyze_quant_best.py`; `python scripts/diag_quant_margin.py`; `python scripts/plot_quant_best.py` | `records/results/quant_best.md`, `quant_margin_diag.md`, `fig_quant_best_*.png` |
| Entanglement partitions | `python scripts/make_entangle_partitions.py`; `python scripts/make_entangle_partitions_excl.py` | `data/splits/<scenario>-E*.npz` |
| Entanglement Exp 1 (S2) and Exp 2 (all) | `python scripts/entangle_grid.py <seed> > b.sh; bash scripts/gpu_batch.sh b.sh` (seeds 0, 1, 2) | |
| Entanglement Exp 1 (S1, other partitions unused) | `python scripts/entangle_excl_grid.py A > a.txt; python scripts/entangle_excl_grid.py B > b.txt; bash scripts/run_parallel.sh a.txt 15 logs_a; bash scripts/run_parallel.sh b.txt 15 logs_b` | |
| Entanglement analysis | `python scripts/aggregate.py; python scripts/analyze_entangle_exps.py` | `records/results/entanglement_experiments.md` |
| Evaluation tables / docs | `python scripts/build_eval_tables.py` | `outputs/eval_tables.json` -> `records/results/evaluation.md` |
| Reports | `bash report/build.sh --figures` (full report), `bash report/build.sh short` | `report/report.pdf`, `report/short_report.pdf` |

Post-hoc evaluations (`probe_best_models.py`, `eval_perturb.py`, `eval_quant.py`, `diag_quant_margin.py`) need the model checkpoints,
which `unlearn.py` saves to `outputs/checkpoints/` by default (`--save_ckpt 1`); checkpoints are not part of the repository (~45 MB
each), so run the corresponding unlearning commands first.

## 5. Evaluation protocol in brief

- **Accuracy**: forget / retain / test accuracy (eval mode, no augmentation); **avg_gap** to the Retrain model of the same scenario
  and seed. Hyperparameters were selected with `sel_gap` (validation accuracy + validation-negative MIA), never with test data.
- **MIA**: per-sample cross-entropy loss; members = forget images, non-members = test images of the same class(es), balanced;
  logistic regression with 5-fold cross-validation (`src/mu/eval/mia.py`). Ideal = Retrain (~50%).
- **Entanglement**: mean distance of each forget image to its 50 nearest retain images in the Original's 512-d features; forget set
  split into low / medium / high thirds; Exp 1 unlearns each third separately, Exp 2 sequentially (low -> high).
- **Embedding separability** (S1): linear probe and nearest class centroid on frozen features, 10-NN purity, t-SNE.
- **Robustness**: Gaussian-perturbed forget images; simulated weight-only per-channel INT8 / INT6 / INT4 quantization.
- **Cost**: unlearning wall-clock time and peak GPU memory, recorded in every `metrics.json`.

## References

Implementations follow (and borrow small pieces from) the reference code of the Weight Saliency (SalUn) and RUM papers.

1. Kurmanji et al., "Towards Unbounded Machine Unlearning," NeurIPS 2023 (Teacher-Student / SCRUB, GA + Descent / NegGrad+) - code: <https://github.com/meghdadk/SCRUB>.
2. Chen et al., "Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision Boundary," CVPR 2023.
3. Fan et al., "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in both Image Classification and Generation," ICLR 2024 - code: <https://github.com/OPTML-Group/Unlearn-Saliency>.
4. Zhao et al., "What Makes Unlearning Hard and What to Do About It," NeurIPS 2024 (RUM) - code: <https://github.com/kairanzhao/RUM>.
5. "An Illusion of Unlearning? Assessing Machine Unlearning Through Internal Representations," PMLR 2026.
6. "The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples," NeurIPS 2025.
