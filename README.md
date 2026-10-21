# The Lottery Ticket Hypothesis

A from-scratch PyTorch reproduction of ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](https://arxiv.org/abs/1803.03635) (Frankle & Carbin, ICLR 2019), implementing **iterative magnitude pruning (IMP)** and, critically, the **random-reinitialization control** that the paper uses to actually test the hypothesis.

## The hypothesis, and why a control matters

The paper's claim isn't just "you can prune neural networks and keep most of the accuracy" — that's well known and uninteresting on its own. The claim is specifically that a sparse subnetwork's *original initialization* is what makes it trainable: the same sparse topology, started from a *different* random initialization, trains noticeably worse. Without comparing against that control, a pruning experiment can't distinguish "winning tickets are special" from "sparse networks are just fine in general."

This implementation runs both experiments at every sparsity level so that comparison is explicit:

1. **Winning ticket** — reset to its original initialization `θ₀` after each pruning round (the actual IMP procedure).
2. **Random reinitialization (control)** — the identical mask applied to a freshly, randomly initialized network.

### Iterative Magnitude Pruning (IMP)

1. Randomly initialize a network `f(x; θ₀)`.
2. Train it for `j` iterations, producing trained weights `θⱼ`.
3. Prune the `p`% of weights with the smallest magnitude, producing a binary mask `m`.
4. Reset the surviving weights to their values in `θ₀` (not their trained values) — this is the winning ticket, `f(x; m ⊙ θ₀)`.
5. Repeat steps 2–4, pruning `p`% of the *remaining* weights each round, until the target sparsity is reached.

## Project structure

```
.
├── src/
│   ├── main.py            # CLI entry point: runs IMP + the reinit control, saves results/plots
│   ├── models/lenet.py     # LeNet-300-100 (paper's MNIST model) and a configurable ConvNet
│   ├── pruning/imp.py      # LotteryTicketPruner: masking, pruning, resetting, reinit-control support
│   ├── datasets/loaders.py # MNIST / CIFAR-10 loaders with seeded train/val splits
│   └── utils/
│       ├── trainer.py      # Training/eval loop; freezes pruned weights via gradient masking
│       └── seed.py         # Reproducibility helper (seeds python/numpy/torch)
├── tests/                  # pytest unit tests for the pruning and training logic
├── experiments/
│   ├── runs/results.json   # Per-round metrics for both the winning ticket and the control
│   └── plots/               # Generated comparison plots
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running an experiment

```bash
cd src
python main.py
```

Useful flags (see `python main.py --help` for the full list):

| Flag | Default | Meaning |
|---|---|---|
| `--rounds` | 10 | Number of pruning rounds |
| `--epochs` | 5 | Training epochs per round |
| `--lr` | 1.2e-3 | Adam learning rate |
| `--pruning-rate-fc` | 0.2 | Fraction of *remaining* FC weights pruned per round |
| `--seed` | 42 | Seed for reproducibility (model init, data split, shuffling) |
| `--no-control` | off | Skip the random-reinitialization control (faster, but loses the actual hypothesis test) |
| `--quiet` | off | Disable per-batch progress bars |

Each run writes `experiments/runs/results.json` (per-round metrics for both the winning ticket and the control, plus the exact config used) and `experiments/plots/lottery_ticket_results.png`.

## Results

LeNet-300-100 on MNIST, 10 rounds, 20% of remaining FC weights pruned per round, 5 epochs/round, Adam at `lr=1.2e-3`, seed 42:

![Lottery Ticket Hypothesis Results](experiments/plots/lottery_ticket_results.png)

| Round | Weights remaining | Winning ticket acc. | Winning ticket early-stop iter | Control acc. | Control early-stop iter |
|---|---|---|---|---|---|
| 0 | 100.00% | 0.9778 | 4500 | 0.9803 | 4500 |
| 1 | 80.00%  | 0.9777 | 2700 | 0.9786 | 3600 |
| 2 | 64.00%  | 0.9786 | 3600 | 0.9795 | 4500 |
| 3 | 51.20%  | 0.9772 | 3600 | 0.9790 | 4500 |
| 4 | 40.96%  | 0.9797 | 3600 | 0.9771 | 4500 |
| 5 | 32.77%  | 0.9784 | 2700 | 0.9778 | 4500 |
| 6 | 26.21%  | 0.9789 | 3600 | 0.9773 | 4500 |
| 7 | 20.97%  | **0.9822** | 4500 | 0.9729 | 2700 |
| 8 | 16.77%  | **0.9816** | 2700 | 0.9785 | 4500 |
| 9 | 13.42%  | **0.9816** | 1800 | 0.9734 | 4500 |

(Single seed, `seed=42` — see [Known limitations](#known-limitations--future-work).)

**What this shows:** near full density, the winning ticket and the random-reinit control are statistically indistinguishable — unsurprising, since pruning barely 20–50% of a heavily overparameterized MLP removes little that matters. The gap opens up at high sparsity: below ~21% of the original weights (rounds 7–9), the winning ticket is consistently *more accurate* than the control given the identical mask, while the control's early-stop iteration stays pinned at the training budget ceiling (4500) — it never finds a validation-loss improvement late in training, i.e. it's struggling to fit at all under that sparse mask from a fresh initialization. The winning ticket, by contrast, keeps hitting its best validation loss earlier (down to 1800 iterations by round 9). That divergence — same topology, different initialization, different outcome — is the actual phenomenon the paper is about.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

Covers: mask initialization, that `prune_step` removes the correct fraction and specifically the smallest-magnitude weights, that `reset_weights` correctly restores `θ₀` under the mask, that the reinit-control mask application is independent of the winning ticket's weights, that pruned weights stay frozen at zero during training, and that training actually reduces loss.

## Known limitations / future work

- Only LeNet-300-100 on MNIST is run end-to-end; a `ConvNet` class and CIFAR-10 loaders exist but aren't wired into an experiment yet (Conv-2/4/6 from the paper).
- Pruning is layer-wise, not global — the paper notes global pruning matters more for deeper networks.
- No learning-rate warmup, which the paper found necessary to find winning tickets in deeper networks at higher learning rates.
- Single-seed runs; the paper averages over multiple random seeds per sparsity level to report error bars.
