# GNNs Don't Need Backprop

This is the official repository for the paper
[`GNNs Don't Need Backprop`](https://pwelke.de/publications/goupil2026gnnsdontneedbackprop.pdf),
published as an oral at ESANN 2026.

The repository contains the code for the main sampling pipeline, the trained
baseline, and the repeated-sampling variant used in the paper.

## Repository Layout

- `src/run.py`
  Runs the main pipeline: sample random GNN encoders, select one with ALI on
  the training split, freeze it, extract graph features, then train an MLP.

- `src/run_baseline.py`
  Runs the end-to-end baseline where the GNN encoder and classifier are trained
  jointly.

- `src/run_multi.py`
  Runs the repeated-sampling variant, where a new random encoder is selected
  for each training run.

- `src/`
  Contains the dataset loading code, ALI metric implementation, random MPNN
  sampling code, baseline models, and MLP training utilities.

- `config/`
  Contains reference YAML configurations for the main experiments.

## Dependencies

You can install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Depending on your setup, you may still prefer to install `torch` and
`torch-geometric` separately with wheels that match your CPU or CUDA
environment.

## Running

Run from the repository root.

Default runs using the reference configs in `config/`:

```bash
python src/run.py
python src/run_baseline.py
python src/run_multi.py
```

You can also pass a YAML config file:

```bash
python src/run.py --config /path/to/config.yaml
python src/run_baseline.py --config /path/to/config_baseline.yaml
python src/run_multi.py --config /path/to/config_multi.yaml
```

Reference configs:
- [`config/config.yaml`](config/config.yaml)
- [`config/config_baseline.yaml`](config/config_baseline.yaml)
- [`config/config_multi.yaml`](config/config_multi.yaml)
