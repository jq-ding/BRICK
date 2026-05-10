# BRICK

> **Let Brain Rhythm Shape Machine Intelligence for Connecting Dots on Graphs**
> 
> Jiaqi Ding, Tingting Dan, Zhixuan Zhou, Guorong Wu
> 
> **NeurIPS 2025**

## Overview

`BRICK` (**B**rain **R**hythm-**I**nspired **C**onnection on **K**nowledge graphs) is a graph learning framework that takes inspiration from a fundamental property of biological intelligence: the brain communicates and integrates information through **rhythmic neural oscillations and synchronization**, not through static message passing.

Most graph neural networks rely on iterative message passing between neighbors, which suffers from oversmoothing and limited expressiveness on long-range dependencies. BRICK reformulates information propagation as a system of **coupled oscillators with controlled synchronization**, where node states evolve according to a Kuramoto-style dynamics and global structure emerges from local phase coupling. This brain-rhythm-inspired view of graph computation produces representations that are robust, long-range aware, and biologically plausible.

The framework supports three downstream tasks out of the box:
- **Brain data analysis** (graph-level tasks on functional connectivity)
- **Unsupervised clustering** of nodes
- **Node-level prediction**

## Repository Structure

```
BRICK/
├── source/
│   ├── data/
│   │   ├── create_dataset.py
│   │   └── dataset.py            # Data loaders for different brain datasets
│   ├── modules/
│   │   ├── GST.py                # Graph Scattering Transform
│   │   └── kuramoto_solver.py    # Kuramoto solver for oscillator synchronization
│   ├── brick.py                  # Main BRICK model
│   └── utils.py
├── toy_example.py                # Minimal example to verify the installation
├── train_brain.py                # Train and evaluate on brain data
├── train_cluster.py              # Unsupervised node clustering
└── train_node.py                 # Node-level prediction
```


## Datasets

BRICK has been validated on:

- **Brain data**: HCP (task-state and resting-state fMRI), ADNI, OASIS, PPMI
- **Standard graph benchmarks**: Cora, Citeseer, Pubmed, OGB

See `source/data/dataset.py` for the full list of dataset loaders.

## Citation

```bibtex
@inproceedings{ding2025let,
  title     = {Let Brain Rhythm Shape Machine Intelligence for Connecting Dots on Graphs},
  author    = {Ding, Jiaqi and Dan, Tingting and Zhou, Zhixuan and Wu, Guorong},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025}
}
```

## Contact

Jiaqi Ding — `jiaqid@cs.unc.edu`
[Personal website](https://jq-ding.github.io/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=5h5qru8AAAAJ)
