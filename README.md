# ACActive
Activity cliff aware active learning with multi-view molecular representation for accelerating hit discovery.

This repository contains an active learning framework for molecular hit discovery.
The model leverages multi-view molecular representations and an activity cliff prediction module to guide iterative sample selection.
By prioritizing informative compounds, the framework aims to improve screening efficiency with limited labels.

# Installation
Install dependency from environment.yml
```bash
conda env create -f environment.yml
```

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-geometric
```
# Data

# Experiment
```bash
python experiments/main.py -dataset ALDH1 -acq exploitation
```

# Prediction
```bash
# regression
python experiments/evaluation.py --input ./data/input.csv --input_unlabel ./data/input_unlabel.csv --output ./result/output.csv

# classification
python experiments/evaluation.py --input ./data/input_classification.csv --input_unlabel ./data/input_unlabel.csv --assay_active_values active act a --assay_inactive_values inactive inact i --output ./result/output.csv
```
The input is training data annotated with assay values.  
`input.csv` must contain two columns: `smiles` and `y` (assay value).

For **classification** tasks, the model requires `assay_active_values` and `assay_inactive_values`.

- `assay_active_values` contains label names corresponding to **active** compounds.
- `assay_inactive_values` contains label names corresponding to **inactive** compounds.

The `input_unlabel` file contains compounds to be experimentally tested.  
`input_unlabel.csv` must contain a `smiles` column.

The output contains a score for each molecule in `input_unlabel.csv`, indicating its selection priority.  
`output.csv` includes two columns: `smiles` and `score`.

# Reference
Traversing chemical space with active deep learning for low-data drug discovery, Nature Computational Science, 2024
```bash
@article{van2024traversing,
  title={Traversing chemical space with active deep learning for low-data drug discovery},
  author={van Tilborg, Derek and Grisoni, Francesca},
  journal={Nature Computational Science},
  volume={4},
  number={10},
  pages={786--796},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```