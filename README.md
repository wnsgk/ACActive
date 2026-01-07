# ACActive
Activity cliff aware active learning with multi-view molecular representation for accelerating hit discovery.

This repository contains an active learning framework for molecular hit discovery.
The model leverages multi-view molecular representations and an activity cliff prediction module to guide iterative sample selection.
By prioritizing informative compounds, the framework aims to improve screening efficiency with limited labels.

# Installation

## Cuda 12.1
Install dependency from environment.yml
```bash
conda env create -f environment.yml
```

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-geometric
```

## Cuda 11.8

```bash
# 1. create environment directory
conda create -n acactive python=3.9 -y

# 2. activate
conda activate acactive

# 3. PyTorch CUDA 11.8
conda install pytorch==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. 
conda install -c conda-forge -y numpy=1.26.0 scipy=1.13.1 scikit-learn=1.6.1 pandas=1.5.3 matplotlib-base=3.9.4 pillow=11.3.0 rdkit=2024.09.6 h5py=3.13.0 pyyaml=6.0.2 tqdm=4.67.1 joblib=1.5.1

# 5. lightgbm/xgboost
conda install lightgbm xgboost -c conda-forge -y

# 6. PyTorch Geometric (CUDA 11.8)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-geometric==2.6.1

# 7. other dependencies
pip install transformers==4.49.0 unimol-tools==0.1.4.post1 \
addict==2.4.0 \
aiohttp==3.13.2 aiosignal==1.4.0 async-timeout==5.0.1 \
annotated-types==0.7.0 attrs==25.4.0 \
charset-normalizer==3.4.4 click==8.1.8 \
eval-type-backport==0.3.1 frozenlist==1.8.0 fsspec==2025.10.0 \
gitdb==4.0.12 gitpython==3.1.45 \
hf-xet==1.2.0 huggingface-hub==0.29.3 \
idna==3.11 llvmlite==0.43.0 multidict==6.7.0 numba==0.60.0 \
platformdirs==4.4.0 propcache==0.4.1 protobuf==6.33.2 psutil==7.1.3 \
pydantic==2.12.5 pydantic-core==2.41.5 \
rdkit-pypi==2022.9.5 regex==2025.11.3 requests==2.32.5 \
safetensors==0.7.0 seaborn==0.13.2 sentry-sdk==2.47.0 \
smmap==5.0.2 tokenizers==0.21.1 \
typing-inspection==0.4.2 \
urllib3==2.6.2 wandb==0.21.3 yarl==1.22.0
```

# Data

# Experiment
```bash
python experiments/main.py -dataset ALDH1 -acq exploitation
```

# Prediction
```bash
# classification
python experiments/evaluation.py --input ./data/input_classification.csv --input_unlabel ./data/input_unlabel.csv --assay_active_values active act a --assay_inactive_values inactive inact i --output ./result/output.csv

# regression
python experiments/evaluation.py --input ./data/input.csv --input_unlabel ./data/input_unlabel.csv --output ./result/output.csv

```
The input is training data annotated with assay values.  
`input.csv` must contain two columns: `smiles` and `y` (assay value).

For **classification** tasks, the model requires `assay_active_values` and `assay_inactive_values`.

- `assay_active_values` contains label names corresponding to **active** compounds.
- `assay_inactive_values` contains label names corresponding to **inactive** compounds.

For **regression** tasks, higher y values indicate better performance. Add `--is_reverse` when lower y values indicate better performance.

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