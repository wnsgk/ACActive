/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/labeled.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_4.csv \
    --input_val_col label \
    --input_smiles_col smiles \
    --input_unlabel_smiles_col smiles

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/assay.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_assay.csv \
    --input_val_col IC50 \
    --input_smiles_col SMILES \
    --input_unlabel_smiles_col smiles

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/assay.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_assay2.csv \
    --input_val_col LD50 \
    --input_smiles_col Smiles \
    --input_unlabel_smiles_col smiles

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/pubchem_CDK7_bioactivity.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_assay3.csv \
    --input_val_col Activity \
    --input_smiles_col Canonical_SMILES \
    --input_unlabel_smiles_col smiles \
    --assay_active_values Active act \
    --assay_inactive_values Inactive ina

### 2026 01 07 run
/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/ACActive/experiments/evaluation.py \
    --input ./data/input.csv \
    --input_unlabel ./data/input_unlabel.csv \
    --output ./result_2/output_acactive_20260107.csv

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/ACActive/experiments/evaluation.py \
    --input ./data/input.csv \
    --input_unlabel ./data/input_unlabel.csv \
    --output ./result_2/output_acactive_reverse_20260107.csv \
    --is_reverse
    
/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/input.csv \
    --input_unlabel ./data/input_unlabel.csv \
    --output ./result_2/output_20260107.csv \
    --input_val_col y \
    --input_smiles_col smiles \
    --input_unlabel_smiles_col smiles

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/input.csv \
    --input_unlabel ./data/input_unlabel.csv \
    --output ./result_2/output_reverse_20260107.csv \
    --input_val_col y \
    --input_smiles_col smiles \
    --input_unlabel_smiles_col smiles \
    --is_reverse

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/assay.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_assay_regression_reverse_20260107.csv \
    --input_val_col LD50 \
    --input_smiles_col Smiles \
    --input_unlabel_smiles_col smiles \
    --is_reverse

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/assay.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_assay_regression_20260107.csv \
    --input_val_col LD50 \
    --input_smiles_col Smiles \
    --input_unlabel_smiles_col smiles

/envs/drugvlab/envs/tool_hit_ranking_acactive/bin/python /envs/drugvlab/tools/tool_hit_ranking_acactive/experiments/evaluation.py \
    --input ./data/pubchem_CDK7_bioactivity.csv \
    --input_unlabel ./data/unlabeled.csv \
    --output ./result_2/output_assay_binary_20260107.csv \
    --input_val_col Activity \
    --input_smiles_col Canonical_SMILES \
    --input_unlabel_smiles_col smiles \
    --assay_active_values Active act \
    --assay_inactive_values Inactive ina