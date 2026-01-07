"""

This script contains the main active learning loop that runs all experiments.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from math import ceil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
from active_learning.nn_contrastive import RfEnsemble, Ensemble_triple, AcqModel
# from active_learning.nn_rnn_loss_sum import RfEnsemble, Ensemble_triple, AcqModel
from active_learning.data_prep import MasterDataset, MasterDataset2Labeled, MasterDataset2Unlabeled
from active_learning.data_handler import Handler
from active_learning.utils import Evaluate, to_torch_dataloader, to_torch_dataloader_multi
from active_learning.acquisition import Acquisition, logits_to_pred
import os
import random
import logging
from collections import defaultdict
import torch.nn.functional as F

INFERENCE_BATCH_SIZE = 128
TRAINING_BATCH_SIZE = 64
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from sklearn.model_selection import train_test_split
def stratified_index_split_with_positive(y: torch.Tensor, test_size=0.2, random_state=42):
    # y가 GPU에 있으면 CPU로 옮기기 (train_test_split은 numpy 기반)
    if y.dim() > 1 and y.size(1) == 1:
        y = y.view(-1)

    # numpy 변환 (train_test_split은 numpy만 받음)
    if y.is_cuda:
        y_np = y.cpu().numpy()
    else:
        y_np = y.numpy()

    n = len(y_np)
    indices = torch.arange(n)

    # stratified split 시도
    train_idx, valid_idx = train_test_split(
        indices.numpy(), test_size=test_size, stratify=y_np, random_state=random_state
    )

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    valid_idx = torch.tensor(valid_idx, dtype=torch.long)

    # valid에 positive가 하나도 없으면 강제 조정
    if y[valid_idx].sum() == 0:
        pos_idx = torch.where(y == 1)[0]
        neg_idx = torch.where(y == 0)[0]

        # positive 하나를 valid로 이동
        chosen_pos = pos_idx[torch.randint(len(pos_idx), (1,))]
        remaining_pos = pos_idx[pos_idx != chosen_pos]

        n_valid = int(n * test_size)
        remaining_needed = n_valid - 1
        chosen_neg = neg_idx[torch.randperm(len(neg_idx))[:remaining_needed]]

        valid_idx = torch.cat([chosen_pos, chosen_neg])
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[valid_idx] = False
        train_idx = torch.arange(n)[train_mask]
    return train_idx, valid_idx

def active_learning(dir, n_start: int = 64, acquisition_method: str = 'exploration', max_screen_size: int = None,
                    batch_size: int = 16, architecture: str = 'gcn', seed: int = 0, bias: str = 'random',
                    optimize_hyperparameters: bool = False, ensemble_size: int = 1, retrain: bool = True,
                    anchored: bool = True, dataset: str = 'ALDH1', scrambledx: bool = False,
                    scrambledx_seed: int = 1, cycle_threshold=1, beta=0, start = 0, feature = '',
                    hidden = 512, at_hidden = 64, layer = '', cycle_rnn = 0, lmda = 0.01, 
                    input='./data/input.csv', input_unlabel='./data/input_unlabel.csv', output='./result/output.csv',
                    assay_active = None, assay_inactive = None, input_val_col='y', input_unlabel_val_col='score', input_smiles_col='smiles', input_unlabel_smiles_col='smiles', is_reverse=False) -> pd.DataFrame:
    """
    :param n_start: number of molecules to start out with
    :param acquisition_method: acquisition method, as defined in active_learning.acquisition
    :param max_screen_size: we stop when this number of molecules has been screened
    :param batch_size: number of molecules to add every cycle
    :param architecture: 'gcn', 'mlp', or 'rf'
    :param seed: int 1-20
    :param bias: 'random', 'small', 'large'
    :param optimize_hyperparameters: Bool
    :param ensemble_size: number of models in the ensemble, default is 10
    :param scrambledx: toggles randomizing the features
    :param scrambledx_seed: seed for scrambling the features
    :return: dataframe with results
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning,
                            message="X does not have valid feature names")
    set_seed(seed)

    # Load the datasets
    representation = 'ecfp' if architecture in ['mlp', 'rf', 'lgb', 'xgb', 'svm'] else 'graph'

    ds_screen = MasterDataset2Labeled('screen', representation=representation, feature = feature, dataset=dataset, scramble_x=scrambledx,
                              scramble_x_seed=scrambledx_seed, input=input, assay_active=assay_active, assay_inactive=assay_inactive, input_val_col=input_val_col, input_smiles_col=input_smiles_col, is_reverse=is_reverse)
    ds_test = MasterDataset2Unlabeled('test', representation=representation, feature = feature, dataset=dataset, input=input_unlabel, assay_active=assay_active, assay_inactive=assay_inactive, input_unlabel_val_col=input_unlabel_val_col, input_unlabel_smiles_col=input_unlabel_smiles_col)

    # Initiate evaluation trackers

    dir_name = f"{dir}/{seed}"

    # Define some variables
    hits_discovered, total_mols_screened, all_train_smiles = [], [], []
    max_screen_size = len(ds_screen) if max_screen_size is None else max_screen_size

    # build test loader
    x_test, y_test, smiles_test, fp_test = ds_test.all()

    n_cycles = ceil((max_screen_size - n_start) / batch_size)
    # exploration_factor = 1 / lambd^x. To achieve a factor of 1 at the last cycle: lambd = 1 / nth root of 2
    lambd = 1 / (2 ** (1/n_cycles))

    unlabel_df = pd.read_csv(input_unlabel)
    ACQ = Acquisition(
        method='evaluation_exploitation',
        seed=seed,
        lambd=lambd,
        unlabel_df=unlabel_df,
        input_unlabel_smiles_col=input_unlabel_smiles_col,
    )
    # While max_screen_size has not been achieved, do some active learning in cycles
    result = pd.DataFrame(columns=[input_unlabel_smiles_col, 'score'])
    prediction_list = defaultdict(list)
    
    valid_loader = None
    cycle=0

    # Get the train and screen data for this cycle
    x_train, y_train, smiles_train, fp_train = ds_screen.all()
    x_screen, y_screen, smiles_screen, fp_screen = ds_screen.all()
    x_test, y_test, smiles_test, fp_test = ds_test.all()


    # Update some tracking variables
    all_train_smiles.append(';'.join(smiles_train.tolist()))
    # hits_discovered.append(sum(y_train).item())
    hits = smiles_train[np.where(y_train == 1)]
    total_mols_screened.append(len(y_train))

    train_loader = to_torch_dataloader_multi(fp_train, x_train, y_train,
                                    batch_size=INFERENCE_BATCH_SIZE,
                                    shuffle=False, pin_memory=True, classification=assay_active is not None)
    train_loader_balanced = to_torch_dataloader_multi(fp_train, x_train, y_train,
                                                batch_size=TRAINING_BATCH_SIZE,
                                                shuffle=False, pin_memory=True, classification=assay_active is not None)
    screen_loader = to_torch_dataloader_multi(fp_screen, x_screen, y_screen,
                                        batch_size=INFERENCE_BATCH_SIZE,
                                        shuffle=False, pin_memory=True, classification=assay_active is not None)
    x_test, y_test, smiles_test, fp_test = ds_test.all()
    test_loader = to_torch_dataloader_multi(fp_test, x_test, y_test,
                                    batch_size=INFERENCE_BATCH_SIZE,
                                    shuffle=False, pin_memory=True, classification=assay_active is not None)

    ########################################################################
    # Initiate and train the model (optimize if specified)
    print("Training model")
    if retrain or cycle == 0:
        n_hidden = 128 if cycle < 0 else 1024
        # if cycle < start:
        M = Ensemble_triple(seed=seed, ensemble_size=ensemble_size, architecture=architecture, hidden = hidden, at_hidden = at_hidden, layer = layer,
                            in_feats = [len(fp_train[idx][0]) for idx in range(len(fp_train))], 
                            n_hidden = n_hidden, anchored=anchored, cycle=cycle, lmda=lmda, assay_active=assay_active)
        # if cycle == 0:
        M.train(train_loader_balanced, valid_loader, cycle=cycle_threshold, verbose=False)

    # Do inference of the train/test/screen data
    print("Train/test/screen inference")

    screen_logits_N_K_C_2 = None
    screen_logits_N_K_C, screen_logits_N_K_C_2 = M.predict_cliff(test_loader, train_loader, dir_name, 'screen', cycle=cycle_rnn)

    # If this is the second to last cycle, update the batch size, so we end at max_screen_size
    # if len(train_idx) + batch_size > max_screen_size:
    #     batch_size = max_screen_size - len(train_idx)

    # Select the molecules to add for the next cycle
    print("Sample acquisition")
    print('hit'+str(cycle)+' : ', len(hits))

    ACQ = Acquisition(
        method='evaluation_exploitation',
        seed=seed,
        lambd=lambd,
        unlabel_df=unlabel_df,
        input_unlabel_smiles_col=input_unlabel_smiles_col,
    )
    # get output directory name only eg. output = './results/ALDH1/gcn/exploration/0/random/0' then make directory ./results/ALDH1/gcn/exploration/0/random/
    if os.path.dirname(output) != '':
        os.makedirs(os.path.dirname(output), exist_ok=True)
    picks = ACQ.acquire(screen_logits_N_K_C, smiles_test, screen_loss=screen_logits_N_K_C_2, hits=hits, n=batch_size, seed=seed, cycle=cycle, y_screen = y_test, dir_name=dir_name, beta=beta, cliff=cycle_rnn, cycle_threshold=cycle_threshold, output=output, classification=assay_active is not None)


    return picks
