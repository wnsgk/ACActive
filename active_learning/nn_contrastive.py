"""

This script contains all models:

    - MLP: a simple feed forward multi-layer perceptron. Supports weight anchoring - Pearce et al. (2018)
    - GCN: a simple graph convolutional NN - Kipf & Welling (2016). Supports weight anchoring - Pearce et al. (2018)
    - Model: A wrapper class that contains a train(), and predict() loop
    - Ensemble: Class that ensembles n Model classes. Contains a train() method and an predict() method that outputs
        logits_N_K_C, defined as [N, num_inference_samples, num_classes]. Also has an optimize_hyperparameters() method.

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

from copy import deepcopy
import sys
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_add_pool, BatchNorm, GATConv, GINConv
from tqdm.auto import trange
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from active_learning.hyperopt import optimize_hyperparameters
from typing import Optional, List
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import random
from rdkit import Chem
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem import AllChem

sys.path.append('.')

# class ResidualBlock(torch.nn.Module):
#     def __init__(self, dim, dropout=0.2):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(dim, dim)
#         self.bn1 = torch.nn.BatchNorm1d(dim)
#         self.fc2 = torch.nn.Linear(dim, dim)
#         self.bn2 = torch.nn.BatchNorm1d(dim)
#         self.dropout = torch.nn.Dropout(dropout)

#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.fc1(x)))
#         out = self.dropout(out)
#         out = self.bn2(self.fc2(out))
#         out = out + residual  # skip connection
#         out = F.relu(out)
#         return out

class MLP(torch.nn.Module):
    def __init__(self, in_feats: int = 1024, n_hidden: int = 128, n_out: int = 2, n_layers: int = 3, seed: int = 42,
                 lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(in_feats if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)
        # self.dropout = torch.nn.Dropout(0.2)
        
        # self.blocks = torch.nn.ModuleList([
        #     ResidualBlock(n_hidden, dropout=0.1) for _ in range(n_layers)
        # ])

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
            # x = self.dropout(x)
        # for block in self.blocks:
        #     x = block(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
    
    def embedding(self, x: Tensor) -> Tensor:
        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 2:
                break
            x = F.relu(x)
            # x = self.dropout(x)
            a += 1
        
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 512, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 2, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GCNConv(n_hidden, n_hidden))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
    
    def embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 2:
                break
            x = F.relu(x)
            a += 1
        
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 512, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 2, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GATConv(n_hidden, n_hidden, add_self_loops=True, negative_slope=0.2,
                                      heads=8, concat=False))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
    
    def embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 2:
                break
            x = F.relu(x)
            a += 1
        
        return x

class GIN(torch.nn.Module):
    def __init__(self, in_feats: int = 130, n_hidden: int = 1024, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 3, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        self.atom_embedding = torch.nn.Linear(in_feats, n_hidden)

        SimpleMLP = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden))

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GINConv(nn=SimpleMLP))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)
        return x
    
    def embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 2:
                break
            x = F.relu(x)
            a += 1
        
        return x

class Model(torch.nn.Module):
    def __init__(self, architecture: str, in_feats, n_hidden, **kwargs):
        super().__init__()
        assert architecture in ['gcn', 'mlp', 'gat', 'gin']
        self.architecture = architecture
        if architecture == 'mlp':
            self.model = MLP(in_feats=in_feats, n_hidden = n_hidden, **kwargs)
        elif architecture == 'gcn':
            self.model = GCN(**kwargs)
        elif architecture == 'gin':
            self.model = GIN(**kwargs)
        else:
            self.model = GAT(**kwargs)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.loss_fn = torch.nn.NLLLoss()

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr,
                                          weight_decay=self.model.weight_decay)

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if self.model.anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def mixup_data(self, x, y, alpha=1.0, device='cuda'):
        """Mixup 데이터 생성"""
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss 계산"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:

        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        scaler = torch.cuda.amp.GradScaler()
        a = 0
        self.model.train()
        for _ in bar:
            running_loss = 0
            items = 0

            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):

                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y = batch.y
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x, y = batch[0].to(self.device), batch[1].to(self.device)
                        y_hat = self.model(x)
                        # mixed_inputs, targets_a, targets_b, lam = self.mixup_data(x, y, alpha=0.4)
                        # y_hat = self.model(mixed_inputs)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    # loss = self.mixup_criterion(self.loss_fn, y_hat, targets_a.squeeze(), targets_b.squeeze(), lam)
                    loss = self.loss_fn(y_hat, y.squeeze())

                    # pos_idx = (y.squeeze() == 1).nonzero(as_tuple=True)[0]
                    # neg_idx = (y.squeeze() == 0).nonzero(as_tuple=True)[0]

                    # rank_loss = 0.0
                    # if len(pos_idx) > 0 and len(neg_idx) > 0:
                    #     pos_scores = y_hat[pos_idx]
                    #     neg_scores = y_hat[neg_idx[torch.randint(len(neg_idx), (len(pos_idx),))]]
                    #     rank_loss = torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()
                    
                    # class_loss = self.loss_fn(y_hat, y.squeeze())
                    # lambda_rank = 0.02  # 가중치, 0.1~1.0 정도 튜닝
                    # loss = class_loss + lambda_rank * rank_loss
                    # print(loss, a)
                    # a += 1

                    if self.model.anchored:
                        # Calculate the total anchored L2 loss
                        l2_loss = 0
                        for param_name, params in self.model.named_parameters():
                            anchored_param = self.model.anchor_weights[param_name]

                            l2_loss += (self.model.l2_lambda / len(y)) * torch.mul(params - anchored_param,
                                                                                   params - anchored_param).sum()

                        # Add anchored loss to regular loss according to Pearce et al. (2018)
                        loss = loss + l2_loss
                    scaler.scale(loss).backward()
                    # loss.backward()
                    scaler.step(self.optimizer)
                    # self.optimizer.step()
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)
            self.epoch += 1

    def predict(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        # self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats

    def embedding(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        # self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model.embedding(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model.embedding(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats


class Ensemble(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, ensemble_size: int = 5, seed: int = 0, architecture: str = 'mlp', in_feats = 1024, n_hidden = 1024, **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, 3)
        # self.models = {0: Model(seed=self.seeds[0], architecture=architecture, in_feats= in_feats, n_hidden = n_hidden, **kwargs)}
        self.models = {i: Model(seed=s, architecture=architecture, in_feats= in_feats, n_hidden = n_hidden, **kwargs) for i, s in enumerate(self.seeds)}

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, **kwargs) -> None:
        for i, m in self.models.items():
            m.train(dataloader, **kwargs)

    def predict(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C
    
    def embedding(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = self.models[0].embedding(dataloader)
        # logits_N_K_C = torch.stack([m.embedding(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"
    
class GCN_multi(torch.nn.Module):
    def __init__(self, in_feats: Optional[List[int]] = None, n_hidden: int = 512, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 2, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        gnn_feature = 130

        self.atom_embedding = torch.nn.Linear(gnn_feature, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GCNConv(n_hidden, n_hidden))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear((n_hidden // 2) * (len(in_feats) + 1) if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

        self.fp_norms = torch.nn.ModuleList([torch.nn.LayerNorm(in_dim) for in_dim in in_feats])
        self.fp_projects = torch.nn.ModuleList([torch.nn.Linear(in_dim, n_hidden // 2) for in_dim in in_feats])

        self.gcn_proj = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden // 2),
            torch.nn.LayerNorm(n_hidden // 2),
            torch.nn.ReLU()
        )

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, xs: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        if xs is not None:
            projected = []
            for i, (norm, proj) in enumerate(zip(self.fp_norms, self.fp_projects)):
                xs_i = xs[:, i, :]
                projected.append(F.relu(proj(norm(xs_i))))
            projected.append(self.gcn_proj(x))
            x = torch.cat(projected, dim=-1)
            
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)
        return x
    
    def embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor, xs: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        if xs is not None:
            projected = []
            for i, (norm, proj) in enumerate(zip(self.fp_norms, self.fp_projects)):
                xs_i = xs[:, i, :]
                projected.append(F.relu(proj(norm(xs_i))))
            projected.append(self.gcn_proj(x))
            x = torch.cat(projected, dim=-1)

        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 1:
                break
            x = F.relu(x)
            a += 1
        
        return x

class GAT_multi(torch.nn.Module):
    def __init__(self, in_feats: Optional[List[int]] = None, n_hidden: int = 512, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 2, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        gnn_feature = 130

        self.atom_embedding = torch.nn.Linear(gnn_feature, n_hidden)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GATConv(n_hidden, n_hidden, add_self_loops=True, negative_slope=0.2,
                                      heads=8, concat=False))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear((n_hidden // 2) * (len(in_feats) + 1) if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

        self.fp_norms = torch.nn.ModuleList([torch.nn.LayerNorm(in_dim) for in_dim in in_feats])
        self.fp_projects = torch.nn.ModuleList([torch.nn.Linear(in_dim, n_hidden // 2) for in_dim in in_feats])

        self.gcn_proj = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden // 2),
            torch.nn.LayerNorm(n_hidden // 2),
            torch.nn.ReLU()
        )

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, xs: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        if xs is not None:
            projected = []
            for i, (norm, proj) in enumerate(zip(self.fp_norms, self.fp_projects)):
                xs_i = xs[:, i, :]
                projected.append(F.relu(proj(norm(xs_i))))
            projected.append(self.gcn_proj(x))
            x = torch.cat(projected, dim=-1)
            
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)
        return x
    
    def embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor, xs: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        if xs is not None:
            projected = []
            for i, (norm, proj) in enumerate(zip(self.fp_norms, self.fp_projects)):
                xs_i = xs[:, i, :]
                projected.append(F.relu(proj(norm(xs_i))))
            projected.append(self.gcn_proj(x))
            x = torch.cat(projected, dim=-1)

        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 1:
                break
            x = F.relu(x)
            a += 1
        
        return x

class GIN_multi(torch.nn.Module):
    def __init__(self, in_feats: Optional[List[int]] = None, n_hidden: int = 512, num_conv_layers: int = 3, lr: float = 3e-4,
                 epochs: int = 50, n_out: int = 2, n_layers: int = 2, seed: int = 42, anchored: bool = True,
                 l2_lambda: float = 3e-4, weight_decay: float = 0):

        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay

        gnn_feature = 130

        self.atom_embedding = torch.nn.Linear(gnn_feature, n_hidden)

        SimpleMLP = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(n_hidden, n_hidden))

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(GINConv(nn=SimpleMLP))
            self.norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear((n_hidden // 2) * (len(in_feats) + 1) if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))

        self.out = torch.nn.Linear(n_hidden, n_out)

        self.fp_norms = torch.nn.ModuleList([torch.nn.LayerNorm(in_dim) for in_dim in in_feats])
        self.fp_projects = torch.nn.ModuleList([torch.nn.Linear(in_dim, n_hidden // 2) for in_dim in in_feats])

        self.gcn_proj = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_hidden // 2),
            torch.nn.LayerNorm(n_hidden // 2),
            torch.nn.ReLU()
        )

    def reset_parameters(self):
        self.atom_embedding.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, xs: Tensor) -> Tensor:
        # Atom Embedding:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        if xs is not None:
            projected = []
            for i, (norm, proj) in enumerate(zip(self.fp_norms, self.fp_projects)):
                xs_i = xs[:, i, :]
                projected.append(F.relu(proj(norm(xs_i))))
            projected.append(self.gcn_proj(x))
            x = torch.cat(projected, dim=-1)
            
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)
        return x
    
    def embedding(self, x: Tensor, edge_index: Tensor, batch: Tensor, xs: Tensor) -> Tensor:
        x = F.elu(self.atom_embedding(x))

        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)

        # Perform global pooling by sum pooling
        x = global_add_pool(x, batch)

        if xs is not None:
            projected = []
            for i, (norm, proj) in enumerate(zip(self.fp_norms, self.fp_projects)):
                xs_i = xs[:, i, :]
                projected.append(F.relu(proj(norm(xs_i))))
            projected.append(self.gcn_proj(x))
            x = torch.cat(projected, dim=-1)

        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 1:
                break
            x = F.relu(x)
            a += 1
        
        return x
    
class SampleAwareAttention(torch.nn.Module):
    def __init__(self, d, n_feature, hidden_dim=128):
        super().__init__()
        self.query_proj = torch.nn.Linear(d * n_feature, hidden_dim)
        self.key_proj   = torch.nn.Linear(d, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, features):
        # features: list of [batch, d]
        x = torch.stack(features, dim=1)  # (batch, n_feature, d)

        # 샘플별 query: concat 전체 feature → (batch, d*n_feature)
        concat = x.view(x.size(0), -1)     # (batch, n_feature*d)
        query = self.query_proj(concat)    # (batch, hidden_dim)

        # block별 key
        keys = self.key_proj(x)            # (batch, n_feature, hidden_dim)

        # attention score
        scores = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1) / self.scale
        weights = F.softmax(scores, dim=1) # (batch, n_feature)

        # weighted sum
        weighted = x * weights.unsqueeze(-1)
        fused = weighted.view(x.size(0), -1)
        # fused = weighted.sum(dim=1)  # (batch, d)
        return fused, weights
    
class MLP_triple(torch.nn.Module):
    def __init__(self, in_feats: Optional[List[int]] = None, n_hidden: int = 128, hidden = 512, at_hidden=64, layer = '', n_out: int = 2, n_layers: int = 2, 
                 seed: int = 42, lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0, dropout_ratio=0, classification=False):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)
        self.proj_dim = hidden

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(in_dim) for in_dim in in_feats])
        self.projects = torch.nn.ModuleList([torch.nn.Linear(in_dim, self.proj_dim) for in_dim in in_feats])
        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        self.attention_dim = at_hidden
        self.attention_layer = 2 if layer == '_layer2' else 1
        self.attention = True
        self.att_weight = None
        self.att_weight2 = None
        self.feat_len = len(in_feats)
        self.classification = classification

        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(self.proj_dim * len(in_feats) if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)
        self.regout = torch.nn.Linear(n_hidden, 1)
        self.layernorm = torch.nn.BatchNorm1d(self.proj_dim)
        self.layernorms = torch.nn.LayerNorm(self.proj_dim * len(in_feats))

        self.attn = SampleAwareAttention(self.proj_dim, len(in_feats), self.attention_dim)
        self.attn2 = SampleAwareAttention(self.proj_dim, len(in_feats), self.attention_dim)

        self.gate = torch.nn.Linear(len(in_feats)*self.proj_dim, len(in_feats))

        init_logits = [0.25, 0.25, 0.25, 0.25]
        self.gates = torch.nn.ParameterDict({
            str(name): torch.nn.Parameter(torch.tensor([val])) for name, val in enumerate(init_logits)
            # name: torch.nn.Parameter(torch.randn(1)) for name in range(len(in_feats))
        })
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, *xs: Tensor, return_feature=False) -> Tensor:
        projected = []
        for x, norm, proj in zip(xs, self.norms, self.projects):
            projected.append(F.relu(proj(x)))

        ##### attention #####
        if self.attention:
            x, self.att_weight = self.attn(projected)
            if self.attention_layer == 2:
                x = x.view(x.size(0), len(projected), self.proj_dim)
                x = list(x.unbind(dim=1))
                x, self.att_weight2 = self.attn2(x)
        else:
            x = torch.cat(projected, dim=-1)

        # each = False
        # if each:
        #     g = torch.softmax(self.gate(x), dim=-1)
        #     # g = torch.sigmoid(self.gate(x))
        #     scaled_feats = [g[:, i].unsqueeze(-1) * projected[i] 
        #                     for i in range(len(projected))]
        #     x = torch.cat(scaled_feats, dim=-1)  # (batch, 128*n_feats)
        # else:
        #     logits = torch.cat([p for p in self.gates.values()])         # (n_feats,)
        #     g = torch.softmax(logits, dim=0)

        #     scaled_feats = [g[i] * projected[i] 
        #                     for i in range(len(projected))]
        #     x = torch.cat(scaled_feats, dim=-1)  # (batch, 128*n_feats)
        
        # x = self.layernorm(x)
        feature = x
        # x = self.dropout(x)
        for lin, norm in zip(self.fc, self.fc_norms):
            x = F.relu(norm(lin(x)))
            # feature.append(x)
        # for block in self.blocks:
        #     x = block(x)
        if not self.classification:
            x = self.regout(x)
        else:
            x = self.out(x)
            x = F.log_softmax(x, 1)
        if return_feature:
            return x, feature
        return x
    
    # def get_features(self, *xs: Tensor) -> Tensor:
    #     projected = []
    #     for x, norm, proj in zip(xs, self.norms, self.projects):
    #         projected.append(F.relu(proj(x)))

    #     ##### attention #####
    #     if self.attention:
    #         x, self.att_weight = self.attn(projected)
    #         if self.attention_layer == 2:
    #             x = x.view(x.size(0), len(projected), self.proj_dim)
    #             x = list(x.unbind(dim=1))
    #             x, self.att_weight2 = self.attn2(x)
    #     else:
    #         x = torch.cat(projected, dim=-1)
        
    #     x1 = self.fc_norms[0](self.fc[0](x))
    #     x2 = self.fc_norms[1](self.fc[1](x1))
        
    #     return [x1, x2]

    
    def embedding(self, *xs: Tensor) -> Tensor:
        projected = []
        for x, norm, proj in zip(xs, self.norms, self.projects):
            projected.append(F.relu(proj(norm(x))))
        a=0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = norm(lin(x))
            if a == 1:
                break
            x = F.relu(x)
            a += 1
        
        return x
    
class CliffPredictionModule(torch.nn.Module):
    def __init__(self, in_dims, in_feats, hidden_dim=128):
        super().__init__()

        self.cliff_head = torch.nn.Sequential(
            torch.nn.Linear(in_dims*in_feats, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)  # loss scalar 출력
        )

    def forward(self, h):
        # h_i shape: [B, 1, embed_dim]
        h_i = h.unsqueeze(1)
        # h_j shape: [1, B, embed_dim]
        h_j = h.unsqueeze(0)
        abs_diff = torch.abs(h_i - h_j)
        
        # 3-2. AC 예측 헤드 통과
        # pred_cliff_logits shape: [B, B, 1] ---squeeze---> [B, B]
        pred_cliff_logits = self.cliff_head(abs_diff).squeeze(-1)
        
        # 학습 시에는 두 예측 결과를 모두 반환
        return pred_cliff_logits
    
    def forward(self, h1, h2):
        """
        두 개의 임베딩 셋 (h1, h2)을 입력받아
        그 '차이'에 대한 Cliff 예측값을 반환합니다.
        
        - 학습 시: h1=h(B,dim), h2=h(B,dim) -> [B, B] 반환
        - 평가 시: h1=H_U(B,dim), h2=H_L(L,dim) -> [B, L] 반환
        """
        
        # 2. 차이 벡터 계산 로직 (내장)
        # h1_i shape: [B, 1, embed_dim]
        # h2_j shape: [1, L, embed_dim] (L은 B와 같을 수 있음)
        h1_i = h1.unsqueeze(1)
        h2_j = h2.unsqueeze(0)
        
        # abs_diff shape: [B, L, embed_dim]
        abs_diff = torch.abs(h1_i - h2_j)
        
        # 3. MLP 헤드 통과
        # pred_cliff_logits shape: [B, L, 1] ---squeeze---> [B, L]
        pred_cliff_logits = self.cliff_head(abs_diff).squeeze(-1)
        
        return pred_cliff_logits
    
class LossPredictionModule(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim=128):
        super().__init__()
        self.projs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d, hidden_dim),
                torch.nn.ReLU(inplace=True)
            ) for d in in_dims
        ])

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * len(in_dims), hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1)  # loss scalar 출력
        )

    def forward(self, feats):
        """
        feats: list of mid-layer feature maps (각각 shape (B, C, H, W))
        return: predicted loss (B,)
        """
        zs = [proj(f) for proj, f in zip(self.projs, feats)]
        z = torch.cat(zs, dim=-1)
        # print(self.head(z))
        # l_hat = self.head(z).squeeze(-1)
        l_hat = F.softplus(self.head(z)).squeeze(-1)
        return l_hat

    def pairwise_ranking_loss(self, l_hat, l_real, margin=0.2):
        """
        l_hat: predicted losses (B,)
        l_real: true losses (B,)
        margin: hinge margin (0.1~0.3 recommended for NLL)
        """

        B = l_hat.size(0)
        # (B, B) 쌍 행렬 만들기
        diff_real = l_real.unsqueeze(0) - l_real.unsqueeze(1)
        diff_pred = l_hat.unsqueeze(0) - l_hat.unsqueeze(1)

        # sign matrix (+1, -1)
        s = torch.sign(diff_real)

        # margin hinge
        loss_mat = F.relu(-s * diff_pred + margin)

        # 대각선 (i==j) 제거
        mask = ~torch.eye(B, dtype=torch.bool, device=l_hat.device)
        loss = loss_mat[mask].mean()
        # print(l_hat)
        # print(l_real)
        # print(loss)
        return loss.mean()

class Model_triple(torch.nn.Module):
    def __init__(self, architecture: str, in_feats, n_hidden, hidden, at_hidden, layer, cycle, lmda=0, classification = False, **kwargs):
        super().__init__()
        assert architecture in ['gcn', 'mlp', 'gat', 'gin']
        self.architecture = architecture
        n_hidden = 512
        if architecture == 'mlp':
            self.model = MLP_triple(in_feats=in_feats, n_hidden = n_hidden, hidden = hidden, at_hidden=at_hidden, layer=layer, classification=classification, **kwargs)
        elif architecture == 'gcn':
            self.model = GCN_multi(in_feats=in_feats, **kwargs)
        elif architecture == 'gin':
            self.model = GIN_multi(in_feats=in_feats, **kwargs)
        else:
            self.model = GAT_multi(in_feats=in_feats, **kwargs)

        self.loss_prediction_module = LossPredictionModule(in_dims=[n_hidden, n_hidden])

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.classification = classification
        if self.classification:
            self.loss_fn = torch.nn.NLLLoss()
        else:
            self.loss_fn = torch.nn.MSELoss()
        self.cycle = cycle
        self.lmda = lmda
        self.sim_threshold = 0.6
        # self.loss_fn = torch.nn.NLLLoss(reduction = 'none')

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        # self.loss_prediction_module = self.loss_prediction_module.to(self.device)
        self.cliff_prediction_module = CliffPredictionModule(in_dims=hidden, in_feats=len(in_feats), hidden_dim=128).to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr,
        #                                   weight_decay=self.model.weight_decay)
        self.optimizer = torch.optim.Adam([
            # 그룹 1: 메인 모델의 파라미터
            {'params': self.model.parameters(), 
            'lr': self.model.lr, 
            'weight_decay': self.model.weight_decay},
            
            # 그룹 2: Cliff 모듈의 파라미터
            {'params': self.cliff_prediction_module.parameters(), 
            'lr': self.model.lr,  # (우선 동일한 lr로 시작)
            'weight_decay': self.model.weight_decay}
        ])
        # self.opt_lpm  = torch.optim.Adam(self.loss_prediction_module.parameters(), lr=3e-4)

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if self.model.anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def mixup_data(self, x, y, alpha=1.0, device='cuda'):
        """Mixup 데이터 생성"""
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup loss 계산"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def label_based_triplet_loss_cosine(
        self,
        embeddings,
        labels,
        margin=0.3,
        tau_sim=0.5   # similarity threshold
    ):
        """
        embeddings: (N, d)
        labels: (N,)
        margin: triplet margin
        tau_sim: similarity threshold (only pairs with sim > tau_sim are used)
        """
        N = embeddings.size(0)
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        # print(sim_matrix)

        # Mask for similarity threshold
        sim_mask = sim_matrix > tau_sim

        # Label equality matrix
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        label_not_equal = ~label_equal

        # Combine: only pairs with sim > tau_sim
        pos_mask = sim_mask & label_equal
        neg_mask = sim_mask & label_not_equal

        losses = []
        a = 0
        b = 0
        for i in range(N):
            pos_idx = torch.where(pos_mask[i])[0]
            neg_idx = torch.where(neg_mask[i])[0]
            # a += len(pos_idx)
            # b += len(neg_idx)

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            s_ap = sim_matrix[i, pos_idx]
            s_an = sim_matrix[i, neg_idx]

            # hardest mining
            hard_pos = s_ap.min()   # 가장 낮은 positive sim
            hard_neg = s_an.max()   # 가장 높은 negative sim

            loss_i = F.relu(hard_neg - hard_pos + margin)
            losses.append(loss_i)
        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return torch.stack(losses).mean()
        
    def train2(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:
        # epochs = 50
        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        scaler = torch.cuda.amp.GradScaler()
        a = 0
        self.model.train()
        self.loss_prediction_module.train()
        use_loss_prediction = True

        for epoch in bar:
            running_loss = 0
            items = 0
            # if epoch < 50:

            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()
                self.opt_lpm.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    feat_list = None
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y = batch.y
                        if hasattr(batch, "fp"):
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch, batch.fp)
                        else:
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        *xs, y = [t.to(self.device) for t in batch]
                        if use_loss_prediction:
                            y_hat, feat_list = self.model(*xs, return_feature=True)
                        else:
                            y_hat = self.model(*xs)
                        # mixed_inputs, targets_a, targets_b, lam = self.mixup_data(x, y, alpha=0.4)
                        # y_hat = self.model(mixed_inputs)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    # loss = self.mixup_criterion(self.loss_fn, y_hat, targets_a.squeeze(), targets_b.squeeze(), lam)
                    # loss_raw = self.loss_fn(y_hat, y.squeeze())
                    loss = self.loss_fn(y_hat, y.squeeze())
                    
                    # loss = loss_raw.mean()
                    print('loss:', loss)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)
            self.epoch += 1

    def calculate_embedding_similarity_matrix(self, h):
        """
        GNN 임베딩(h) 간의 BxB 코사인 유사도 행렬을 계산합니다.
        
        입력:
        - h (Tensor): GNN 인코더의 출력 [B, embed_dim]
        """
        
        # 1. 임베딩 L2 정규화
        #    (코사인 유사도 = 정규화된 벡터의 내적)
        h_norm = F.normalize(h, p=2, dim=1)
        
        # 2. 행렬 곱(matmul)으로 코사인 유사도 행렬 계산
        #    (h_norm @ h_norm.T)
        #    shape: [B, embed_dim] @ [embed_dim, B] -> [B, B]
        sim_matrix = torch.matmul(h_norm, h_norm.t())
        
        # 수치적 안정성을 위해 값을 -1.0과 1.0 사이로 클리핑
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)
        
        return sim_matrix

    def calculate_fp_similarity_matrix(self, fp_batch_U, fp_batch_L):
        """
        (평가/AL용) Unlabeled 배치 [B, dim]와 Labeled 셋 [L, dim]를 입력받아,
        '두 셋 간의' Tanimoto 유사도 행렬 [B, L]를 반환합니다.
        """
        fp_U_float = fp_batch_U.float() # [B, dim]
        fp_L_float = fp_batch_L.float() # [L, dim]
        
        # [B, dim] @ [dim, L] -> [B, L]
        inter = torch.matmul(fp_U_float, fp_L_float.t())
        
        sums_U = fp_U_float.sum(dim=1).unsqueeze(1) # [B, 1]
        sums_L = fp_L_float.sum(dim=1).unsqueeze(0) # [1, L]
        
        # [B, L]
        union = sums_U + sums_L - inter
        
        eps = 1e-6
        sim_matrix = inter / (union + eps)
        
        return sim_matrix
    
    def train(self, dataloader: DataLoader, valid_loader, epochs: int = None, cycle = 0, verbose: bool = True) -> None:
        # epochs = 70
        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        self.model.train()
        self.cliff_prediction_module.train()
        scaler = torch.cuda.amp.GradScaler()
        criterion_cliff = torch.nn.BCEWithLogitsLoss()
        best_valid_loss = float('inf')
        best_epoch = 0
        for epoch in bar:
            running_loss = 0
            items = 0
            self.model.train()
            self.cliff_prediction_module.train()
            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y = batch.y
                        if hasattr(batch, "fp"):
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch, batch.fp)
                        else:
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        *xs, y = [t.to(self.device) for t in batch]
                        y_hat, embeddings = self.model(*xs, return_feature = True)
                        y_hat_cliff = self.cliff_prediction_module(embeddings, embeddings)

                        true_sim =self.calculate_fp_similarity_matrix(xs[0], xs[0])
                        true_activity = y.view(-1, 1)
                        true_act_i = true_activity.unsqueeze(1)
                        true_act_j = true_activity.unsqueeze(0)
                        true_delta_act = torch.abs(true_act_i - true_act_j)
                        true_delta_act = true_delta_act.squeeze(-1)

                        CLIFF_SIM_THRESHOLD = 0.6
                        CLIFF_ACT_THRESHOLD = 1.0
    
                        true_cliff_labels = (true_sim > self.sim_threshold) & \
                                            (true_delta_act >= CLIFF_ACT_THRESHOLD)
                        true_cliff_labels = true_cliff_labels.float()

                        mask = (true_sim > self.sim_threshold)
                        if mask.sum() == 0:
                            loss_cliff = torch.tensor(0.0).to(self.device)
                        else:
                            # 3. 관심 영역의 예측값과 정답만 필터링합니다.
                            y_hat_masked = y_hat_cliff[mask]
                            labels_masked = true_cliff_labels[mask]

                            num_positives = labels_masked.sum()
                            
                            if num_positives == 0:
                                criterion_cliff = torch.nn.BCEWithLogitsLoss()
                            else:
                                # 5. '관심 영역 내'의 불균형을 기반으로 pos_weight 계산
                                num_negatives = labels_masked.numel() - num_positives
                                pos_weight_value = num_negatives / num_positives
                                
                                criterion_cliff = torch.nn.BCEWithLogitsLoss(
                                    pos_weight=torch.tensor(pos_weight_value).to(self.device)
                                )

                            loss_cliff = criterion_cliff(y_hat_masked, labels_masked)
                        # loss_cliff = criterion_cliff(y_hat_cliff, true_cliff_labels)
                        lamda = self.lmda

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    # loss = self.mixup_criterion(self.loss_fn, y_hat, targets_a.squeeze(), targets_b.squeeze(), lam)
                    if self.classification:
                        loss = self.loss_fn(y_hat, y.squeeze())
                    else:
                        loss = self.loss_fn(y_hat.squeeze(), y.squeeze())
                    # print(loss, lamda*loss_cliff)
                    if self.cycle >= cycle:
                        loss = loss + lamda*loss_cliff
                    else:
                        loss = loss
                    # if epoch %10 == 0:
                    #     print(loss, triplet_loss*0.1)
                    # loss = loss + 0.1*triplet_loss
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)

            self.epoch += 1

    def predict(self, dataloader: DataLoader, dir_name = '', mode = '', cycle=0) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        self.model.eval()
        attention_weight = []
        attention_weight2 = []
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        if hasattr(batch, "fp"):
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch, batch.fp)
                        else:
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        *xs, y = [t.to(self.device) for t in batch]
                        y_hat = self.model(*xs)
                        if self.model.attention:
                            if self.model.att_weight is not None:
                                attention_weight.append(self.model.att_weight.cpu())
                            if self.model.att_weight2 is not None:
                                attention_weight2.append(self.model.att_weight2.cpu())
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)
        if mode != '' and self.model.attention:
            if self.model.att_weight is not None:
                mean_weight = torch.cat(attention_weight, dim=0).mean(dim=0).numpy()
                df = pd.DataFrame(mean_weight.reshape(1, -1), 
                        columns=[f"feature_{i}" for i in range(self.model.feat_len)])
                df.to_csv(f"{dir_name}/attention_weights_{mode}.csv", mode='a', index=False, header=not os.path.exists(f"{dir_name}/attention_weights_{mode}.csv"))
            if self.model.att_weight2 is not None:
                mean_weight2 = torch.cat(attention_weight2, dim=0).mean(dim=0).numpy()
                df = pd.DataFrame(mean_weight2.reshape(1, -1), 
                        columns=[f"feature_{i}" for i in range(self.model.feat_len)])
                df.to_csv(f"{dir_name}/attention_weights2_{mode}.csv", mode='a', index=False, header=not os.path.exists(f"{dir_name}/attention_weights2_{mode}.csv"))
        return y_hats

    def mc_dropout(self, dataloader: DataLoader, p=0.2) -> Tensor:
        y_hats = torch.tensor([]).to(self.device)
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = p
        self.model.train()
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    *xs, y = [t.to(self.device) for t in batch]
                    y_hat = self.model(*xs)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0
        return y_hats
    
    def embedding(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        if hasattr(batch, "fp"):
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch, batch.fp)
                        else:
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        *xs, y = [t.to(self.device) for t in batch]
                        y_hat = self.model(*xs)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats
    def predict_cliff(self, dataloader: DataLoader, train_dataloader: DataLoader, dir_name = '', mode = '', cycle=0) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        y_hats_cliff = torch.tensor([]).to(self.device)
        train_fp_list = []
        train_embeddings_list = []
        train_y_true_list = []
        self.model.eval()
        self.cliff_prediction_module.eval()
        all_final_scores = []
        all_final_scores1 = []
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in train_dataloader:
                    *xs, y = [t.to(self.device) for t in batch]
                    y_hat, embeddings = self.model(*xs, return_feature = True)

                    # xs[0]가 핑거프린트 배치(예: [B, 2048])라고 가정
                    train_fp_list.append(xs[0])
                    # embeddings 배치(예: [B, 512])
                    train_embeddings_list.append(embeddings)
                    train_y_true_list.append(y.squeeze())
                train_fp = torch.cat(train_fp_list, dim=0).to(self.device)
                train_embeddings = torch.cat(train_embeddings_list, dim=0).to(self.device)
                train_y_true = torch.cat(train_y_true_list, dim=0).to(self.device)
                lst = []
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        if hasattr(batch, "fp"):
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch, batch.fp)
                        else:
                            y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        *xs, y = [t.to(self.device) for t in batch]
                        y_hat, embeddings = self.model(*xs, return_feature = True)
                        y_hat_cliff = self.cliff_prediction_module(embeddings, train_embeddings)
                        cliff_probs = torch.sigmoid(y_hat_cliff)

                        sim_matrix = self.calculate_fp_similarity_matrix(xs[0], train_fp)
                        mask = (sim_matrix > self.sim_threshold)

                        train_y_true_broadcast = train_y_true.unsqueeze(0)

                        conditional_scores = torch.where(
                            train_y_true_broadcast == 1, 
                            1.0 - cliff_probs,          
                            cliff_probs                 
                        )                               
                        if cycle == 2:
                            conditional_scores = conditional_scores - 0.5
                        mx = True
                        if mx:
                            conditional_scores[~mask] = -float('inf')

                            final_scores_batch, _ = torch.max(conditional_scores, dim=1) # dim=1 (L 차원)
                        else:
                            masked_scores = conditional_scores.masked_fill(~mask, float('nan'))
                            final_scores_batch = torch.nanmean(masked_scores, dim=1)
                            final_scores_batch = torch.nan_to_num(final_scores_batch, nan=-float('inf'))
                        
                        final_scores_batch[torch.isinf(final_scores_batch)] = 0.0
                        all_final_scores.append(final_scores_batch)

                        conditional_scores = torch.where(
                            train_y_true_broadcast == 1,  # Labeled Set Y_TRUE가 1이면
                            1.0 - cliff_probs,          # 1 - score (Smooth 원함)
                            0.5                 # Labeled Set Y_TRUE가 0이면
                        )                               # score (Cliff 원함)
                        if cycle == 2:
                            conditional_scores = conditional_scores - 0.5
                        mx = True
                        if mx:
                            conditional_scores[~mask] = -float('inf')

                            final_scores_batch, _ = torch.max(conditional_scores, dim=1)
                        else:
                            masked_scores = conditional_scores.masked_fill(~mask, float('nan'))
                            final_scores_batch = torch.nanmean(masked_scores, dim=1)
                            final_scores_batch = torch.nan_to_num(final_scores_batch, nan=-float('inf'))
                        
                        final_scores_batch[torch.isinf(final_scores_batch)] = 0.0
                        all_final_scores1.append(final_scores_batch)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)
                final_scores_tensor = torch.cat(all_final_scores, dim=0)
                final_scores_tensor1 = torch.cat(all_final_scores1, dim=0)
                lst1 = final_scores_tensor1.tolist()
                lst1 = list(filter(lambda x: x != 0.0, lst1))
                lst1.sort()
                # print(lst1)
        return y_hats, final_scores_tensor

class Ensemble_triple(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, ensemble_size: int = 5, seed: int = 0, architecture: str = 'mlp', hidden = 512, at_hidden = 64, layer = '',
                 in_feats: Optional[List[int]] = None, n_hidden = 1024, cycle=0, lmda=0, assay_active = None, **kwargs) -> None:
        self.ensemble_size = 1
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, 10)
        classification = assay_active is not None
        self.models = {0: Model_triple(seed=self.seeds[0], architecture=architecture, in_feats=in_feats, n_hidden=n_hidden, hidden = hidden, at_hidden=at_hidden, layer=layer, 
                                       cycle=cycle, lmda=lmda, classification = classification, **kwargs)}
        # self.models = {i: Model_triple(seed=s, architecture=architecture, in_feats=in_feats, n_hidden=n_hidden, hidden = hidden, at_hidden=at_hidden, layer=layer, 
        #                                cycle=cycle, lmda=lmda, **kwargs) for i, s in enumerate(self.seeds[:5])}
        # self.models = {i: Model_triple(seed=s, architecture=architecture, in_feats=in_feats, n_hidden=n_hidden, **kwargs) for i, s in enumerate(self.seeds)}

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, valid_loader, cycle=0, **kwargs) -> None:
        for i, m in self.models.items():
            m.train(dataloader, valid_loader, cycle=cycle, **kwargs)

    def predict(self, dataloader, dir_name = '', mode='', cycle=0, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict(dataloader, dir_name, mode, cycle) for m in self.models.values()], 1)

        return logits_N_K_C
    
    def predict_loss(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict_loss(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C
    
    def predict_cliff(self, dataloader, train_dataloader, dir_name = '', mode='', cycle=0, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        results_list = []
        for m in self.models.values():
            # result_tuple = (y_hats_model_i, scores_model_i)
            result_tuple = m.predict_cliff(dataloader, train_dataloader, dir_name, mode, cycle)
            results_list.append(result_tuple)

        y_hats_list, scores_list = zip(*results_list)

        ensemble_y_hats = torch.stack(y_hats_list, dim=1)
        ensemble_scores = torch.stack(scores_list, dim=1)

        return ensemble_y_hats, ensemble_scores
    
    def embedding(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = self.models[0].embedding(dataloader)
        # logits_N_K_C = torch.stack([m.embedding(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C

    def mc_dropout(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        preds = []
        for _ in range(10):
            logits_N_K_C = torch.stack([m.mc_dropout(dataloader) for m in self.models.values()], 1)
            # probs = torch.softmax(logits_N_K_C, dim=-1)
            # class1_probs = probs[:, :, 1]  # [N, 1] → class 1 확률만 추출
            class1_probs = logits_N_K_C[:, :, 1]  # [N, 1] → class 1 확률만 추출
            preds.append(class1_probs.squeeze(1))  # [N]
        # 쌓기: [10, N]
        preds = torch.stack(preds, dim=0)

        # Monte Carlo 평균 및 표준편차
        mu_1 = preds.mean(dim=0)   # [N]
        sigma_1 = preds.std(dim=0) # [N]

        return sigma_1

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"
    
class TrajAcqNet(torch.nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.rnn = torch.nn.GRU(1, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)
    def forward(self, seqs, lengths):
        # seqs: (batch, L, 1), lengths: 각 시퀀스 길이
        packed = pack_padded_sequence(seqs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h = self.rnn(packed)
        # 마지막 hidden state h 사용 (num_layers, batch, hidden)
        out = self.fc(h[-1])   # (batch, 1)
        return out.squeeze(-1)
    
class AcqModel(torch.nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TrajAcqNet(hidden_dim=hidden_dim).to(self.device)

    def make_dataset_from_list(self, score_list, label_list):
        """
        list 기반 입력:
        score_list: [[s11, s12, ...], [s21, s22, ...], ...]
        label_list: [1, 0, 1, ...]

        return: [(seq_tensor, label_float), ...]
        """
        data = []
        assert len(score_list) == len(label_list)
        
        for seq, label in zip(score_list, label_list):
            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            label_float = float(label)
            data.append((seq_tensor, label_float))
        
        return data
    
    def prefix_augment(score_list, label_list, shuffle=False):
        aug_scores, aug_labels = [], []
        assert len(score_list) == len(label_list)
        
        for seq, label in zip(score_list, label_list):
            for k in range(1, len(seq) + 1):
                aug_scores.append(seq[:k])   # prefix sequence
                aug_labels.append(label)     # label은 그대로 복사
        if shuffle:
            combined = list(zip(aug_scores, aug_labels))
            random.shuffle(combined)
            aug_scores, aug_labels = zip(*combined)
            aug_scores, aug_labels = list(aug_scores), list(aug_labels)
        return aug_scores, aug_labels
    
    def collate_fn(self, batch):
        seqs, labels = zip(*batch)
        lengths = [len(s) for s in seqs]
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        return padded.unsqueeze(-1), torch.tensor(lengths), torch.tensor(labels, dtype=torch.float32)

    def train(self, score_list, label):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # 데이터 준비
        # score_list, label = self.prefix_augment(score_list, label)
        train_data = self.make_dataset_from_list(score_list, label)

        # dataloader
        loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=self.collate_fn)

        # 학습
        for epoch in range(100):
            self.model.train()
            total_loss = 0
            for seqs, lengths, labels in loader:
                seqs, lengths, labels = seqs.to(self.device), lengths.to(self.device), labels.to(self.device)
                preds = self.model(seqs, lengths)
                loss = F.binary_cross_entropy_with_logits(preds, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1:02d} | loss: {total_loss/len(loader):.4f}")
            
    def predict(self, data_list):
        """
        list of trajectory (e.g. [[1,2,3], [2,3,4], [0.5,1.0]]) 를 입력받아
        각 시퀀스의 hit 확률을 tensor로 반환
        """
        self.model.eval()
        
        # tensor 변환
        seqs = [torch.tensor(seq, dtype=torch.float32) for seq in data_list]
        lengths = [len(s) for s in seqs]
        
        # padding
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        seqs_tensor = padded.unsqueeze(-1).to(self.device)
        lengths_tensor = torch.tensor(lengths).to(self.device)
        
        # 예측
        with torch.no_grad():
            probs = torch.sigmoid(self.model(seqs_tensor, lengths_tensor))
        
        # 그대로 tensor 반환 (float 변환 안 함)
        return probs
# class FeatureGating(nn.Module):
#     def __init__(self, d, n_feature):
#         super().__init__()
#         self.gate = torch.nn.Linear(d, 1)
#         self.n_feature = n_feature

#     def forward(self, features):  
#         # features: list of [batch, d]
#         x = torch.stack(features, dim=1)      # (batch, n_feature, d)
#         scores = self.gate(x).squeeze(-1)     # (batch, n_feature)
#         weights = F.softmax(scores, dim=1)    # (batch, n_feature)
#         fused = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, d)
#         return fused, weights


class MLP_test(torch.nn.Module):
    def __init__(self, in_feats: int = 1024, n_hidden: int = 1024, n_out: int = 2, n_layers: int = 3, seed: int = 42,
                 lr: float = 3e-4, epochs: int = 50, anchored: bool = True, l2_lambda: float = 3e-4,
                 weight_decay: float = 0):
        super().__init__()
        self.seed, self.lr, self.l2_lambda, self.epochs, self.anchored = seed, lr, l2_lambda, epochs, anchored
        self.weight_decay = weight_decay
        torch.manual_seed(seed)

        self.fc = torch.nn.ModuleList()
        self.fc_norms = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(in_feats if i == 0 else n_hidden, n_hidden))
            self.fc_norms.append(BatchNorm(n_hidden, allow_single_element=True))
        self.out = torch.nn.Linear(n_hidden, n_out)

    def reset_parameters(self):
        for lin, norm in zip(self.fc, self.fc_norms):
            lin.reset_parameters()
            norm.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)

        x = self.out(x)
        x = F.log_softmax(x, 1)

        return x
    
    def embedding(self, x: Tensor) -> Tensor:
        a = 0
        for lin, norm in zip(self.fc, self.fc_norms):
            x = lin(x)
            x = norm(x)
            if a == 2:
                break
            x = F.relu(x)
            a += 1
        
        return x

class Model_test(torch.nn.Module):
    def __init__(self, architecture: str, in_feats, **kwargs):
        super().__init__()
        assert architecture in ['gcn', 'mlp', 'gat', 'gin']
        self.architecture = architecture
        if architecture == 'mlp':
            self.model = MLP_test(in_feats=in_feats, **kwargs)
        elif architecture == 'gcn':
            self.model = GCN(**kwargs)
        elif architecture == 'gin':
            self.model = GIN(**kwargs)
        else:
            self.model = GAT(**kwargs)

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.loss_fn = torch.nn.NLLLoss()
        # self.loss_fn = torch.nn.NLLLoss(reduction="none")

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr,
                                          weight_decay=self.model.weight_decay)

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if self.model.anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    # def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:

    #     bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
    #     scaler = torch.cuda.amp.GradScaler()

    #     for _ in bar:
    #         running_loss = 0
    #         items = 0

    #         for idx, batch in enumerate(dataloader):

    #             self.optimizer.zero_grad()

    #             with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):

    #                 if self.architecture in ['gcn', 'gat', 'gin']:
    #                     batch.to(self.device)
    #                     y = batch.y
    #                     y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
    #                 else:
    #                     x, y = batch[0].to(self.device), batch[1].to(self.device)
    #                     y_hat = self.model(x)

    #                 if len(y_hat) == 0:
    #                     y_hat = y_hat.unsqueeze(0)
    #                 loss = self.loss_fn(y_hat, y.squeeze())

    #                 if self.model.anchored:
    #                     # Calculate the total anchored L2 loss
    #                     l2_loss = 0
    #                     for param_name, params in self.model.named_parameters():
    #                         anchored_param = self.model.anchor_weights[param_name]

    #                         l2_loss += (self.model.l2_lambda / len(y)) * torch.mul(params - anchored_param,
    #                                                                                params - anchored_param).sum()

    #                     # Add anchored loss to regular loss according to Pearce et al. (2018)
    #                     loss = loss + l2_loss

    #                 scaler.scale(loss).backward()
    #                 # loss.backward()
    #                 scaler.step(self.optimizer)
    #                 # self.optimizer.step()
    #                 scaler.update()

    #                 running_loss += loss.item()
    #                 items += len(y)

    #         epoch_loss = running_loss / items
    #         bar.set_postfix(loss=f'{epoch_loss:.4f}')
    #         self.train_loss.append(epoch_loss)
    #         self.epoch += 1

    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:

        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()

        for _ in bar:
            running_loss = 0
            items = 0

            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):

                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y = batch.y
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x, y, w = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                        y_hat = self.model(x)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    loss = self.loss_fn(y_hat, y.squeeze())
                    # loss = loss.mean()
                    # loss = (loss * w).mean()

                    if self.model.anchored:
                        # Calculate the total anchored L2 loss
                        l2_loss = 0
                        for param_name, params in self.model.named_parameters():
                            anchored_param = self.model.anchor_weights[param_name]

                            l2_loss += (self.model.l2_lambda / len(y)) * torch.mul(params - anchored_param,
                                                                                   params - anchored_param).sum()

                        # Add anchored loss to regular loss according to Pearce et al. (2018)
                        loss = loss + l2_loss

                    scaler.scale(loss).backward()
                    # loss.backward()
                    scaler.step(self.optimizer)
                    # self.optimizer.step()
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)
            self.epoch += 1

    def predict(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats

    def embedding(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model.embedding(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model.embedding(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats

class Ensemble_test(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, in_feats = 1024, ensemble_size: int = 5, seed: int = 0, architecture: str = 'mlp', **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, 10)
        # self.models = {0: Model_test(seed=self.seeds[0], architecture=architecture, **kwargs)}
        self.models = {i: Model_test(seed=s, architecture=architecture, in_feats=in_feats, **kwargs) for i, s in enumerate(self.seeds)}

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, **kwargs) -> None:
        for i, m in self.models.items():
            m.train(dataloader, **kwargs)

    def predict(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C
    
    def embedding(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = self.models[0].embedding(dataloader)
        # logits_N_K_C = torch.stack([m.embedding(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"
    
from transformers import RobertaTokenizerFast, RobertaModel
import torch


class ChemBERTaClassifier(torch.nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", num_labels=2):
        super().__init__()
        self.chemberta = RobertaModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.chemberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
        return self.classifier(cls_rep)


class BERT(torch.nn.Module):
    def __init__(self, in_feats, **kwargs):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.loss_fn = torch.nn.NLLLoss()
        # self.loss_fn = torch.nn.NLLLoss(reduction="none")

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr,
                                          weight_decay=self.model.weight_decay)

        # Save initial weights in the model for the anchored regularization and move them to the gpu
        if self.model.anchored:
            self.model.anchor_weights = deepcopy({i: j for i, j in self.model.named_parameters()})
            self.model.anchor_weights = {i: j.to(self.device) for i, j in self.model.anchor_weights.items()}

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:

        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        scaler = torch.cuda.amp.GradScaler()

        for _ in bar:
            running_loss = 0
            items = 0

            for idx, batch in enumerate(dataloader):

                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):

                    x, y, w = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                    y_hat = self.model(x)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    loss = self.loss_fn(y_hat, y.squeeze())

                    scaler.scale(loss).backward()
                    # loss.backward()
                    scaler.step(self.optimizer)
                    # self.optimizer.step()
                    scaler.update()

                    running_loss += loss.item()
                    items += len(y)

            epoch_loss = running_loss / items
            bar.set_postfix(loss=f'{epoch_loss:.4f}')
            self.train_loss.append(epoch_loss)
            self.epoch += 1

    def predict(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats

    def embedding(self, dataloader: DataLoader) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                for batch in dataloader:
                    if self.architecture in ['gcn', 'gat', 'gin']:
                        batch.to(self.device)
                        y_hat = self.model.embedding(batch.x.float(), batch.edge_index, batch.batch)
                    else:
                        x = batch[0].to(self.device)
                        y_hat = self.model.embedding(x)
                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)
                    y_hats = torch.cat((y_hats, y_hat), 0)

        return y_hats

class Ensemble_bert(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, in_feats = 1024, ensemble_size: int = 5, seed: int = 0, architecture: str = 'mlp', **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, 10)
        # self.models = {0: Model_test(seed=self.seeds[0], architecture=architecture, **kwargs)}
        self.models = {i: Model_test(seed=s, architecture=architecture, in_feats=in_feats, **kwargs) for i, s in enumerate(self.seeds)}

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, **kwargs) -> None:
        for i, m in self.models.items():
            m.train(dataloader, **kwargs)

    def predict(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C
    
    def embedding(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = self.models[0].embedding(dataloader)
        # logits_N_K_C = torch.stack([m.embedding(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"

class CosineContrastiveModel(torch.nn.Module):
    def __init__(self, device, input_dim=1024, embedding_dim=64, seed=0):
        super(CosineContrastiveModel, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, embedding_dim)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: (batch_size, input_dim)
        return: normalized embedding (batch_size, embedding_dim)
        """
        z = self.encoder(x)
        z = F.normalize(z, dim=1)  # L2 normalize
        logits = self.classifier(z)
        return z, logits

    def cosine_similarity(self, z1, z2):
        return F.cosine_similarity(z1, z2, dim=1)

    def supervised_contrastive_loss(self, embeddings, labels, temperature=0.07):
        """
        Supervised Contrastive Loss (Khosla et al., 2020)
        Args:
            embeddings: (batch_size, embedding_dim)
            labels: (batch_size,)  - integer labels
            temperature: scaling factor
        """
        device = embeddings.device
        embeddings = F.normalize(embeddings, dim=1)

        # similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

        # mask: same class -> 1, else 0
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # self-comparison 제거
        self_mask = torch.eye(mask.shape[0], dtype=torch.bool).to(device)
        mask = mask * (1 - self_mask.float())

        # scale by temperature
        logits = sim_matrix / temperature

        # exp(similarity) for negatives
        exp_logits = torch.exp(logits) * (~self_mask)

        pos_counts = mask.sum(dim=1)
        valid_mask = pos_counts > 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # log_prob for positives
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        # loss
        loss = -mean_log_prob_pos.mean()
        return loss

    def predict(self, dataloader):
        self.eval()
        decision = []
        with torch.no_grad():
            for data in dataloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                z, logits = self.forward(inputs)
                decision.append(logits.cpu())
        return torch.cat(decision)

    def get_embedding(self, dataloader):
        self.eval()
        decision = []
        embedding = []
        with torch.no_grad():
            for data in dataloader:
                inputs, _ = data
                inputs = inputs.to(self.device)
                z, logits = self.forward(inputs)
                decision.append(logits.cpu())
                embedding.append(z.cpu())
        return torch.cat(decision), torch.cat(embedding)


class Ensemble_cosine(torch.nn.Module):
    """ Ensemble of GCNs"""
    def __init__(self, device, ensemble_size: int = 10, seed: int = 0, architecture: str = 'mlp', **kwargs) -> None:
        self.ensemble_size = 5
        self.device = device
        self.architecture = architecture
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, ensemble_size)
        self.models = {}
        for i, s in enumerate(self.seeds):
            self.models[i] = CosineContrastiveModel(device, seed=s).to(device)

    def optimize_hyperparameters(self, x, y: DataLoader, **kwargs):
        # raise NotImplementedError
        best_hypers = optimize_hyperparameters(x, y, architecture=self.architecture, **kwargs)
        # # re-init model wrapper with optimal hyperparameters
        self.__init__(ensemble_size=self.ensemble_size, seed=self.seed, **best_hypers)

    def train(self, dataloader: DataLoader, **kwargs) -> None:
        for i, m in self.models.items():
            lr = 5e-4
            optimizer = torch.optim.Adam(m.parameters(), lr=lr)
            m.to(self.device)
            m.train()
            for epoch in range(50):
                for data in dataloader:
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    z, score = m(inputs)
                    bce_loss = torch.nn.BCELoss()(score, labels.float())
                    con_loss = m.supervised_contrastive_loss(z, labels, temperature=0.07)
                    loss = bce_loss 
                    # loss = con_loss*0.2 + bce_loss 
                    loss.backward()
                    optimizer.step()

    def predict(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)

        return logits_N_K_C

    def embedding(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        predict = []
        embedding = []
        aca_embedding = []
        for i, m in enumerate(self.models.values()):
            result = m.get_embedding(dataloader)
            predict.append(result[0])
            if i == 0:
                embedding.append(result[1])
                aca_embedding.append(result[1])
        logits_N_K_C = torch.stack(predict, 1)

        return logits_N_K_C, torch.cat(embedding), torch.cat(aca_embedding)

    def embedding_dropout(self, dataloader, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        predict = []
        embedding = []
        aca_embedding = []
        for m in self.models.values():
            for i in range(5):
                result = m.get_embedding_dropout(dataloader)
                predict.append(result[0])
                if i == 0:
                    embedding.append(result[1])
                    aca_embedding.append(result[1])
        logits_N_K_C = torch.stack(predict, 1)

        return logits_N_K_C, torch.cat(embedding), torch.cat(aca_embedding)

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} Classifiers"
    
class RfEnsemble():
    """ Ensemble of RFs"""
    def __init__(self, ensemble_size: int = 10, seed: int = 0, architecture = 'rf', **kwargs) -> None:
        self.ensemble_size = ensemble_size
        self.seed = seed
        rng = np.random.default_rng(seed=seed)
        self.seeds = rng.integers(0, 1000, ensemble_size)
        self.mode = architecture
        if self.mode == 'rf':
            self.models = {i: RandomForestClassifier(random_state=s, class_weight="balanced", **kwargs) for i, s in enumerate(self.seeds)}
        elif self.mode == 'xgb':
            self.models = {i: XGBClassifier(random_state=s, objective="binary:logistic", eta=0.1, scale_pos_weight=20) for i, s in enumerate(self.seeds)}
        elif self.mode == 'lgb':
            self.models = {i: LGBMClassifier(random_state=s, is_unbalance=True, verbosity = -1) for i, s in enumerate(self.seeds)}
        elif self.mode == 'svm':
            self.models = {i: SVC(random_state=s, probability=True, class_weight="balanced", **kwargs) for i, s in enumerate(self.seeds)}

    def train(self, x, y, **kwargs) -> None:
        for i, m in self.models.items():
            m.fit(x, y)

    def predict(self, x, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        # logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)
        eps = 1e-10  # we need to add this so we don't get divide by zero errors in our log function

        y_hats = []
        for m in self.models.values():
            # if self.mode == 'rf' or self.mode == 'svm':
            y_hat = torch.tensor(m.predict_proba(x) + eps)
            # elif self.mode == 'xgb':
            #     y_hat = torch.tensor(m.predict(x) + eps)
            # elif self.mode == 'lgb':
            #     y_hat = torch.tensor(m.predict_proba(x) + eps)
            # elif self.mode == 'svm':
            if y_hat.shape[1] == 1:  # if only one class if predicted with the RF model, add a column of zeros
                y_hat = torch.cat((y_hat, torch.zeros((y_hat.shape[0], 1))), dim=1)
            y_hats.append(y_hat)

        logits_N_K_C = torch.stack(y_hats, 1)

        logits_N_K_C = torch.log(logits_N_K_C)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} RF Classifiers"
