
from typing import Any
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections import OrderedDict
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import torch
import h5py
from config import ROOT_DIR
from active_learning.utils import molecular_graph_featurizer as smiles_to_graph, scramble_features, smiles_to_ecfp, smiles_to_murcko_ecfp, \
    get_tanimoto_matrix, check_featurizability, scramble_graphs, get_brics_fp, get_brics


def resolve_column_name(df: pd.DataFrame, column: str, path: str) -> str:
    if column in df.columns:
        return column
    matches = [col for col in df.columns if col.lower() == column.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Column '{column}' is ambiguous in {path}. Matches: {matches}")
    raise KeyError(f"Column '{column}' not found in {path}. Available columns: {list(df.columns)}")


def canonicalize(smiles: str, sanitize: bool = True):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=sanitize))


def get_data(random_state: int = 42, dataset: str = 'ALDH1'):

    # read smiles from file and canonicalize them
    with open(os.path.join(ROOT_DIR, f'data/{dataset}/original/inactives.smi')) as f:
        inactives = [canonicalize(smi.strip().split()[0]) for smi in f.readlines()]
    with open(os.path.join(ROOT_DIR, f'data/{dataset}/original/actives.smi')) as f:
        actives = [canonicalize(smi.strip().split()[0]) for smi in f.readlines()]

    # remove duplicates:
    inactives = list(set(inactives))
    actives = list(set(actives))

    # remove intersecting molecules:
    intersecting_mols = np.intersect1d(inactives, actives)
    inactives = [smi for smi in inactives if smi not in intersecting_mols]
    actives = [smi for smi in actives if smi not in intersecting_mols]

    # remove molecules that have scaffolds that cannot be kekulized or featurized
    inactives_, actives_ = [], []
    for smi in tqdm(actives):
        try:
            if Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False)) is not None:
                if check_featurizability(smi):
                    actives_.append(smi)
        except:
            pass
    for smi in tqdm(inactives):
        try:
            if Chem.MolFromSmiles(smi_to_scaff(smi, includeChirality=False)) is not None:
                if check_featurizability(smi):
                    inactives_.append(smi)
        except:
            pass

    # add to df
    df = pd.DataFrame({'smiles': inactives_ + actives_,
                       'y': [0] * len(inactives_) + [1] * len(actives_)})

    # shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def split_data(df: pd.DataFrame, random_state: int = 42, screen_size: int = 50000, test_size: int = 10000,
               dataset: str = 'ALDH1') -> (pd.DataFrame, pd.DataFrame):

    from sklearn.model_selection import train_test_split
    df_screen, df_test = train_test_split(df, stratify=df['y'].tolist(), train_size=screen_size, test_size=test_size,
                                          random_state=random_state)

    # write to csv
    df_screen.to_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/screen.csv'), index=False)
    df_test.to_csv(os.path.join(ROOT_DIR, f'data/{dataset}/original/test.csv'), index=False)

    return df_screen, df_test

from transformers import AutoTokenizer, AutoModel
from unimol_tools import UniMolRepr


class MasterDataset:
    """ Dataset that holds all data in an indexable way """
    def __init__(self, name: str, df: pd.DataFrame = None, dataset: str = 'ALDH1', nbits=1024, feature = '', representation: str = 'ecfp', root: str = 'data',
                 overwrite: bool = False, scramble_x: bool = False, scramble_x_seed: int = 1) -> None:

        assert representation in ['ecfp', 'graph', 'scaffold'], f"'representation' must be 'ecfp' or 'graph', not {representation}"
        self.representation = representation
        self.pth = os.path.join(ROOT_DIR, root, dataset, name)


        # If not done already, process all data. Else just load it
        if not os.path.exists(self.pth) or overwrite:
            assert df is not None, "You need to supply a dataframe with 'smiles' and 'y' values"
            os.makedirs(os.path.join(root, dataset, name), exist_ok=True)
            self.process(df)
            self.smiles_index, self.index_smiles, self.smiles, self.x, self.y, self.graphs = self.load()
        else:
            self.smiles_index, self.index_smiles, self.smiles, self.x, self.y, self.graphs = self.load()

        feature_map = {'cb':'chemberta', 'mf':'molformer', 'um':'unimol', 'ba':'brics_all', 'bp':'brics_pos', 'bas':'brics_all_sim', 'bps':'brics_pos_sim','fp+cb':'chemberta', 
                       'fp+mf':'molformer', 'fp+um':'unimol', 'fp+ba':'brics_all', 'fp+bp':'brics_pos', 'fp+bas':'brics_all_sim', 'fp+ps':'brics_pos_sim', 
                       'fingerprint':'', '':'', 'fp':'fingerprint', 'gcn':'gcn', 'gat':'gat', 'gin':'gin', '0':'0'}

        # self.x = smiles_to_ecfp(self.smiles, radius=3, nbits=256, silent=False)
        # self.x = np.load(f'./DrugCLIP/{dataset}_embedding.npy')
        self.murcko = self.x #smiles_to_murcko_ecfp(self.smiles, silent=False)
        self.feature = []
        self.brics = None

        feature = [x for x in feature.split("+")]

        self.feature_name = [feature_map[f_i] for f_i in feature]
        # self.feature_name = ['fingerprint', 'unimol', feature_map[feature]]
        # self.feature_name = ['molformer', 'unimol', feature_map[feature]]
        for f in self.feature_name:
            print(f)
            if f == 'fingerprint':
                self.feature.append(self.x)
            elif f == '0':
                self.feature.append(np.zeros(512))
            elif f[0] != 'g':
                self.feature.append(self.get_feature(f, dataset, name))
            
        if scramble_x:
            if representation == 'ecfp':
                self.x = scramble_features(self.x, seed=scramble_x_seed)
            if representation == 'scaffold':
                self.x = scramble_features(self.x, seed=scramble_x_seed)
            if representation == 'graph':
                self.graphs = scramble_graphs(self.graphs, seed=scramble_x_seed)
                self.x = scramble_features(self.x, seed=scramble_x_seed)
    
    def update_brics(self, train_smiles, train_positive_smiles):
        for i, f in enumerate(self.feature_name):
            if 'brics' in f:
                if f == 'brics_all':
                    self.feature[i] = get_brics_fp(train_smiles, self.brics, self.smiles)
                elif f == 'brics_pos':
                    self.feature[i] = get_brics_fp(train_positive_smiles, self.brics, self.smiles)
                elif f == 'brics_all_sim':
                    self.feature[i] = get_brics_fp(train_smiles, self.brics, self.smiles, sim=True)
                elif f == 'brics_pos_sim':
                    self.feature[i] = get_brics_fp(train_positive_smiles, self.brics, self.smiles, sim=True)
                print('fp_size: ', len(self.feature[i][0]))
    
    def get_feature(self, feature, dataset, name):
        result = None
        if feature in ['chemberta', 'molformer']:
            if feature == 'chemberta':
                if os.path.exists(f"/data2/project/junha/traversing_chem_space/data/{dataset}/chemberta_embeddings_{name}.npy"):
                    return np.load(f"/data2/project/junha/traversing_chem_space/data/{dataset}/chemberta_embeddings_{name}.npy")
                model_name = "seyonec/ChemBERTa-zinc-base-v1"
            elif feature == 'molformer':
                if os.path.exists(f"/data2/project/junha/traversing_chem_space/data/{dataset}/molformer_embeddings_{name}.npy"):
                    return np.load(f"/data2/project/junha/traversing_chem_space/data/{dataset}/molformer_embeddings_{name}.npy")
                model_name = "ibm/MoLFormer-XL-both-10pct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = torch.device("cuda:1")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            batch_size = 1024
            embedding = []
            smiles_list = list(self.smiles)
            for i in range(0, len(smiles_list), batch_size):
                inputs = self.tokenizer(smiles_list[i:i+batch_size], return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 보통 [CLS] 토큰 또는 평균 pooling을 사용
                if feature == 'chemberta':
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()   # [CLS] 토큰 (batch, hidden_dim)
                elif feature == 'molformer':
                    embeddings = outputs.pooler_output.cpu().numpy()
                embedding.append(embeddings)

            # self.x = np.concatenate(embedding, axis=0)
            result = np.concatenate(embedding, axis=0)
            np.save(f"/data2/project/junha/traversing_chem_space/data/{dataset}/chemberta_embeddings_{name}.npy", result)
            # self.x = np.concatenate([self.x, self.chemberta], axis=1)

        ################### unimol ################
        elif feature == 'unimol':
            result = np.load(f"/data2/project/junha/traversing_chem_space/data/{dataset}/unimol_embeddings.npy")
            # self.x = np.load(f"/data2/project/junha/traversing_chem_space/data/{dataset}/unimol_embeddings.npy")
            # self.x = np.concatenate([self.x, self.chemberta], axis=1)
        elif 'brics' in feature:
            self.brics = get_brics(self.smiles, dataset, split=name)
            result = self.x
        return result

    def process(self, df: pd.DataFrame) -> None:

        print('Processing data ... ', flush=True, file=sys.stderr)

        index_smiles = OrderedDict({i: smi for i, smi in enumerate(df.smiles)})
        smiles_index = OrderedDict({smi: i for i, smi in enumerate(df.smiles)})
        smiles = np.array(df.smiles.tolist())
        x = smiles_to_ecfp(smiles, silent=False)
        y = torch.tensor(df.y.tolist())
        graphs = [smiles_to_graph(smi, y=y.type(torch.LongTensor)) for smi, y in tqdm(zip(smiles, y))]

        torch.save(index_smiles, os.path.join(self.pth, 'index_smiles'))
        torch.save(smiles_index, os.path.join(self.pth, 'smiles_index'))
        torch.save(smiles, os.path.join(self.pth, 'smiles'))
        torch.save(x, os.path.join(self.pth, 'x'))
        torch.save(y, os.path.join(self.pth, 'y'))
        torch.save(graphs, os.path.join(self.pth, 'graphs'))

    def load(self) -> (dict, dict, np.ndarray, np.ndarray, np.ndarray, list):

        print('Loading data ... ', flush=True, file=sys.stderr)

        index_smiles = torch.load(os.path.join(self.pth, 'index_smiles'))
        smiles_index = torch.load(os.path.join(self.pth, 'smiles_index'))
        smiles = torch.load(os.path.join(self.pth, 'smiles'))
        x = torch.load(os.path.join(self.pth, 'x'))
        y = torch.load(os.path.join(self.pth, 'y'))
        graphs = torch.load(os.path.join(self.pth, 'graphs'))

        return smiles_index, index_smiles, smiles, x, y, graphs

    def __len__(self) -> int:
        return len(self.smiles)

    def all(self):
        return self[range(len(self.smiles))]

    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]
        if self.representation == 'ecfp':
            return self.x[idx], self.y[idx], self.smiles[idx], [self.feature[i][idx] for i in range(len(self.feature))]
        if self.representation == 'graph':
            return [self.graphs[i] for i in idx], self.y[idx], self.smiles[idx], [self.feature[i][idx] for i in range(len(self.feature))]
        if self.representation == 'scaffold':
            return self.x[idx], self.y[idx], self.smiles[idx], self.x[idx], self.murcko[idx]


class MasterDataset2Labeled: # Test가 아닌경우
    """Test가 아닌경우"""
    """ Dataset that holds all data in an indexable way """
    def __init__(self, name: str, df: pd.DataFrame = None, dataset: str = 'ALDH1', nbits=1024, feature = '', representation: str = 'ecfp', root: str = 'data',
                 overwrite: bool = False, scramble_x: bool = False, scramble_x_seed: int = 1, input='./data/input.csv', assay_active = None, assay_inactive = None, input_val_col='y', input_smiles_col='smiles', is_reverse=False) -> None:

        assert representation in ['ecfp', 'graph', 'scaffold'], f"'representation' must be 'ecfp' or 'graph', not {representation}"
        self.mode = name
        self.representation = representation
        self.pth = input
        self.assay_active = assay_active
        self.assay_inactive = assay_inactive
        self.input_val_col = input_val_col
        self.input_smiles_col = input_smiles_col
        self.is_reverse = is_reverse

        self.smiles, self.x, self.y = self.load()

        feature_map = {'cb':'chemberta', 'mf':'molformer', 'um':'unimol', 'ba':'brics_all', 'bp':'brics_pos', 'bas':'brics_all_sim', 'bps':'brics_pos_sim','fp+cb':'chemberta', 
                       'fp+mf':'molformer', 'fp+um':'unimol', 'fp+ba':'brics_all', 'fp+bp':'brics_pos', 'fp+bas':'brics_all_sim', 'fp+ps':'brics_pos_sim', 
                       'fingerprint':'', '':'', 'fp':'fingerprint', 'gcn':'gcn', 'gat':'gat', 'gin':'gin', '0':'0'}

        self.murcko = self.x #smiles_to_murcko_ecfp(self.smiles, silent=False)
        feature = 'fp+mf+um'
        self.feature = []
        self.brics = None

        feature = [x for x in feature.split("+")]

        self.feature_name = [feature_map[f_i] for f_i in feature]
        for f in self.feature_name:
            if f == 'fingerprint':
                self.feature.append(self.x)
            elif f == '0':
                self.feature.append(np.zeros(512))
            elif f[0] != 'g':
                self.feature.append(self.get_feature(f, dataset, name))
    
    def update_brics(self, train_smiles, train_positive_smiles):
        for i, f in enumerate(self.feature_name):
            if 'brics' in f:
                if f == 'brics_all':
                    self.feature[i] = get_brics_fp(train_smiles, self.brics, self.smiles)
                elif f == 'brics_pos':
                    self.feature[i] = get_brics_fp(train_positive_smiles, self.brics, self.smiles)
                elif f == 'brics_all_sim':
                    self.feature[i] = get_brics_fp(train_smiles, self.brics, self.smiles, sim=True)
                elif f == 'brics_pos_sim':
                    self.feature[i] = get_brics_fp(train_positive_smiles, self.brics, self.smiles, sim=True)
                print('fp_size: ', len(self.feature[i][0]))
    
    def get_feature(self, feature, dataset, name):
        result = None
        if feature in ['chemberta', 'molformer']:
            if feature == 'chemberta':
                model_name = "seyonec/ChemBERTa-zinc-base-v1"
            elif feature == 'molformer':
                model_name = "ibm/MoLFormer-XL-both-10pct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = torch.device("cuda:1")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            batch_size = 1024
            embedding = []
            smiles_list = list(self.smiles)
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="Embedding", unit="batch"):

                inputs = self.tokenizer(smiles_list[i:i+batch_size], return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 보통 [CLS] 토큰 또는 평균 pooling을 사용
                if feature == 'chemberta':
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()   # [CLS] 토큰 (batch, hidden_dim)
                elif feature == 'molformer':
                    embeddings = outputs.pooler_output.cpu().numpy()
                embedding.append(embeddings)
            result = np.concatenate(embedding, axis=0)

        ################### unimol ################
        elif feature == 'unimol':
            repr_model = UniMolRepr(
                data_type='molecule',   # 'molecule', 'oled', 'pocket' 등
                remove_hs=False,
                model_name='unimolv2',  # 또는 'unimolv2'
                model_size='84m',       # unimolv2 쓸 때만 의미 있음
                # pretrained_model_path=None, pretrained_dict_path=None 이면
                # 알아서 HuggingFace에서 가중치 다운로드
            )
            
            result = np.array(repr_model.get_repr(list(self.smiles))['cls_repr'])
        elif 'brics' in feature:
            self.brics = get_brics(self.smiles, dataset, split=name)
            result = self.x
        return result

    def load(self) -> (np.ndarray, np.ndarray, np.ndarray):

        print('Loading data ... ', flush=True, file=sys.stderr)

        csv = pd.read_csv(self.pth)
        smiles_col = resolve_column_name(csv, self.input_smiles_col, self.pth)
        smiles = np.array(csv[smiles_col])
        x = smiles_to_ecfp(smiles, silent=False)
        if self.mode != 'test' and self.assay_active is not None:
            csv.loc[csv[self.input_val_col].isin(self.assay_active), self.input_val_col] = 1
            csv.loc[csv[self.input_val_col].isin(self.assay_inactive), self.input_val_col] = 0
            csv[self.input_val_col] = csv[self.input_val_col].astype(int)
        
        elif self.mode != 'test' and self.assay_active is None:
            csv[self.input_val_col] = csv[self.input_val_col].replace([np.inf, -np.inf], np.nan)
            mu = csv[self.input_val_col].mean()
            sigma = csv[self.input_val_col].std(ddof=0)   # population std (권장)

            csv[self.input_val_col] = (csv[self.input_val_col] - mu) / sigma
            if self.is_reverse:
                csv[self.input_val_col] = -csv[self.input_val_col]

        y = np.array(csv[self.input_val_col])

        return smiles, x, y

    def __len__(self) -> int:
        return len(self.smiles)

    def all(self):
        return self[range(len(self.smiles))]

    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]
        if self.representation == 'ecfp':
            return self.x[idx], self.y[idx], self.smiles[idx], [self.feature[i][idx] for i in range(len(self.feature))]

class MasterDataset2Unlabeled: # Test인 경우
    """Test인 경우"""
    """ Dataset that holds all data in an indexable way """
    def __init__(self, name: str, df: pd.DataFrame = None, dataset: str = 'ALDH1', nbits=1024, feature = '', representation: str = 'ecfp', root: str = 'data',
                 overwrite: bool = False, scramble_x: bool = False, scramble_x_seed: int = 1, input='./data/input.csv', assay_active = None, assay_inactive = None, input_unlabel_val_col='score', input_unlabel_smiles_col='smiles') -> None:

        assert representation in ['ecfp', 'graph', 'scaffold'], f"'representation' must be 'ecfp' or 'graph', not {representation}"
        self.mode = name
        self.representation = representation
        self.pth = input
        self.input_unlabel_val_col = input_unlabel_val_col
        self.input_unlabel_smiles_col = input_unlabel_smiles_col

        self.smiles, self.x, self.y = self.load()

        feature_map = {'cb':'chemberta', 'mf':'molformer', 'um':'unimol', 'ba':'brics_all', 'bp':'brics_pos', 'bas':'brics_all_sim', 'bps':'brics_pos_sim','fp+cb':'chemberta', 
                       'fp+mf':'molformer', 'fp+um':'unimol', 'fp+ba':'brics_all', 'fp+bp':'brics_pos', 'fp+bas':'brics_all_sim', 'fp+ps':'brics_pos_sim', 
                       'fingerprint':'', '':'', 'fp':'fingerprint', 'gcn':'gcn', 'gat':'gat', 'gin':'gin', '0':'0'}

        self.murcko = self.x #smiles_to_murcko_ecfp(self.smiles, silent=False)
        feature = 'fp+mf+um'
        self.feature = []
        self.brics = None

        feature = [x for x in feature.split("+")]

        self.feature_name = [feature_map[f_i] for f_i in feature]
        for f in self.feature_name:
            if f == 'fingerprint':
                self.feature.append(self.x)
            elif f == '0':
                self.feature.append(np.zeros(512))
            elif f[0] != 'g':
                self.feature.append(self.get_feature(f, dataset, name))
    
    def update_brics(self, train_smiles, train_positive_smiles):
        for i, f in enumerate(self.feature_name):
            if 'brics' in f:
                if f == 'brics_all':
                    self.feature[i] = get_brics_fp(train_smiles, self.brics, self.smiles)
                elif f == 'brics_pos':
                    self.feature[i] = get_brics_fp(train_positive_smiles, self.brics, self.smiles)
                elif f == 'brics_all_sim':
                    self.feature[i] = get_brics_fp(train_smiles, self.brics, self.smiles, sim=True)
                elif f == 'brics_pos_sim':
                    self.feature[i] = get_brics_fp(train_positive_smiles, self.brics, self.smiles, sim=True)
                print('fp_size: ', len(self.feature[i][0]))
    
    def get_feature(self, feature, dataset, name):
        result = None
        if feature in ['chemberta', 'molformer']:
            if feature == 'chemberta':
                model_name = "seyonec/ChemBERTa-zinc-base-v1"
            elif feature == 'molformer':
                model_name = "ibm/MoLFormer-XL-both-10pct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            device = torch.device("cuda:1")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
            batch_size = 1024
            embedding = []
            smiles_list = list(self.smiles)
            for i in tqdm(range(0, len(smiles_list), batch_size), desc="Embedding", unit="batch"):

                inputs = self.tokenizer(smiles_list[i:i+batch_size], return_tensors="pt", padding=True, truncation=True).to(device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # 보통 [CLS] 토큰 또는 평균 pooling을 사용
                if feature == 'chemberta':
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()   # [CLS] 토큰 (batch, hidden_dim)
                elif feature == 'molformer':
                    embeddings = outputs.pooler_output.cpu().numpy()
                embedding.append(embeddings)
            result = np.concatenate(embedding, axis=0)

        ################### unimol ################
        elif feature == 'unimol':
            repr_model = UniMolRepr(
                data_type='molecule',   # 'molecule', 'oled', 'pocket' 등
                remove_hs=False,
                model_name='unimolv2',  # 또는 'unimolv2'
                model_size='84m',       # unimolv2 쓸 때만 의미 있음
                # pretrained_model_path=None, pretrained_dict_path=None 이면
                # 알아서 HuggingFace에서 가중치 다운로드
            )
            
            result = np.array(repr_model.get_repr(list(self.smiles))['cls_repr'])
        elif 'brics' in feature:
            self.brics = get_brics(self.smiles, dataset, split=name)
            result = self.x
        return result

    def load(self) -> (np.ndarray, np.ndarray, np.ndarray):

        print('Loading data ... ', flush=True, file=sys.stderr)

        csv = pd.read_csv(self.pth)
        smiles_col = resolve_column_name(csv, self.input_unlabel_smiles_col, self.pth)
        smiles = np.array(csv[smiles_col])
        x = smiles_to_ecfp(smiles, silent=False)
        if self.mode != 'test' and self.assay_active is not None:
            csv.loc[csv[self.input_val_col].isin(self.assay_active), self.input_val_col] = 1
            csv.loc[csv[self.input_val_col].isin(self.assay_inactive), self.input_val_col] = 0
            csv[self.input_val_col] = csv[self.input_val_col].astype(int)
        y = np.zeros(len(csv), dtype=float)
        return smiles, x, y

    def __len__(self) -> int:
        return len(self.smiles)

    def all(self):
        return self[range(len(self.smiles))]

    def __getitem__(self, idx):
        if type(idx) is int:
            idx = [idx]
        if self.representation == 'ecfp':
            return self.x[idx], self.y[idx], self.smiles[idx], [self.feature[i][idx] for i in range(len(self.feature))]

def smi_to_scaff(smiles: str, includeChirality: bool = False):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=includeChirality)


def similarity_vectors(df_screen, df_test, root: str = 'data', dataset: str = 'ALDH1'):

    print("Computing Tanimoto matrix for all test molecules")
    S = get_tanimoto_matrix(df_test['smiles'].tolist(), verbose=True, scaffolds=False, zero_diag=True, as_vector=True)
    save_hdf5(1-S, f'{ROOT_DIR}/{root}/{dataset}/test/tanimoto_distance_vector')
    del S

    print("Computing Tanimoto matrix for all screen molecules")
    S = get_tanimoto_matrix(df_screen['smiles'].tolist(), verbose=True, scaffolds=False, zero_diag=True, as_vector=True)
    save_hdf5(1 - S, f'{ROOT_DIR}/{root}/{dataset}/screen/tanimoto_distance_vector')
    del S


def save_hdf5(obj: Any, filename: str):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('obj', data=obj)
    hf.close()


def load_hdf5(filename: str) -> Any:
    hf = h5py.File(filename, 'r')
    obj = np.array(hf.get('obj'))
    hf.close()

    return obj
