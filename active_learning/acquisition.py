"""

This script contains a selection of sample acquisition methods for active learning.
All functions here operate on model predictions in the form of logits_N_K_C = [N, num_inference_samples, num_classes].
Here N is a molecule, K are the number of sampled predictions (i.e., 10 for a 10-model ensemble), and C = 2 ([0, 1]):

    Author: Derek van Tilborg, Eindhoven University of Technology, May 2023

"""

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as dist
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as ECFPbitVec
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit import Chem
import math
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import os

class Acquisition:
    def __init__(self, method: str, seed: int = 42, **kwargs):

        self.acquisition_method = {'random': self.random_pick,
                                   'exploration': greedy_exploration,
                                   'exploitation': greedy_exploitation,
                                   'round_exploitation': round_exploitation,
                                   'dynamic': dynamic_exploration,
                                   'dynamicbald': dynamic_exploration_bald,
                                   'bald': bald,
                                   'similarity': similarity_search,
                                   'loss_prediction': loss_prediction,
                                   'epig': epig,
                                   'ei': expected_improvement,
                                   'pi': probability_of_improvement,
                                   'ts': thompson_sampling,
                                   'boundre': greedy_boundre,
                                   'boundre_fp': greedy_boundre,
                                   'boundre_ac': greedy_boundre,
                                   'boundre_bald': bald_boundre,
                                   'cosine': greedy_boundre,
                                   'evaluation_exploitation': evaluation_exploitation}

        assert method in self.acquisition_method.keys(), f"Specified 'method' not available. " \
                                                         f"Select from: {self.acquisition_method.keys()}"

        self.method = method
        self.params = kwargs
        self.rng = np.random.default_rng(seed=seed)
        self.iteration = 0

    def acquire(self, logits_N_K_C: Tensor, smiles: np.ndarray[str], hits: np.ndarray[str], screen_loss=None, y_screen='', dir_name='', cliff=0, beta=0, cycle_threshold=0, n: int = 1, seed: int = 0, cycle=0,output='', classification=False) -> \
            np.ndarray[str]:

        self.iteration += 1

        return self.acquisition_method[self.method](logits_N_K_C=logits_N_K_C, smiles=smiles, screen_loss=screen_loss, n=n, y_screen=y_screen, dir_name=dir_name, hits=hits,
                                                    iteration=self.iteration, seed=seed, cycle=cycle, cliff=cliff, beta=beta, cycle_threshold=cycle_threshold, output=output, classification = classification,
                                                    **self.params)

    def __call__(self, *args, **kwargs) -> np.ndarray[str]:
        return self.acquire(*args, **kwargs)

    def random_pick(self, smiles: np.ndarray[str], n: int = 1, return_smiles: bool = True, **kwargs) -> np.ndarray:
        """ select n random samples """
        picks_idx = self.rng.integers(0, len(smiles), n)

        return smiles[picks_idx] if return_smiles else picks_idx

@torch.no_grad()
def epig_scores_all(
    logits_pool_N_K_C: torch.Tensor,                  # [N,K,C]
    M_subsample: int = 100000,
    n_chunk: int = 2048,
    m_chunk: int = 256,
    eps: float = 1e-12,
    use_float64: bool = True,
):
    device = logits_pool_N_K_C.device
    N, K, C = logits_pool_N_K_C.shape
    acc_dtype = torch.float64 if use_float64 else logits_pool_N_K_C.dtype

    # 풀에서 무작위 M개 선택 (재현성 원하면 generator 지정)
    idx = torch.randperm(N, device=device)[:min(M_subsample, N)]
    logits_tgt = logits_pool_N_K_C[idx]
                              # [M',K,C]
    # 확률 및 주변분포(타깃) 미리 계산
    probs_tgt_full = F.softmax(logits_tgt, dim=-1)                       # [M,K,C]
    M = probs_tgt_full.shape[0]
    py_tgt_full  = probs_tgt_full.mean(dim=1).clamp_min(eps)             # [M,C]
    log_py_tgt_full = torch.log(py_tgt_full)                              # [M,C]

    # --- 출력 초기화 ---
    epig = torch.zeros(N, dtype=acc_dtype, device=device)

    # --- 후보 N을 chunk로 나눠 처리 ---
    for n0 in range(0, N, n_chunk):
        n1 = min(n0 + n_chunk, N)
        logits_pool_chunk = logits_pool_N_K_C[n0:n1]                     # [n,K,C]
        probs_pool_chunk  = F.softmax(logits_pool_chunk, dim=-1)         # [n,K,C]
        py_pool_chunk     = probs_pool_chunk.mean(dim=1).clamp_min(eps)  # [n,C]
        log_py_pool_chunk = torch.log(py_pool_chunk)                      # [n,C]

        # 이 청크의 누적
        epig_chunk = torch.zeros(n1 - n0, dtype=acc_dtype, device=device)

        # --- 타깃 M도 chunk로 ---
        for m0 in range(0, M, m_chunk):
            m1 = min(m0 + m_chunk, M)
            probs_tgt = probs_tgt_full[m0:m1]                            # [m,K,C]
            py_tgt    = py_tgt_full[m0:m1]                               # [m,C]
            log_py_tgt= log_py_tgt_full[m0:m1]                           # [m,C]

            # 공동예측분포: (1/K) * sum_k p_k(y|x) p_k(y*|x*)
            # einsum: [n,K,C] x [m,K,C] -> [n,m,C,C]
            joint = torch.einsum('nkc,mkd->nmcd', probs_pool_chunk, probs_tgt) / K
            joint = joint.clamp_min(eps)
            log_joint = torch.log(joint)

            # KL( joint || py_pool ⊗ py_tgt )
            term = (log_joint
                    - log_py_pool_chunk[:, None, :, None]
                    - log_py_tgt[None, :, None, :])                      # [n,m,C,C]
            kl_nm = (joint * term).sum(dim=(2,3))                        # [n,m]
            epig_chunk += kl_nm.sum(dim=1).to(acc_dtype)                 # [n]

        # x* 기대값으로 정규화
        epig[n0:n1] = epig_chunk / M

    return epig  # [N], 모든 후보의 EPIG 점수

def epig(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    scores = epig_scores_all(logits_N_K_C)
    picks_idx = torch.argsort(scores, descending=True)[:n]

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]


def probability_of_improvement(
    logits_N_K_C: torch.Tensor,
    smiles: np.ndarray[str],
    n: int = 10,
    **kwargs
) -> np.ndarray[str]:
    """
    Select top-n molecules using Probability of Improvement (PI).
    
    Args:
        logits: [N, K, C] tensor (N candidates, K stochastic forward passes, C classes)
        smiles: numpy array of SMILES strings (length N)
        pos_class: target class index (usually 1 for positive/hit)
        n: number of molecules to select
        xi: exploration parameter for PI

    Returns:
        np.ndarray of selected SMILES strings (length n)
    """
    # 확률 변환
    pos_class = 1
    xi = 0.01

    probs = F.softmax(logits_N_K_C, dim=-1)          # [N, K, C]
    pos_probs = probs[..., pos_class]          # [N, K]

    # 평균, 표준편차
    p_mean = pos_probs.mean(dim=1)             # [N]
    p_std = pos_probs.std(dim=1)               # [N]

    # 현재까지 best 확률 (mean 기준으로 정의)
    p_best = p_mean.max().item()

    # PI 계산
    eps = 1e-8
    z = (p_mean - p_best - xi) / (p_std + eps)

    normal = dist.Normal(torch.tensor(0.0, device=logits_N_K_C.device),
                         torch.tensor(1.0, device=logits_N_K_C.device))
    pi = normal.cdf(z)                         # [N]

    # 상위 n개 선택
    picks_idx = torch.topk(pi, n).indices.cpu().numpy()
    return np.array([smiles[picks_idx]]) if n == 1 else smiles[picks_idx]

def expected_improvement(
    logits_N_K_C: torch.Tensor,
    smiles: np.ndarray[str],
    n: int = 10,
    **kwargs
) -> np.ndarray[str]:
    """
    Select top-n molecules using Expected Improvement (EI).
    
    Args:
        logits: [N, K, C] tensor (N candidates, K stochastic forward passes, C classes)
        smiles: numpy array of SMILES strings (length N)
        pos_class: target class index (usually 1 for positive/hit)
        n: number of molecules to select
        xi: exploration parameter for EI

    Returns:
        np.ndarray of selected SMILES strings (length n)
    """
    
    pos_class = 1
    xi = 0.01

    # 확률 변환
    probs = F.softmax(logits_N_K_C, dim=-1)          # [N, K, C]
    pos_probs = probs[..., pos_class]          # [N, K]

    # 평균, 표준편차
    p_mean = pos_probs.mean(dim=1)             # [N]
    p_std = pos_probs.std(dim=1)               # [N]

    # 현재까지 best 확률 (mean 기준으로 정의)
    p_best = p_mean.max().item()

    # EI 계산
    eps = 1e-8
    z = (p_mean - p_best - xi) / (p_std + eps)

    normal = dist.Normal(torch.tensor(0.0, device=logits_N_K_C.device),
                         torch.tensor(1.0, device=logits_N_K_C.device))
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))

    improvement = p_mean - p_best - xi
    ei = improvement * cdf + p_std * pdf       # [N]

    # 상위 n개 선택
    picks_idx = torch.topk(ei, n).indices.cpu().numpy()
    return np.array([smiles[picks_idx]]) if n == 1 else smiles[picks_idx]

def get_pmean_pstd_from_logits(logits, pos_class=1):
    """
    logits: [N, K, C]
    pos_class: positive 클래스 index
    """
    probs = F.softmax(logits, dim=-1)          # [N, K, C]
    pos_probs = probs[..., pos_class]          # [N, K]
    p_mean = pos_probs.mean(dim=1)             # [N]
    p_std  = pos_probs.std(dim=1)              # [N]
    return p_mean, p_std

def logits_to_pred(logits_N_K_C: Tensor, return_prob: bool = True, return_uncertainty: bool = True) -> (Tensor, Tensor):
    """ Get the probabilities/class vector and sample uncertainty from the logits """

    mean_probs_N_C = torch.mean(torch.exp(logits_N_K_C), dim=1)
    uncertainty = mean_sample_entropy(logits_N_K_C)

    if return_prob:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat


def logit_mean(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """ Logit mean with the logsumexp trick - Kirch et al., 2019, NeurIPS """

    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])


def entropy(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Calculates the Shannon Entropy """

    return -torch.sum((torch.exp(logits_N_K_C) * logits_N_K_C).double(), dim=dim, keepdim=keepdim)


def mean_sample_entropy(logits_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Calculates the mean entropy for each sample given multiple ensemble predictions - Kirch et al., 2019, NeurIPS"""

    sample_entropies_N_K = entropy(logits_N_K_C, dim=dim, keepdim=keepdim)
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)

    return entropy_mean_N


def mutual_information(logits_N_K_C: Tensor) -> Tensor:
    """ Calculates the Mutual Information - Kirch et al., 2019, NeurIPS """
    # this term represents the entropy of the model prediction (high when uncertain)
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)

    # This term is the expectation of the entropy of the model prediction for each draw of model parameters
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)

    I = mean_entropy_N - entropy_mean_N

    return I

def draw_fig(y_screen, idx, dir_name, cycle):
    y_screen_sorted = y_screen[idx.cpu().numpy()]

    indices = np.arange(1, len(y_screen) + 1)

    cumulative_pos = np.cumsum(y_screen_sorted)

    positive_rate = cumulative_pos / indices

    os.makedirs(dir_name+f"/plot", exist_ok=True)

    # plt.figure(figsize=(8,5))
    # plt.plot(indices, positive_rate, label="Positive rate", color="red")
    # plt.xlabel("Index")
    # plt.ylabel("Positive Rate")
    # plt.title("Cumulative Positive Rate by Index")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.savefig(dir_name + f'/plot/cum_pos{cycle}.png')
    # plt.cla()
    # plt.clf()
    # plt.close()
    for window in [100, 1000]:
        mov_avg = np.array([
            y_screen_sorted[i:min(i+window, len(y_screen))].float().mean()
            for i in range((len(y_screen) - window + 1))
        ])
        x_vals = np.arange(1, len(y_screen) - window + 2)

        plt.figure(figsize=(8,5))
        plt.plot(x_vals[::5], mov_avg[::5], label=f"Moving avg (window={window})")
        plt.xlabel("Prediction score (sorted)")
        plt.ylabel("Positive rate (moving avg)")
        plt.title("Local positive rate across prediction scores")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(dir_name + f'/plot/window_{window}_{cycle}.png')
        plt.cla()
        plt.clf()
        plt.close()

def boltzmann_sample_torch(scores: torch.Tensor, temperature: float = 0.5, k: int = 64) -> torch.Tensor:
    """
    PyTorch 텐서와 볼츠만 분포를 사용하여 후보군에서 K개의 인덱스를 확률적으로 샘플링합니다.

    Args:
        scores (torch.Tensor): 각 분자(후보)의 모델 점수 텐서 (1D).
        temperature (float): 볼츠만 분포의 온도 매개변수 (T).
        k (int): 최종적으로 샘플링할 분자의 개수 (기본값 64).

    Returns:
        torch.Tensor: 선택된 k개 분자의 인덱스 텐서 (비복원 추출).
    """
    sorted_indices = scores.argsort(descending=True)
    top_M_original_indices = sorted_indices[:1000]
    top_M_scores = scores[top_M_original_indices]
    # 1. 볼츠만 지수 계산 (S/T)
    # scores.div(temperature) -> scores / T
    scaled_scores = top_M_scores.div(temperature)
    # 2.2. 최대값 찾기 (오버플로우 방지 핵심)
    max_scaled_score = scaled_scores.max()
    
    # 2.3. 안정적인 볼츠만 지수 계산: exp((S_i / T) - Max(S_j / T))
    # 입력값에서 최대값을 빼줌으로써 exp()의 인자를 음수/0 근처로 유지
    boltzmann_exponentials = (scaled_scores - max_scaled_score).exp()
    probabilities = boltzmann_exponentials / boltzmann_exponentials.sum()
    
    # 3. 확률에 따라 k개 비복원 샘플링 (Sampling without replacement)
    # PyTorch의 multinomial 함수 사용
    # probabilities 텐서는 반드시 CPU에 있어야 함 (torch.multinomial 제약 사항)
    sorted_indices = scores.argsort(descending=True)
    
    # 2.3. 선택된 확률 합산
    print(f'top64: {(probabilities[:64]).sum().item()}, top100: {(probabilities[:100]).sum().item()}') # .item()으로 Python float으로 변환

    if probabilities.is_cuda:
        probabilities_cpu = probabilities.cpu()
    else:
        probabilities_cpu = probabilities
        
    local_sampled_indices = torch.multinomial(
        input=probabilities_cpu,
        num_samples=k,
        replacement=False # 비복원 추출 (가장 중요)
    )
    sampled_indices = top_M_original_indices[local_sampled_indices]
    
    if scores.is_cuda:
        sampled_indices = sampled_indices.to(scores.device)
        
    return sampled_indices

def greedy_exploitation(logits_N_K_C: Tensor, smiles: np.ndarray[str], y_screen, dir_name, cycle, beta=0, n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    # print(logits_N_K_C)
    # mean_probs_hits = torch.mean(logits_N_K_C, dim=1)
    probs = F.softmax(logits_N_K_C, dim=-1)
    mean_probs_hits = torch.mean(probs, dim=1)[:, 1]
    # mean_probs_hits = probs[:, 0, 1]
    UCB = beta>0
    # mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    if not UCB:
        idx = torch.argsort(mean_probs_hits, descending=True)

        picks_idx = idx[:n]
        picks_idx1 = idx
        if dir_name != '':
            df = pd.DataFrame({
                "idx": picks_idx1.cpu().numpy(),
                "prob": mean_probs_hits[picks_idx1].cpu().numpy(),
                "label": y_screen.numpy()[picks_idx1.cpu().numpy()],
                "smiles": smiles[picks_idx1.cpu().numpy()]
            })
            df.to_csv(dir_name + f"/sorted_results{cycle}.csv", index=False, float_format="%.6f")
    else:
        # beta = 0.5
        sigma = probs[:, :, 1].std(dim=1)
        ucb = mean_probs_hits + beta * sigma
        idx = torch.argsort(ucb, descending=True)

        picks_idx = idx[:n]
        if dir_name != '':
            df = pd.DataFrame({
                "idx": picks_idx.cpu().numpy(),
                "ucb": ucb[picks_idx].cpu().numpy(),
                "prob": mean_probs_hits[picks_idx].cpu().numpy(),
                "sigma": sigma[picks_idx].cpu().numpy(),
                "label": y_screen.numpy()[picks_idx.cpu().numpy()]
            })
            df.to_csv(dir_name + f"/sorted_results{cycle}.csv", index=False, float_format="%.6f")

        # draw_fig(y_screen, idx, dir_name, cycle)

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]

def thompson_sampling(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:
    # logits: [N, K, C]
    pos_class=1
    probs = F.softmax(logits_N_K_C, dim=-1)
    pos_probs = probs[..., pos_class]  # [N, K]

    K = pos_probs.shape[1]
    k = torch.randint(0, K, (1,)).item()   # 하나의 posterior 샘플 선택
    scores = pos_probs[:, k]               # [N]

    # 상위 n개 index 뽑기
    picks_idx = torch.topk(scores, n).indices.cpu().numpy()

    return np.array([smiles[picks_idx]]) if n == 1 else smiles[picks_idx]

def pick_top_and_boltzmann(mean_probs_hits: torch.Tensor, top_k=32, sample_k=32, T=0.2):
    """
    mean_probs_hits: shape [N], probabilities or scores
    top_k: number of top deterministic picks
    sample_k: number of stochastic picks from the rest
    T: temperature
    """

    # --- 1) top_k deterministic selection ---
    # sort by descending mean_probs_hits
    sorted_vals, sorted_idxs = torch.sort(mean_probs_hits, descending=True)
    top_idxs = sorted_idxs[:top_k]

    # --- 2) remaining pool ---
    remaining_idxs = sorted_idxs[top_k:]  # indices of the lower scores
    remaining_scores = mean_probs_hits[remaining_idxs]

    # --- 3) Boltzmann sample from the remaining ---
    # ensure sample_k <= len(remaining_idxs)
    sample_k = min(sample_k, len(remaining_idxs))

    boltzmann_selected_local = boltzmann_sample_torch(remaining_scores, temperature=T, k=sample_k)

    # boltzmann_selected_local는 remaining 중에서의 local index이므로
    boltzmann_selected_global = remaining_idxs[boltzmann_selected_local]

    # --- 4) 최종 합치기 ---
    final_idxs = torch.cat([top_idxs, boltzmann_selected_global], dim=0)

    return final_idxs

def round_exploitation(logits_N_K_C: Tensor, smiles: np.ndarray[str], screen_loss, y_screen, dir_name, cycle, cliff, cycle_threshold, beta=0, n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    # print(logits_N_K_C)
    # mean_probs_hits = torch.mean(logits_N_K_C, dim=1)
    # logits_ucb = logits_N_K_C.clone()  # 복사본 만들어서 수정
    # logits_ucb[:, :, 1] = logits_N_K_C[:, :, 1] + 1 * screen_loss.unsqueeze(1)
    probs = F.softmax(logits_N_K_C, dim=-1)
    probs = torch.mean(probs, dim=1)[:, 1]
    # probs = torch.mean(probs, dim=1)[:, 1]
    # prob_std = probs[:, :, 1] * (1 + screen_loss)
    # sigma = 0 if beta == 0 else prob_std.std(dim=1)
    # if beta > 0:
    #     probs = probs[:, 0, 1]
    #     screen_loss = screen_loss[:, 0]
    
    # mean_probs_hits = torch.mean(probs, dim=1)[:, 1]
    # probs2 = F.softmax(logits_N_K_C, dim=-1)
    # mean_probs_hits2 = torch.mean(probs2, dim=1)[:, 1]
    screen_loss = screen_loss.squeeze().float()
    idx = torch.argsort(probs, descending=True)

    # mean_probs_hits = probs.clone()
    prev_pick_idx = idx[:n]
    prev_pick_idx1 = idx
    if cliff == 0:
        mean_probs_hits = probs.clone()
    else:
        if cycle >= cycle_threshold:
            mean_probs_hits = probs.clone() * (1 + screen_loss)# + beta*sigma
        else:
            mean_probs_hits = probs.clone() # + 0.1 * screen_loss

    idx = torch.argsort(mean_probs_hits, descending=True)

    picks_idx = idx[:n]
    cliff_pos = torch.nonzero(screen_loss > 0).view(-1)
    cliff_neg = torch.nonzero(screen_loss < 0).view(-1)
    if dir_name != '':
        picks_idx1 = idx
        # df = pd.DataFrame({
        #     "idx": picks_idx1.cpu().numpy(),
        #     "hit_prob": mean_probs_hits[picks_idx1].cpu().numpy(),
        #     "model_prob": probs[picks_idx1].cpu().numpy(),
        #     "sigma": screen_loss[picks_idx1].cpu().numpy(),
        #     "label": y_screen.numpy()[picks_idx1.cpu().numpy()],
        #     "smiles": smiles[picks_idx1.cpu().numpy()]
        # })
        # df.to_csv(dir_name + f"/sorted_results{cycle}.csv", index=False, float_format="%.6f")
        # df = pd.DataFrame({
        #     "idx": prev_pick_idx1.cpu().numpy(),
        #     "hit_prob": mean_probs_hits[prev_pick_idx1].cpu().numpy(),
        #     "model_prob": probs[prev_pick_idx1].cpu().numpy(),
        #     "sigma": screen_loss[prev_pick_idx1].cpu().numpy(),
        #     "label": y_screen.numpy()[prev_pick_idx1.cpu().numpy()],
        #     "smiles": smiles[prev_pick_idx1.cpu().numpy()]
        # })
        # df.to_csv(dir_name + f"/sorted_original_results{cycle}.csv", index=False, float_format="%.6f")
        # cp = 0
        # cn = 0
        # for pi in prev_pick_idx.cpu().numpy():
        #     if pi not in picks_idx.cpu().numpy():
        #         if screen_loss[pi] < 0:
        #             cn += 1
        # for pi in picks_idx.cpu().numpy():
        #     if pi in prev_pick_idx.cpu().numpy():
        #         if screen_loss[pi] > 0:
        #             cp += 1
        # y_cliff_pos = y_screen.numpy()[cliff_pos.cpu().numpy()]
        # y_cliff_neg = y_screen.numpy()[cliff_neg.cpu().numpy()]
        # df = pd.DataFrame({
        #     "cliff_pos_pos": [np.sum(y_cliff_pos)],
        #     "cliff_pos_neg": [len(y_cliff_pos) - np.sum(y_cliff_pos)],
        #     "cliff_pos_ratio": [0 if len(y_cliff_pos) == 0 else np.sum(y_cliff_pos) / len(y_cliff_pos)],
        #     "in_pos": [cp],
        #     "cliff_neg_pos": [np.sum(y_cliff_neg)],
        #     "cliff_neg_neg": [len(y_cliff_neg) - np.sum(y_cliff_neg)],
        #     "cliff_neg_ratio": [0 if len(y_cliff_neg) == 0 else np.sum(y_cliff_neg) / len(y_cliff_neg)],
        #     "not_in_neg": [cn]
        # })
        # df.to_csv(dir_name + f"/cliff_score.csv", mode='a', index=False, float_format="%.4f", header=False if os.path.isfile(dir_name + f"/cliff_score.csv") else True)
    picks_idx1 = idx
    df = pd.DataFrame({
        "idx": picks_idx1.cpu().numpy(),
        "hit_prob": mean_probs_hits[picks_idx1].cpu().numpy(),
        "model_prob": probs[picks_idx1].cpu().numpy(),
        "sigma": screen_loss[picks_idx1].cpu().numpy(),
        "label": y_screen.numpy()[picks_idx1.cpu().numpy()],
        "smiles": smiles[picks_idx1.cpu().numpy()]
    })
    df.to_csv(dir_name + f"/sorted_results{cycle}.csv", index=False, float_format="%.6f")
    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]

def evaluation_exploitation(logits_N_K_C: Tensor, smiles: np.ndarray[str], screen_loss, classification=False, output='./result/output.csv', n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    alpha = 1.5
    if classification:
        probs = F.softmax(logits_N_K_C, dim=-1)
        probs = torch.mean(probs, dim=1)[:, 1]
    else:
        probs = F.sigmoid(alpha * logits_N_K_C.squeeze())
        
    screen_loss = screen_loss.squeeze().float()
    idx = torch.argsort(probs, descending=True)

    # mean_probs_hits = probs.clone()
    mean_probs_hits = probs.clone() * (1 + screen_loss)# + beta*sigma

    idx = torch.argsort(mean_probs_hits, descending=True)
    idx = idx.squeeze()
    mean_probs_hits = mean_probs_hits.squeeze()

    picks_idx = idx[:n]

    picks_idx1 = idx
    picks_np = picks_idx1.cpu().numpy()
    scores_np = mean_probs_hits[picks_idx1].cpu().numpy()
    unlabel_df = kwargs.get("unlabel_df")
    smiles_col = kwargs.get("input_unlabel_smiles_col", "smiles")
    if unlabel_df is not None:
        if len(unlabel_df) != len(smiles):
            raise ValueError(
                f"input_unlabel length mismatch: df={len(unlabel_df)} vs smiles={len(smiles)}"
            )
        df = unlabel_df.iloc[picks_np].copy()
        df["score"] = scores_np
        df.insert(0, "ranking", np.arange(1, len(df) + 1))
    else:
        df = pd.DataFrame({
            smiles_col: smiles[picks_np],
            "score": scores_np,
        })
        df.insert(0, "ranking", np.arange(1, len(df) + 1))
    df.to_csv(output, float_format="%.6f", index=False)
    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]

def loss_prediction(logits_N_K_C: Tensor, smiles: np.ndarray[str], screen_loss, y_screen, dir_name, cycle, n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n most samples with the most variance in hit classification """

    probs = F.softmax(logits_N_K_C, dim=-1)
    mean_probs_hits = torch.mean(probs, dim=1)[:, 1]
    score = screen_loss.squeeze() * (1.0 - mean_probs_hits)
    # mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    idx = torch.argsort(score, descending=True)
    if dir_name != '':
        df = pd.DataFrame({
            "idx": idx.cpu().numpy(),
            "score": score[idx].cpu().numpy(),
            "screen_loss": screen_loss.squeeze()[idx].cpu().numpy(),
            "hit_prob": mean_probs_hits[idx].cpu().numpy(),
            "label": y_screen.numpy()[idx.cpu().numpy()]
        })
        df.to_csv(dir_name + f"/sorted_results_loss{cycle}.csv", index=False)

        draw_fig(y_screen, idx, dir_name, cycle)
    picks_idx = idx[:n]

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]

def mutual_information_boundre(logits_N_K_C: Tensor) -> Tensor:
    """ Calculates the Mutual Information - Kirch et al., 2019, NeurIPS """
    # this term represents the entropy of the model prediction (high when uncertain)
    entropy_mean_N = entropy(logits_N_K_C, dim=-1, keepdim=False)
    # This term is the expectation of the entropy of the model prediction for each draw of model parameters
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)

    I = mean_entropy_N - entropy_mean_N

    return I

def bald_boundre(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:

    I = mutual_information_boundre(torch.sigmoid(logits_N_K_C))

    picks_idx = torch.argsort(I, descending=False)[:n]

    return smiles[picks_idx.cpu()]

def std_boundre(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:

    mean_d = logits_N_K_C.mean(dim=1).squeeze()
    std_d = logits_N_K_C.std(dim=1).squeeze()

    picks_idx = torch.argsort(std_d, descending=True)[:n]

    return smiles[picks_idx.cpu()]

def std_bound_boundre(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:

    mean_d = logits_N_K_C.mean(dim=1).squeeze()
    std_d = logits_N_K_C.std(dim=1).squeeze()

    lambda_weight = 1.5
    score = -torch.abs(mean_d - 1.0) + lambda_weight * std_d

    picks_idx = torch.argsort(score, descending=True)[:n]

    return smiles[picks_idx.cpu()]

def mean_boundre(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:

    mean_d = logits_N_K_C.mean(dim=1).squeeze()

    lambda_weight = 1.5
    score = -torch.abs(mean_d - 1.0)

    picks_idx = torch.argsort(score, descending=True)[:n]

    return smiles[picks_idx.cpu()]

def greedy_(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    # print(logits_N_K_C)
    # mean_probs_hits = torch.mean(logits_N_K_C, dim=1).squeeze()
    mean_probs_hits = logits_N_K_C
    # mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    picks_idx = torch.argsort(mean_probs_hits, descending=True)[:n]

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]
 
def greedy_boundre(logits_N_K_C: Tensor, smiles: np.ndarray[str], y_screen, dir_name, cycle, n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n highest predicted samples """
    # print(logits_N_K_C)
    mean_probs_hits = torch.mean(logits_N_K_C, dim=1).squeeze()
    idx = torch.argsort(mean_probs_hits, descending=True)
    if dir_name != '':
        df = pd.DataFrame({
            "idx": idx.cpu().numpy(),
            "hit_prob": mean_probs_hits[idx].cpu().numpy(),
            "label": y_screen.numpy()[idx.cpu().numpy()]
        })
        df.to_csv(dir_name + f"/sorted_results{cycle}.csv", index=False)

        draw_fig(y_screen, idx, dir_name, cycle)
    # mean_probs_hits = logits_N_K_C
    # mean_probs_hits = torch.mean(torch.exp(logits_N_K_C), dim=1)[:, 1]
    picks_idx = idx[:n]

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]


def greedy_exploration(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, **kwargs) -> np.ndarray[str]:
    """ Get the n most samples with the most variance in hit classification """

    entropy_mean_N = mean_sample_entropy(logits_N_K_C)
    # sd_mean_N = torch.std(torch.exp(logits_N_K_C), dim=1)[:, 1]
    picks_idx = torch.argsort(entropy_mean_N, descending=True)[:n]

    return np.array([smiles[picks_idx.cpu()]]) if n == 1 else smiles[picks_idx.cpu()]



def dynamic_exploration(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, lambd: float = 0.95,
                        iteration: int = 0, **kwargs) -> np.ndarray[str]:
    """ starts with 100% exploration, approaches the limit of 100% exploitation. The speed in which we stop
    exploring depends on lambda. For example, a lambda of 0.9 will require 44 iterations to reach full exploitation,
    a lambda of 0.5 will get there in only 7 iterations """

    exploitation_factor = (1/(lambd ** iteration)) - 1
    n_exploit = round(n * exploitation_factor)
    n_explore = n - n_exploit

    exploitative_picks = greedy_exploitation(logits_N_K_C, smiles, n=n_exploit)
    explorative_picks = greedy_exploration(logits_N_K_C, smiles, n=n_explore)

    return np.concatenate((exploitative_picks, explorative_picks))


def dynamic_exploration_bald(logits_N_K_C: Tensor, smiles: np.ndarray[str], n: int = 1, lambd: float = 0.95,
                             iteration: int = 0, **kwargs) -> np.ndarray[str]:
    """ starts with 100% exploration, approaches the limit of 100% exploitation. The speed in which we stop
    exploring depends on lambda. For example, a lambda of 0.9 will require 44 iterations to reach full exploitation,
    a lambda of 0.5 will get there in only 7 iterations """

    exploitation_factor = (1/(lambd ** iteration)) - 1
    n_exploit = round(n * exploitation_factor)
    n_explore = n - n_exploit

    exploitative_picks = greedy_exploitation(logits_N_K_C, smiles, n=n_exploit)
    explorative_picks = bald(logits_N_K_C, smiles, n=n_explore)

    return np.concatenate((exploitative_picks, explorative_picks))


def bald(logits_N_K_C: Tensor, smiles: np.ndarray[str], y_screen, dir_name, n: int = 1, seed: int = 0, cycle = 0, **kwargs) -> np.ndarray[str]:
    """ Get the n molecules with the lowest Mutual Information """
    I = mutual_information(logits_N_K_C)

    idx = torch.argsort(I, descending=False)

    if dir_name != '':
        draw_fig(y_screen, idx, dir_name, cycle)

    picks_idx = idx[:n]
    return smiles[picks_idx.cpu()]


def similarity_search(hits: np.ndarray[str], smiles: np.ndarray[str], n: int = 1, radius: int = 2, nBits: int = 1024,
                      **kwargs) -> np.ndarray[str]:
    """ 1. Compute the similarity of all screen smiles to all hit smiles
        2. take the n screen smiles with the highest similarity to any hit """

    fp_hits = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in hits]
    fp_smiles = [ECFPbitVec(Chem.MolFromSmiles(smi), radius=radius, nBits=nBits) for smi in smiles]

    m = np.zeros([len(hits), len(smiles)], dtype=np.float16)
    for i in range(len(hits)):
        m[i] = BulkTanimotoSimilarity(fp_hits[i], fp_smiles)

    # get the n highest similarity smiles to any hit
    picks_idx = np.argsort(np.max(m, axis=0))[::-1][:n]
    # return smiles[picks_idx]
########################################

    sorted_idx = np.argsort(np.max(m, axis=0))[::-1]
    seen = set()
    picks = []
    for idx in sorted_idx:
        smi = smiles[idx]
        if smi not in seen:
            seen.add(smi)
            picks.append(smi)
        if len(picks) == n:
            break
#########################################
    return np.array(picks)
