import torch
import os
import numpy as np

_self_score_log_counter = 0

def normalize_vector(tensor, normalize_type="l2", dim=-1, keepdim=True):

    if normalize_type == "false" or normalize_type is False:
        return tensor
    elif normalize_type == "l1":
        norm = torch.norm(tensor, p=1, dim=dim, keepdim=keepdim)
        return torch.where((norm == 0).expand_as(tensor),
                          torch.zeros_like(tensor),
                          tensor / norm)
    elif normalize_type == "l2":
        norm = torch.norm(tensor, p=2, dim=dim, keepdim=keepdim)
        return torch.where((norm == 0).expand_as(tensor),
                          torch.zeros_like(tensor),
                          tensor / norm)


def ReduceDimRep(in_rep, dim_reduce):
    if dim_reduce is None or str(dim_reduce).lower() in ['none', 'null']:
        return in_rep
    
    try:
        dim_reduce = int(dim_reduce)
    except (ValueError, TypeError):
        return in_rep
    
    nonzero_mask = (in_rep != 0)
    nonzero_counts = nonzero_mask.sum(dim=1)
    
    out_rep = torch.zeros_like(in_rep)
    
    for i in range(in_rep.shape[0]):
        sample_rep = in_rep[i]
        nonzero_count = nonzero_counts[i].item()
        
        if nonzero_count == 0:
            continue
        elif nonzero_count <= dim_reduce:
            out_rep[i] = sample_rep
        else:
            vals, inds = torch.topk(sample_rep, dim_reduce, dim=0)
            out_rep[i].scatter_(dim=0, index=inds, src=vals)
    
    return out_rep

def SqueezeRep(in_rep, k=None):
    nonzero_cnt = torch.count_nonzero(in_rep, dim=1)
    max_nonzero_cnt = nonzero_cnt.max().item()
    vals, inds = torch.sort(in_rep, dim=1, descending=True)
    squeeze_k = max_nonzero_cnt
    if k is not None and int(max_nonzero_cnt) > k:
        squeeze_k = int(k)
    out_vals = vals[:, :squeeze_k]
    out_inds = inds[:, :squeeze_k]
    return out_vals, out_inds


def get_agg_feat(in_rep, agg="max", agg_mode="len"):
    agg_feat = None
    if agg_mode == "len":
        if agg == "sum":
            agg_feat = torch.sum(in_rep, dim=1)
        elif agg == "max":
            agg_feat, _ = torch.max(in_rep, dim=1)
        elif agg == "eos":
            agg_feat = in_rep[:, -1, :]
        elif agg == "last":
            agg_feat = in_rep[:, -2, :]
        else:
            raise NotImplementedError
    elif agg_mode == "dim":
        if agg == "avg":
            agg_feat = torch.mean(in_rep, dim=-1)
        elif agg == "max":
            agg_feat, _ = torch.max(in_rep, dim=-1)
        else:
            raise NotImplementedError

    return agg_feat


def get_sparse_score(q_rep, d_rep, in_batch_mode=True, self_normalize="false", d_self_score=False, d_raw_text_mask=None, bm25_vectors=None, self_score_mode="bm25"):
    global _self_score_log_counter
    score_dict = {}
    
    d_raw_text_mask_new = normalize_vector(d_raw_text_mask, self_normalize, dim=-1) if d_raw_text_mask is not None else None

    if in_batch_mode:
        score_dict["in_batch_score"] = torch.matmul(q_rep, d_rep.t())  
        if d_self_score:
            if self_score_mode == "d_rep":
                score_dict["d_in_batch_self_score"] = torch.matmul(d_rep, d_rep.t())
                _self_score_log_counter += 1
                
            elif self_score_mode == "bm25" and bm25_vectors is not None:
                bm25_vectors = bm25_vectors.to(d_rep.device).to(d_rep.dtype)
                score_dict["d_in_batch_self_score"] = torch.matmul(bm25_vectors, d_rep.t())
                _self_score_log_counter += 1
                
            else:
                score_dict["d_in_batch_self_score"] = torch.matmul(d_raw_text_mask_new, d_rep.t())
                _self_score_log_counter += 1

    else:
        raise NotImplementedError  # TODO
    return score_dict

class StageBasedDimController:

    def __init__(self, 
                 q_stages=[512, 256, 128],  
                 d_stages=[1024, 512, 256], 
                 convergence_ratio=0.9,     
                 patience=50):              
        
        self.q_stages = q_stages
        self.d_stages = d_stages
        self.current_stage = 0
        self.convergence_ratio = convergence_ratio
        self.patience = patience
        self.stable_steps = 0
        self.sample_dims_history = []
        
    def update(self, q_rep, d_rep):

        q_dims = torch.count_nonzero(q_rep, dim=-1)  # [batch_size]
        d_dims = torch.count_nonzero(d_rep, dim=-1)  # [batch_size]
        
        self.sample_dims_history.append({
            'q_dims': q_dims.cpu().numpy(),
            'd_dims': d_dims.cpu().numpy()
        })
        
        if len(self.sample_dims_history) > self.patience:
            self.sample_dims_history.pop(0)
            
        if len(self.sample_dims_history) >= self.patience:
            can_advance = self._check_stage_convergence()
            if can_advance and self.current_stage < len(self.q_stages) - 1:
                self.current_stage += 1
                self.sample_dims_history = []  

        
        return self.q_stages[self.current_stage], self.d_stages[self.current_stage]
    
    def _check_stage_convergence(self):

        current_q_window = self.q_stages[self.current_stage]
        current_d_window = self.d_stages[self.current_stage]
        
        all_q_dims = []
        all_d_dims = []
        for batch_dims in self.sample_dims_history:
            all_q_dims.extend(batch_dims['q_dims'])
            all_d_dims.extend(batch_dims['d_dims'])
        
        q_converged_ratio = np.mean(np.array(all_q_dims) <= current_q_window)
        d_converged_ratio = np.mean(np.array(all_d_dims) <= current_d_window)
        

        
        return (q_converged_ratio >= self.convergence_ratio and 
                d_converged_ratio >= self.convergence_ratio)
    
    def get_current_dims(self):
        return self.q_stages[self.current_stage], self.d_stages[self.current_stage]
