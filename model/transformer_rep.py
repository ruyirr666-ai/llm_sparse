from abc import ABC
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from model.factory_rep import ReduceDimRep, get_agg_feat, get_sparse_score
from math import inf
from model.tool.loader import download_model_from_hub
import os


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



class TransformerRep(torch.nn.Module):

    def __init__(self, model_type_or_dir, out_hidden=False):

        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir, output_hidden_states=out_hidden)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        print('self.transformer: ', self.transformer)

    def forward(self, **tokens):
        out = self.transformer(**tokens)
        return out


class QwenTransformerRep(torch.nn.Module):

    def __init__(self, model_type_or_dir, torch_dtype, out_hidden=False, use_bidirectional_attention=False, use_dual_pass_concat=False):

        super().__init__()
        self.use_dual_pass_concat = use_dual_pass_concat
        
        self.transformer = AutoModelForCausalLM.from_pretrained(model_type_or_dir,
                                                                torch_dtype=torch_dtype,
                                                                trust_remote_code=True,
                                                                output_hidden_states=out_hidden)
        

        if use_bidirectional_attention and hasattr(self.transformer.config, 'is_decoder'):
            self.transformer.config.is_decoder = False
            print(f"is_decoder = False")
        elif use_bidirectional_attention:
            print(f"model not support is_decoder configuration, bidirectional attention may not be effective")
        else:
            print(f"use default causal attention")
        
        if use_dual_pass_concat:
            print(f"use dual pass concat mode")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir,
                                                       trust_remote_code=True,
                                                       pad_token='<|endoftext|>',
                                                       padding_side="left")
        self.tokenizer.padding_side = 'left'
        print('self.transformer: ', self.transformer)

    def forward(self, **tokens):
        if self.use_dual_pass_concat:
            input_ids = tokens["input_ids"]  # [batch_size, seq_len]
            attention_mask = tokens["attention_mask"]  # [batch_size, seq_len]
            
            dual_input_ids, dual_attention_mask = self._create_dual_sequence(input_ids, attention_mask)
            
            dual_tokens = {
                "input_ids": dual_input_ids,
                "attention_mask": dual_attention_mask
            }
            
            out = self.transformer(**dual_tokens)
            
            out = self._extract_second_sequence_output(out, input_ids, attention_mask)
            
        else:
            out = self.transformer(**tokens)
        return out
    
    def _create_dual_sequence(self, input_ids, attention_mask):
        batch_size, original_seq_len = input_ids.shape
        device = input_ids.device
        
        new_input_ids = []
        new_attention_mask = []
        
        for i in range(batch_size):
            valid_mask = attention_mask[i] == 1
            valid_tokens = input_ids[i][valid_mask]  
            
            dual_tokens = torch.cat([valid_tokens, valid_tokens])
            
            if len(dual_tokens) > original_seq_len:
                dual_tokens = dual_tokens[:original_seq_len]
            
            pad_length = original_seq_len - len(dual_tokens)
            padded_tokens = torch.cat([
                torch.full((pad_length,), self.tokenizer.pad_token_id, device=device, dtype=input_ids.dtype),
                dual_tokens
            ])
            
            padded_mask = torch.cat([
                torch.zeros(pad_length, device=device, dtype=attention_mask.dtype),
                torch.ones(len(dual_tokens), device=device, dtype=attention_mask.dtype)
            ])
            
            new_input_ids.append(padded_tokens)
            new_attention_mask.append(padded_mask)
        
        return torch.stack(new_input_ids), torch.stack(new_attention_mask)
    
    def _extract_second_sequence_output(self, model_output, original_input_ids, original_attention_mask):
        logits = model_output["logits"]  # [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        second_logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=logits.dtype, device=device)
        
        second_hidden_states = None
        if "hidden_states" in model_output and model_output["hidden_states"] is not None:
            hidden_states = model_output["hidden_states"]  # tuple of [batch_size, seq_len, hidden_size]
            second_hidden_states = []
            
        for i in range(batch_size):
            valid_positions = torch.where(original_attention_mask[i] == 1)[0]
            if len(valid_positions) == 0:
                continue
                
            total_valid_length = len(valid_positions)
            
            second_seq_start_offset = total_valid_length // 2
            
            current_valid_start = torch.where(torch.diff(torch.cat([torch.tensor([0], device=device), 
                                                                   (original_attention_mask[i] == 1).float()])) == 1)[0]
            if len(current_valid_start) > 0:
                current_valid_start = current_valid_start[0]
            else:
                current_valid_start = 0
                
            second_seq_abs_start = current_valid_start + second_seq_start_offset
            second_seq_length = total_valid_length - second_seq_start_offset
            
            if second_seq_abs_start + second_seq_length <= seq_len:
                source_start = second_seq_abs_start
                source_end = second_seq_abs_start + second_seq_length
                target_start = current_valid_start
                target_end = current_valid_start + second_seq_length
                
                second_logits[i, target_start:target_end] = logits[i, source_start:source_end]
                
                if second_hidden_states is not None:
                    for layer_idx, layer_hidden in enumerate(hidden_states):
                        if i == 0 and layer_idx == 0:  
                            second_hidden_states = [torch.zeros_like(layer_hidden) for _ in range(len(hidden_states))]
                        second_hidden_states[layer_idx][i, target_start:target_end] = layer_hidden[i, source_start:source_end]

        new_output = {
            "logits": second_logits
        }
        
        if second_hidden_states is not None:
            new_output["hidden_states"] = tuple(second_hidden_states)
            
        for key, value in model_output.items():
            if key not in ["logits", "hidden_states"]:
                new_output[key] = value
                
        return new_output


class SiameseBase(torch.nn.Module, ABC):
    """
    SiameseBase
    """

    def __init__(self, model_type_or_dir, model_type_or_dir_q=None, freeze_d_model=False, agg="last",
                 model_type="qw", torch_dtype=None,
                 special_token_list=None,
                 out_hidden=False, hidden_process=False, hidden_agg="max", use_bidirectional_attention=False, use_dual_pass_concat=False):
        super().__init__()
        self.freeze_d_model = freeze_d_model
        self.agg = agg
        self.model_type = model_type  
        
        self._use_mixed_precision = False

        if isinstance(torch_dtype, str):
            if torch_dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif torch_dtype == "float16":
                torch_dtype = torch.float16
            elif torch_dtype == "float32":
                torch_dtype = torch.float32
            elif torch_dtype == "auto":
                torch_dtype = "auto"
            else:
                print(f"Warning: Unknown torch_dtype string '{torch_dtype}', using default")
                torch_dtype = None

        if model_type_or_dir is not None:
            model_type_or_dir = download_model_from_hub(model_type_or_dir)
        if model_type_or_dir_q is not None:
            model_type_or_dir_q = download_model_from_hub(model_type_or_dir_q)

        if model_type == "bert":
            self.transformer_rep = TransformerRep(model_type_or_dir, out_hidden)
            self.transformer_rep_q = TransformerRep(model_type_or_dir_q, out_hidden) if model_type_or_dir_q is not None else None
            self.tokenizer = self.transformer_rep.tokenizer if self.transformer_rep is not None else None
            self.special_token_list = special_token_list
        elif model_type in ["qw", "qwen"]:
            self.transformer_rep = QwenTransformerRep(model_type_or_dir, torch_dtype, out_hidden, use_bidirectional_attention, use_dual_pass_concat)
            self.transformer_rep_q = QwenTransformerRep(model_type_or_dir_q, torch_dtype,
                                                        out_hidden, use_bidirectional_attention, use_dual_pass_concat) if model_type_or_dir_q is not None else None
            self.tokenizer = self.transformer_rep.tokenizer if self.transformer_rep is not None else None
            self.special_token_list = [self.tokenizer.pad_token_id]
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Supported types: 'bert', 'qw', 'qwen'")

        assert not (freeze_d_model and model_type_or_dir_q is None)
        if freeze_d_model:
            self.transformer_rep.requires_grad_(False)
        self.output_dim = self.transformer_rep.transformer.config.vocab_size

        self.out_hidden = out_hidden
        self.hidden_process = hidden_process
        self.hidden_agg = hidden_agg

        self.hidden_size = self.transformer_rep.transformer.config.hidden_size  # 2048
        self.hidden2vocab = self.Linear(self.hidden_size, self.output_dim) if self.out_hidden else None
        
        self._training_step = 0
        
        self.dim_controller = None

    def init_dynamic_dim_controller(self, config):
        from .factory_rep import StageBasedDimController
        
        if config.get("dynamic_dim_reduce", False):
            q_stages = config.get("q_dim_stages", [512, 256, 128])
            d_stages = config.get("d_dim_stages", [1024, 512, 256])
            convergence_ratio = config.get("convergence_ratio", 0.9)
            patience = config.get("patience", 50)
            
            self.dim_controller = StageBasedDimController(
                q_stages=q_stages,
                d_stages=d_stages,
                convergence_ratio=convergence_ratio,
                patience=patience
            )

    def pad_special_token(self, vector, pad_val=0, pad_mode='len', input_ids=None):
        for token in self.special_token_list:
            if pad_mode == 'len':
                mask = input_ids == token
                vector[mask] = pad_val
            elif pad_mode == 'dim':
                vector[..., token] = pad_val
        return vector

    def Linear(self, in_rep, out_rep, bias=True):
        m = torch.nn.Linear(in_rep, out_rep, bias)
        torch.nn.init.xavier_uniform_(m.weight)
        
        use_bfloat16 = (self.model_type in ["qw", "qwen"] and 
                       not getattr(self, '_use_mixed_precision', False))
        
        if use_bfloat16:
            m.weight.data = m.weight.data.bfloat16()
            if bias:
                torch.nn.init.constant_(m.bias, 0.0)
                m.bias.data = m.bias.data.bfloat16()
        else:
            if bias:
                torch.nn.init.constant_(m.bias, 0.0)
        
        return m

    def encode_(self, tokens, is_q=False):
        transformer = self.transformer_rep
        if is_q and self.transformer_rep_q is not None:
            transformer = self.transformer_rep_q
        return transformer(**tokens)

    def encode(self, tokens, is_q, out_hidden=False):
        # past_key_values: 24 X 2 X [4, 30, 16, 128]
        model_output = self.encode_(tokens, is_q)
        
        logits = model_output["logits"]  # [4, 30, voc_size]
            
        input_ids = tokens["input_ids"]  # [4, 30]
        input_mask = self.pad_special_token(torch.ones_like(input_ids), pad_val=0, pad_mode='len', input_ids=input_ids)
        tmp = torch.log(1 + torch.relu(logits)) * input_mask.unsqueeze(-1)
        agg_feat = get_agg_feat(tmp, agg=self.agg, agg_mode="len")
        out = {}
        out["agg_feat"] = agg_feat
        out["input_mask"] = input_mask

        if out_hidden:
            hidden_states = model_output["hidden_states"]  # 25 X 1 X [4, 30, 2048]
            hidden_state = None
            if self.hidden_process:
                hidden_state = torch.log(1 + torch.relu(hidden_states[-1]))
            else:
                hidden_state = hidden_states[-1]  # [4, 30, 2048]
            out["hidden_state"] = hidden_state
            tmp = hidden_state * input_mask.unsqueeze(-1)
            hidden_agg_feat = get_agg_feat(tmp, agg=self.hidden_agg, agg_mode="len")
            out["hidden_agg_feat"] = hidden_agg_feat
        return out

    def train(self, mode=True):
        if self.transformer_rep_q is None:
            self.transformer_rep.train(mode)
        else:
            self.transformer_rep_q.train(mode)
            mode_d = False if not mode else not self.freeze_d_model
            self.transformer_rep.train(mode_d)
        if self.hidden2vocab is not None:
            self.hidden2vocab.train(mode)

    def forward(self, **kwargs):
        if self.training:
            self._training_step += 1
        
        should_debug = (self._training_step <= 5 or self._training_step % 200 == 0) if self.training else False
        
        out = {}
        do_q, do_d = "q_kwargs" in kwargs, "d_kwargs" in kwargs
        no_q_expansion, no_d_expansion = "no_q_expansion" in kwargs, "no_d_expansion" in kwargs
        expansion_only_q, expansion_only_d = "expansion_only_q" in kwargs, "expansion_only_d" in kwargs
        is_q_fix = "is_q_fix" in kwargs

        q_dim_reduce = kwargs["q_dim_reduce"] if "q_dim_reduce" in kwargs else None
        d_dim_reduce = kwargs["d_dim_reduce"] if "d_dim_reduce" in kwargs else None

        d_add_comp_logit = "d_add_comp_logit" in kwargs
        d_out_hidden = "d_out_hidden" in kwargs
        q_add_comp_logit = "q_add_comp_logit" in kwargs  
        q_out_hidden = "q_out_hidden" in kwargs

        in_batch_mode = "in_batch_mode" in kwargs
        d_self_score = "d_self_score" in kwargs
        
        q_normalize_score = kwargs.get("q_normalize_score", "false")
        d_normalize_score = kwargs.get("d_normalize_score", "false") 
        self_normalize = kwargs.get("self_normalize", "false")
        
        bm25_vectors = kwargs.get("bm25_vectors", None)
        self_score_mode = kwargs.get("self_score_mode", "bm25")

        q_rep_raw = None
        d_rep_raw = None
        q_rep = None
        d_rep = None
        q_p = None
        d_p = None
        q_raw_text_mask = None
        d_raw_text_mask = None
        if do_q:
            q_input_ids = kwargs["q_kwargs"]["input_ids"]
            out.update({"q_input_ids": q_input_ids})
            q_raw_text_mask = self.pad_special_token(
                torch.zeros(q_input_ids.shape[0], self.output_dim).float().to(torch.device("cuda")).scatter(dim=-1,
                                                                                                            index=q_input_ids,
                                                                                                            src=torch.ones_like(
                                                                                                                q_input_ids).float()),
                pad_val=0, pad_mode='dim')
            
            q_encode_out = None
            q_hidden_agg_feat = None
            q_hidden_state = None
            if not is_q_fix:
                if q_out_hidden:
                    q_encode_out = self.encode(kwargs["q_kwargs"], is_q=True, out_hidden=True)
                    q_hidden_agg_feat = q_encode_out["hidden_agg_feat"]
                    q_hidden_state = q_encode_out["hidden_state"]
                else:
                    q_encode_out = self.encode(kwargs["q_kwargs"], is_q=True)
                q_agg_feat = q_encode_out["agg_feat"]
                q_rep_raw = q_agg_feat
            else:
                q_rep_raw = q_raw_text_mask.to(torch.device("cuda"))
            out.update({"q_rep_raw": q_rep_raw})
            out.update({"q_raw_text_mask": q_raw_text_mask})
            

            
            #q_rep_sig = torch.sigmoid(q_rep_raw)
            q_rep_sig = q_rep_raw


            
            q_p = None
            if not is_q_fix and q_out_hidden:
                if self.hidden_agg == "max":
                    q_hidden_w = q_hidden_agg_feat
                    # q_input_mask = q_encode_out["input_mask"]
                    # q_hidden_w = torch.matmul(q_hidden_state.transpose(-1, -2), F.softmax(
                    #     q_hidden_agg_feat.unsqueeze(-1).masked_fill(q_input_mask.unsqueeze(-1) == 0, -inf), dim=1)).squeeze(-1)
                elif self.hidden_agg == "last":
                    q_hidden_w = q_hidden_agg_feat
                else:
                    raise ValueError(f"Unsupported hidden_agg mode: {self.hidden_agg}")
                
                # q_p = torch.sigmoid(self.hidden2vocab(q_hidden_w))
                q_hidden_w = q_hidden_w.to(self.hidden2vocab.weight.dtype)
                q_p = self.hidden2vocab(q_hidden_w)
                
 

            q_rep_tmp_1 = None
            if not is_q_fix and q_add_comp_logit and q_out_hidden:
      
                q_rep_max = torch.max(q_rep_sig, dim=-1, keepdim=True)[0] 
                q_rep_tmp_1 = q_rep_sig + (q_rep_max - q_p) * q_raw_text_mask
                #q_rep_tmp_1 = q_rep_sig + q_p * q_raw_text_mask
                out.update({"q_p": q_p})
                
                
                
    
            else:
                if not is_q_fix:
                    q_rep_tmp_1 = q_rep_sig
                    
                    
                    
                    
                    
                else:
                    q_rep_tmp_1 = q_rep_raw

            q_rep_tmp_2 = None
            if not is_q_fix and no_q_expansion:
                q_rep_tmp_2 = q_rep_tmp_1 * q_raw_text_mask

            elif not is_q_fix and expansion_only_q:
                q_rep_tmp_2 = q_rep_tmp_1 * (1 - q_raw_text_mask)

            else:
                q_rep_tmp_2 = q_rep_tmp_1

            q_rep_tmp_3 = None
            current_q_dim_reduce = q_dim_reduce  
            
            if self.dim_controller is not None and do_q and do_d:
                q_rep_tmp_3 = q_rep_tmp_2  
            elif q_dim_reduce is not None:
                
                if should_debug:
                    q_tmp2_nonzero = torch.count_nonzero(q_rep_tmp_2, dim=-1).float()
                    print(f"DEBUG fixed dimension reduction input distribution: min={q_tmp2_nonzero.min():.1f}, max={q_tmp2_nonzero.max():.1f}, mean={q_tmp2_nonzero.mean():.1f}")
                
                q_rep_tmp_3 = ReduceDimRep(q_rep_tmp_2, q_dim_reduce)
                
                if should_debug:
                    q_tmp3_nonzero = torch.count_nonzero(q_rep_tmp_3, dim=-1).float()
                    print(f"DEBUG fixed dimension reduction output distribution: min={q_tmp3_nonzero.min():.1f}, max={q_tmp3_nonzero.max():.1f}, mean={q_tmp3_nonzero.mean():.1f}")
                    print(f"DEBUG q_rep_tmp_3 (fixed reduction): shape={q_rep_tmp_3.shape}, non-zero={torch.count_nonzero(q_rep_tmp_3, dim=-1).float().mean():.1f}")
            else:
                q_rep_tmp_3 = q_rep_tmp_2

            q_rep = normalize_vector(q_rep_tmp_3, q_normalize_score, dim=-1)
            if should_debug:
                print(f"DEBUG q_rep (standardization mode: {q_normalize_score}): non-zero={torch.count_nonzero(q_rep, dim=-1).float().mean():.1f}")

            out.update({"q_rep": q_rep})

        if do_d:
            d_input_ids = kwargs["d_kwargs"]["input_ids"]
            out.update({"d_input_ids": d_input_ids})
            d_encode_out = None
            d_hidden_agg_feat = None
            d_hidden_state = None
            if d_out_hidden:
                d_encode_out = self.encode(kwargs["d_kwargs"], is_q=False, out_hidden=True)
                d_hidden_agg_feat = d_encode_out["hidden_agg_feat"]
                d_hidden_state = d_encode_out["hidden_state"]
            else:
                d_encode_out = self.encode(kwargs["d_kwargs"], is_q=False)
            d_input_mask = d_encode_out["input_mask"]
            d_rep_raw = d_encode_out["agg_feat"]
            out.update({"d_rep_raw": d_rep_raw})

            d_raw_text_mask = self.pad_special_token(
                torch.zeros_like(d_rep_raw).float().scatter(dim=-1, index=d_input_ids,
                                                            src=torch.ones_like(d_input_ids).float()), pad_val=0,
                pad_mode='dim').to(d_rep_raw.dtype)
            out.update({"d_raw_text_mask": d_raw_text_mask})
            
            if should_debug:
                print(f"DEBUG d_rep_raw: shape={d_rep_raw.shape}, non-zero={torch.count_nonzero(d_rep_raw, dim=-1).float().mean():.1f}")
            
            d_rep_sig = d_rep_raw
            
            if should_debug:
                print(f"DEBUG d_rep_sig: shape={d_rep_sig.shape}, non-zero={torch.count_nonzero(d_rep_sig, dim=-1).float().mean():.1f}")
            
            p = None
            if d_out_hidden:
                if self.hidden_agg == "max":
                    d_hidden_w = d_hidden_agg_feat
                    # d_hidden_w = torch.matmul(d_hidden_state.transpose(-1, -2), F.softmax(
                    #     d_hidden_agg_feat.unsqueeze(-1).masked_fill(d_input_mask.unsqueeze(-1) == 0, -inf), dim=1)).squeeze(-1)
                elif self.hidden_agg == "last":
                    
                    d_hidden_w = d_hidden_agg_feat
                else:
                    raise ValueError(f"Unsupported hidden_agg mode: {self.hidden_agg}")
                
                p = torch.sigmoid(self.hidden2vocab(d_hidden_w))
                d_hidden_w = d_hidden_w.to(self.hidden2vocab.weight.dtype)
                p = self.hidden2vocab(d_hidden_w)
                
                if should_debug:
                    print(f"DEBUG d_p: shape={p.shape}, non-zero={torch.count_nonzero(p, dim=-1).float().mean():.1f}")

            d_rep_tmp_1 = None
            if d_add_comp_logit and d_out_hidden:
                
                d_rep_max = torch.max(d_rep_sig, dim=-1, keepdim=True)[0]  
                d_rep_tmp_1 = d_rep_sig + (d_rep_max - p) * d_raw_text_mask
                out.update({"d_p": p})
                
                if should_debug:
                    print(f"DEBUG d_rep_tmp_1 (after sparsity recovery): shape={d_rep_tmp_1.shape}, non-zero={torch.count_nonzero(d_rep_tmp_1, dim=-1).float().mean():.1f}")
            else:
                d_rep_tmp_1 = d_rep_sig
                
                if should_debug:
                    print(f"DEBUG d_rep_tmp_1 (fallback, after sparsity recovery): shape={d_rep_tmp_1.shape}, non-zero={torch.count_nonzero(d_rep_tmp_1, dim=-1).float().mean():.1f}")

            d_rep_tmp_2 = None
            if no_d_expansion:
            
                d_rep_tmp_2 = d_rep_tmp_1 * d_raw_text_mask

            elif expansion_only_d:
                
                d_rep_tmp_2 = d_rep_tmp_1 * (1 - d_raw_text_mask)

            else:
                
                d_rep_tmp_2 = d_rep_tmp_1



            d_rep_reduced = None
            if d_dim_reduce is not None and self.dim_controller is None:

                d_rep_reduced = ReduceDimRep(d_rep_tmp_2, d_dim_reduce)
                
                
            else:
                d_rep_reduced = d_rep_tmp_2
                
            d_rep = normalize_vector(d_rep_reduced, d_normalize_score, dim=-1)


            out.update({"d_rep": d_rep})

        if self.dim_controller is not None and do_q and do_d:
            current_q_dim, current_d_dim = self.dim_controller.update(q_rep_tmp_2, d_rep_tmp_2)
            
            if current_q_dim < q_rep_tmp_2.shape[-1]:
                q_rep_tmp_3 = ReduceDimRep(q_rep_tmp_2, current_q_dim)

            else:
                q_rep_tmp_3 = q_rep_tmp_2

            

            if current_d_dim < d_rep_tmp_2.shape[-1]:
                d_rep_reduced = ReduceDimRep(d_rep_tmp_2, current_d_dim)

            else:
                d_rep_reduced = d_rep_tmp_2

            

            q_rep = normalize_vector(q_rep_tmp_3, q_normalize_score, dim=-1)
            d_rep = normalize_vector(d_rep_reduced, d_normalize_score, dim=-1)
            

            out.update({"q_rep": q_rep, "d_rep": d_rep})

        if do_d and do_q:
            q_rep_ = q_rep.to(dtype=d_rep.dtype)
            if not in_batch_mode:
                out.update({"label": kwargs["label"]})
            

            hard_neg_rep = None
            if "hard_neg_1_kwargs" in kwargs:

                hard_neg_encode_out = self.encode(kwargs["hard_neg_1_kwargs"], is_q=False, out_hidden=d_out_hidden)
                hard_neg_rep_raw = hard_neg_encode_out["agg_feat"]
                

                hard_neg_rep_sig = hard_neg_rep_raw
                hard_neg_rep_tmp_1 = hard_neg_rep_sig
                

                if d_dim_reduce is not None:
                    hard_neg_rep_reduced = ReduceDimRep(hard_neg_rep_tmp_1, d_dim_reduce)
                else:
                    hard_neg_rep_reduced = hard_neg_rep_tmp_1
                

                hard_neg_rep = normalize_vector(hard_neg_rep_reduced, d_normalize_score, dim=-1)
                

            
            score_dict = get_sparse_score(q_rep_, d_rep, in_batch_mode=in_batch_mode, self_normalize=self_normalize,
                                          d_self_score=d_self_score,
                                          d_raw_text_mask=d_raw_text_mask,
                                          bm25_vectors=bm25_vectors,
                                          self_score_mode=self_score_mode)
            

            if hard_neg_rep is not None:
                hard_neg_score = torch.sum(q_rep_ * hard_neg_rep, dim=-1)  # [batch_size]
                score_dict["hard_neg_scores"] = hard_neg_score.unsqueeze(1)  # [batch_size, 1]

            out.update({"score_dict": score_dict})

        return out
