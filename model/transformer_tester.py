import torch
from collections import defaultdict
from tqdm.auto import tqdm
from model.tester import BaseTester
from model.utils import parse_dict, to_list
from model.factory_rep import SqueezeRep


class TransformerTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_forward(self, batch, test_mode, is_q=False):
        if test_mode in ("only_one", "only_one_bge"):
            if is_q:
                q_kwargs = parse_dict(batch, "t_")
                q_d_args = {"q_kwargs": q_kwargs}

                if self.config.get("no_q_expansion", False):
                    q_d_args["no_q_expansion"] = True
                if self.config.get("expansion_only_q", False):
                    q_d_args["expansion_only_q"] = True
                is_q_fix = self.config.get("no_q_weight", False) and self.config.get("no_q_expansion", False)
                if is_q_fix:
                    q_d_args["is_q_fix"] = True

                if "q_dim_reduce" in self.config and self.config["q_dim_reduce"] is not None:
                    if not is_q_fix:
                        q_d_args["q_dim_reduce"] = self.config["q_dim_reduce"]
                if self.config.get("q_add_comp_logit", False):
                    q_d_args["q_add_comp_logit"] = True
                if self.config.get("q_out_hidden", False):
                    q_d_args["q_out_hidden"] = True
                if self.config.get("q_normalize_score", False):
                    q_d_args["q_normalize_score"] = self.config.get("q_normalize_score")
            else:
                d_kwargs = parse_dict(batch, "t_")
                q_d_args = {"d_kwargs": d_kwargs}

                if self.config.get("no_d_expansion", False):
                    q_d_args["no_d_expansion"] = True
                if self.config.get("expansion_only_d", False):
                    q_d_args["expansion_only_d"] = True
                if "d_dim_reduce" in self.config and self.config["d_dim_reduce"] is not None:
                    q_d_args["d_dim_reduce"] = self.config["d_dim_reduce"]

                if self.config.get("d_add_comp_logit", False):
                    q_d_args["d_add_comp_logit"] = True
                if self.config.get("d_out_hidden", False):
                    q_d_args["d_out_hidden"] = True
                if self.config.get("d_self_score", False):
                    q_d_args["d_self_score"] = True
                if self.config.get("d_normalize_score", False):
                    q_d_args["d_normalize_score"] = self.config.get("d_normalize_score")
                if self.config.get("self_normalize", False):
                    q_d_args["self_normalize"] = self.config.get("self_normalize")
        elif test_mode == "qw_pairs":
            raise NotImplementedError  # TODO
        else:
            raise NotImplementedError  # TODO

        out_q_d = self.model(**q_d_args)
        out = {}
        for k, v in out_q_d.items():
            out[k] = v
        return out

    def test_iterations(self):
        self.model.eval()
        with torch.no_grad():
            test_res = self.evaluate_test(data_loader=self.test_loader)
        try:
            torch.distributed.barrier()
        except:
            print('Dist.barrier Error Occurs, But Ignored')
            pass

    def evaluate_test(self, data_loader):
        out_d = defaultdict(float)
        i = 0
        for batch in tqdm(data_loader):
            i += 1
            inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"q_id", "d_id", "label", "t_id"}}
            test_mode = self.config.get("test_mode", None)
            test_q = self.config.get("test_q", False)
            test_d = self.config.get("test_d", False)
            assert not (test_q and test_d)
            is_q = test_q
            out = self.test_forward(inputs, test_mode=test_mode, is_q=is_q)
            if self.table_writer is not None:
                save_topk = self.config.get("save_topk", False)
                if test_mode in ("only_one", "only_one_bge"):
                    if is_q:
                        q_input_ids = out["q_input_ids"]
                        q_rep_raw = out["q_rep_raw"]
                        q_raw_text_mask = out["q_raw_text_mask"]
                        q_rep = out["q_rep"]
                        q_p = out["q_p"] if "q_p" in out else None

                        q_rep_raw_val, q_rep_raw_ind = SqueezeRep(q_rep_raw, save_topk) if save_topk else SqueezeRep(q_rep_raw)
                        q_raw_text_val, q_raw_text_ind = SqueezeRep(q_raw_text_mask)
                        q_rep_val, q_rep_ind = SqueezeRep(q_rep, save_topk) if save_topk else SqueezeRep(q_rep)
                        q_p_val, q_p_ind = None, None
                        if q_p is not None:
                            q_p_val, q_p_ind = SqueezeRep(q_p, save_topk) if save_topk else SqueezeRep(q_p)

                        q_input_ids = q_input_ids.tolist()
                        q_rep_raw_val = q_rep_raw_val.tolist()
                        q_rep_raw_ind = q_rep_raw_ind.tolist()
                        q_raw_text_val = q_raw_text_val.tolist()
                        q_raw_text_ind = q_raw_text_ind.tolist()
                        q_rep_val = q_rep_val.tolist()
                        q_rep_ind = q_rep_ind.tolist()
                        q_p_val = q_p_val.tolist() if q_p_val is not None else None
                        q_p_ind = q_p_ind.tolist() if q_p_ind is not None else None

                        q_rep_raw_l0 = torch.count_nonzero(q_rep_raw, dim=-1).float().tolist()
                        q_raw_text_l0 = torch.count_nonzero(q_raw_text_mask, dim=-1).float().tolist()
                        q_rep_l0 = torch.count_nonzero(q_rep, dim=-1).float().tolist()
                        q_p_l0 = torch.count_nonzero(q_p, dim=-1).float().tolist() if q_p is not None else None

                        q_id = to_list(batch["t_id"])

                        q_id = [str(i) for i in q_id]
                        q_input_ids = [str(i) for i in q_input_ids]
                        q_rep_raw_val = [str(i) for i in q_rep_raw_val]
                        q_rep_raw_ind = [str(i) for i in q_rep_raw_ind]
                        q_raw_text_val = [str(i) for i in q_raw_text_val]
                        q_raw_text_ind = [str(i) for i in q_raw_text_ind]
                        q_rep_val = [str(i) for i in q_rep_val]
                        q_rep_ind = [str(i) for i in q_rep_ind]
                        q_p_val = [str(i) for i in q_p_val] if q_p_val is not None else [0] * len(q_id)
                        q_p_ind = [str(i) for i in q_p_ind] if q_p_ind is not None else [0] * len(q_id)
                        q_rep_raw_l0 = [str(i) for i in q_rep_raw_l0]
                        q_raw_text_l0 = [str(i) for i in q_raw_text_l0]
                        q_rep_l0 = [str(i) for i in q_rep_l0]
                        q_p_l0 = [str(i) for i in q_p_l0] if q_p_l0 is not None else [0] * len(q_id)

                        d_id = [0] * len(q_id)
                        d_input_ids = [0] * len(q_id)
                        d_rep_raw_val = [0] * len(q_id)
                        d_rep_raw_ind = [0] * len(q_id)
                        d_raw_text_val = [0] * len(q_id)
                        d_raw_text_ind = [0] * len(q_id)
                        d_rep_val = [0] * len(q_id)
                        d_rep_ind = [0] * len(q_id)
                        d_p_val = [0] * len(q_id)
                        d_p_ind = [0] * len(q_id)
                        d_rep_raw_l0 = [0] * len(q_id)
                        d_raw_text_l0 = [0] * len(q_id)
                        d_rep_l0 = [0] * len(q_id)
                        d_p_l0 = [0] * len(q_id)
                    else:
                        d_input_ids = out["d_input_ids"]
                        d_rep_raw = out["d_rep_raw"]
                        d_raw_text_mask = out["d_raw_text_mask"]
                        d_rep = out["d_rep"]
                        d_p = out["d_p"] if "d_p" in out else None

                        d_rep_raw_val, d_rep_raw_ind = SqueezeRep(d_rep_raw, save_topk) if save_topk else SqueezeRep(d_rep_raw)
                        d_raw_text_val, d_raw_text_ind = SqueezeRep(d_raw_text_mask)
                        d_rep_val, d_rep_ind = SqueezeRep(d_rep, save_topk) if save_topk else SqueezeRep(d_rep)
                        d_p_val, d_p_ind = None, None
                        if d_p is not None:
                            d_p_val, d_p_ind = SqueezeRep(d_p, save_topk) if save_topk else SqueezeRep(d_p)

                        d_input_ids = d_input_ids.tolist()
                        d_rep_raw_val = d_rep_raw_val.tolist()
                        d_rep_raw_ind = d_rep_raw_ind.tolist()
                        d_raw_text_val = d_raw_text_val.tolist()
                        d_raw_text_ind = d_raw_text_ind.tolist()
                        d_rep_val = d_rep_val.tolist()
                        d_rep_ind = d_rep_ind.tolist()
                        d_p_val = d_p_val.tolist() if d_p_val is not None else None
                        d_p_ind = d_p_ind.tolist() if d_p_ind is not None else None

                        d_rep_raw_l0 = torch.count_nonzero(d_rep_raw, dim=-1).float().tolist()
                        d_raw_text_l0 = torch.count_nonzero(d_raw_text_mask, dim=-1).float().tolist()
                        d_rep_l0 = torch.count_nonzero(d_rep, dim=-1).float().tolist()
                        d_p_l0 = torch.count_nonzero(d_p, dim=-1).float().tolist() if d_p is not None else None

                        d_id = to_list(batch["t_id"])

                        d_id = [str(i) for i in d_id]
                        d_input_ids = [str(i) for i in d_input_ids]
                        d_rep_raw_val = [str(i) for i in d_rep_raw_val]
                        d_rep_raw_ind = [str(i) for i in d_rep_raw_ind]
                        d_raw_text_val = [str(i) for i in d_raw_text_val]
                        d_raw_text_ind = [str(i) for i in d_raw_text_ind]
                        d_rep_val = [str(i) for i in d_rep_val]
                        d_rep_ind = [str(i) for i in d_rep_ind]
                        d_p_val = [str(i) for i in d_p_val] if d_p_val is not None else [0] * len(d_id)
                        d_p_ind = [str(i) for i in d_p_ind] if d_p_ind is not None else [0] * len(d_id)
                        d_rep_raw_l0 = [str(i) for i in d_rep_raw_l0]
                        d_raw_text_l0 = [str(i) for i in d_raw_text_l0]
                        d_rep_l0 = [str(i) for i in d_rep_l0]
                        d_p_l0 = [str(i) for i in d_p_l0] if d_p_l0 is not None else [0] * len(d_id)

                        q_id = [0] * len(d_id)
                        q_input_ids = [0] * len(d_id)
                        q_rep_raw_val = [0] * len(d_id)
                        q_rep_raw_ind = [0] * len(d_id)
                        q_raw_text_val = [0] * len(d_id)
                        q_raw_text_ind = [0] * len(d_id)
                        q_rep_val = [0] * len(d_id)
                        q_rep_ind = [0] * len(d_id)
                        q_p_val = [0] * len(d_id)
                        q_p_ind = [0] * len(d_id)
                        q_rep_raw_l0 = [0] * len(d_id)
                        q_raw_text_l0 = [0] * len(d_id)
                        q_rep_l0 = [0] * len(d_id)
                        q_p_l0 = [0] * len(d_id)
                elif test_mode == "qw_pairs":
                    raise NotImplementedError  # TODO
                else:
                    raise NotImplementedError  # TODO

                tuples = list(
                    zip(q_id, q_input_ids,
                        q_rep_raw_val, q_rep_raw_ind, q_raw_text_val, q_raw_text_ind, q_rep_val, q_rep_ind, q_p_val, q_p_ind,
                        q_rep_raw_l0, q_raw_text_l0, q_rep_l0, q_p_l0,
                        d_id, d_input_ids,
                        d_rep_raw_val, d_rep_raw_ind, d_raw_text_val, d_raw_text_ind, d_rep_val, d_rep_ind, d_p_val, d_p_ind,
                        d_rep_raw_l0, d_raw_text_l0, d_rep_l0, d_p_l0))
                print('write begin!')
                self.table_writer.write(tuples)
                print('write end!')

                
        return out_d
