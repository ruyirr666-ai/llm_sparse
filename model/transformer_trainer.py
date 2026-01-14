import os
import torch
from tqdm.auto import tqdm
from model.tool import amp
from model.tool.utils import custom_save
from model.trainer import TrainerIter
from model.utils import parse_dict
from model.utils import makedir


class TransformerTrainer(TrainerIter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitoring_ckpt = self.config.get("monitoring_ckpt", None)
        to_write = 'iter,epoch,all_loss,loss,self_loss,ui_loss,reg_loss,lr,l0_q_raw,l0_d_raw,l0_q,l0_d,l0_q_raw_text,l0_d_raw_text'
        if self.monitoring_ckpt is not None:
            raise NotImplementedError  # TODO
        to_write += '\n'

        self.train_res_handler = None
        if int(os.environ['RANK']) == 0:
            self.train_res_handler = open(os.path.join(self.checkpoint_dir, "train_perf.txt"), "w")
            self.train_res_handler.write(to_write)
            self.train_res_handler.close()

        if hasattr(self.model, 'module'):
            self.model.module.init_dynamic_dim_controller(self.config)
        else:
            self.model.init_dynamic_dim_controller(self.config)



    def forward(self, batch):
        q_kwargs = parse_dict(batch, "q_")
        d_kwargs = parse_dict(batch, "d_")
        q_d_args = {"q_kwargs": q_kwargs, "d_kwargs": d_kwargs}

        hard_neg_1_kwargs = parse_dict(batch, "hard_neg_1_") if "hard_neg_1_input_ids" in batch else None
        hard_neg_2_kwargs = parse_dict(batch, "hard_neg_2_") if "hard_neg_2_input_ids" in batch else None
        
        if hard_neg_1_kwargs is not None:
            q_d_args["hard_neg_1_kwargs"] = hard_neg_1_kwargs
        if hard_neg_2_kwargs is not None:
            q_d_args["hard_neg_2_kwargs"] = hard_neg_2_kwargs

        if self.config.get("no_q_expansion", False):
            q_d_args["no_q_expansion"] = True
        if self.config.get("no_d_expansion", False):
            q_d_args["no_d_expansion"] = True
        if self.config.get("expansion_only_q", False):
            q_d_args["expansion_only_q"] = True
        if self.config.get("expansion_only_d", False):
            q_d_args["expansion_only_d"] = True
        is_q_fix = self.config.get("no_q_weight", False) and self.config.get("no_q_expansion", False)
        if is_q_fix:
            q_d_args["is_q_fix"] = True

        if "q_dim_reduce" in self.config and self.config["q_dim_reduce"] is not None:
            if not is_q_fix:
                q_d_args["q_dim_reduce"] = self.config["q_dim_reduce"]
        if "d_dim_reduce" in self.config and self.config["d_dim_reduce"] is not None:
            q_d_args["d_dim_reduce"] = self.config["d_dim_reduce"]
        if self.config.get("d_add_comp_logit", False):
            q_d_args["d_add_comp_logit"] = True
        if self.config.get("d_out_hidden", False):
            q_d_args["d_out_hidden"] = True

        if self.config.get("in_batch_mode", False):
            q_d_args["in_batch_mode"] = True
        else:
            label = batch["label"].to(self.device)
            q_d_args["label"] = label
        if self.config.get("d_self_score", False):
            q_d_args["d_self_score"] = True
        if self.config.get("q_normalize_score", False):
            q_d_args["q_normalize_score"] = self.config.get("q_normalize_score")
        if self.config.get("d_normalize_score", False):
            q_d_args["d_normalize_score"] = self.config.get("d_normalize_score")
        if self.config.get("self_normalize", False):
            q_d_args["self_normalize"] = self.config.get("self_normalize")

        if "bm25_vectors" in batch and batch["bm25_vectors"] is not None:
            q_d_args["bm25_vectors"] = batch["bm25_vectors"]
        
        if self.config.get("self_score_mode", False):
            q_d_args["self_score_mode"] = self.config.get("self_score_mode")

        out_q_d = self.model(**q_d_args)
        out = {}
        for k, v in out_q_d.items():
            out[k] = v
        return out

    def train_iterations(self):
        is_final_ckpt_saved = False
        mpm = amp.MixedPrecisionManager(self.fp16)
        
        if hasattr(self.model, 'module'):
            if hasattr(self.model.module, '_use_mixed_precision'):
                self.model.module._use_mixed_precision = self.fp16
        else:
            if hasattr(self.model, '_use_mixed_precision'):
                self.model._use_mixed_precision = self.fp16
        
        for i in tqdm(range(self.start_iteration, self.nb_iterations)):
            self.model.train()
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)

            epoch = -1.0
            with mpm.context():
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.to(self.device)
                epoch = (int(self.config["train_batch_size"]) * i) / self.config["train_pairs"]

                out = self.forward(batch)
                monitor_losses = {}
                loss = self.loss["loss"](out).mean()  # torch.bfloat16
                monitor_losses["loss"] = loss.item()
                self_loss = 0
                if "d_self_loss" in self.loss:
                    d_self_weight = self.config.get("d_self_weight", 0)
                    raw_self_loss = self.loss["d_self_loss"](out).mean()
                    self_loss = raw_self_loss * d_self_weight
                    loss += self_loss
                    if i <= 5 or i % 200 == 0:
                        print(f"Rank {os.environ.get('RANK', 0)}: iter={i}, raw_self_loss={raw_self_loss.item():.6f}, d_self_weight={d_self_weight}, final_self_loss={self_loss.item():.6f}")
                monitor_losses["self_loss"] = self_loss.item() if "d_self_loss" in self.loss else -1.0

                ui_loss = 0
                if "ui_loss" in self.loss:
                    ui_weight = self.config.get("ui_weight", 1.0)
                    
                    try:
                        raw_ui_loss = self.loss["ui_loss"](out)
                        
                        if i <= 5:
                            print(f"DEBUG: raw_ui_loss type={type(raw_ui_loss)}, shape={raw_ui_loss.shape if hasattr(raw_ui_loss, 'shape') else 'None'}, value={raw_ui_loss}")
                            print(f"DEBUG: ui_weight type={type(ui_weight)}, value={ui_weight}")
                        
                        if hasattr(raw_ui_loss, 'dim') and raw_ui_loss.dim() > 0:
                            raw_ui_loss = raw_ui_loss.mean()
                        
                        ui_weight = float(ui_weight)
                        
                        ui_loss = raw_ui_loss * ui_weight
                        loss += ui_loss
                        
                        if i <= 5 or i % 200 == 0:
                            print(f"Rank {os.environ.get('RANK', 0)}: iter={i}, raw_ui_loss={raw_ui_loss.item():.6f}, ui_weight={ui_weight}, final_ui_loss={ui_loss.item():.6f}")
                    
                    except Exception as e:
                        print(f"ERROR in UI loss calculation: {e}")
                        print(f"raw_ui_loss: {raw_ui_loss if 'raw_ui_loss' in locals() else 'Not calculated'}")
                        print(f"ui_weight: {ui_weight}")
                        raise e
                        
                monitor_losses["ui_loss"] = ui_loss.item() if "ui_loss" in self.loss else -1.0

                if self.regularizer is not None:
                    if "train" in self.regularizer:
                        reg_losses = {}
                        for reg in self.regularizer["train"]:
                            lambda_q = self.regularizer["train"][reg]["lambdas"]["lambda_q"].step() if "lambda_q" in \
                                            self.regularizer["train"][reg]["lambdas"] else False
                            lambda_d = self.regularizer["train"][reg]["lambdas"]["lambda_d"].step() if "lambda_d" in \
                                            self.regularizer["train"][reg]["lambdas"] else False
                            targeted_rep = self.regularizer["train"][reg]["targeted_rep"]
                            reg_losses[reg] = 0
                            if lambda_q:
                                reg_losses[reg] += (self.regularizer["train"][reg]["loss"](out["q_{}".format(targeted_rep)]) * lambda_q).mean()
                            if lambda_d:
                                reg_losses[reg] += (self.regularizer["train"][reg]["loss"](out["d_{}".format(targeted_rep)]) * lambda_d).mean()
                            loss += sum(reg_losses.values())
                            monitor_losses["{}_loss".format(reg)] = sum(reg_losses.values()).item() if (lambda_q or lambda_d) else -1.0
                        monitor_losses["all_loss"] = loss.item()
                    with torch.no_grad():
                        for reg in self.regularizer["eval"]:
                            monitor_losses["{}_q".format(reg)] = self.regularizer["eval"][reg]["loss"](out["q_rep"]).mean().item()
                            monitor_losses["{}_d".format(reg)] = self.regularizer["eval"][reg]["loss"](out["d_rep"]).mean().item()
                            monitor_losses["{}_q_raw".format(reg)] = self.regularizer["eval"][reg]["loss"](out["q_rep_raw"]).mean().item()
                            monitor_losses["{}_d_raw".format(reg)] = self.regularizer["eval"][reg]["loss"](out["d_rep_raw"]).mean().item()
                            monitor_losses["{}_q_raw_text".format(reg)] = self.regularizer["eval"][reg]["loss"](out["q_raw_text_mask"]).mean().item()
                            monitor_losses["{}_d_raw_text".format(reg)] = self.regularizer["eval"][reg]["loss"](out["d_raw_text_mask"]).mean().item()

            loss = loss / self.config["gradient_accumulation_steps"]
            mpm.backward(loss)
            if i % self.config["gradient_accumulation_steps"] == 0:
                mpm.step(self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()
                    monitor_losses["lr"] = self.scheduler.get_last_lr()[0]  # for iter=i-1
            if i % self.record_frequency == 0:
                self.save_checkpoint(step=i, epoch=epoch, perf=loss, save_type='normal')
                if epoch >= 1:  
                    self.save_checkpoint(step=i, epoch=epoch, perf=loss, save_type='final')

                if self.validation:
                    raise NotImplementedError  # TODO

                print_head_iter = (self.start_iteration + self.record_frequency) - (self.start_iteration % self.record_frequency)
                if int(os.environ['RANK']) == 0:
                    if i == print_head_iter:
                        if self.monitoring_ckpt is None:
                            print('%-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-20s %s' % (
                                'Iter', 'Epoch', 'All_Loss', 'Loss', 'Self_Loss', 'UI_Loss', 'Reg_Loss',
                                'L0_q_raw', 'L0_d_raw', 'L0_q', 'L0_d', 'L0_q_raw_text', 'L0_d_raw_text'))
                        else:
                            raise NotImplementedError  # TODO

                    if self.monitoring_ckpt is None:
                        print('%-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-15s %-20s %s' % (
                             i, "{:.3f}".format(epoch),
                             "{:.5f}".format(monitor_losses["all_loss"]),
                             "{:.5f}".format(monitor_losses["loss"]),
                             "{:.8f}".format(monitor_losses["self_loss"]),  
                             "{:.6f}".format(monitor_losses["ui_loss"]),  
                             "{:.5f}".format(monitor_losses.get("FLOPS_loss", -1.0)),
                             "{:.1f}".format(monitor_losses["L0_q_raw"]),
                             "{:.1f}".format(monitor_losses["L0_d_raw"]),
                             "{:.1f}".format(monitor_losses["L0_q"]),
                             "{:.1f}".format(monitor_losses["L0_d"]),
                             "{:.1f}".format(monitor_losses["L0_q_raw_text"]),
                             "{:.1f}".format(monitor_losses["L0_d_raw_text"])))
                    else:
                        raise NotImplementedError  # TODO
                    self.train_res_handler = open(os.path.join(self.checkpoint_dir, "train_perf.txt"), "a")
                    self.train_res_handler.write("{},{:.3f},{:.5f},{:.5f},{:.8f},{:.6f},{:.5f},{:.7f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}".format(
                        i, epoch,
                        monitor_losses["all_loss"],
                        monitor_losses["loss"],
                        monitor_losses["self_loss"],  
                        monitor_losses["ui_loss"],    
                        monitor_losses.get("FLOPS_loss", -1.0),
                        monitor_losses["lr"],
                        monitor_losses["L0_q_raw"],
                        monitor_losses["L0_d_raw"],
                        monitor_losses["L0_q"],
                        monitor_losses["L0_d"],
                        monitor_losses["L0_q_raw_text"],
                        monitor_losses["L0_d_raw_text"]))

                    if self.monitoring_ckpt is not None:
                        raise NotImplementedError  # TODO
                    self.train_res_handler.write("\n")
                    self.train_res_handler.close()
            try:
                torch.distributed.barrier()
            except:
                print('Dist.barrier Error Occurs, But Ignored')
                pass

            if int(os.environ['RANK']) == 0:
                try:
                    sparsity_csv_path = os.path.join(self.checkpoint_dir, "sparsity_every_step.csv")
                    if i == self.start_iteration:
                        with open(sparsity_csv_path, "w") as f:
                            f.write("iter,epoch,loss,l0_q_raw,l0_d_raw,l0_q,l0_d\n")
                    
                    sparsity_data = f"{i},{epoch:.3f},{monitor_losses['loss']:.5f}," \
                                   f"{monitor_losses['L0_q_raw']:.1f},{monitor_losses['L0_d_raw']:.1f}," \
                                   f"{monitor_losses['L0_q']:.1f},{monitor_losses['L0_d']:.1f}\n"
                    with open(sparsity_csv_path, "a") as f:
                        f.write(sparsity_data)
                except Exception as e:
                    print(f"Error in sparsity CSV recording: {e}")

    def save_checkpoint(self, step, epoch, perf, save_type='normal'):
        if int(os.environ['RANK']) == 0:
            custom_save_obj = custom_save(self.config)
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model

            state = {"step": step,
                     "epoch": epoch,
                     # "perf": perf,
                     "model_state_dict": model_to_save.state_dict(),
                     # "optimizer_state_dict": self.optimizer.state_dict(),
                     # "config": self.config,
                     # "regularizer": self.regularizer,
                     }
            # if self.scheduler is not None:
            #     state["scheduler_state_dict"] = self.scheduler.state_dict()
            ckpt_dir = ''
            if save_type == 'normal':
                ckpt_dir = os.path.join(self.checkpoint_dir, "model_ckpt_{}".format(state["step"]))
                makedir(ckpt_dir)
                custom_save_obj.save_checkpoint(state,
                                                os.path.join(ckpt_dir, "model_ckpt_{}.pth".format(state["step"])))
                try:
                    formal_ckpt_dir = os.path.join(self.checkpoint_dir, "model_ckpt_{}_formal".format(state["step"]))
                    makedir(formal_ckpt_dir)
                    print(f"Rank {os.environ.get('RANK', 0)}: Saving formal checkpoint to {formal_ckpt_dir}")
                    model_to_save.transformer_rep.transformer.save_pretrained(formal_ckpt_dir)
                    print(f"Rank {os.environ.get('RANK', 0)}: transformer saved successfully")
                    model_to_save.transformer_rep.tokenizer.save_pretrained(formal_ckpt_dir)
                    print(f"Rank {os.environ.get('RANK', 0)}: tokenizer saved successfully")
                    print(f"Rank {os.environ.get('RANK', 0)}: formal checkpoint saved successfully")
                except Exception as e:
                    print(f"Rank {os.environ.get('RANK', 0)}: formal checkpoint saving failed: {e}")
                    print(f"Rank {os.environ.get('RANK', 0)}: training will continue, but formal checkpoint is incomplete")
            elif save_type == 'final':
                ckpt_dir = os.path.join(self.checkpoint_dir, "model_final_checkpoint")
                makedir(ckpt_dir)
                custom_save_obj.save_checkpoint(state, os.path.join(ckpt_dir, "model_final_checkpoint.pth"))
                formal_ckpt_dir = os.path.join(self.checkpoint_dir, "model_final_checkpoint_formal")
                makedir(formal_ckpt_dir)
                model_to_save.transformer_rep.transformer.save_pretrained(formal_ckpt_dir)
                model_to_save.transformer_rep.tokenizer.save_pretrained(formal_ckpt_dir)
            else:
                raise NotImplementedError
        else:
            pass
