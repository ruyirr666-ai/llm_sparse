import torch


class InBatchPairwiseNLL:


    def __init__(self, d_self_score=False):
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.d_self_score = d_self_score

    def __call__(self, out_d):
        in_batch_scores = None
        if self.d_self_score:
            in_batch_scores = out_d["score_dict"]["d_in_batch_self_score"]
        else:
            in_batch_scores = out_d["score_dict"]["in_batch_score"]
        
    
        if "hard_neg_scores" in out_d["score_dict"]:
   
            hard_neg_scores = out_d["score_dict"]["hard_neg_scores"]  
            batch_size = in_batch_scores.shape[0]
            

            positive_scores = torch.diag(in_batch_scores).unsqueeze(1)  
            

            batch_neg_scores = []
            for i in range(batch_size):

                row_scores = in_batch_scores[i]
                neg_scores = torch.cat([row_scores[:i], row_scores[i+1:]])
                batch_neg_scores.append(neg_scores)
            batch_neg_scores = torch.stack(batch_neg_scores)  
            
  
            all_scores = torch.cat([positive_scores, hard_neg_scores, batch_neg_scores], dim=1)
            

            log_probs = self.logsoftmax(all_scores)
            

            loss = -log_probs[:, 0]
            return torch.mean(loss)
        else:

            nb_columns = in_batch_scores.shape[1]
            nb_gpus = int(in_batch_scores.shape[0] / nb_columns)

            scores = self.logsoftmax(in_batch_scores)
            return torch.mean(-scores[torch.arange(in_batch_scores.shape[0]),
                                      torch.arange(nb_columns).repeat(nb_gpus)])





