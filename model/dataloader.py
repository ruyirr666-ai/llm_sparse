import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from model.utils import rename_keys
import os


class DataLoaderWrapper(DataLoader):
    def __init__(self, tokenizer, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = tokenizer
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        raise NotImplementedError("must implement this method")


class QwDataLoader(DataLoaderWrapper):
    """
    QwDataLoader
    """

    def __init__(self, tokenizer, max_length, config=None, **kwargs):
        self.config = config
        super().__init__(tokenizer, max_length, **kwargs)

    def collate_fn(self, batch):

        if isinstance(batch[0], dict) and 'bm25_vector' in batch[0]:

            q = [item['query'] for item in batch]
            d = [item['title'] for item in batch]
            

            bm25_vectors = torch.stack([item['bm25_vector'] for item in batch])
            hard_neg_1 = None
        else:

            if len(batch[0]) >= 5:

                q = default_collate(
                    [str(r[1]) if not isinstance(r[1], str) and not hasattr(r[1], 'decode') else r[1] if isinstance(r[1], str) else r[1].decode() 
                     for r in batch])
                d = default_collate(
                    [str(r[3]) if not isinstance(r[3], str) and not hasattr(r[3], 'decode') else r[3] if isinstance(r[3], str) else r[3].decode() 
                     for r in batch])
                

                use_hard_negatives = self.config.get("use_hard_negatives", False) if self.config else False
                if use_hard_negatives:

                    hard_neg_1 = default_collate(
                        [str(r[4]) if not isinstance(r[4], str) and not hasattr(r[4], 'decode') else r[4] if isinstance(r[4], str) else r[4].decode() 
                         for r in batch])
                else:

                    hard_neg_1 = None
            else:

                q = default_collate(
                    [str(r[1]) if not isinstance(r[1], str) and not hasattr(r[1], 'decode') else r[1] if isinstance(r[1], str) else r[1].decode() 
                     for r in batch])
                d = default_collate(
                    [str(r[3]) if not isinstance(r[3], str) and not hasattr(r[3], 'decode') else r[3] if isinstance(r[3], str) else r[3].decode() 
                     for r in batch])
                hard_neg_1 = None
            
            bm25_vectors = None


        tokenizer_class_name = self.tokenizer.__class__.__name__
        is_bert_tokenizer = 'Bert' in tokenizer_class_name or 'bert' in tokenizer_class_name.lower()
        

        if is_bert_tokenizer:

            q_tokens = self.tokenizer(list(q),
                                     add_special_tokens=True,
                                     max_length=self.max_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_attention_mask=True,
                                     return_tensors='pt')
            
            d_tokens = self.tokenizer(list(d),
                                     add_special_tokens=True,
                                     max_length=self.max_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_attention_mask=True,
                                     return_tensors='pt')
            
            result = {
                "q_input_ids": q_tokens["input_ids"],
                "q_attention_mask": q_tokens["attention_mask"],
                "d_input_ids": d_tokens["input_ids"],
                "d_attention_mask": d_tokens["attention_mask"],
                "bm25_vectors": bm25_vectors  
            }

            if hard_neg_1 is not None:
                hard_neg_1_tokens = self.tokenizer(list(hard_neg_1),
                                                 add_special_tokens=True,
                                                 max_length=self.max_length,
                                                 padding='max_length',
                                                 truncation=True,
                                                 return_attention_mask=True,
                                                 return_tensors='pt')
                result["hard_neg_1_input_ids"] = hard_neg_1_tokens["input_ids"]
                result["hard_neg_1_attention_mask"] = hard_neg_1_tokens["attention_mask"]
        
        else:

            q_tokens = self.tokenizer(list(q),
                                     add_special_tokens=True,
                                     max_length=self.max_length - 1,
                                     return_attention_mask=True)
            q_ = {}
            q_["input_ids"] = [sublist + [self.tokenizer.eos_token_id] for sublist in q_tokens["input_ids"]]
            q_["input_ids"] = [[self.tokenizer.pad_token_id] * (self.max_length - len(sublist)) + sublist for sublist in q_["input_ids"]]
            q_["attention_mask"] = [sublist + [1] for sublist in q_tokens['attention_mask']]
            q_["attention_mask"] = [[0] * (self.max_length - len(sublist)) + sublist for sublist in q_["attention_mask"]]
            

            d_tokens = self.tokenizer(list(d),
                                     add_special_tokens=True,
                                     max_length=self.max_length - 1,
                                     return_attention_mask=True)
            d_ = {}
            d_["input_ids"] = [sublist + [self.tokenizer.eos_token_id] for sublist in d_tokens["input_ids"]]
            d_["input_ids"] = [[self.tokenizer.pad_token_id] * (self.max_length - len(sublist)) + sublist for sublist in d_["input_ids"]]
            d_["attention_mask"] = [sublist + [1] for sublist in d_tokens['attention_mask']]
            d_["attention_mask"] = [[0] * (self.max_length - len(sublist)) + sublist for sublist in d_["attention_mask"]]

            result = {
                "q_input_ids": torch.tensor(q_["input_ids"]),
                "q_attention_mask": torch.tensor(q_["attention_mask"]),
                "d_input_ids": torch.tensor(d_["input_ids"]),
                "d_attention_mask": torch.tensor(d_["attention_mask"]),
                "bm25_vectors": bm25_vectors  
            }
            

            if hard_neg_1 is not None:
                hard_neg_1_tokens = self.tokenizer(list(hard_neg_1),
                                                 add_special_tokens=True,
                                                 max_length=self.max_length - 1,
                                                 return_attention_mask=True)
                hard_neg_1_ = {}
                hard_neg_1_["input_ids"] = [sublist + [self.tokenizer.eos_token_id] for sublist in hard_neg_1_tokens["input_ids"]]
                hard_neg_1_["input_ids"] = [[self.tokenizer.pad_token_id] * (self.max_length - len(sublist)) + sublist for sublist in hard_neg_1_["input_ids"]]
                hard_neg_1_["attention_mask"] = [sublist + [1] for sublist in hard_neg_1_tokens['attention_mask']]
                hard_neg_1_["attention_mask"] = [[0] * (self.max_length - len(sublist)) + sublist for sublist in hard_neg_1_["attention_mask"]]
                
                result["hard_neg_1_input_ids"] = torch.tensor(hard_neg_1_["input_ids"])
                result["hard_neg_1_attention_mask"] = torch.tensor(hard_neg_1_["attention_mask"])
            
        return result


class TextCollectionDataLoader(DataLoaderWrapper):
    """
    TextCollectionDataLoader
    """
    
    def __init__(self, tokenizer, max_length, config=None, **kwargs):
        self.config = config
        super().__init__(tokenizer, max_length, **kwargs)

    def collate_fn(self, batch):
        print('RANK: {}, LOCAL_RANK: {}, WORLD_SIZE: {}, batch: {}'.format(int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE']), len(batch)))

        t = default_collate(
            [str(r[1]) if not isinstance(r[1], str) and not hasattr(r[1], 'decode') else r[1] if isinstance(r[1], str) else r[1].decode() for r in batch])  # text
        t_id = default_collate(
            [str(r[0]) if not isinstance(r[0], str) and not hasattr(r[0], 'decode') else r[0] if isinstance(r[0], str) else r[0].decode() for r in batch])  # text_id


        tokenizer_class_name = self.tokenizer.__class__.__name__
        is_bert_tokenizer = 'Bert' in tokenizer_class_name or 'bert' in tokenizer_class_name.lower()
        
        if is_bert_tokenizer:

            t_tokens = self.tokenizer(list(t),
                                     add_special_tokens=True,
                                     max_length=self.max_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_attention_mask=True,
                                     return_tensors='pt')
            
            return {
                "t_input_ids": t_tokens["input_ids"],
                "t_attention_mask": t_tokens["attention_mask"],
                "t_id": torch.tensor([int(i) for i in t_id], dtype=torch.long)
            }
        
        else:

            t = self.tokenizer(list(t),
                               add_special_tokens=True,
                               max_length=self.max_length - 1,
                               return_attention_mask=True)
            t_ = {}
            t_["input_ids"] = [sublist + [self.tokenizer.eos_token_id] for sublist in t["input_ids"]]
            t_["input_ids"] = [[self.tokenizer.pad_token_id] * (self.max_length - len(sublist)) + sublist for sublist in
                               t_["input_ids"]]
            t_["attention_mask"] = [sublist + [1] for sublist in t['attention_mask']]
            t_["attention_mask"] = [[0] * (self.max_length - len(sublist)) + sublist for sublist in t_["attention_mask"]]

            sample = {**rename_keys(t_, "t")}

            return {**{k: torch.tensor(v) for k, v in sample.items()},
                    "t_id": torch.tensor([int(i) for i in t_id], dtype=torch.long)}
