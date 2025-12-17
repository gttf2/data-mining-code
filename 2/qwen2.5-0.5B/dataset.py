import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    """
    预分词版本的数据集：
    - 在 __init__ 一次性把所有文本 token 化
    - __getitem__ 只做索引，不再每次 encode_plus
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.max_len = max_len

        # 先把 tokenizer 配好 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ★ 一次性批量分词，存在内存里
        encodings = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",   # 直接得到 [N, L] 的张量
        )

        self.input_ids = encodings["input_ids"]           # shape: [N, L]
        self.attention_mask = encodings["attention_mask"] # shape: [N, L]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 这里就只是做索引，几乎没有 CPU 开销
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }
