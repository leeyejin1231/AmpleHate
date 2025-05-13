import pandas as pd
import torch
from utils import NERProcessor
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class CustomNERDataset(Dataset):
    def __init__(self, csv_file, tokenizer, ner_tagger=None, use_ner=True):
        """
        csv_file: 데이터 파일 경로 (post, label)
        tokenizer: BERT 토크나이저
        ner_tagger: NER 태깅 모델
        """
        self.csv_file = csv_file

        if ".tsv" in csv_file:
            self.data = pd.read_csv(csv_file, delimiter="\t")
        else:
            self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.processor = NERProcessor(tokenizer, ner_tagger, use_ner)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if "ihc" in self.csv_file:
            class2int = {'not_hate':0 ,'implicit_hate': 1}
            text, label = row["post"], class2int[row["class"]]
        elif "sbic" in self.csv_file:
            class2int = {'not_offensive':0 ,'offensive': 1}
            text, label = row["post"], class2int[row["offensiveLABEL"]]
        elif "dyna" in self.csv_file:
            class2int = {'nothate':0 ,'hate': 1}
            text, label = row["text"], class2int[row["label"]]
        elif "SST" in self.csv_file:
            text, label = row['sentence'], row['label']
            text = row['sentence']
        elif "IMDB" in self.csv_file:
            class2int = {'positive':0 ,'negative': 1}
            text, label = row['review'], class2int[row["sentiment"]]
        else:
            text, label = row["post"], row["label"]

        # 토큰화 및 Head-Token 인덱스 추출
        token_ids, head_token_idx, attention_mask = self.processor.tokenize_and_encode(text)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        head_token_idx = torch.tensor(head_token_idx, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)   
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": token_ids,
            "head_token_idx": head_token_idx,
            "label": label,
            "attention_mask": attention_mask
        }

def set_seed(seed=42):
    """
    랜덤 시드 고정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU 사용 시에도 시드 고정

    # CUDNN 설정 (연산 속도 vs 재현성 선택)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """
    DataLoader에서 batch 단위로 데이터를 처리하는 함수
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    # head_token_idx: [batch, variable_len] -> pad to [batch, max_num_head]
    head_token_idxs = [item["head_token_idx"] for item in batch]
    head_token_idx = pad_sequence(head_token_idxs, batch_first=True, padding_value=0)
    labels = torch.stack([item["label"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "head_token_idx": head_token_idx,
        "labels": labels,
        "attention_mask": attention_mask
    }

# 데이터 로더 설정
def get_dataloader(csv_file, tokenizer, ner_tagger, use_ner, batch_size=16, shuffle=True, seed=42):
    set_seed(seed)
    print("---Start dataload---")
    g = torch.Generator()   
    g.manual_seed(seed)
    dataset = CustomNERDataset(csv_file, tokenizer, ner_tagger, use_ner)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, generator=g)
    print("---End dataload---")
    return dataloader
