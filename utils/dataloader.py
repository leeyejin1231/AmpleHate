import pandas as pd
import torch
from utils import NERProcessor
from torch.utils.data import Dataset, DataLoader

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
        if "sbic" in self.csv_file:
            class2int = {'not_offensive':0 ,'offensive': 1}
            text, label = row["post"], class2int[row["offensiveLABEL"]]
        elif "dyna" in self.csv_file:
            class2int = {'nothate':0 ,'hate': 1}
            text, label = row["text"], class2int[row["label"]]
        else:
            text, label = row["post"], row["label"]

        # 토큰화 및 Head-Token 인덱스 추출
        token_ids, head_token_idx = self.processor.tokenize_and_encode(text)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        head_token_idx = torch.tensor(head_token_idx, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": token_ids,
            "head_token_idx": head_token_idx,
            "label": label
        }

def collate_fn(batch):
    """
    DataLoader에서 batch 단위로 데이터를 처리하는 함수
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    head_token_idx = torch.stack([item["head_token_idx"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids": input_ids,
        "head_token_idx": head_token_idx,
        "labels": labels
    }

# 데이터 로더 설정
def get_dataloader(csv_file, tokenizer, ner_tagger, use_ner, batch_size=16, shuffle=True):
    print("---Start dataload---")
    dataset = CustomNERDataset(csv_file, tokenizer, ner_tagger, use_ner)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    print("---End dataload---")
    return dataloader
