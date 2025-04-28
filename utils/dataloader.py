import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomNERDataset(Dataset):
    def __init__(self, csv_file, tokenizer):
        """
        csv_file: 데이터 파일 경로 (post, label)
        tokenizer: BERT 토크나이저
        """
        self.csv_file = csv_file

        if ".tsv" in csv_file:
            self.data = pd.read_csv(csv_file, delimiter="\t")
        else:
            self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer

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
        encoded = self.tokenizer.encode_plus(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        token_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        # label = torch.tensor(label, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "label": label
        }

def collate_fn(batch):
    """
    DataLoader에서 batch 단위로 데이터를 처리하는 함수
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 데이터 로더 설정
def get_dataloader(csv_file, tokenizer, batch_size=16, shuffle=True):
    print("---Start dataload---")
    dataset = CustomNERDataset(csv_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    print("---End dataload---")
    return dataloader
