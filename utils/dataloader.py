import pandas as pd
import torch
from utils import NERProcessor
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import tqdm

class CustomNERDataset(Dataset):
    def __init__(self, csv_file, tokenizer, ner_tagger=None, use_ner=True):
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
        else:
            text, label = row["post"], row["label"]

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    # Get max length in batch
    max_len = max(len(item["head_token_idx"]) for item in batch)
    
    # Pad sequences to max length
    padded_head_token_idx = []
    for item in batch:
        head_idx = item["head_token_idx"]
        padding_len = max_len - len(head_idx)
        padded_idx = torch.cat([head_idx, torch.zeros(padding_len, dtype=torch.long)])
        padded_head_token_idx.append(padded_idx)
    
    input_ids = torch.stack([item["input_ids"] for item in batch])
    head_token_idx = torch.stack(padded_head_token_idx)
    labels = torch.stack([item["label"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "head_token_idx": head_token_idx,
        "labels": labels,
        "attention_mask": attention_mask
    }

def get_dataloader(csv_file, tokenizer, ner_tagger, use_ner, batch_size=16, shuffle=True, seed=42):
    set_seed(seed)
    print("---Start dataload---")
    g = torch.Generator()   
    g.manual_seed(seed)
    dataset = CustomNERDataset(csv_file, tokenizer, ner_tagger, use_ner)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, generator=g)
    print("---End dataload---")
    return dataloader

def ner_coverage_statistics(dataset):
    total = len(dataset)
    ner_applied = 0
    for i in tqdm(range(total)):
        item = dataset[i]
        if not (len(item["head_token_idx"]) == 1 and item["head_token_idx"][0].item() == 0):
            ner_applied += 1
    print(f"NER tagging applied: {ner_applied}/{total} ({ner_applied/total*100:.2f}%)")
