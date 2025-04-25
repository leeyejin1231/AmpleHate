import torch.optim as optim
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
from transformers import BertTokenizer
from utils import NERTagger, get_dataloader, iter_product
from model import CustomBERT, ContrastiveLossCosine, ILCBertClassifier
from config import train_config
from easydict import EasyDict as edict

device = torch.device('cuda')


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


def train_epoch(dataloader, model, optimizer, criterion, loss_type):
    print("---Start train!---")
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        for param in model.bert.parameters():
            param.requires_grad = False

        optimizer.zero_grad()
        loss, _ = model(input_ids, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(dataloader, model):
    print("---Start Valid!---")
    model.eval()
    all_preds = [[] for _ in range(log.param.num_layers)]
    all_labels = []
    correct_counts = [0] * log.param.num_layers
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits_list = model(input_ids)

            for idx, logits in enumerate(logits_list):
                preds = torch.argmax(logits, dim=-1)
                correct_counts[idx] += (preds == labels).sum().item()
                all_preds[idx].extend(preds.cpu().tolist())
            
            all_labels.extend(labels.cpu().tolist())
            total += labels.size(0)
                
    accuracies = [correct / total for correct in correct_counts]
    f1_scores = []
    for idx in range(log.param.num_layers):
        # Skip empty predictions
        if len(all_preds[idx]) == 0:
            f1_scores.append(0.0)
            continue
        f1_scores.append(f1_score(all_labels, all_preds[idx], average='binary', pos_label=1))
    best_layer = int(torch.tensor(accuracies).argmax().item())
    
    return accuracies, f1_scores, best_layer


def train(log):
    set_seed(log.param.SEED)

    tokenizer = BertTokenizer.from_pretrained(log.param.model_type)
    ner_tagger = NERTagger()
    MODEL_SAVE_PATH = f"./save/{log.param.dataset}/best_model.pth"
    criterion = {"lambda_loss":log.param.lambda_loss, "cross-entropy": nn.CrossEntropyLoss(), "contrastive-learning":ContrastiveLossCosine()}

    if "ihc" in log.param.dataset:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.tsv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/valid.tsv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size)
    elif "SST" in log.param.dataset:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.tsv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/dev.tsv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size)
    else:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.csv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/valid.csv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size)

    model = ILCBertClassifier(log.param.model_type).to(device)
    # optimizer = optim.AdamW(model.parameters(), lr=log.param.learning_rate)
    optimizer = optim.AdamW(model.ilc_classifiers.parameters(), lr=log.param.learning_rate)

    best_f1_score = 0.0
    num_epochs = log.param.nepoch
    df = {"param":{}, "train":{"loss":[], "f1":[]}}
    df["param"]["dataset"] = log.param.dataset
    df["param"]["train_batch_size"] = log.param.train_batch_size
    df["param"]["learning_rate"] = log.param.learning_rate
    df["param"]["loss"] = log.param.loss
    df["param"]["SEED"] = log.param.SEED
    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}")
        train_loss = train_epoch(train_loader, model, optimizer, criterion, log.param.loss)
        accuracies, f1_scores, best_layer = evaluate(valid_loader, model)
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {accuracies[best_layer]:.4f}, F1-Score: {f1_scores[best_layer]:.4f}, Best Layer: {best_layer}")
        
        # df["train"]["loss"].append(train_loss)
        # df["train"]["f1"].append(f1_scores[best_layer])

        if f1_scores[best_layer] > best_f1_score:
            best_f1_score = f1_scores[best_layer]
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            # df["stop_epoch"] = epoch+1
            # df["valid_f1_score"] = f1_score[best_layer]
            # df["valid_loss"] = train_loss
            print(f"=== Model saved at epoch {epoch+1} with F1-score: {f1_scores[best_layer]:.4f} ===")
    
    # with open(f'save/{log.param.dataset}/log.json', 'w') as file:
    #     json.dump(df, file)

if __name__ == '__main__':
    tuning_param = train_config.tuning_param
    
    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name
        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        train(log)