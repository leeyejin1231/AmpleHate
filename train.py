import torch.optim as optim
import torch
import json
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
from transformers import BertTokenizer, AutoTokenizer
from utils import NERTagger, get_dataloader, iter_product
from model import CustomBERT, ContrastiveLossCosine
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


def train_epoch(dataloader, model, optimizer, criterion, loss_type, log):
    print("---Start train!---")
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        head_token_idx = batch["head_token_idx"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, head_token_idx, attention_mask)
        if loss_type == "contrastive-learning":
            loss = (criterion["lambda_loss"]*criterion["cross-entropy"](outputs,labels)) + ((1-criterion["lambda_loss"])*criterion["contrastive-learning"](outputs,labels))
        else:
            loss = criterion["cross-entropy"](outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def best_threshold(probs: np.ndarray, labels:np.ndarray, grid=np.linspace(0.05, 0.95, 19)):
    best_t, best_f1 = 0.5, 0
    for t in grid:
        f1 = f1_score(labels, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def evaluate(dataloader, model, log):
    # print("---Start Valid!---")
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            head_token_idx = batch["head_token_idx"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, head_token_idx, attention_mask)
            preds = torch.softmax(outputs, dim=1)[:, 1]

            predictions.extend(preds.cpu())
            true_labels.extend(labels.cpu().numpy())

    t = best_threshold(np.array(predictions), true_labels)
    y_pred = (np.array(predictions) >= t).astype(int)
    acc = accuracy_score(true_labels, y_pred)
    f1 = f1_score(true_labels, y_pred, average="macro")
    best_t, accuracy, f1 = float(t), float(acc), float(f1)
    
    return accuracy, f1, best_t

def train_steps(dataloader, valid_loader, model, optimizer, criterion, loss_type, log, df, MODEL_SAVE_PATH):
    print("---Start train by step!---")
    model.train()
    total_loss = 0
    step = 0
    best_f1_score = 0.0
    # eval_steps = getattr(log.param, 'eval_steps', 100)
    eval_steps = 50
    step_losses = []
    step_f1s = []
    step_thresholds = []
    stop_step = 0
    valid_f1_score = 0.0
    valid_loss = 0.0
    valid_threshold = 0.0

    for epoch in range(log.param.nepoch):
        print(f"epoch {epoch+1}")
        for batch in dataloader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            head_token_idx = batch["head_token_idx"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, head_token_idx, attention_mask)
            if loss_type == "contrastive-learning":
                loss = (criterion["lambda_loss"]*criterion["cross-entropy"](outputs,labels)) + ((1-criterion["lambda_loss"])*criterion["contrastive-learning"](outputs,labels))
            else:
                loss = criterion["cross-entropy"](outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # step별 평가 및 저장
            if step % eval_steps == 0:
                avg_loss = total_loss / eval_steps
                acc, f1, best_t = evaluate(valid_loader, model, log)
                print(f"Step {step}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
                step_losses.append(avg_loss)
                step_f1s.append(f1)
                step_thresholds.append(best_t)
                total_loss = 0

                if f1 > best_f1_score:
                    best_f1_score = f1
                    torch.save({"model":model.state_dict(), "threshold":best_t}, MODEL_SAVE_PATH)
                    stop_step = step
                    valid_f1_score = f1
                    valid_loss = avg_loss
                    valid_threshold = best_t
                    print(f"=== Model saved at step {step} with F1-score: {f1:.4f} ===")

    # 마지막 남은 loss 기록
    if total_loss > 0:
        avg_loss = total_loss / (step % eval_steps)
        acc, f1, best_t = evaluate(valid_loader, model, log)
        print(f"Step {step}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        step_losses.append(avg_loss)
        step_f1s.append(f1)
        step_thresholds.append(best_t)
        if f1 > best_f1_score:
            best_f1_score = f1
            torch.save({"model":model.state_dict(), "threshold":best_t}, MODEL_SAVE_PATH)
            stop_step = step
            valid_f1_score = f1
            valid_loss = avg_loss
            valid_threshold = best_t
            print(f"=== Model saved at step {step} with F1-score: {f1:.4f} ===")

    df["train"]["step_loss"] = step_losses
    df["train"]["step_f1"] = step_f1s
    df["train"]["step_threshold"] = step_thresholds
    df["stop_step"] = stop_step
    df["valid_f1_score"] = valid_f1_score
    df["valid_loss"] = valid_loss
    df["valid_threshold"] = valid_threshold
    return df

def train(log):
    set_seed(log.param.SEED)

    tokenizer = AutoTokenizer.from_pretrained(log.param.model_type)
    ner_tagger = NERTagger()
    os.makedirs(f"./save/{log.param.dataset}/seed_{log.param.SEED}/lambda_{log.param.e}", exist_ok=True)
    MODEL_SAVE_PATH = f"./save/{log.param.dataset}/seed_{log.param.SEED}/lambda_{log.param.e}/best_model.pth"
    criterion = {"lambda_loss":log.param.lambda_loss, "cross-entropy": nn.CrossEntropyLoss(), "contrastive-learning":ContrastiveLossCosine()}

    if "ihc" in log.param.dataset:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.tsv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size, seed=log.param.SEED)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/valid.tsv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size, seed=log.param.SEED)
    elif "SST" in log.param.dataset:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.tsv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size, seed=log.param.SEED)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/dev.tsv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size, seed=log.param.SEED)
    elif "dynahate" in log.param.dataset or "sbic" in log.param.dataset:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.csv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size, seed=log.param.SEED)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/dev.csv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size, seed=log.param.SEED)
    else:
        train_loader = get_dataloader(f"./data/{log.param.dataset}/train.csv", tokenizer, ner_tagger=ner_tagger, use_ner=True,  batch_size=log.param.train_batch_size, seed=log.param.SEED)
        valid_loader = get_dataloader(f"./data/{log.param.dataset}/valid.csv", tokenizer, ner_tagger=None, use_ner=False, batch_size=log.param.train_batch_size, seed=log.param.SEED)

    model = CustomBERT(log.param.model_type, hidden_dim=log.param.hidden_size, e=log.param.e).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=log.param.learning_rate)

    df = {"param":{}, "train":{}}
    df["param"]["dataset"] = log.param.dataset
    df["param"]["train_batch_size"] = log.param.train_batch_size
    df["param"]["learning_rate"] = log.param.learning_rate
    df["param"]["loss"] = log.param.loss
    df["param"]["SEED"] = log.param.SEED

    df = train_steps(train_loader, valid_loader, model, optimizer, criterion, log.param.loss, log, df, MODEL_SAVE_PATH)

    with open(f"./save/{log.param.dataset}/seed_{log.param.SEED}/lambda_{log.param.e}/log.json", 'w') as file:
        json.dump(df, file)

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