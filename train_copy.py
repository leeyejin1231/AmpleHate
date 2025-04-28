import torch, numpy as np, random, json
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer
from utils import get_dataloader, iter_product
from config import train_config as config
from model import BERT_ILC_Binary
from easydict import EasyDict as edict

device = torch.device("cuda")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, criterion, optimizer):
    model.train(); tot=0
    for batch in tqdm(train_loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)

        optimizer.zero_grad()
        logits_dict = model(input_ids, attention_mask)
        loss = 0

        for l,logits in logits_dict.items():
            layer_loss = criterion(logits, labels)
            probe = model.probes[str(l)]
            if model.reg_type=="l1":
                layer_loss += model.reg_weight*probe.weight.abs().sum()
            else:
                layer_loss += model.reg_weight*probe.weight.norm()
            loss += layer_loss
        
        loss /= len(logits_dict)
        loss.backward()
        optimizer.step()
        tot+=loss.item()

    return tot/len(train_loader)


def best_threshold(probs: np.ndarray, labels: np.ndarray,
                   grid=np.linspace(0.05, 0.95, 19)):
    """
    probs  : (N,)  sigmoid 확률
    labels : (N,)  0/1
    grid   : 탐색할 임계값 리스트
    return : (best_thr, best_f1)
    """
    best_t, best_f1 = 0.5, 0
    for t in grid:
        f1 = f1_score(labels, (probs >= t).astype(int))
        acc = accuracy_score(labels, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1, acc


@torch.no_grad()
def evaluate(model, valid_loader):
    model.eval()
    # 레이어별 확률·라벨 저장
    layer_probs   = {l: [] for l in model.probe_layers}
    layer_labels  = {l: [] for l in model.probe_layers}

    for batch in tqdm(valid_loader, desc="eval"):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y    = batch["labels"].cpu().numpy()

        logits_dict = model(ids, mask)
        for l, logits in logits_dict.items():
            p = torch.sigmoid(logits).cpu().numpy()  # 확률
            layer_probs[l].append(p)
            layer_labels[l].append(y)

    # 레이어별 best threshold & F1
    best_thr, best_f1, best_acc = {}, {}, {}
    for l in model.probe_layers:
        probs  = np.concatenate(layer_probs[l])
        labels = np.concatenate(layer_labels[l])
        t, f1, acc  = best_threshold(probs, labels)
        best_thr[l], best_f1[l], best_acc[l] = float(t), float(f1), float(acc)

    best_layer = max(best_f1, key=best_f1.get)
    return best_f1, best_acc, best_layer, best_thr

# def evaluate(model, valid_loader):
#     model.eval()
#     layer_scores = {l:[] for l in model.probe_layers}
#     layer_accuracy = {l:[] for l in model.probe_layers}

#     for batch in tqdm(valid_loader, desc="eval"):
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].cpu().numpy()

#         logits_dict = model(input_ids, attention_mask)

#         for l,logits in logits_dict.items():
#             preds = (torch.sigmoid(logits).cpu().numpy() >= .5).astype(int)
#             layer_scores[l].append(f1_score(labels, preds, zero_division=0))
#             layer_accuracy[l].append(accuracy_score(labels, preds))

#     mean_f1 = {l:float(np.mean(v)) for l,v in layer_scores.items()}
#     mean_accuracy = {l:float(np.mean(v)) for l,v in layer_accuracy.items()}
#     best_l = max(mean_f1, key=mean_f1.get)

#     return mean_f1, mean_accuracy, best_l


def train(log):
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(log.param.model_type)
    train_loader = get_dataloader(f"./dataset/{log.param.dataset}/train.csv", tokenizer, batch_size=16)
    valid_loader = get_dataloader("./dataset/ood.csv", tokenizer, batch_size=16, shuffle=False)

    ckpt   = torch.load(f"./save/{log.param.dataset}/best_ilc_binary_2.pth", map_location=device)
    best_l = ckpt["best_layer"]
    threshold = ckpt["threshold"][best_l]
    print(f"Loaded checkpoint.  best_layer = {best_l}, best_threshold = {threshold}")

    model = BERT_ILC_Binary(log.param.model_type).to(device)
    model.load_state_dict(ckpt["state"])

    # model = BERT_ILC_Binary(bert_name=log.param.model_type, probe_layers = [8, 9, 10, 11]).to(device)
    optimizer = optim.AdamW(model.probes.parameters(), lr=log.param.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_f1=0.6463
    best_layer=None
    epochs = log.param.nepoch + 21

    for epoch in range(21, epochs):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        mean_f1, accuracy, cur_best, best_threshold = evaluate(model, valid_loader)
        print(f"[{epoch}] loss {tr_loss:.4f}  best layer {cur_best}  accuracy {accuracy[cur_best]:.4f}   F1 {mean_f1[cur_best]:.4f}")
        if mean_f1[cur_best] > best_f1:
            best_f1, best_layer, best_threshold = mean_f1[cur_best], cur_best, best_threshold
            torch.save({"state":model.state_dict(),"best_layer":best_layer, "threshold":best_threshold},f"save/{log.param.dataset}/best_ilc_binary.pth")
            print(" ** checkpoint saved **")
    print("Finished. best layer=",best_layer," F1=",best_f1)


if __name__ == "__main__":
    tuning_param = config.tuning_param
    
    param_list = [config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) 

    for param_com in param_list[1:]: # as first element is just name
        log = edict()
        log.param = config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        train(log)

