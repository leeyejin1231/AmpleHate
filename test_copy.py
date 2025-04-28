#!/usr/bin/env python
# test.py
import torch, json, numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer
from utils import get_dataloader, iter_product
from model import BERT_ILC_Binary
from config import test_config as config
from easydict import EasyDict as edict
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(model, loader, best_layer, threshold):
    model.eval()
    y_true, y_pred = [], []

    for batch in tqdm(loader, desc="Test"):
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        label = batch["labels"].cpu().numpy()        # (B,)
        logits = model(ids, mask)[best_layer]        # (B,)
        pred   = (torch.sigmoid(logits) >= threshold).cpu().numpy().astype(int)

        y_true.append(label)
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="binary", zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    return acc, f1, cm


def main(log):
    tokenizer = AutoTokenizer.from_pretrained(log.param.model_type)
    if log.param.dataset == "ihc":
        test_loader = get_dataloader(f"./dataset/ihc/test.tsv", tokenizer, batch_size=log.param.eval_batch_size, shuffle=False)
    else:
        test_loader = get_dataloader(f"./dataset/{log.param.dataset}/test.csv", tokenizer, batch_size=log.param.eval_batch_size, shuffle=False)

    ckpt   = torch.load(f"./save/{log.param.model_path}/best_ilc_binary.pth", map_location=DEVICE)
    best_l = ckpt["best_layer"]
    threshold = ckpt["threshold"][best_l]
    print(f"Loaded checkpoint.  best_layer = {best_l}, best_threshold = {threshold}")

    model = BERT_ILC_Binary(log.param.model_type).to(DEVICE)
    model.load_state_dict(ckpt["state"])

    acc, f1, cm = evaluate(model, test_loader, best_l, threshold)
    print(f"\n=== Test Results (layer {best_l}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion matrix (TN FP / FN TP):\n", cm)

    with open(f"save/{log.param.model_path}/test_result.json", "w") as fp:
        json.dump({
            "layer":       int(best_l),
            "accuracy":    float(acc),
            "f1":          float(f1),
            "conf_matrix": cm.tolist()
        }, fp, indent=2)
    print(f"Saved_{log.param.dataset}/test_result.json")


if __name__ == "__main__":
    tuning_param = config.tuning_param
    
    param_list = [config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) 

    for param_com in param_list[1:]: # as first element is just name
        log = edict()
        log.param = config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        main(log)