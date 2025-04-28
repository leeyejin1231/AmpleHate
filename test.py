import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import pandas as pd
import random
from transformers import BertTokenizer, BertModel
from utils import get_dataloader, iter_product
from model import CustomBERT
from config import test_config
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


def test_model(dataloader, model):
    print("---Start Test!---")
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    return accuracy, f1, predictions, true_labels


def test(log):
    set_seed(log.param.SEED)

    tokenizer = BertTokenizer.from_pretrained(log.param.model_type)
    model = CustomBERT(log.param.model_type, hidden_dim=log.param.hidden_size, e=log.param.e).to(device)

    if "ihc" in log.param.dataset or "SST" in log.param.dataset:
        test_loader = get_dataloader(f"./dataset/{log.param.dataset}/test.tsv", tokenizer, batch_size=16, shuffle=False)
    else:
        test_loader = get_dataloader(f"./dataset/{log.param.dataset}/test.csv", tokenizer, batch_size=16, shuffle=False)
    model.load_state_dict(torch.load(f"./save/{log.param.model_path}/best_model.pth"))
    model.to(device)

    test_accuracy, test_f1, predictions, _ = test_model(test_loader, model)

    df = pd.DataFrame()
    df["label"] = predictions
    # df.to_csv(f'./save/{log.param.dataset}/SST-2.tsv', sep="\t", index=False)

    print(f"Dataset: {log.param.dataset}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}")
    print(f"Test F1-Score: {test_f1*100:.2f}")


if __name__ == '__main__':
    tuning_param = test_config.tuning_param
    
    param_list = [test_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name
        log = edict()
        log.param = test_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

    test(log)