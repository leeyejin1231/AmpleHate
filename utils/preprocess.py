from transformers import pipeline
import torch
import re

class NERTagger:
    def __init__(self, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
        """
        사전 학습된 NER 모델을 불러옴 (CONLL-03 데이터셋 기반)
        """
        device = torch.device('cuda')
        self.ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)

    def extract_named_entities(self, text):
        """
        입력된 텍스트에서 특정 개체명을 추출하여 리스트로 반환
        """
        entities = self.ner_pipeline(text)
        target_entities = []

        for entity in entities:
            if entity["entity_group"] in ["ORG", "NORP", "GPE", "LOC", "EVENT"]:
                target_entities.append(entity["word"])

        return target_entities


class NERProcessor:
    def __init__(self, tokenizer, ner_tagger=None, use_ner=True):
        self.tokenizer = tokenizer
        self.ner_tagger = ner_tagger
        self.use_ner = use_ner

    def extract_head_tokens(self, text):
        """
        문장에서 '[ORG]' 바로 오른쪽 단어(들)를 모두 리스트로 반환.
        """
        # 정규표현식으로 [ORG] 뒤 단어 추출
        # 예: "foo [ORG] bar baz [ORG] qux" → ['bar', 'qux']
        return re.findall(r'\[ORG\]\s*(\\w+)', text)

    def tokenize_and_encode(self, text):
        """
        문장을 BERT Tokenizer를 이용해 토큰화하고, Head-Token의 인덱스(들) 리스트를 반환.
        """
        head_tokens = self.extract_head_tokens(text)
        text = text.replace("[ORG]", "")
        tokens = self.tokenizer.tokenize(text)
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        token_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 각 head_token이 토큰화된 시퀀스에서 어디에 위치하는지 모두 찾아 리스트로 반환
        head_token_idx = []
        for ht in head_tokens:
            try:
                idx = tokens.index(ht) + 1  # [CLS]가 0번째이므로 +1
                head_token_idx.append(idx)
            except ValueError:
                continue  # 못 찾으면 무시

        if not head_token_idx:
            head_token_idx = [0]  # 없으면 [CLS]를 Head-Token으로

        return token_ids, head_token_idx, attention_mask
