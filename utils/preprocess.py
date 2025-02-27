from transformers import pipeline
import torch

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

    def extract_head_token(self, text):
        """
        NER을 적용하여 특정 개체명을 Head-Token으로 선정.
        Head-Token이 없으면 [CLS]를 사용.
        """
        if not self.use_ner:
            return None
        
        entities = self.ner_tagger.extract_named_entities(text)
        if len(entities) > 0:
            return entities[0]  # 첫 번째 개체명을 Head-Token으로 사용
        return None  # 개체명이 없는 경우

    def tokenize_and_encode(self, text):
        """
        문장을 BERT Tokenizer를 이용해 토큰화하고, Head-Token의 인덱스를 반환.
        """
        head_token = self.extract_head_token(text)
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text, truncation=True, padding="max_length", max_length=512)

        # Head-Token이 존재하는 경우 인덱스를 찾음
        if head_token:
            try:
                head_token_idx = tokens.index(head_token) + 1  # [CLS]가 0번째이므로 +1
            except ValueError:
                head_token_idx = 0  # 찾지 못한 경우 기본값 ([CLS]) 설정
        else:
            head_token_idx = 0  # 개체명이 없으면 [CLS]를 Head-Token으로 설정

        return token_ids, head_token_idx
