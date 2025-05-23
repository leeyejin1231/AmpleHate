from transformers import pipeline
import torch

class NERTagger:
    def __init__(self, model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
        device = torch.device('cuda')
        self.ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)

    def extract_named_entities(self, text):
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
        if not self.use_ner:
            return []
        entities = self.ner_tagger.extract_named_entities(text)
        return entities 

    def tokenize_and_encode(self, text):
        head_tokens = self.extract_head_tokens(text)
        tokens = self.tokenizer.tokenize(text)
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        token_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        head_token_idx = []
        for ht in head_tokens:
            try:
                idx = tokens.index(ht) + 1  # [CLS] = 0
                head_token_idx.append(idx)
            except ValueError:
                continue  

        if not head_token_idx:
            head_token_idx = [0]

        return token_ids, head_token_idx, attention_mask
