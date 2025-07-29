from transformers import pipeline
import torch
import random
import pandas as pd

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

    def analyze_dataset_ner_coverage(self, data_file):
        """
        데이터셋에서 NER 태깅된 엔티티가 존재하는 데이터의 개수와 비율을 분석
        """
        print(f"=== NER Coverage Analysis for {data_file} ===")
        
        # 데이터 파일 읽기
        if ".tsv" in data_file:
            data = pd.read_csv(data_file, delimiter="\t")
        else:
            data = pd.read_csv(data_file)
        
        total_samples = len(data)
        samples_with_entities = 0
        total_entities = 0
        
        # 데이터셋 타입에 따라 텍스트 컬럼 결정
        if "ihc" in data_file:
            text_column = "post"
        elif "sbic" in data_file:
            text_column = "post"
        elif "dyna" in data_file:
            text_column = "text"
        elif "SST" in data_file:
            text_column = "sentence"
        elif "IMDB" in data_file:
            text_column = "review"
        else:
            text_column = "post"
        
        print(f"Analyzing {total_samples} samples...")
        
        # 각 샘플에 대해 NER 태깅 수행
        for idx, row in data.iterrows():
            if idx % 1000 == 0:  # 진행상황 출력
                print(f"Processing... {idx}/{total_samples}")
                
            text = row[text_column]
            entities = self.extract_named_entities(text)
            
            if entities:  # 엔티티가 하나라도 있으면
                samples_with_entities += 1
                total_entities += len(entities)
        
        # 결과 계산 및 출력
        coverage_ratio = samples_with_entities / total_samples * 100
        avg_entities_per_sample = total_entities / total_samples
        avg_entities_per_tagged_sample = total_entities / samples_with_entities if samples_with_entities > 0 else 0
        
        print(f"\n=== NER Coverage Analysis Results ===")
        print(f"Total samples: {total_samples}")
        print(f"Samples with NER entities: {samples_with_entities}")
        print(f"Samples without NER entities: {total_samples - samples_with_entities}")
        print(f"Coverage ratio: {coverage_ratio:.2f}%")
        print(f"Total entities found: {total_entities}")
        print(f"Average entities per sample: {avg_entities_per_sample:.2f}")
        print(f"Average entities per tagged sample: {avg_entities_per_tagged_sample:.2f}")
        print("=" * 40)
        
        return {
            "total_samples": total_samples,
            "samples_with_entities": samples_with_entities,
            "coverage_ratio": coverage_ratio,
            "total_entities": total_entities,
            "avg_entities_per_sample": avg_entities_per_sample,
            "avg_entities_per_tagged_sample": avg_entities_per_tagged_sample
        }


class NERProcessor:
    def __init__(self, tokenizer, data_length, ner_tagger=None, use_ner=True, head_token_ratio=0.2):
        self.tokenizer = tokenizer
        self.ner_tagger = ner_tagger
        self.use_ner = use_ner
        self.head_token_ratio = head_token_ratio
        
        # 전체 데이터 중 head_token을 적용할 샘플들의 인덱스를 미리 정의
        num_head_token_samples = int(data_length * head_token_ratio)
        self.head_token_sample_indices = set(random.sample(range(data_length), num_head_token_samples))
        
        # 현재 처리 중인 샘플의 인덱스를 추적
        self.current_sample_idx = 0

    def extract_head_tokens(self, text):
        """
        NER을 적용하여 특정 개체명들을 Head-Token으로 선정.
        Head-Token이 없으면 빈 리스트 반환.
        """
        if not self.use_ner:
            return []
        entities = self.ner_tagger.extract_named_entities(text)
        return entities  # 여러 개체명 모두 반환

    def tokenize_and_encode(self, text):
        """
        문장을 BERT Tokenizer를 이용해 토큰화하고, Head-Token들의 인덱스 리스트를 반환.
        미리 정의된 20%의 샘플에 대해서만 head_token을 부여하며, 부여할 때는 토큰 중에서 랜덤으로 하나만 선택.
        """
        tokens = self.tokenizer.tokenize(text)
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=512)
        token_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # 현재 샘플이 미리 정의된 20%에 해당하는지 확인
        if self.current_sample_idx in self.head_token_sample_indices:
            # 실제 토큰들의 인덱스 범위 찾기 (특수 토큰 제외)
            # [CLS]는 0번, [SEP]와 [PAD]는 제외하고 실제 텍스트 토큰들 중에서 선택
            valid_indices = []
            for i, token_id in enumerate(token_ids):
                # [CLS]: 101, [SEP]: 102, [PAD]: 0 (BERT 기본 토큰 ID)
                if token_id not in [0, 101, 102] and attention_mask[i] == 1:
                    valid_indices.append(i)
            
            if valid_indices:
                # 실제 토큰들 중에서 랜덤으로 하나 선택
                head_token_idx = [random.choice(valid_indices)]
            else:
                # 유효한 토큰이 없으면 [CLS] 사용
                head_token_idx = [0]
        else:
            # 80%의 경우 [CLS]를 기본값으로 설정
            head_token_idx = [0]

        # 다음 샘플을 위해 인덱스 증가
        self.current_sample_idx += 1

        return token_ids, head_token_idx, attention_mask
