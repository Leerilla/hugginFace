text = "I think I felt very bad after seeing this."
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# Preprocess (프리프로세싱) : 전처리를 의미
# 모델에 입력으로 전달되는 텍스트 데이터를 사전에 처리하는 단계
# 이 단계에서는 텍스트를 모델이 이해할 수 있는 형식으로 변환하기 위해 토큰화가 수행됨
model_name = "stevhliu/my_awesome_model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 레이블 정보 지정 (포스트프로세싱)
# 모델의 출력 결과를 해석하기 쉽도록 레이블 정보를 설정하는 단계
# 모델이 출력한 숫자형 결과를 사람이 이해할 수 있는 레이블로 변환함
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Inference (인퍼런스)
# 학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행하는 단계
# 이 단계에서는 텍스트 데이터를 입력으로 모델에 전달하고 감정 분석 결과를 반환함
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


results = classifier(text)
print(results)
