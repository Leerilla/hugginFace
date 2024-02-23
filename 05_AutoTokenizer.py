# 코드 설명
# 위 코드는 "nlptown/bert-base-multilingual-uncased-sentiment"을 사용하여 
# "We are very happy to show you the 🤗 Transformers library." 문장을 토큰화 하는 코드이다.
# 토크나이저는 문장을 모델이 이해할 수 있는 토큰으로 변환하는 역할을 한다.

# 모든 모델이 네이밍 규칙을 따르는 것은 아니며 해당 모델에서 이러한 의미로 네이밍을 가졌다.
# 모델 네이밍 
# nlptown/bert-base-multilingual-uncased-sentiment
# nlptown : 모델을 만든 제작자를 의미한다.
# bert(Bidirectional Encoder Representations from Tansformers) : 모델 아키텍처를 나타낸다.
# base : 모델의 크기를 의미하며 기본 크기의 모델이다.
# multilingual : 모델이 어려 언어를 지원한다는 것을 의미함
# uncased : 대소문자 구분을 하지 않는다.
# sentiment : 모델이 감정 분석에 특화되어 있다는 것을 의미함

from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)
