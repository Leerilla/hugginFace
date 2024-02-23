# DOCS Task Guides의 Text Classification을 확인한다.
# loda lMdb dataSet 테스트를 위한 데이터 로드
from datasets import load_dataset

imdb = load_dataset("imdb")

# Preprocess
# 데이터 분석이나 머신러닝 모델 학습을 위해 데이터를 준비하는 과정이다.
# 데이터를 모델에 적합한 형식으로 변환하고, 모델 학습에 방해가 되는 문제를 해결하기 위해서 수행하는 과정이다.
# 주로 데이터를 모델에 적합한 형식으로 변환하고 모델 학습에 중요한 특징을 추출한다.
from transformers import AutoTokenizer,DataCollatorWithPadding 

# BERT 모델은 자연어 처리 분야에서 널리 사용하는 강력한 모델이지만, 모델의 크기가 크고 학습속도가 느린 단점을 가지고 있다.
# 이를 개선하기 위해 google에서 distilBert 모델을 개발하여 학습속도를 줄였다.
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

# 텍스트 토큰화 : 텍스트를 의미 있는 단위로 분리한다.
# 시퀀스 길이 자르기 : 텍스트의 토큰 수를 모델이 처리할 수 있는 최대 길이 이하로 제한 하는 과정이다.
def preprocess_function(examples):
    return tokenizer(examples["text"], max_length=512,truncation=True)

# 위 과정에서 자른 시퀀스를 map 함수에 전달하여 데이터 셋을 준비하는 과정이다.
# batched를 통해 데이터 셋에 요소를 한 번에 여러 개씩 처리할 수 있도록 적용한다.
tokenized_imdb = imdb.map(preprocess_function, batched=True)

# 콜레이션 : 데이터셋의 여러 요소들을 하나의 배치로 합쳐 모델에 입력을 제공하는 과정
# 동적 패딩 :  시퀀스 데이터의 길이가 다를 때 모델에 입력으로 제공하기 전에 시퀀스들을 일괄적으로 동일한 길이로 만드는 방법이다.

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

from transformers import pipeline
model_name = "my_awesome_model"  # trainer로 사용된 모델의 이름
classifier = pipeline("sentiment-analysis", model=model_name)
print(classifier(text))