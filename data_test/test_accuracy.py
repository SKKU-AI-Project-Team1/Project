import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import torch
import torchvision
from transformers import BertModel
from transformers import AlbertModel
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = 'kykim/bert-kor-base'

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='file path 받기')

# 입력받을 인자값 등록
parser.add_argument('--d', required=True, help='Test File Path')
parser.add_argument('--m', required=True, help='Model Path')
args = parser.parse_args()
data_path =args.d
model_path = args.m

print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')

# 시드 고정
random.seed(42)
# 토크나이저 관련 경고 무시하기 위하여 설정
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# device 지정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')




class FineTuningBertModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):
        # 부모클래스 초기화
        super().__init__()
        # 사전학습 모델 지정
        self.bert = BertModel.from_pretrained(bert_pretrained)

        # dropout 설정
        self.dropout = nn.Dropout(p=dropout_rate)
        # 최종 출력층 정의
        self.fc = nn.Linear(768, 489)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 입력을 pre-trained bert model 로 대입
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 결과의 last_hidden_state 가져옴
        last_hidden_state = output['last_hidden_state']
        # last_hidden_state[:, 0, :]는 [CLS] 토큰을 가져옴
        x = self.dropout(last_hidden_state[:, 0, :])
        # FC 을 거쳐 최종 출력
        x = self.fc(x)
        return x
    
# 저장한 state_dict를 로드 합니다.
BERT_model_test = FineTuningBertModel(MODEL_NAME).to(device)
BERT_model_test.load_state_dict(torch.load(model_path))  # 모델 가중치 불러오기


class Predictor():
    def __init__(self, model, tokenizer, labels: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels

    def predict(self, sentence):
        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )
        tokens.to(device)
        prediction = self.model(**tokens) 

        prediction = F.softmax(prediction, dim=1) 

        output = prediction.argmax(dim=1).item()

        prob, result = prediction.max(dim=1)[0].item(), self.labels[output]
        return result




tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

labels = {}
for i in range(489):
  labels[i] = i
predictor = Predictor(BERT_model_test, tokenizer, labels)

cm =[[0 for i in range(489)] for j in range(489)]
y_pred = []

def test(predictor, path, model_path):
    test_data=pd.read_csv(path)
    test_data=test_data[["abstract", "ipc_subclass_num"]]
    count =0
    answer_count=0
    print(f"total length : {len(test_data)}")
    for i in range(len(test_data)):
        count +=1
        if count%500==0:
            print(f"current {count}")
        sentence=test_data.iloc[i]["abstract"]
        answer = test_data.iloc[i]["ipc_subclass_num"]
        result=predictor.predict(sentence)
        cm[answer][result] +=1
        # y_pred.append(result)
        if answer==result:
            answer_count +=1
      
    prob = answer_count/count
    print(f'path : {path}, model : {model_path} test_accuracy는 {prob*100:.3f}% 입니다.')
    line = f'path : {path}, model : {model_path} test_accuracy: {prob*100:.3f}%\n'
    file = open("test_log.txt","a")
    file.write(line)
    file.close

test(predictor, data_path, model_path)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confunsion.jpg',format='jpeg')