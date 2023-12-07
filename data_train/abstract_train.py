import torch
import torchvision
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizerFast, AlbertModel
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig
from transformers import BertModel
from transformers import AlbertModel
from sklearn.model_selection import StratifiedKFold

print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')
# 토크나이저 관련 경고 무시하기 위하여 설정
os.environ["TOKENIZERS_PARALLELISM"] = 'true'

# device 지정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')


train = pd.read_csv('../../data/fillNAN_train_data.csv')
test = pd.read_csv('../../data/fillNAN_valid_data.csv')
print('train',train.shape)
print('test',test.shape)

MODEL_NAME = 'kykim/bert-kor-base'
# MODEL_NAME='kykim/albert-kor-base'




class PatentDataset(Dataset):

    def __init__(self, dataframe, tokenizer_pretrained):
        # sentence, label 컬럼으로 구성된 데이터프레임 전달
        self.data = dataframe
        # Huggingface 토크나이저 생성

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_pretrained)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # 클래스의 인덱스에 접근할 때 자동으로 호출되는 메서드
        # iloc : 행번호로 선택하는 방법
        sentence = self.data.iloc[idx]['abstract']
        label = self.data.iloc[idx]['ipc_subclass_num']

        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)       # 2D -> 1D : 1,512 -> 512
        attention_mask = tokens['attention_mask'].squeeze(0) # 2D -> 1D
        token_type_ids = torch.zeros_like(attention_mask)

        # input_ids, attention_mask, token_type_ids 이렇게 3가지 요소를 반환하도록 합니다.
        # input_ids: 토큰
        # attention_mask: 실제 단어가 존재하면 1, 패딩이면 0 (패딩은 0이 아닐 수 있습니다)
        # token_type_ids: 문장을 구분하는 id. 단일 문장인 경우에는 전부 0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)
    
# 토크나이저 지정
tokenizer_pretrained = MODEL_NAME

skf  =StratifiedKFold(n_splits=5)
train_data = None
test_data = None
for train_index, val_index in skf.split(train, train['ipc_subclass_num']):
    X_train=train['abstract'][val_index]
    Y_train = train['ipc_subclass_num'][val_index]
    train20_data=pd.concat([X_train, Y_train], axis=1)
    train_data = PatentDataset(train20_data, tokenizer_pretrained)
    break
for train_index, val_index in skf.split(test, test['ipc_subclass_num']):
    X_train = test['abstract'][val_index]
    Y_train = test['ipc_subclass_num'][val_index]
    test20_data=pd.concat([X_train, Y_train], axis=1)
    test_data = PatentDataset(test20_data, tokenizer_pretrained)
    break


# DataLoader로 이전에 생성한 Dataset를 지정하여, batch 구성, shuffle, num_workers 등을 설정합니다.
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8)


inputs, labels = next(iter(train_loader))
inputs = {k: v.to(device) for k, v in inputs.items()}
config = BertConfig.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME).to(device)
output = model(**inputs)
fc = nn.Linear(768, 2)
fc.to(device)
fc_output = fc(output['last_hidden_state'][:, 0, :])


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
    
BERT_model = FineTuningBertModel(MODEL_NAME)
BERT_model.to(device)

# loss 정의: CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# 옵티마이저 정의: bert.paramters()와 learning_rate 설정
optimizer = optim.Adam(BERT_model.parameters(), lr=1e-5)

from tqdm import tqdm  # Progress Bar 출력

def model_train(model, data_loader, loss_fn, optimizer, device):
    # 모델을 훈련모드로 설정합니다. training mode 일 때 Gradient 가 업데이트 됩니다. 반드시 train()으로 모드 변경을 해야 합니다.
    model.train()

    # loss와 accuracy 계산을 위한 임시 변수 입니다. 0으로 초기화합니다.
    running_loss = 0
    corr = 0
    counts = 0


    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)

    # mini-batch 학습
    for idx, (inputs, labels) in enumerate(prograss_bar):

        inputs = {k:v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)


        optimizer.zero_grad()

        output = model(**inputs)

        loss = loss_fn(output, labels)

        loss.backward()

        optimizer.step()

        _, index = output.max(dim=1)

        corr += index.eq(labels).sum().item()
        counts += len(labels)


        running_loss += loss.item() * labels.size(0)


        prograss_bar.set_description(f"training loss: {running_loss/(idx+1):.5f}, training accuracy: {corr / counts:.5f}")
    acc = corr / len(data_loader.dataset)


    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn, device):

    model.eval()

    with torch.no_grad():

        corr = 0
        running_loss = 0

        # evaluation진행
        for inputs, labels in data_loader:

            inputs = {k:v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)


            output = model(**inputs) # (8,2)


            _, index = output.max(dim=1)


            corr += torch.sum(index.eq(labels)).item()


            running_loss += loss_fn(output, labels).item() * labels.size(0)

        acc = corr / len(data_loader.dataset)


        return running_loss / len(data_loader.dataset), acc
    
# 최대 Epoch
num_epochs = 6

# checkpoint로 저장할 모델의 이름을 정의 합니다.
model_name = 'bert-kor-base'

min_loss = np.inf

# Epoch 별 훈련 및 검증을 수행합니다.
for epoch in range(num_epochs):
    # Model Training
    # 훈련 손실과 정확도를 반환 받습니다.
    train_loss, train_acc = model_train(BERT_model, train_loader, loss_fn, optimizer, device)

    # 검증 손실과 검증 정확도를 반환 받습니다.
    val_loss, val_acc = model_evaluate(BERT_model, test_loader, loss_fn, device)

    # val_loss 가 개선되었다면 min_loss를 갱신하고 model의 가중치(weights)를 저장합니다.
    if val_loss < min_loss:
        print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
        min_loss = val_loss
        torch.save(BERT_model.state_dict(), f'{model_name}.pth')

    # Epoch 별 결과를 출력합니다.
    print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
    line = f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}\n'
    file = open("log.txt","a")
    file.write(line)
    file.close
torch.save(BERT_model.state_dict(), 'abstract_base_model.pth')