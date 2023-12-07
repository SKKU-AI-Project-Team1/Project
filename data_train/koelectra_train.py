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
from transformers import AutoTokenizer, AutoModel


print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')

os.environ["TOKENIZERS_PARALLELISM"] = 'true'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스: {device}')


train = pd.read_csv('../../data/sampled20_train_data.csv')
test = pd.read_csv('../../data/valid_data.csv')
print('train',train.shape)
print('test',test.shape)


MODEL_NAME='beomi/KcELECTRA-base-v2022'




class PatentDataset(Dataset):

    def __init__(self, dataframe, tokenizer_pretrained):
        self.data = dataframe

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['invention_title']
        label = self.data.iloc[idx]['ipc_subclass_num']

        # 토큰화 처리
        tokens = self.tokenizer(
            sentence,                # 1개 문장
            return_tensors='pt',     # 텐서로 반환
            truncation=True,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )

        input_ids = tokens['input_ids'].squeeze(0)      
        attention_mask = tokens['attention_mask'].squeeze(0) 
        token_type_ids = torch.zeros_like(attention_mask)


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }, torch.tensor(label)
    

tokenizer_pretrained = MODEL_NAME


train_data = PatentDataset(train, tokenizer_pretrained)
test_data = PatentDataset(test, tokenizer_pretrained)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=8)
print('train_loader',train_loader)
print('test_loader',test_loader)


inputs, labels = next(iter(train_loader))


inputs = {k: v.to(device) for k, v in inputs.items()}





# 모델 생성
model = AutoModel.from_pretrained(MODEL_NAME).to(device)


output = model(**inputs)

fc = nn.Linear(768, 2)
fc.to(device)
fc_output = fc(output['last_hidden_state'][:, 0, :])
print(fc_output.shape)
print(fc_output)
print(fc_output.argmax(dim=1))

class FineTuningElectraModel(nn.Module):
    def __init__(self, bert_pretrained, dropout_rate=0.5):

        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_pretrained)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(768, 489)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = output['last_hidden_state']
        x = self.dropout(last_hidden_state[:, 0, :])
        x = self.fc(x)
        return x
    
BERT_model = FineTuningElectraModel(MODEL_NAME)
BERT_model.to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(BERT_model.parameters(), lr=1e-5)

from tqdm import tqdm 

def model_train(model, data_loader, loss_fn, optimizer, device):
    model.train()

    running_loss = 0
    corr = 0
    counts = 0

    prograss_bar = tqdm(data_loader, unit='batch', total=len(data_loader), mininterval=1)


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


        for inputs, labels in data_loader:

            inputs = {k:v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)


            output = model(**inputs) # (8,2)


            _, index = output.max(dim=1)

            corr += torch.sum(index.eq(labels)).item()


            running_loss += loss_fn(output, labels).item() * labels.size(0)

        # validation 정확도 계산
        acc = corr / len(data_loader.dataset)


        return running_loss / len(data_loader.dataset), acc
    
# 최대 Epoch을 지정합니다.
num_epochs = 6

# checkpoint로 저장할 모델의 이름을 정의 합니다.
model_name = 'koelectra-base'

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
    print(f'koelectra epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')
    line = f'koelectra epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}\n'
    file = open("log.txt","a")
    file.write(line)
    file.close
torch.save(BERT_model.state_dict(), 'koelectra_base_model.pth')