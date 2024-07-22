import re
import os
import json
import urllib.request
from dotenv import load_dotenv
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# .env 파일에서 환경 변수 로드
load_dotenv()

X_NCP_ID =  os.getenv('X_NCP_ID')
X_NCP_KEY = os.getenv('X_NCP_KEY')

response_data = {
    "source": "en",
    "target": "ko",
}

# headers 정의
headers = {
    "X-NCP-APIGW-API-KEY-ID": X_NCP_ID,
    "X-NCP-APIGW-API-KEY": X_NCP_KEY
}

# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

# 단어 사전 생성
all_dialogues = ' '.join([' '.join(dialog['dialog']) for dialog in dataset['train']])
vocab = set(all_dialogues.split())
word_to_ix = {word: i+1 for i, word in enumerate(vocab)}
word_to_ix['<PAD>'] = 0
vocab_size = len(word_to_ix)

# TextClassifier 모델 정의 (학습 시 사용한 것과 동일해야 함)
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        output = self.fc(x[:, -1, :])
        return output

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextClassifier(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256, num_classes=7).to(device)
model.load_state_dict(torch.load('GPT/model_pytorch/text_classifier.pth'))
model.eval()  # 평가 모드로 설정

# Dialog Act 매핑
act_mapping = {
    0: 'inform',
    1: 'question',
    2: 'directive',
    3: 'commissive',
    4: 'expressive'
}

def preprocess_input(text):
    # 입력 텍스트를 전처리하는 함수
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # 구두점 제거
    return text

def text_to_tensor(text, word_to_ix):
    # 텍스트를 텐서로 변환하는 함수
    words = text.split()
    indices = [word_to_ix.get(word, word_to_ix['<PAD>']) for word in words]
    return torch.tensor([indices])

def predict_dialog_act(text):
    # 입력 텍스트의 Dialog Act를 예측하는 함수
    preprocessed_text = preprocess_input(text)
    text_tensor = text_to_tensor(preprocessed_text, word_to_ix)
    text_tensor = text_tensor.to(device)

    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
        dialog_act = act_mapping[predicted.item()]

    return dialog_act

def naver_translation(dialog_act):
    response_data_str = f"source={response_data['source']}&target={response_data['target']}&text={dialog_act}"
    url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"

    request = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(request, data=response_data_str.encode("utf-8"))
    rescode = response.getcode()

    if rescode == 200:
        response_body = response.read()
        response_json = json.loads(response_body.decode('utf-8'))
        translated_text = response_json['message']['result']['translatedText']
        return translated_text
    else:
        return f"Error Code: {str(rescode)}"


# 채팅봇 실행
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    dialog_act = predict_dialog_act(user_input)
    output = f"{dialog_act} type"
    print(
        f"bot : {output}\n"
        f"bot : {naver_translation(output)}"
    )