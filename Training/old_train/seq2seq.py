import os
import re
import logging
from tqdm import tqdm
from datetime import datetime
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim


def preprocess_dialogue(dialogue):
    # 불필요한 기호 제거 및 소문자 변환
    return re.sub(r'[^a-zA-Z0-9\s]', '', dialogue).lower()

def load_and_preprocess_dataset():
    dataset = load_dataset("li2017dailydialog/daily_dialog")
    print("Train:", dataset['train'].num_rows)
    print("Validation:", dataset["validation"].num_rows)
    print("Test:", dataset["test"].num_rows)

    train_dialogues = []
    val_dialogues = []
    test_dialogues = []

    # 대화 내용, 대화 행위, 감정 레이블 통합
    for dialogue in dataset['train']:
        processed_dialogue = preprocess_dialogue(' '.join(dialogue['dialog']))
        train_dialogues.append((processed_dialogue, dialogue['act'], dialogue['emotion']))

    for dialogue in dataset['validation']:
        processed_dialogue = preprocess_dialogue(' '.join(dialogue['dialog']))
        val_dialogues.append((processed_dialogue, dialogue['act'], dialogue['emotion']))

    for dialogue in dataset['test']:
        processed_dialogue = preprocess_dialogue(' '.join(dialogue['dialog']))
        test_dialogues.append((processed_dialogue, dialogue['act'], dialogue['emotion']))

    return train_dialogues, val_dialogues, test_dialogues

train_dialogues, val_dialogues, test_dialogues = load_and_preprocess_dataset()

# 단어 사전 생성
vocab = set(word for dialogue, _, _ in train_dialogues for word in dialogue.split())
word_to_ix = {word: i + 1 for i, word in enumerate(vocab)}  # 0은 패딩을 위해 예약
word_to_ix['<PAD>'] = 0
word_to_ix['<UNK>'] = len(word_to_ix) + 1  # Unknown token 추가

# 레이블 인코딩
act_labels = set(act for _, acts, _ in train_dialogues for act in acts)  # acts에서 각 요소를 분리
emotion_labels = set(emotion for _, _, emotions in train_dialogues for emotion in emotions)  # emotions에서 각 요소를 분리

act_to_ix = {act: i for i, act in enumerate(act_labels)}
emotion_to_ix = {emotion: i for i, emotion in enumerate(emotion_labels)}

# 데이터 인코딩 함수
def encode_dataset(dialogues):
    encoded_data = []
    for dialogue, acts, emotions in dialogues:
        act = acts[0] if isinstance(acts, list) else acts
        emotion = emotions[0] if isinstance(emotions, list) else emotions
        encoded_dialogue = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in dialogue.split()]  # <UNK> 처리
        encoded_data.append((encoded_dialogue, act_to_ix[act], emotion_to_ix[emotion]))
    return encoded_data

encoded_train = encode_dataset(train_dialogues)
encoded_val = encode_dataset(val_dialogues)
encoded_test = encode_dataset(test_dialogues)


class DialogDataset(Dataset):
    def __init__(self, dialogues, word_to_ix):
        self.dialogues = dialogues
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue, act, emotion = self.dialogues[idx]
        dialogue_ix = [self.word_to_ix.get(w, self.word_to_ix['<PAD>']) for w in dialogue.split()]
        return torch.tensor(dialogue_ix), act, emotion

def collate_fn(batch):
    dialogues = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    acts = [item[1] for item in batch]
    emotions = [item[2] for item in batch]
    return dialogues, acts, emotions

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        output = self.fc(x)
        return output

# 로그 디렉토리 및 파일 설정
log_dir = 'Training/log'
os.makedirs(log_dir, exist_ok=True)  # 디렉토리 생성
log_filename = datetime.now().strftime('%y%m%d_%H%M%S_Seq2Seq.log')
log_filepath = os.path.join(log_dir, log_filename)

# 로깅 설정
logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(message)s')

# CUDA 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델, 손실 함수, 옵티마이저 정의 (모델을 CUDA로 이동)
model = Seq2Seq(vocab_size=len(word_to_ix), embedding_dim=128, hidden_dim=256).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 패딩 인덱스 무시
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)  # L2 정규화 추가

# 데이터셋 생성
train_dataset = DialogDataset(encode_dataset(train_dialogues), word_to_ix)
val_dataset = DialogDataset(encode_dataset(val_dialogues), word_to_ix)
test_dataset = DialogDataset(encode_dataset(test_dialogues), word_to_ix)

# DataLoader에 num_workers를 설정하여 병렬 데이터 로딩
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=8)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, num_workers=8)

# 정확도 계산 함수
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=2)  # 각 시간 단계에서 예측된 단어
    predicted = predicted[:, 1:]  # 첫 단어는 제외
    targets = targets[:, 1:]  # 첫 단어는 제외

    # 예측과 목표의 형태가 다를 경우, 목표를 예측과 같은 길이로 잘라냅니다.
    if predicted.size(1) != targets.size(1):
        min_len = min(predicted.size(1), targets.size(1))
        predicted = predicted[:, :min_len]
        targets = targets[:, :min_len]

    correct = (predicted == targets).float()  # 정답과 비교
    accuracy = correct.sum() / (targets != 0).sum()  # 패딩을 제외한 정확도
    return accuracy.item()

# 학습 루프
num_epochs = 50  # 에포크 수를 조정
min_loss_threshold = 0.2000  # 손실 기준 설정
epoch = 0

while epoch < num_epochs:
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    running_accuracy = 0.0
    
    # tqdm을 사용하여 진행 바 생성
    for dialogues, acts, emotions in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        # 데이터를 CUDA로 이동
        dialogues = dialogues.to(device)

        optimizer.zero_grad()
        outputs = model(dialogues[:, :-1])  # 마지막 단어 제외
        loss = criterion(outputs.view(-1, outputs.size(-1)), dialogues[:, 1:].contiguous().view(-1))  # 첫 단어 제외
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, dialogues)

    # 평균 손실과 정확도 계산
    avg_loss = running_loss / len(train_dataloader)
    avg_accuracy = running_accuracy / len(train_dataloader)

    log_message = f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}'
    print(log_message)  # 콘솔에 출력
    logging.info(log_message)  # 로그 파일에 기록

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_accuracy = 0.0
        for val_dialogues, val_acts, val_emotions in val_dataloader:
            val_dialogues = val_dialogues.to(device)
            val_outputs = model(val_dialogues[:, :-1])
            val_loss += criterion(val_outputs.view(-1, val_outputs.size(-1)), val_dialogues[:, 1:].contiguous().view(-1)).item()
            val_accuracy += calculate_accuracy(val_outputs, val_dialogues)
        val_loss /= len(val_dataloader)
        val_accuracy /= len(val_dataloader)

        log_message = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}'
        print(log_message)
        logging.info(log_message)

        # 최소 손실 기준 달성 시 학습 중단
        if val_loss < min_loss_threshold:
            print(f'Minimum loss threshold of {min_loss_threshold} reached. Training stopped.')
            break

        epoch += 1

    # 테스트 단계
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_accuracy = 0.0
        for test_dialogues, test_acts, test_emotions in test_dataloader:
            test_dialogues = test_dialogues.to(device)
            test_outputs = model(test_dialogues[:, :-1])
            test_loss += criterion(test_outputs.view(-1, test_outputs.size(-1)), test_dialogues[:, 1:].contiguous().view(-1)).item()
            test_accuracy += calculate_accuracy(test_outputs, test_dialogues)
        test_loss /= len(test_dataloader)
        test_accuracy /= len(test_dataloader)

        log_message = f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}'
        print(log_message)
        logging.info(log_message)

# 모델 저장
torch.save(model.state_dict(), 'GPT/model_pytorch/seq2seq_model.pth')
