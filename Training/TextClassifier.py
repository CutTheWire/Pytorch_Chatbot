import os
import logging
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")
print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))
print("Test:", len(dataset["test"]))

# 데이터 전처리
def preprocess_data(data):
    dialogues = [' '.join(dialogue['dialog']) for dialogue in data]
    acts = [dialogue['act'] for dialogue in data]
    emotions = [dialogue['emotion'] for dialogue in data]
    return dialogues, acts, emotions

train_dialogues, train_acts, train_emotions = preprocess_data(dataset['train'])
val_dialogues, val_acts, val_emotions = preprocess_data(dataset['validation'])
test_dialogues, test_acts, test_emotions = preprocess_data(dataset['test'])

# 단어 사전 생성
vocab = set(' '.join(train_dialogues).split())
word_to_ix = {word: i+1 for i, word in enumerate(vocab)}  # 0은 패딩을 위해 예약
word_to_ix['<PAD>'] = 0

class DialogDataset(Dataset):
    def __init__(self, dialogues, acts, emotions, word_to_ix):
        self.dialogues = dialogues
        self.acts = acts
        self.emotions = emotions
        self.word_to_ix = word_to_ix

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx].split()
        dialogue_ix = [self.word_to_ix.get(w, self.word_to_ix['<PAD>']) for w in dialogue]
        act = torch.tensor(self.acts[idx][0], dtype=torch.long)  # 첫 번째 act만 사용
        emotion = torch.tensor(self.emotions[idx][0], dtype=torch.long)  # 첫 번째 emotion만 사용
        return torch.tensor(dialogue_ix), act, emotion

def collate_fn(batch):
    dialogues, acts, emotions = zip(*batch)
    dialogues_padded = pad_sequence(dialogues, batch_first=True, padding_value=0)
    return dialogues_padded, torch.stack(acts), torch.stack(emotions)

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
    
# 정확도 계산 함수
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

# 로그 디렉토리 및 파일 설정
log_dir = 'Training/log'
os.makedirs(log_dir, exist_ok=True)  # 디렉토리 생성
log_filename = datetime.now().strftime('%y%m%d_%H%M%S_TextClassifier.log')
log_filepath = os.path.join(log_dir, log_filename)

# 로깅 설정
logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(message)s')

# CUDA 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델, 손실 함수, 옵티마이저 정의 (모델을 CUDA로 이동)
model = TextClassifier(vocab_size=len(word_to_ix), embedding_dim=128, hidden_dim=256, num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터셋 생성
train_dataset = DialogDataset(train_dialogues, train_acts, train_emotions, word_to_ix)
val_dataset = DialogDataset(val_dialogues, val_acts, val_emotions, word_to_ix)
test_dataset = DialogDataset(test_dialogues, test_acts, test_emotions, word_to_ix)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# 학습 루프
num_epochs = 99999
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
        acts = acts.to(device)
        emotions = emotions.to(device)

        optimizer.zero_grad()
        outputs = model(dialogues)
        loss = criterion(outputs, acts)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, acts)

    # 평균 손실과 정확도 계산
    avg_loss = running_loss / len(train_dataloader)
    avg_accuracy = running_accuracy / len(train_dataloader)

    log_message = f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}'
    print(log_message)  # 콘솔에 출력
    logging.info(log_message)  # 로그 파일에 기록

    # 손실이 기준 이하로 떨어지면 학습 종료
    if avg_loss < min_loss_threshold:
        stop_message = f'Loss has reached below {min_loss_threshold:.4f}. Stopping training.'
        print(stop_message)
        logging.info(stop_message)
        break

    epoch += 1  # 에포크 증가

# 모델 저장
torch.save(model.state_dict(), 'GPT/model_pytorch/text_classifier.pth')