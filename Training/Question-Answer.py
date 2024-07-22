import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import os
from datetime import datetime


# 데이터셋 클래스 정의
class ChatbotDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx]
        input_encoding = self.tokenizer.encode_plus(
            item['input'],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        target_encoding = self.tokenizer.encode_plus(
            item['output'],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'target_ids': target_encoding['input_ids'].flatten(),
            'target_attention_mask': target_encoding['attention_mask'].flatten()
        }

    def __len__(self):
        return len(self.data)

# 모델 정의
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size):
        super(ChatbotModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.LSTM(768, 768, batch_first=True)
        self.fc_out = nn.Linear(768, vocab_size)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        decoder_output, _ = self.decoder(bert_output.last_hidden_state)
        output = self.fc_out(decoder_output)
        return output
    
# 로그 파일 경로 생성
def create_log_file():
    os.makedirs("Training/log", exist_ok=True)
    current_time = datetime.now().strftime("%y%m%d_%H")
    log_file_path = f"Training/log/{current_time}_Question-Answer_Training.log"
    return log_file_path

# 로그 파일에 쓰기
def log_training_progress(log_file_path, message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    NUM_EPOCHS = 1000
    LEARNING_RATE = 5e-5

    # JSON 파일 로드
    with open('reddit/data/model_openai_responses.json', 'r') as f:
        data = json.load(f)

    print(f"총 데이터 개수: {len(data)}")
    # BERT 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 데이터셋과 데이터로더 생성
    dataset = ChatbotDataset(data, tokenizer)
    def collate_fn(batch):
        return {
            'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
            'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0),
            'target_ids': pad_sequence([item['target_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
            'target_attention_mask': pad_sequence([item['target_attention_mask'] for item in batch], batch_first=True, padding_value=0)
        }
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=10, pin_memory=True, collate_fn=collate_fn)
    
    # 모델, 손실 함수, 옵티마이저 정의
    model = ChatbotModel(tokenizer.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    # 로그 파일 경로 생성
    log_file_path = create_log_file()
    def monitor_training_progress(epoch, num_epochs, loss, start_time, device):
        current_time = time.time()
        elapsed_time = current_time - start_time
        gpu_memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        gpu_memory_cached = torch.cuda.max_memory_cached(device) / (1024 ** 2)
        gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        gpu_memory_utilization = ((gpu_memory_used + gpu_memory_cached) / gpu_memory_total) * 100
        # 로그 메시지 생성
        log_message = (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, "
                       f"Elapsed Time: {elapsed_time:.2f} seconds, "
                       f"GPU Memory Used: {gpu_memory_used:.2f} MB, "
                       f"GPU Memory Cached: {gpu_memory_cached:.2f} MB, "
                       f"Total GPU Memory Utilization: {gpu_memory_utilization:.2f}%")
        
        print(log_message)  # 콘솔에 출력
        log_training_progress(log_file_path, log_message)  # 파일에 기록

    # 학습 루프
    device = torch.device('cuda' if torch.cuda.is_available() else sys.exit(1))
    model.to(device)

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_attention_mask = batch['target_attention_mask'].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        monitor_training_progress(epoch, NUM_EPOCHS, avg_loss, start_time, device)

        if avg_loss <= 0.2:
            print(f"Loss reached the target of 0.2. Stopping training.")
            break

    # 모델 저장
    torch.save(model.state_dict(), 'GPT/model_pytorch/chatbot_model.pth')