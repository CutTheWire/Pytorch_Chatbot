import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from torch.amp import autocast, GradScaler

# CUDA 사용 가능 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# CPU 코어 수 확인 및 설정
num_workers = 8  # 8개의 CPU 코어 사용

# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

# T5 토크나이저와 모델 초기화
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# 커스텀 데이터셋 클래스 정의
class DialogDataset(Dataset):
    def __init__(self, dialogues, acts, emotions, tokenizer, max_length=35): # max_length = 최대 토큰 길이
        self.dialogues = dialogues
        self.acts = acts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        act = self.acts[idx]
        emotion = self.emotions[idx]

        input_text = "dialogue: " + " ".join(dialogue[:-1])
        target_text = dialogue[-1]

        for i, (a, e) in enumerate(zip(act[:-1], emotion[:-1])):
            input_text += f" turn{i+1}_act: {a} turn{i+1}_emotion: {e}"

        input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": input_encoding.input_ids.flatten(),
            "attention_mask": input_encoding.attention_mask.flatten(),
            "labels": target_encoding.input_ids.flatten(),
        }

# 데이터셋 생성
train_dataset = DialogDataset(dataset['train']['dialog'], dataset['train']['act'], dataset['train']['emotion'], tokenizer)
val_dataset = DialogDataset(dataset['validation']['dialog'], dataset['validation']['act'], dataset['validation']['emotion'], tokenizer)
test_dataset = DialogDataset(dataset['test']['dialog'], dataset['test']['act'], dataset['test']['emotion'], tokenizer)

# DataLoader 생성 (배치 크기 증가)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=True)

# 옵티마이저 설정
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 그래디언트 스케일러 초기화 (혼합 정밀도 훈련용)
scaler = GradScaler('cuda')

# 학습 루프
num_epochs = 200
best_val_loss = float('inf')
best_val_acc = 0.0 
accumulation_steps = 4  # 그래디언트 누적 스텝 수
output_dir = "saved_model"  # 모델 저장 경로

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_acc = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for i, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # 혼합 정밀도 훈련
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
            logits = outputs.logits

        # 정확도 계산
        pred_ids = torch.argmax(logits, dim=-1)
        acc = (pred_ids == labels).float().mean()

        # 그래디언트 스케일링 및 역전파
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_acc += acc.item() * accumulation_steps
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_acc:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            # 정확도 계산
            pred_ids = torch.argmax(logits, dim=-1)
            acc = (pred_ids == labels).float().mean()

            total_val_loss += loss.item()
            total_val_acc += acc.item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.4f}")

    # 최고 성능 모델 저장
    if avg_val_acc > best_val_acc:
        best_val_loss = avg_val_loss
        best_val_acc = avg_val_acc
        print("Saving best model...")
        torch.save(model.state_dict(), os.path.join(output_dir, "best_t5_dialog_model.pth"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best_t5_dialog_model"))

print("모델 학습 완료!")

# 테스트 데이터로 평가
model.eval()
total_test_loss = 0
total_test_acc = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        # 정확도 계산
        pred_ids = torch.argmax(logits, dim=-1)
        acc = (pred_ids == labels).float().mean()

        total_test_loss += loss.item()
        total_test_acc += acc.item()

avg_test_loss = total_test_loss / len(test_loader)
avg_test_acc = total_test_acc / len(test_loader)
print(f"Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_acc:.4f}")

# 모델 테스트
print("모델 테스트 중...")
model.eval()
test_input = "dialogue: Hello, how are you? I'm fine, thank you. How about you? turn1_act: 3 turn1_emotion: 0 turn2_act: 4 turn2_emotion: 0"
input_ids = tokenizer(test_input, return_tensors="pt").input_ids.to(device)

with autocast('cuda'):
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("생성된 대화:")
print(generated_text)

# 대화 생성 함수
@torch.no_grad()
def generate_response(dialogue_history, acts, emotions):
    input_text = "dialogue: " + " ".join(dialogue_history)
    for i, (act, emotion) in enumerate(zip(acts, emotions)):
        input_text += f" turn{i+1}_act: {act} turn{i+1}_emotion: {emotion}"
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    with autocast('cuda'):
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 대화 테스트
print("\n대화 테스트:")
dialogue_history = ["Hello, how are you?", "I'm fine, thank you. How about you?"]
acts = [3, 4]
emotions = [0, 0]

for _ in range(3):  # 3턴의 대화 생성
    response = generate_response(dialogue_history, acts, emotions)
    print(f"Model: {response}")
    dialogue_history.append(response)
    acts.append(0)  # 임의의 act 값
    emotions.append(0)  # 임의의 emotion 값
    
    user_input = input("You: ")
    dialogue_history.append(user_input)
    acts.append(0)  # 임의의 act 값
    emotions.append(0)  # 임의의 emotion 값

# 멀티 GPU 사용을 위한 설정 (선택적)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    # JIT 컴파일을 위한 모델 최적화 (선택적)
    model = torch.jit.script(model)

print("최적화된 모델로 학습 및 추론 완료!")
