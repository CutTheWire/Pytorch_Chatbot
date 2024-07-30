import os
import torch
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Tuple

# CUDA 사용 가능 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# CPU 코어 수 확인 및 설정
num_workers = 8

# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

# T5 토크나이저 초기화
tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)

# 학습 설정
num_epochs = 10  # 에폭 수 감소
best_val_loss = float('inf')
patience = 3  # 조기 종료를 위한 인내심
no_improve = 0
output_dir = "saved_model"

# BLEU 점수 계산 설정
smoother = SmoothingFunction().method1
weights = (0.5, 0.3, 0.2, 0)  # 1-gram, 2-gram, 3-gram에 가중치 부여, 4-gram은 제외

# 그래디언트 스케일러 초기화 (혼합 정밀도 훈련용)
scaler = GradScaler()

class DialogDataset(Dataset):
    """
    대화 데이터셋을 처리하는 커스텀 데이터셋 클래스
    """
    def __init__(self, dialogues: List[list], acts: List[list], emotions: List[list], tokenizer: T5Tokenizer, max_length: int = 36):
        self.dialogues = dialogues
        self.acts = acts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_data = self.preprocess_data()

    def preprocess_data(self) -> List[dict]:
        """
        대화 데이터를 토크나이저를 사용하여 인코딩합니다.
        """
        encoded_data = []
        for dialogue, act, emotion in zip(self.dialogues, self.acts, self.emotions):
            input_text = "dialogue: " + " ".join(dialogue[:-1])
            target_text = dialogue[-1]

            for i, (a, e) in enumerate(zip(act, emotion)):
                input_text += f" turn{i+1}_act: {a} turn{i+1}_emotion: {e}"

            input_encoding = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            encoded_data.append({
                "input_ids": input_encoding.input_ids.flatten(),
                "attention_mask": input_encoding.attention_mask.flatten(),
                "labels": target_encoding.input_ids.flatten(),
            })
        return encoded_data

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> dict:
        return self.encoded_data[idx]

def augment_data(dialogue: List[str], act: List[int], emotion: List[int]) -> Tuple[List[str], List[int], List[int]]:
    """
    간단한 데이터 증강: 대화의 순서를 뒤집음
    """
    return dialogue[::-1], act[::-1], emotion[::-1]

class T5DialogTrainer:
    """
    T5 모델을 학습하고 평가하는 클래스
    """
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        self.scheduler = None
        self.scaler = GradScaler('cuda')

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, output_dir: str, patience: int):
        """
        모델을 학습하고 검증합니다.
        """
        best_val_loss = float('inf')
        no_improve = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            total_acc = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                pred_ids = torch.argmax(logits, dim=-1)
                acc = (pred_ids == labels).float().mean()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_acc += acc.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})

            avg_train_loss = total_loss / len(train_loader)
            avg_train_acc = total_acc / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}, Average Train Accuracy: {avg_train_acc:.4f}")

            avg_val_loss, avg_val_acc, avg_bleu = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {avg_val_loss:.4f}, Average Validation Accuracy: {avg_val_acc:.4f}, Average BLEU: {avg_bleu:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                print("Saving best model...")
                self.save_model(output_dir, "best_t5_dialog_model")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        print("Training complete!")

    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """
        모델을 검증합니다.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_acc = 0
        total_bleu = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                pred_ids = torch.argmax(logits, dim=-1)
                acc = (pred_ids == labels).float().mean()

                pred_text = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                label_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                batch_bleu = 0
                for pred, label in zip(pred_text, label_text):
                    batch_bleu += sentence_bleu([label.split()], pred.split(), smoothing_function=smoother, weights=weights)
                batch_bleu /= len(pred_text)

                total_val_loss += loss.item()
                total_val_acc += acc.item()
                total_bleu += batch_bleu

        avg_val_loss = total_val_loss / len(loader)
        avg_val_acc = total_val_acc / len(loader)
        avg_bleu = total_bleu / len(loader)

        return avg_val_loss, avg_val_acc, avg_bleu

    def save_model(self, output_dir: str, model_name: str):
        """
        모델을 저장합니다.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_save_path = os.path.join(output_dir, model_name)
        self.model.module.save_pretrained(model_save_path) if torch.cuda.device_count() > 1 else self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")

def prepare_data(dataset) -> Tuple[DialogDataset, DialogDataset]:
    """
    데이터셋을 분리하여 학습용 및 검증용 데이터를 준비합니다.
    """
    dialogues = [dialogue['dialog'] for dialogue in dataset['train']]
    acts = [dialogue['act'] for dialogue in dataset['train']]
    emotions = [dialogue['emotion'] for dialogue in dataset['train']]

    train_dialogues = dialogues[:int(len(dialogues) * 0.8)]
    val_dialogues = dialogues[int(len(dialogues) * 0.8):]
    train_acts = acts[:int(len(acts) * 0.8)]
    val_acts = acts[int(len(acts) * 0.8):]
    train_emotions = emotions[:int(len(emotions) * 0.8)]
    val_emotions = emotions[int(len(emotions) * 0.8):]

    train_data = DialogDataset(train_dialogues, train_acts, train_emotions, tokenizer)
    val_data = DialogDataset(val_dialogues, val_acts, val_emotions, tokenizer)

    return train_data, val_data

if __name__ == "__main__":
    # 데이터 준비
    train_data, val_data = prepare_data(dataset)

    # 데이터 로더 준비
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=num_workers)

    # 모델 초기화
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # 학습기 초기화 및 학습 스케줄러 설정
    trainer = T5DialogTrainer(model, tokenizer, device)
    num_training_steps = len(train_loader) * num_epochs
    trainer.scheduler = get_linear_schedule_with_warmup(trainer.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # 학습 시작
    trainer.train(train_loader, val_loader, num_epochs, output_dir, patience)
