import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
import time
from torch.cuda.amp import autocast, GradScaler

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

        output_texts = item['output']
        target_encodings = [
            self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            ) for text in output_texts
        ]

        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'target_ids': torch.stack([encoding['input_ids'].flatten() for encoding in target_encodings]),
            'target_attention_mask': torch.stack([encoding['attention_mask'].flatten() for encoding in target_encodings])
        }

    def __len__(self):
        return len(self.data)

class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, num_answers):
        super(ChatbotModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.decoder = nn.LSTM(768, 768, batch_first=True)
        self.fc_out = nn.Linear(768, vocab_size)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        decoder_output, _ = self.decoder(bert_output.last_hidden_state)
        output = self.fc_out(decoder_output)
        return output

if __name__ == "__main__":
    NUM_EPOCHS = 1000
    LEARNING_RATE = 5e-5

    with open('reddit/data_dialog/reddit_data.json', 'r') as f:
        data = json.load(f)

    print(f"총 데이터 개수: {len(data)}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = ChatbotDataset(data, tokenizer)
    def collate_fn(batch):
        return {
            'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
            'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0),
            'target_ids': torch.stack([item['target_ids'] for item in batch]),
            'target_attention_mask': torch.stack([item['target_attention_mask'] for item in batch])
        }

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=10, pin_memory=True, collate_fn=collate_fn)
    
    model = ChatbotModel(tokenizer.vocab_size, num_answers=10)  # 10개의 답변을 지원하도록 설정
    criterion = nn.CrossEntropyLoss()  # 다중 클래스 문제에 적합한 손실 함수
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

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

            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.view(-1, tokenizer.vocab_size), target_ids.view(-1))  # 손실 계산
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}')

        if avg_loss <= 0.3:
            print(f"Loss reached the target of 0.3. Stopping training.")
            break

    # 모델 저장
    torch.save(model.state_dict(), 'GPT/model_pytorch/chatbot_model.pth')