import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
import numpy as np
from typing import List, Dict

# Define custom dataset
class DialogueDataset(Dataset):
    def __init__(self, dialogues: List[str], acts: List[int], emotions: List[int], tokenizer, max_length: int = 128):
        self.dialogues = dialogues
        self.acts = acts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.dialogues[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['act'] = torch.tensor(self.acts[idx])
        item['emotion'] = torch.tensor(self.emotions[idx])
        return item

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence([item['token_type_ids'] for item in batch], batch_first=True, padding_value=0)
    acts = torch.tensor([item['act'] for item in batch])
    emotions = torch.tensor([item['emotion'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'act': acts, 'emotion': emotions}

# Define the model
class DialogueModel(nn.Module):
    def __init__(self, num_acts: int, num_emotions: int):
        super(DialogueModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.act_classifier = nn.Linear(self.bert.config.hidden_size, num_acts)
        self.emotion_classifier = nn.Linear(self.bert.config.hidden_size, num_emotions)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        act_logits = self.act_classifier(cls_output)
        emotion_logits = self.emotion_classifier(cls_output)
        return act_logits, emotion_logits

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        act_labels = batch['act'].to(device)
        emotion_labels = batch['emotion'].to(device)

        act_logits, emotion_logits = model(input_ids, attention_mask, token_type_ids)
        act_loss = criterion(act_logits, act_labels)
        emotion_loss = criterion(emotion_logits, emotion_labels)
        loss = act_loss + emotion_loss

        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            act_labels = batch['act'].to(device)
            emotion_labels = batch['emotion'].to(device)

            act_logits, emotion_logits = model(input_ids, attention_mask, token_type_ids)
            act_loss = criterion(act_logits, act_labels)
            emotion_loss = criterion(emotion_logits, emotion_labels)
            loss = act_loss + emotion_loss

            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("li2017dailydialog/daily_dialog")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create Datasets
    train_dataset = DialogueDataset(dataset['train']['dialog'], dataset['train']['act'], dataset['train']['emotion'], tokenizer)
    val_dataset = DialogueDataset(dataset['validation']['dialog'], dataset['validation']['act'], dataset['validation']['emotion'], tokenizer)

    # Data Loaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = DialogueModel(num_acts=5, num_emotions=7).to(device)

    # Optimizer and Criterion
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    epochs = 3
    for epoch in range(epochs):
        train_model(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate_model(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

    # Save the model
    torch.save(model.state_dict(), "dialogue_model.pt")
