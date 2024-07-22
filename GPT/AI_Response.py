import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 모델 정의 (이전 코드와 동일)
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

# 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = ChatbotModel(tokenizer.vocab_size)
model.load_state_dict(torch.load('GPT/model_pytorch/text_classifier.pth'))
model.eval()
model.to(device)

# 입력 텍스트
input_text = "who are you?"

# 입력 텍스트를 토큰화
input_encoding = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=512,
    return_token_type_ids=False,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt',
)

input_ids = input_encoding['input_ids'].to(device)
attention_mask = input_encoding['attention_mask'].to(device)

# 모델 예측
with torch.no_grad():
    outputs = model(input_ids, attention_mask)

# 예측된 토큰을 텍스트로 변환
predicted_ids = torch.argmax(outputs, dim=-1)
predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(f"입력 텍스트: {input_text}")
print(f"예측된 텍스트: {predicted_text}")
