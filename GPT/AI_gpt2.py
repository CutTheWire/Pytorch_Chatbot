from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import torch

# 로컬 모델 경로 설정
model_dir = "d:/AI_BOT/GPT2/model"

# 모델과 토크나이저 로드
try:
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
except EnvironmentError as e:
    print(f"모델을 로드하는 중에 오류가 발생했습니다: {e}")
    exit(1)

def generate_response(prompt, max_length=150, temperature=0.7, top_p=0.9):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # attention_mask 생성
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,  # attention_mask 추가
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        pad_token_id=tokenizer.eos_token_id,  # pad_token_id 설정
        temperature=temperature,  # temperature 설정
        top_p=top_p,  # top_p 설정
        do_sample=True  # do_sample 활성화
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 대화 시작
print("대화형 AI에 오신 것을 환영합니다! 'exit'를 입력하면 종료됩니다.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    # 프롬프트를 구체적으로 작성
    prompt = f"The user said: '{user_input}'. How would you respond?"
    response = generate_response(prompt)
    print(f"Bot: {response}")
