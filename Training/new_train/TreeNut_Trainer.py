import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# CUDA 사용 가능 여부 확인 및 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 및 토크나이저 불러오기
model_path = "saved_model/best_t5_dialog_model.pth"
tokenizer_path = "saved_model/best_t5_dialog_model"

tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# 대화 생성 함수
@torch.no_grad()
def generate_response(dialogue_history, acts, emotions):
    input_text = "dialogue: " + " ".join(dialogue_history)
    for i, (act, emotion) in enumerate(zip(acts, emotions)):
        input_text += f" turn{i+1}_act: {act} turn{i+1}_emotion: {emotion}"
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    with torch.amp.autocast('cuda'):
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 대화 테스트
print("대화를 시작합니다. 'q'를 입력하면 종료됩니다.")

dialogue_history = []
acts = []
emotions = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'q':
        break

    dialogue_history.append(user_input)
    acts.append(0)  # 임의의 act 값
    emotions.append(0)  # 임의의 emotion 값

    response = generate_response(dialogue_history, acts, emotions)
    print(f"Model: {response}")

    dialogue_history.append(response)
    acts.append(0)  # 임의의 act 값
    emotions.append(0)  # 임의의 emotion 값

print("대화가 종료되었습니다.")
