import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_prompt_template(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model = model.half()
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("모델이 CUDA(기본 GPU)로 이동되었습니다.")
    else:
        print("CUDA를 사용할 수 없습니다. 모델이 CPU에서 실행됩니다.")
    return tokenizer, model

def generate_response(prompt: str, tokenizer, model) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            top_k=40,
            top_p=0.92,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

class ChatSession:
    def __init__(self, prompt_template: dict):
        self.history = []
        self.prompt_template = prompt_template

    def add_message(self, role: str, message: str):
        prefix = self.prompt_template["user_prefix"] if role == "사용자" else self.prompt_template["bot_prefix"]
        self.history.append(f"{prefix}{message}")

    def get_prompt(self) -> str:
        context = self.prompt_template["context"]
        separator = self.prompt_template["separator"]
        return f"{context}{separator}" + separator.join(self.history) + f"{separator}{self.prompt_template['bot_prefix']}"

    def generate_response(self, user_input: str, tokenizer, model) -> str:
        self.add_message("사용자", user_input)
        prompt = self.get_prompt()
        response = generate_response(prompt, tokenizer, model)
        self.add_message("봇", response)
        return response

    def initialize_conversation(self, tokenizer, model) -> None:
        # 초기 프롬프트만 적용
        initial_prompt = self.prompt_template["context"]
        initial_response = generate_response(initial_prompt, tokenizer, model)
        print(f"Bot: {initial_response}")

if __name__ == "__main__":
    model_dir = "./saved_model"
    prompt_template_path = "Training/new_train/prompt_template.json"

    prompt_template = load_prompt_template(prompt_template_path)
    tokenizer, model = load_model_and_tokenizer(model_dir)

    chat_session = ChatSession(prompt_template)

    # 대화 초기화 (프롬프트 적용)
    chat_session.initialize_conversation(tokenizer, model)

    print("대화 시작! 종료하려면 'exit_00221_UREEwire'를 입력하세요.")
    
    while True:
        user_input = input("You: ")
        if user_input == "exit_00221_UREEwire":
            print("대화를 종료합니다.")
            break

        response = chat_session.generate_response(user_input, tokenizer, model)
        print(f"Bot: {response}")
