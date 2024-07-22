import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# FLAN-T5 모델 로드
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# 전처리된 데이터 파일 경로
processed_file_path = 'reddit/data/processed_reddit_data.json'

# 전처리된 데이터 불러오기
try:
    with open(processed_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {processed_file_path}")

# 데이터셋 토큰화 및 모델 호출 함수
def get_model_response(text):
    try:
        # 입력 텍스트 토큰화
        input_ids = tokenizer.encode(text, return_tensors='pt')
        
        # 모델 출력 생성
        output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        
        # 출력 텍스트 디코딩
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return output_text
    except Exception as e:
        print(f"모델 호출 중 오류 발생: {e}")
        return ""

# 모델 응답 처리
responses = []
total_comments = len(data)

for item in data:
    index = item.get("index")
    value = item.get("value", "")
    response = get_model_response(value)
    responses.append({
        "input": value,
        "output": response
    })
    print(f"진행 상황: {index}/{total_comments} ({index / total_comments * 100:.2f}%)")

# 모델 응답 결과를 JSON 파일로 저장
output_file_path = 'reddit/data/model_flan_t5_responses.json'
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
    print(f"모델 응답 결과가 '{output_file_path}'에 저장되었습니다.")
except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")
