import json
import os
import openai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# API 키가 설정되지 않았을 경우 오류 발생
if not API_KEY:
    raise ValueError("API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정하세요.")

# OpenAI API 키 설정
openai.api_key = API_KEY

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
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",  # openAI 모델 사용
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            max_tokens=128
        )
        return response['choices'][0]['message']['content']
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
output_file_path = 'reddit/data/model_openai_responses.json'
try:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
    print(f"모델 응답 결과가 '{output_file_path}'에 저장되었습니다.")
except Exception as e:
    print(f"파일 저장 중 오류 발생: {e}")
