import re
import string
import json
from collections import OrderedDict
import demoji

# 데이터 파일 경로
file_path = 'reddit/data/reddit_data.txt'

# 데이터 불러오기
with open(file_path, 'r', encoding='utf-8') as f:
    data = f.readlines()

# 데이터 전처리 함수
def preprocess_text(text):
    # 이모지 제거
    text = demoji.replace(text, '')
    # 소문자로 변환
    text = text.lower()
    # URL, HTML 태그 제거
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    # 구두점 제거
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    return text.strip()  # 공백 제거

# 모든 댓글 데이터 전처리
data = [preprocess_text(comment) for comment in data if comment.strip()]

# 중복 제거
unique_data = list(OrderedDict.fromkeys(data))

# 빈 문자열 제외
filtered_data = [comment for comment in unique_data if comment]

# 인덱스 9000개까지만 포함
limited_data = filtered_data[:9000]

# JSON 형식의 데이터로 변환
json_data = [{"index": idx + 1, "value": comment} for idx, comment in enumerate(limited_data)]

# JSON 파일로 저장
processed_file_path = 'reddit/data/processed_reddit_data.json'

with open(processed_file_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"전처리된 데이터가 {processed_file_path}에 저장되었습니다.")
