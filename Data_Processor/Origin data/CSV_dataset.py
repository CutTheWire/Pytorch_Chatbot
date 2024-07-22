import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

# 각 데이터셋에서 각 문장의 토큰 길이를 기록하고 최대 문장도 찾는 함수
def get_longest_token_length_and_sentence(dataset_split):
    longest_lengths = []
    longest_sentences = []

    for item in dataset_split:
        dialog = item['dialog']  # 대화 리스트 가져오기
        max_length = 0  # 최대 토큰 길이 초기화
        max_sentence = ""  # 최대 길이 문장 초기화
        
        for line in dialog:
            # 띄어쓰기를 기준으로 토큰화하고 길이 측정
            tokens = line.split()
            length = len(tokens)
            
            # 최대 길이와 문장 업데이트
            if length > max_length:
                max_length = length
                max_sentence = line
        
        longest_lengths.append(max_length)  # 각 대화의 최대 길이 추가
        longest_sentences.append(max_sentence)  # 각 대화의 최대 문장 추가

    return longest_lengths, longest_sentences

# 각 데이터셋에서 가장 긴 대화의 토큰 길이와 문장 찾기
train_lengths, train_sentences = get_longest_token_length_and_sentence(dataset['train'])
validation_lengths, validation_sentences = get_longest_token_length_and_sentence(dataset['validation'])
test_lengths, test_sentences = get_longest_token_length_and_sentence(dataset['test'])

# 결과를 데이터프레임으로 정리
train_df = pd.DataFrame({
    'Length': train_lengths,
    'Sentence': train_sentences
}, index=[f'Dialog {i}' for i in range(len(train_lengths))])

validation_df = pd.DataFrame({
    'Length': validation_lengths,
    'Sentence': validation_sentences
}, index=[f'Dialog {i}' for i in range(len(validation_lengths))])

test_df = pd.DataFrame({
    'Length': test_lengths,
    'Sentence': test_sentences
}, index=[f'Dialog {i}' for i in range(len(test_lengths))])

# CSV 파일 저장 경로 및 이름 설정
save_dir = 'Data_Processor/csv'
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
current_time = datetime.now().strftime("%y%m%d_%H%M%S")
csv_file_name = f"{current_time}_li2017dailydialog.csv"
csv_file_path = os.path.join(save_dir, csv_file_name)

# 각 데이터프레임을 개별 CSV 파일로 저장
train_df.to_csv(os.path.join(save_dir, f'train_{csv_file_name}'), index=False)
validation_df.to_csv(os.path.join(save_dir, f'validation_{csv_file_name}'), index=False)
test_df.to_csv(os.path.join(save_dir, f'test_{csv_file_name}'), index=False)

# 결과 출력
print("Train DataFrame:\n", train_df)
print("\nValidation DataFrame:\n", validation_df)
print("\nTest DataFrame:\n", test_df)
print(f"\nCSV 파일이 저장되었습니다: {csv_file_path}")

