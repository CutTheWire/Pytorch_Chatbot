import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# 현재 파일의 디렉토리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
# 저장할 경로 설정
save_path = os.path.join(current_dir, 'csv')
# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

# 각 데이터셋을 위한 데이터프레임 생성 함수
def create_dataframe(dataset_split):
    dialogs = []
    acts = []
    emotions = []

    for item in dataset_split:
        dialog = item['dialog']  # 대화 리스트 가져오기
        act = item['act']        # 행동 가져오기
        emotion = item['emotion'] # 감정 가져오기
        
        # 각 대화에서 모든 문장을 결합하여 저장
        combined_dialog = ' '.join(dialog)
        
        dialogs.append(combined_dialog)
        acts.append(act)
        emotions.append(emotion)

    return pd.DataFrame({
        'Dialog': dialogs,
        'Act': acts,
        'Emotion': emotions
    })

# 각 데이터셋에서 데이터프레임 생성
train_df = create_dataframe(dataset['train'])
validation_df = create_dataframe(dataset['validation'])
test_df = create_dataframe(dataset['test'])

# CSV 파일 저장 경로 및 이름 설정
os.makedirs(save_path, exist_ok=True)  # 디렉토리가 없으면 생성
csv_file_name = f"li2017dailydialog.csv"
csv_file_path = os.path.join(save_path, csv_file_name)

# 각 데이터프레임을 개별 CSV 파일로 저장
train_df.to_csv(os.path.join(save_path, f'train_{csv_file_name}'), index=False)
validation_df.to_csv(os.path.join(save_path, f'validation_{csv_file_name}'), index=False)
test_df.to_csv(os.path.join(save_path, f'test_{csv_file_name}'), index=False)

# 결과 출력
print("Train DataFrame:\n", train_df)
print("\nValidation DataFrame:\n", validation_df)
print("\nTest DataFrame:\n", test_df)
print(f"\nCSV 파일이 저장되었습니다: {csv_file_path}")
