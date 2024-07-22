import pandas as pd
from datasets import load_from_disk
import matplotlib.pyplot as plt

# 로컬 데이터셋 불러오기
dataset = load_from_disk('Data_Processor/filtered_daily_dialog')

# 각 데이터셋에서 각 문장의 토큰 길이를 기록하는 함수
def get_token_lengths(dataset_split):
    token_lengths = []
    for item in dataset_split:
        dialog = item['dialog']
        max_lengths = [len(line.split()) for line in dialog]
        token_lengths.append(max(max_lengths))
    return token_lengths

# 각 데이터셋에서 대화의 요소 수를 기록하는 함수
def get_dialog_lengths(dataset_split):
    dialog_lengths = []
    for item in dataset_split:
        dialog = item['dialog']
        dialog_lengths.append(len(dialog))
    return dialog_lengths

# 각 데이터셋에서 토큰 길이 및 대화의 요소 수 계산
train_token_lengths = get_token_lengths(dataset['train'])
validation_token_lengths = get_token_lengths(dataset['validation'])
test_token_lengths = get_token_lengths(dataset['test'])

train_dialog_lengths = get_dialog_lengths(dataset['train'])
validation_dialog_lengths = get_dialog_lengths(dataset['validation'])
test_dialog_lengths = get_dialog_lengths(dataset['test'])

# 그래프 그리기
fig, axs = plt.subplots(6, 1, figsize=(15, 36))  # 6행 1열의 서브플롯 생성

# Maximum Token Length 막대 그래프
axs[0].bar(range(len(train_token_lengths)), train_token_lengths, label='Train Max Token Length', color='blue')
axs[0].set_title('Maximum Token Length in Train Dataset')
axs[0].set_xlabel('Dialog Index')
axs[0].set_ylabel('Max Token Length')
axs[0].grid()
axs[0].legend()

axs[1].bar(range(len(validation_token_lengths)), validation_token_lengths, label='Validation Max Token Length', color='orange')
axs[1].set_title('Maximum Token Length in Validation Dataset')
axs[1].set_xlabel('Dialog Index')
axs[1].grid()
axs[1].legend()

axs[2].bar(range(len(test_token_lengths)), test_token_lengths, label='Test Max Token Length', color='green')
axs[2].set_title('Maximum Token Length in Test Dataset')
axs[2].set_xlabel('Dialog Index')
axs[2].grid()
axs[2].legend()

# Dialog Length 막대 그래프
axs[3].bar(range(len(train_dialog_lengths)), train_dialog_lengths, label='Train Dialog Length', color='purple')
axs[3].set_title('Train Dialog Length')
axs[3].set_xlabel('Dialog Index')
axs[3].set_ylabel('Number of Elements')
axs[3].grid()
axs[3].legend()

axs[4].bar(range(len(validation_dialog_lengths)), validation_dialog_lengths, label='Validation Dialog Length', color='gold')
axs[4].set_title('Validation Dialog Length')
axs[4].set_xlabel('Dialog Index')
axs[4].set_ylabel('Number of Elements')
axs[4].grid()
axs[4].legend()

axs[5].bar(range(len(test_dialog_lengths)), test_dialog_lengths, label='Test Dialog Length', color='red')
axs[5].set_title('Test Dialog Length')
axs[5].set_xlabel('Dialog Index')
axs[5].set_ylabel('Number of Elements')
axs[5].grid()
axs[5].legend()

# 그래프 레이아웃 조정 및 출력
plt.tight_layout()
plt.savefig(f'Data_Processor/Preprocessing data/plt/dialog_analysis.png')  # 그래프를 이미지 파일로 저장
plt.show()  # 그래프를 화면에 표시
