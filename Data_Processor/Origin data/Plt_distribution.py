import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

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

# 고유 값과 빈도 계산
def calculate_frequency(data):
    freq = {}
    for value in data:
        if value in freq:
            freq[value] += 1
        else:
            freq[value] = 1
    return freq

train_token_freq = calculate_frequency(train_token_lengths)
validation_token_freq = calculate_frequency(validation_token_lengths)
test_token_freq = calculate_frequency(test_token_lengths)

train_dialog_freq = calculate_frequency(train_dialog_lengths)
validation_dialog_freq = calculate_frequency(validation_dialog_lengths)
test_dialog_freq = calculate_frequency(test_dialog_lengths)

# 그래프 그리기
fig, axs = plt.subplots(6, 1, figsize=(15, 36))  # 6행 1열의 서브플롯 생성

# Train Token Length 분포 그래프
axs[0].scatter(list(train_token_freq.keys()), list(train_token_freq.values()), color='blue', marker='o')
axs[0].set_title('Distribution of Train Max Token Length')
axs[0].set_xlabel('Token Length')
axs[0].set_ylabel('Frequency')
axs[0].grid()

# Validation Token Length 분포 그래프
axs[1].scatter(list(validation_token_freq.keys()), list(validation_token_freq.values()), color='orange', marker='o')
axs[1].set_title('Distribution of Validation Max Token Length')
axs[1].set_xlabel('Token Length')
axs[1].set_ylabel('Frequency')
axs[1].grid()

# Test Token Length 분포 그래프
axs[2].scatter(list(test_token_freq.keys()), list(test_token_freq.values()), color='green', marker='o')
axs[2].set_title('Distribution of Test Max Token Length')
axs[2].set_xlabel('Token Length')
axs[2].set_ylabel('Frequency')
axs[2].grid()

# Train Dialog Length 분포 그래프
axs[3].scatter(list(train_dialog_freq.keys()), list(train_dialog_freq.values()), color='purple', marker='o')
axs[3].set_title('Distribution of Train Dialog Length')
axs[3].set_xlabel('Number of Elements')
axs[3].set_ylabel('Frequency')
axs[3].grid()

# Validation Dialog Length 분포 그래프
axs[4].scatter(list(validation_dialog_freq.keys()), list(validation_dialog_freq.values()), color='gold', marker='o')
axs[4].set_title('Distribution of Validation Dialog Length')
axs[4].set_xlabel('Number of Elements')
axs[4].set_ylabel('Frequency')
axs[4].grid()

# Test Dialog Length 분포 그래프
axs[5].scatter(list(test_dialog_freq.keys()), list(test_dialog_freq.values()), color='red', marker='o')
axs[5].set_title('Distribution of Test Dialog Length')
axs[5].set_xlabel('Number of Elements')
axs[5].set_ylabel('Frequency')
axs[5].grid()

# 그래프 레이아웃 조정 및 출력
plt.tight_layout()
plt.savefig('Data_Processor/plt/dialog_analysis_distribution.png')  # 그래프를 이미지 파일로 저장
plt.show()  # 그래프를 화면에 표시
