from datasets import load_from_disk, load_dataset

# 데이터셋 불러오기
dataset = load_dataset("li2017dailydialog/daily_dialog")
# # 로컬 데이터셋 불러오기
# dataset = load_from_disk('Data_Processor/filtered_daily_dialog')

# 전체 dataset 출력
print(dataset)

# train 데이터셋의 첫 번째 샘플 출력
print(dataset['train'][0])

# validation 데이터셋의 첫 번째 샘플 출력
print(dataset['validation'][0])

# test 데이터셋의 첫 번째 샘플 출력
print(dataset['test'][0])
