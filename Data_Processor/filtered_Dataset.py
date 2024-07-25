from datasets import load_dataset, DatasetDict

# 데이터셋 로드
dataset = load_dataset("li2017dailydialog/daily_dialog")

# 길이 제한 설정
train_length_min = 2
train_length_max = 60
validation_length_min = 2
validation_length_max = 55
test_length_min = 2
test_length_max = 60

# 필터링 함수
def filter_dataset(dataset, min_length, max_length):
    return dataset.filter(lambda x: all(min_length <= len(line.split()) <= max_length for line in x['dialog']))

# 각 데이터셋 필터링
filtered_train = filter_dataset(dataset['train'], train_length_min, train_length_max)
filtered_validation = filter_dataset(dataset['validation'], validation_length_min, validation_length_max)
filtered_test = filter_dataset(dataset['test'], test_length_min, test_length_max)

# 필터링된 데이터셋을 딕셔너리로 저장
filtered_dataset = DatasetDict({
    'train': filtered_train,
    'validation': filtered_validation,
    'test': filtered_test
})

# 로컬에 저장
output_dir = './Data_Processor/filtered_daily_dialog'
filtered_dataset.save_to_disk(output_dir)

print(filtered_dataset)
print("데이터셋이 성공적으로 저장되었습니다.")
