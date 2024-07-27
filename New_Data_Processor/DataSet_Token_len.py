from datasets import load_dataset
import pandas as pd

# 데이터셋 불러오기
dataset = load_dataset("li2017dailydialog/daily_dialog")
print(dataset)
print(dataset['train'][0])
lengths = {}
extreme_value = {}

for item in ['dialog', 'act', 'emotion']:
    lengths[item] = {
        'train'     : [len(train)       for train       in dataset['train'][item]],
        'validation': [len(validation)  for validation  in dataset['validation'][item]],
        'test'      : [len(test)        for test        in dataset['test'][item]]
    }

for item in ['train', 'validation', 'test']:
    extreme_value[item] = {
        'max' : max(lengths["dialog"][item] + lengths["act"][item] + lengths["emotion"][item]),
        'min' : min(lengths["dialog"][item] + lengths["act"][item] + lengths["emotion"][item])
    }

# 데이터프레임 생성
lengths_df = pd.DataFrame(lengths)
extreme_value_df = pd.DataFrame(extreme_value)

# Excel 파일로 저장
lengths_df.to_excel('New_Data_Processor/excel/dataset_lengths.xlsx', index=True, engine='openpyxl')
extreme_value_df.to_excel('New_Data_Processor/excel/dataset_extreme_value.xlsx', index=True, engine='openpyxl')
