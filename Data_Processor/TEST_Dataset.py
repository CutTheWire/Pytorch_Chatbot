from datasets import load_from_disk, load_dataset

class dataset_check:
    def __init__(self) -> None:
        # 데이터셋 불러오기
        self.dataset_origin = load_dataset("li2017dailydialog/daily_dialog")
        # 로컬 데이터셋 불러오기
        self.dataset_prepro = load_from_disk('Data_Processor/filtered_daily_dialog')

    def checking(self, dataset_name: str = 'both') -> None:
        datasets = {
            'origin': self.dataset_origin,
            'prepro': self.dataset_prepro,
            'both': [self.dataset_origin, self.dataset_prepro]
        }
        selected_datasets = datasets.get(dataset_name, datasets['both'])

        for dataset in selected_datasets:
            print(dataset)# 전체 dataset 출력
            print(dataset['train'][0])# train 데이터셋의 첫 번째 샘플 출력
            print(dataset['validation'][0])# validation 데이터셋의 첫 번째 샘플 출력
            print(dataset['test'][0])# test 데이터셋의 첫 번째 샘플 출력

    def get_summary(self, dataset_name: str = 'both') -> dict:
        datasets = {
            'origin': self.dataset_origin,
            'prepro': self.dataset_prepro,
            'both': [self.dataset_origin, self.dataset_prepro]
        }
        selected_datasets = datasets.get(dataset_name, datasets['both'])
        summary = {}

        for dataset in selected_datasets:
            summary[dataset_name] = {
                'train_size': len(dataset['train']),
                'validation_size': len(dataset['validation']),
                'test_size': len(dataset['test'])
            }
        return summary
            
if __name__ == "__main__":
    checker = dataset_check()
    checker.checking()  # 모든 데이터셋 확인
    summary = checker.get_summary()  # 데이터셋 요약 정보 가져오기
    print(summary)
