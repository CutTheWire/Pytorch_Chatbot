import os
import shutil

def clear_huggingface_cache():
    """
    Hugging Face 모델 캐시 디렉토리를 삭제하는 함수입니다.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"캐시 디렉토리 {cache_dir}가 삭제되었습니다.")
    else:
        print(f"캐시 디렉토리 {cache_dir}를 찾을 수 없습니다.")

if __name__ == "__main__":
    clear_huggingface_cache()
