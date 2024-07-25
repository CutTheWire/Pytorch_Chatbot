import os
import shutil
from git import Repo

def print_tree(path, prefix=""):
    if os.path.isdir(path):
        print(prefix + os.path.basename(path) + "/")
        prefix += " ┣ "
        for item in os.listdir(path):
            print_tree(os.path.join(path, item), prefix)
    else:
        print(prefix + "📜" + os.path.basename(path))

def print_changes(repo_path, commit_hash):
    repo = Repo(repo_path)
    commit = repo.commit(commit_hash)
    
    # 디렉토리 구조 출력
    print_tree(repo_path)
    print()
    
    # 변경된 파일 출력
    for changed_file in commit.stats.files:
        if commit.stats.files[changed_file]['deletions'] > 0 and commit.stats.files[changed_file]['insertions'] > 0:
            print(f"🔄 {changed_file} (수정)")
        elif commit.stats.files[changed_file]['deletions'] > 0:
            print(f"❌ {changed_file} (삭제)")
        elif commit.stats.files[changed_file]['insertions'] > 0:
            print(f"✅ {changed_file} (추가)")

# 사용 예시
repo_url = 'https://github.com/TreeNut-KR/Chatbot_Pytorch.git'
branch_name = 'dataset'
commit_hash = 'ecee9ec'

# 기존 디렉토리 삭제
if os.path.exists('Chatbot_Pytorch'):
    shutil.rmtree('Chatbot_Pytorch')

# 리포지토리 클론
repo = Repo.clone_from(repo_url, 'Chatbot_Pytorch')

# 브랜치 전환
repo.git.checkout(branch_name)

# 변경 내역 출력
print_changes('Chatbot_Pytorch', commit_hash)

# 불러온 Chatbot_Pytorch 디렉토리 삭제
shutil.rmtree('Chatbot_Pytorch')
