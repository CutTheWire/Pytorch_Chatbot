import os
import praw
import json
from dotenv import load_dotenv
import re
import time
import multiprocessing

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수 가져오기
CLIENT_ID = os.getenv('client_id')
CLIENT_SECRET = os.getenv('client_secret')
USER_AGENT = os.getenv('user_agent')

# Reddit API 클라이언트 설정
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

def get_data(subreddit_name, start_rank, end_rank):
    # 특정 서브레딧에서 데이터 수집
    subreddit = reddit.subreddit(subreddit_name)
    data = []

    # 서브레딧의 상위 게시물에 대한 질문과 root 댓글 수집
    for i, post in enumerate(subreddit.top(limit=end_rank), start=start_rank):
        if i < start_rank:
            continue
        if i >= end_rank:
            break
        input_text = f"{post.title} {post.selftext}"
        output_text = []
        post.comments.replace_more(limit=0)
        for comment in post.comments.list():
            # 삭제되거나 이동된 댓글 제외
            if comment.body not in ['[deleted]', '[removed]']:
                cleaned_text = re.sub(r'[^\w\s]', '', comment.body)
                output_text.append(cleaned_text)
                if len(output_text) >= 10:  # 10개 이상의 댓글이 수집되면 중단
                    break
        data.append({
            'input': input_text,
            'output': output_text
        })
        print(f"진행 상황: {i+1}/{end_rank} ({round((i-start_rank+1)/(end_rank-start_rank)*100, 2)}%)", end="\r")
        time.sleep(1)  # 1초 대기 후 다음 요청 보내기

    return data

if __name__ == '__main__':
    # 5개의 프로세스 생성
    pool = multiprocessing.Pool(processes=5)

    # 각 프로세스에 get_data 함수 실행
    data_list = pool.starmap(get_data, [
        ('confession', 0, 20),
        ('confessions', 0, 20),
        ('AMA', 0, 20),
        ('AITAH', 0, 20),
        ('KindVoice', 0, 20)
    ])

    # 데이터 병합
    data = []
    for d in data_list:
        data.extend(d)

    # 수집한 데이터를 JSON 파일로 저장
    with open('reddit/data_dialog/reddit_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("데이터 수집 완료!")
