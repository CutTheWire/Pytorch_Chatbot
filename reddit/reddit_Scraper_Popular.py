import os
import praw
from dotenv import load_dotenv

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

# 특정 서브레딧에서 데이터 수집
subreddit = reddit.subreddit('learnpython')  # 'learnpython'을 원하는 서브레딧 이름으로 변경

# 데이터를 저장할 리스트
data = []

# 서브레딧의 상위 100개의 게시물에 대한 댓글 수집
for i, post in enumerate(subreddit.top(limit=100)):
    post.comments.replace_more(limit=0)
    for comment in post.comments.list():
        data.append(comment.body)
    print(f"진행 상황: {i+1}/100 ({round((i+1)/100*100, 2)}%)", end="\r")

# 수집한 데이터를 파일로 저장
with open('reddit/data/reddit_data.txt', 'w', encoding='utf-8') as f:
    for line in data:
        f.write(f"{line}\n")

print("데이터 수집 완료!")
