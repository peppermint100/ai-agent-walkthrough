"""
Pydantic을 사용하여 구조화된 응답을 받는 예제
영화 리뷰를 제목, 평점, 리뷰 내용으로 구조화하여 받습니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt

# .env 파일 경로 설정 (프로젝트 루트)
env_path = Path(__file__).parent.parent / '.env'
if not env_path.exists():
    env_path = Path(__file__).parent / '.env'

# 환경 변수 로드
load_dotenv(dotenv_path=env_path)

# Gemini API 키 설정
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY 환경 변수를 설정해주세요.\n"
        ".env 파일을 생성하고 GEMINI_API_KEY를 설정하거나, "
        ".env.example 파일을 참고하세요."
    )


# Pydantic 모델 정의
class MovieReview(BaseModel):
    """영화 리뷰 구조"""
    title: str = Field(description="영화 제목")
    rating: int = Field(description="1-10점 사이의 평점", ge=1, le=10)
    review: str = Field(description="100자 이내의 리뷰 내용", max_length=100)


# Gemini 모델 초기화
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# 구조화된 출력을 위한 모델 설정
structured_model = model.with_structured_output(MovieReview)

# YAML 파일에서 PromptTemplate 로드
template_path = Path(__file__).parent / "movie_review_template.yaml"
prompt_template = load_prompt(str(template_path))

# 파이프라인 체인 생성
chain = prompt_template | structured_model


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("Pydantic을 사용한 구조화된 응답 받기")
    print("="*70)

    # 예제 영화들
    movies = ["기생충", "인터스텔라", "타이타닉"]

    for movie_name in movies:
        print(f"\n[영화: {movie_name}]")
        print("-" * 70)

        # 체인 실행
        result: MovieReview = chain.invoke({"movie_name": movie_name})

        # 구조화된 데이터 출력
        print(f"제목: {result.title}")
        print(f"평점: {result.rating}/10")
        print(f"리뷰: {result.review}")

        # Pydantic 모델을 딕셔너리로 변환
        print(f"\n딕셔너리 형태: {result.model_dump()}")

    print("\n" + "="*70)
    print("완료!")
    print("="*70)


if __name__ == "__main__":
    main()
