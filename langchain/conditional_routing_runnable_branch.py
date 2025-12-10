"""
RunnableBranch를 사용한 조건부 라우팅 예제
언어를 감지하여 적절한 체인으로 라우팅합니다.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

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


# 언어 감지 함수
def detect_language(text: str) -> str:
    """
    단어의 언어를 감지합니다.

    Args:
        text: 감지할 단어

    Returns:
        "korean", "english", 또는 "unknown"
    """
    text = text.strip()

    # 한글 유니코드 범위 (U+AC00-U+D7AF)
    if re.search(r'[\uAC00-\uD7AF]+', text):
        return "korean"

    # 영어 알파벳
    if re.search(r'^[a-zA-Z]+$', text):
        return "english"

    return "unknown"


# Pydantic 모델 정의
class SimilarWords(BaseModel):
    """유사한 단어를 담는 구조"""
    similar_words: list[str] = Field(description="유사한 단어 3개의 리스트")
    original_word: str = Field(description="원본 단어")
    language: str = Field(description="언어 (korean 또는 english)")


# Gemini 모델 초기화
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# 구조화된 출력을 위한 모델 설정
structured_model = model.with_structured_output(SimilarWords)

# 한국어 프롬프트 템플릿
korean_prompt = PromptTemplate(
    input_variables=["word"],
    template="""당신은 한국어 어휘 전문가입니다.

주어진 한국어 단어: {word}

위 단어와 의미가 유사한 한국어 단어 3개를 제시해주세요.
단어는 반드시 한국어로만 작성하고, 의미가 비슷하거나 관련된 단어를 선택해주세요.

결과를 JSON 형태로 제공하되:
- similar_words: 유사한 단어 3개를 리스트로
- original_word: "{word}"
- language: "korean"
"""
)

# 영어 프롬프트 템플릿
english_prompt = PromptTemplate(
    input_variables=["word"],
    template="""You are an English vocabulary expert.

Given English word: {word}

Please provide 3 English words that are similar in meaning to the word above.
The words must be in English only, and should have similar or related meanings.

Return the result in JSON format:
- similar_words: list of 3 similar words
- original_word: "{word}"
- language: "english"
"""
)

# 체인 생성
korean_chain = korean_prompt | structured_model
english_chain = english_prompt | structured_model


# 전처리 함수: 입력에 언어 정보 추가
def add_language_info(x):
    """입력에 언어 정보를 추가합니다."""
    word = x.get("word", "")
    language = detect_language(word)
    return {
        "word": word,
        "detected_language": language
    }


# 조건 함수들
def is_korean(x):
    """입력이 한국어인지 확인"""
    is_ko = x.get("detected_language") == "korean"
    if is_ko:
        print(f"  → 한국어 감지: '{x.get('word')}'")
    return is_ko


def is_english(x):
    """입력이 영어인지 확인"""
    is_en = x.get("detected_language") == "english"
    if is_en:
        print(f"  → 영어 감지: '{x.get('word')}'")
    return is_en


def handle_unknown(x):
    """알 수 없는 언어 처리"""
    word = x.get("word", "")
    raise ValueError(
        f"언어를 감지할 수 없습니다: '{word}'\n"
        f"한국어 또는 영어 단어를 입력해주세요."
    )


# RunnableLambda로 함수들을 Runnable로 변환
preprocessor = RunnableLambda(add_language_info)
unknown_handler = RunnableLambda(handle_unknown)

# RunnableBranch 생성
branch = RunnableBranch(
    (is_korean, korean_chain),      # 조건 1: 한국어 → korean_chain
    (is_english, english_chain),    # 조건 2: 영어 → english_chain
    unknown_handler                 # 기본: 오류 발생
)

# 전체 체인: 전처리 → 브랜치
full_chain = preprocessor | branch


def demo1_basic_routing():
    """기본 라우팅 예제"""
    print("\n" + "="*70)
    print("예제 1: 기본 언어별 라우팅")
    print("="*70)

    test_words = ["사랑", "love", "행복", "happy"]

    for word in test_words:
        print(f"\n[입력 단어: {word}]")
        print("-" * 70)

        try:
            result: SimilarWords = full_chain.invoke({"word": word})
            print(f"원본: {result.original_word}")
            print(f"언어: {result.language}")
            print(f"유사 단어:")
            for i, similar_word in enumerate(result.similar_words, 1):
                print(f"  {i}. {similar_word}")

            # Pydantic 모델을 딕셔너리로 변환
            print(f"\n딕셔너리 형태: {result.model_dump()}")
        except Exception as e:
            print(f"오류: {e}")


def demo2_error_handling():
    """오류 처리 예제"""
    print("\n" + "="*70)
    print("예제 2: 알 수 없는 언어 오류 처리")
    print("="*70)

    test_words = ["12345", "こんにちは", "😀"]

    for word in test_words:
        print(f"\n[입력 단어: {word}]")
        print("-" * 70)

        try:
            result = full_chain.invoke({"word": word})
            print(f"결과: {result}")
        except ValueError as e:
            print(f"예상된 오류 발생:")
            print(f"  {e}")
        except Exception as e:
            print(f"오류: {e}")


def demo3_understanding_branch():
    """RunnableBranch 동작 원리 이해"""
    print("\n" + "="*70)
    print("예제 3: RunnableBranch 동작 원리")
    print("="*70)

    print("\nRunnableBranch는 다음과 같이 동작합니다:")
    print("1. 입력을 받습니다")
    print("2. 첫 번째 조건부터 순서대로 평가합니다")
    print("3. True를 반환하는 첫 번째 조건의 체인을 실행합니다")
    print("4. 모든 조건이 False면 기본(default) 체인을 실행합니다")

    print("\n현재 브랜치 구조:")
    print("  조건 1: is_korean → korean_chain")
    print("  조건 2: is_english → english_chain")
    print("  기본: unknown_handler (오류 발생)")

    test_cases = [
        ("기쁨", "한국어 감지 → korean_chain 실행"),
        ("joy", "영어 감지 → english_chain 실행"),
    ]

    for word, expected in test_cases:
        print(f"\n테스트: {word}")
        print(f"예상: {expected}")
        result = full_chain.invoke({"word": word})
        print(f"결과: {', '.join(result.similar_words)}")


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("RunnableBranch를 활용한 조건부 라우팅")
    print("="*70)

    # 모든 데모 실행
    demo1_basic_routing()
    demo2_error_handling()
    demo3_understanding_branch()

    print("\n" + "="*70)
    print("학습 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
