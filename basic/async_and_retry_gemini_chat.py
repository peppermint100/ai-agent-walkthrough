"""
Gemini API를 사용하여 비동기 + 재시도 로직이 포함된 스트리밍 챗봇
- 비동기 방식으로 API 호출
- tenacity를 사용한 자동 재시도 (최대 3번, 1초 간격)
- 50% 확률로 실패 시뮬레이션 (테스트용)
"""
import os
import asyncio
import random
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type
)
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 테스트용 실패 시뮬레이션 플래그
SIMULATE_FAILURE = True  # True로 설정하면 50% 확률로 요청 실패

# .env 파일 경로 설정
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

genai.configure(api_key=api_key)

# 모델 초기화
model = genai.GenerativeModel('gemini-2.5-flash')


def _should_fail():
    """50% 확률로 실패 시뮬레이션"""
    if SIMULATE_FAILURE:
        will_fail = random.random() < 0.9
        if will_fail:
            print("🔴 시뮬레이션: API 호출 실패!", flush=True)
        return will_fail
    return False


def _log_retry_attempt(retry_state):
    """재시도 전에 호출되는 콜백 함수"""
    attempt_number = retry_state.attempt_number
    print(f"\n⚠️  재시도 {attempt_number}번째 - 1초 후 다시 시도합니다...", flush=True)


@retry(
    stop=stop_after_attempt(3),  # 최대 3번 시도 (첫 시도 + 2번 재시도)
    wait=wait_fixed(1),  # 1초 간격
    retry=retry_if_exception_type(Exception),
    before_sleep=_log_retry_attempt
)
async def _get_api_response(question: str):
    """
    재시도 로직이 포함된 API 호출 함수

    Args:
        question: 질문 내용

    Returns:
        API 응답 객체
    """
    # 실패 시뮬레이션
    if _should_fail():
        raise Exception("Simulated API failure - 재시도 중...")

    print("✅ API 호출 성공 - 답변 생성 중...", flush=True)
    # 비동기 스트리밍으로 API 호출
    response = await model.generate_content_async(question, stream=True)
    return response


async def ask_question_stream(question: str):
    """
    비동기 스트리밍 방식으로 답변을 받는 함수

    Args:
        question: 질문 내용

    Yields:
        답변의 각 청크(chunk)
    """
    try:
        # 재시도 로직이 포함된 API 호출
        response = await _get_api_response(question)

        # 스트리밍으로 각 청크를 받아서 yield
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        # 재시도 로직에서 처리되지 않은 예외
        logger.error(f"❌ API 호출 실패: {e}")
        raise


async def main():
    """메인 비동기 함수"""
    print("=" * 60)
    print("Gemini 비동기 + 재시도 스트리밍 챗봇")
    print("(비동기 방식 + 자동 재시도 + 실시간 스트리밍)")
    print(f"테스트 모드: {'활성화 (50% 실패 시뮬레이션)' if SIMULATE_FAILURE else '비활성화'}")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 60)

    while True:
        # 사용자 입력 받기 (동기 방식이지만 문제없음)
        user_input = input("\n질문: ").strip()

        # 종료 조건
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("채팅을 종료합니다. 안녕히 가세요!")
            break

        # 빈 입력 처리
        if not user_input:
            print("질문을 입력해주세요.")
            continue

        # 비동기 스트리밍으로 답변 받기
        print("\n답변: ", end='', flush=True)

        try:
            async for chunk in ask_question_stream(user_input):
                print(chunk, end='', flush=True)
            print()  # 줄바꿈

        except Exception as e:
            print(f"\n\n오류 발생: {str(e)}")
            print("재시도 횟수를 초과했습니다. 나중에 다시 시도해주세요.")


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(main())
