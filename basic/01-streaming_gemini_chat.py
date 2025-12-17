"""
Gemini API를 사용하여 스트리밍 방식으로 질문 답변을 처리하는 예제 코드
일반 방식과 다르게 답변이 생성되는 대로 실시간으로 출력됩니다.
"""
import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일 경로 설정 (프로젝트 루트 또는 현재 디렉토리)
env_path = Path(__file__).parent.parent.parent / '.env'
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


def ask_question_stream(question: str):
    """
    Gemini에게 질문을 보내고 스트리밍 방식으로 답변을 받는 함수

    Args:
        question: 질문 내용

    Yields:
        답변의 각 청크(chunk)
    """
    try:
        # stream=True 옵션을 사용하여 스트리밍 활성화
        response = model.generate_content(question, stream=True)

        # 스트리밍으로 각 청크를 받아서 yield
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"\n오류 발생: {str(e)}"


def main():
    """메인 함수"""
    print("=" * 50)
    print("Gemini 스트리밍 챗봇에 오신 것을 환영합니다!")
    print("(답변이 실시간으로 타이핑됩니다)")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("=" * 50)

    while True:
        # 사용자 입력 받기
        user_input = input("\n질문: ").strip()

        # 종료 조건
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("채팅을 종료합니다. 안녕히 가세요!")
            break

        # 빈 입력 처리
        if not user_input:
            print("질문을 입력해주세요.")
            continue

        # Gemini에게 질문 보내고 스트리밍으로 답변 받기
        print("\n답변: ", end='', flush=True)

        # 스트리밍으로 받은 각 청크를 실시간으로 출력
        for chunk in ask_question_stream(user_input):
            print(chunk, end='', flush=True)

        print()  # 줄바꿈


if __name__ == "__main__":
    main()
