"""
Gemini API를 사용하여 간단한 질문을 보내고 답변을 받는 예제 코드
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
# 일반적으로 사용 가능한 모델명:
# - 'gemini-pro' (안정적, 널리 사용됨)
# - 'gemini-1.5-pro' (최신 고성능)
# - 'gemini-1.5-flash-latest' (경량, 빠른 응답)
# 오류가 발생하면 'gemini-pro'로 변경해보세요
model = genai.GenerativeModel('gemini-2.5-flash')


def list_available_models():
    """사용 가능한 모델 목록 출력 (디버깅용)"""
    try:
        models = genai.list_models()
        print("\n사용 가능한 모델 목록:")
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                print(f"  - {m.name}")
        print()
    except Exception as e:
        print(f"모델 목록 조회 실패: {e}\n")


def ask_question(question: str) -> str:
    """
    Gemini에게 질문을 보내고 답변을 받는 함수
    
    Args:
        question: 질문 내용
        
    Returns:
        Gemini의 답변
    """
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        # 모델 오류인 경우 사용 가능한 모델 목록도 출력
        if "404" in str(e) or "not found" in str(e).lower():
            return f"오류 발생: {str(e)}\n\n사용가능한 모델을 확인하려면 코드에서 list_available_models() 함수를 호출하세요."
        return f"오류 발생: {str(e)}"


def main():
    """메인 함수"""
    print("=" * 50)
    print("Gemini 챗봇에 오신 것을 환영합니다!")
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
        
        # Gemini에게 질문 보내기
        print("\n답변:")
        answer = ask_question(user_input)
        print(answer)


if __name__ == "__main__":
    main()

