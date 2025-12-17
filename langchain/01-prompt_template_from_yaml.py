"""
YAML 파일에서 PromptTemplate을 로드하여 사용하는 예제
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser

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

# Gemini 모델 초기화
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,
)

# YAML 파일에서 PromptTemplate 로드
template_path = Path(__file__).parent / "greeting_template.yaml"
prompt_template = load_prompt(str(template_path))

# StrOutputParser 생성
output_parser = StrOutputParser()

# 파이프라인 체인 생성
chain = prompt_template | model | output_parser


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("YAML 파일에서 PromptTemplate 로드")
    print("="*70)

    # 예제 실행
    input_data = {"name": "서연", "topic": "LangChain과 AI"}
    result = chain.invoke(input_data)

    print(f"\n템플릿 파일: {template_path.name}")
    print(f"입력: {input_data}")
    print(f"\n결과:\n{result}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
