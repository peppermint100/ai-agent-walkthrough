"""
LangChain의 PromptTemplate과 StrOutputParser를 사용하는 예제
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
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
# basic 폴더의 설정을 참고하여 같은 모델 사용
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7,  # 창의성 조절 (0.0 ~ 1.0)
)

# PromptTemplate 생성
# {name}과 {topic}은 변수로, 실행 시 실제 값으로 치환됩니다
prompt_template = PromptTemplate(
    input_variables=["name", "topic"],
    template="""당신은 친근하고 따뜻한 AI 어시스턴트입니다.

사용자 이름: {name}
대화 주제: {topic}

위 정보를 바탕으로 자연스럽고 친근한 인사말을 한국어로 작성해주세요.
인사말은 2-3문장 정도로 작성하고, 사용자의 이름과 관심사를 언급해주세요."""
)

# StrOutputParser 생성
# 모델의 출력을 문자열로 파싱합니다
output_parser = StrOutputParser()


def method1_direct_invoke():
    """
    방법 1: invoke()를 사용한 직접 호출
    각 컴포넌트를 순차적으로 호출하는 방식
    """
    print("\n" + "="*70)
    print("방법 1: invoke()를 사용한 직접 호출")
    print("="*70)

    input_data = {"name": "민수", "topic": "Python 프로그래밍"}

    # 1단계: PromptTemplate
    formatted_prompt = prompt_template.invoke(input_data)

    # 2단계: 모델 호출
    model_output = model.invoke(formatted_prompt)

    # 3단계: 문자열 파싱
    final_output = output_parser.invoke(model_output)

    print(f"\n입력: {input_data}")
    print(f"\n결과:\n{final_output}")

    return final_output


def method2_chain_with_pipe():
    """
    방법 2: | (pipe) 연산자를 사용한 체인 방식
    """
    print("\n" + "="*70)
    print("방법 2: | (pipe) 연산자를 사용한 체인 방식")
    print("="*70)

    # 체인 생성: prompt_template | model | output_parser
    chain = prompt_template | model | output_parser

    # 체인 실행: 한 번의 invoke()로 전체 파이프라인 실행
    input_data = {"name": "지은", "topic": "인공지능과 머신러닝"}
    final_output = chain.invoke(input_data)

    print(f"\n입력: {input_data}")
    print(f"\n결과:\n{final_output}")

    return final_output


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("LangChain PromptTemplate & StrOutputParser 학습")
    print("="*70)

    # 방법 1: 직접 invoke 호출
    method1_direct_invoke()

    # 방법 2: 파이프라인 방식
    method2_chain_with_pipe()

    print("\n" + "="*70)
    print("학습 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
