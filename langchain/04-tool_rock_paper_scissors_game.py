"""
LangChain Tool을 사용한 가위바위보 게임
Tool을 사용하는 LLM과 해설하는 LLM이 협력하여 게임을 진행합니다.
"""
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
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


@tool
def play_rock_paper_scissors() -> str:
    """
    가위바위보 게임에서 AI의 선택을 반환하는 도구입니다.

    이 도구는 가위(scissors), 바위(rock), 보(paper) 중 하나를 무작위로 선택하여 반환합니다.
    게임을 진행할 때 AI의 선택을 결정하기 위해 이 도구를 사용하세요.

    Returns:
        str: "가위", "바위", "보" 중 하나

    Examples:
        >>> play_rock_paper_scissors()
        "가위"
    """
    choices = ["가위", "바위", "보"]
    return random.choice(choices)


def judge_winner(user_choice: str, ai_choice: str) -> str:
    """
    가위바위보 게임의 승부를 판정하는 함수

    Args:
        user_choice: 유저의 선택 (가위/바위/보)
        ai_choice: AI의 선택 (가위/바위/보)

    Returns:
        str: "win" (유저 승리), "lose" (AI 승리), "draw" (무승부)
    """
    if user_choice == ai_choice:
        return "draw"

    win_conditions = {
        "가위": "보",
        "바위": "가위",
        "보": "바위"
    }

    if win_conditions[user_choice] == ai_choice:
        return "win"
    else:
        return "lose"


def print_tool_info():
    """Tool의 정보를 출력하는 함수"""
    print("\n" + "="*70)
    print("Tool 정보 확인")
    print("="*70)
    print(f"Tool 이름: {play_rock_paper_scissors.name}")
    print(f"Tool 설명: {play_rock_paper_scissors.description}")
    print(f"\nTool 전체 정보:")
    print(play_rock_paper_scissors)
    print("="*70)


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("LangChain Tool을 사용한 가위바위보 게임")
    print("="*70)

    # Tool 정보 출력
    print_tool_info()

    # Tool 사용용 LLM 초기화
    tool_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.9,  # 더 창의적인 선택을 위해 높은 temperature
    )

    # Tool을 LLM에 바인딩
    tool_llm_with_tools = tool_llm.bind_tools([play_rock_paper_scissors])

    # 해설용 LLM 초기화
    commentary_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=1.0,  # 재미있는 해설을 위해 높은 temperature
    )

    # 해설용 프롬프트 템플릿
    commentary_template = PromptTemplate(
        input_variables=["user_choice", "ai_choice", "result"],
        template="""당신은 스포츠 해설위원입니다. 가위바위보 게임의 결과를 재미있고 드라마틱하게 해설해주세요.

게임 결과:
- 유저의 선택: {user_choice}
- AI의 선택: {ai_choice}
- 승부 결과: {result}

마치 세계대회 결승전인 것처럼 긴장감 넘치고 재미있게 해설해주세요!
해설은 3-5문장 정도로 작성해주세요."""
    )

    # 해설 체인 생성
    commentary_chain = commentary_template | commentary_llm | StrOutputParser()

    # 게임 시작
    print("\n가위바위보 게임에 오신 것을 환영합니다!")
    print("'가위', '바위', '보' 중 하나를 입력하세요. (종료: 'quit')\n")

    while True:
        user_input = input("당신의 선택: ").strip()

        if user_input.lower() == 'quit':
            print("\n게임을 종료합니다. 감사합니다!\n")
            break

        if user_input not in ["가위", "바위", "보"]:
            print("올바른 선택이 아닙니다. '가위', '바위', '보' 중 하나를 입력하세요.\n")
            continue

        print("\n" + "-"*70)
        print("AI가 선택 중...")
        print("-"*70)

        # Tool 사용용 LLM에게 선택 요청
        tool_prompt = "가위바위보 게임을 위해 play_rock_paper_scissors 도구를 사용하여 당신의 선택을 결정하세요."
        response = tool_llm_with_tools.invoke(tool_prompt)

        # Tool 호출 결과 추출
        ai_choice = None
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            # Tool 실행
            ai_choice = play_rock_paper_scissors.invoke(tool_call['args'])
        else:
            # Fallback: Tool을 사용하지 않은 경우 직접 호출
            ai_choice = play_rock_paper_scissors.invoke({})

        print(f"AI의 선택: {ai_choice}")

        # 승부 판정
        result = judge_winner(user_input, ai_choice)

        result_text = {
            "win": "유저 승리!",
            "lose": "AI 승리!",
            "draw": "무승부!"
        }[result]

        print(f"결과: {result_text}\n")

        # 해설 생성
        print("="*70)
        print("해설위원의 코멘트")
        print("="*70)

        commentary = commentary_chain.invoke({
            "user_choice": user_input,
            "ai_choice": ai_choice,
            "result": result_text
        })

        print(f"\n{commentary}\n")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
