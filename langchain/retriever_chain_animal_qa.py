"""
Retriever와 Chain을 사용한 동물 정보 질의응답 시스템
FAISS VectorStore를 Retriever로 변환하고, Gemini LLM과 Chain으로 연결하여
사용자 질문에 대해 관련 동물 정보를 기반으로 답변합니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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


def create_animal_info_documents():
    """
    동물 정보를 담은 Document 생성
    각 동물의 특징을 2-3문장으로 설명합니다.

    Returns:
        list: Document 객체 리스트
    """
    animal_info = [
        {
            "name": "강아지",
            "description": "강아지는 충성심이 강하고 사람을 잘 따르는 인기 있는 반려동물입니다. "
                          "산책과 놀이를 좋아하며 가족 구성원으로 애정을 주고받습니다. "
                          "훈련이 쉽고 키우기 좋으며 집을 지키는 역할도 할 수 있습니다."
        },
        {
            "name": "고양이",
            "description": "고양이는 독립적이면서도 애교가 많은 인기 반려동물입니다. "
                          "스스로 그루밍을 하며 화장실 훈련이 쉽습니다. "
                          "조용하고 관리가 편하며 실내에서 키우기 매우 적합합니다."
        },
        {
            "name": "토끼",
            "description": "토끼는 온순하고 귀여운 외모를 가진 인기 있는 초식동물입니다. "
                          "건초와 채소를 주식으로 하며 조용하고 키우기 쉬운 성격입니다. "
                          "케이지에서 키울 수 있어 공간 활용이 좋고 관리가 편합니다."
        },
        {
            "name": "햄스터",
            "description": "햄스터는 작고 귀여우며 키우기 매우 쉬운 설치류입니다. "
                          "야행성이며 케이지 안에서 활동합니다. "
                          "공간을 적게 차지하고 초보자에게 최적의 반려동물입니다."
        },
        {
            "name": "금붕어",
            "description": "금붕어는 수조에서 키우는 관상용 물고기로 키우기 쉽습니다. "
                          "조용하고 관리가 간단하며 스트레스가 적습니다. "
                          "먹이 주기와 물갈이만 신경 쓰면 되어 초보자에게 좋습니다."
        },
        {
            "name": "앵무새",
            "description": "앵무새는 말을 배우고 사람과 소통할 수 있는 지능적인 새입니다. "
                          "화려한 깃털과 높은 지능을 가지고 있습니다. "
                          "케이지 관리와 정기적인 교감이 필요하지만 키우는 재미가 있습니다."
        },
        {
            "name": "거북이",
            "description": "거북이는 느리고 조용한 파충류로 키우기 쉽습니다. "
                          "수명이 길고 관리가 비교적 쉽습니다. "
                          "수조나 사육장에서 키우며 자외선 램프가 필요합니다."
        },
        {
            "name": "뱀",
            "description": "뱀은 조용하고 공간을 적게 차지하는 파충류입니다. "
                          "먹이로 쥐나 병아리를 주어야 하며 온도 관리가 중요합니다. "
                          "전문적인 사육 지식이 필요한 특수 반려동물입니다."
        },
        {
            "name": "이구아나",
            "description": "이구아나는 대형 파충류로 성체는 1미터 이상 자랍니다. "
                          "채식을 하며 넓은 사육 공간과 높은 온도가 필요합니다. "
                          "전문적인 지식과 많은 관리 시간이 요구됩니다."
        },
        {
            "name": "타란툴라",
            "description": "타란툴라는 대형 거미로 곤충을 먹이로 합니다. "
                          "탈피 과정이 있으며 독이 있어 주의가 필요합니다. "
                          "전문가용으로 특별한 관리 지식이 요구됩니다."
        }
    ]

    documents = []
    for animal in animal_info:
        # Document 생성: page_content에 이름과 설명을 함께 저장
        content = f"{animal['name']}: {animal['description']}"
        doc = Document(
            page_content=content,
            metadata={"animal_name": animal['name']}
        )
        documents.append(doc)

    return documents


def create_vectorstore_and_retriever():
    """
    VectorStore를 생성하고 Retriever로 변환합니다.

    Returns:
        tuple: (vectorstore, retriever)
    """
    print("\n" + "="*70)
    print("VectorStore와 Retriever 생성")
    print("="*70)

    # HuggingFace Embedding 모델 초기화
    print("\nHuggingFace 임베딩 모델 로딩 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Document 생성
    documents = create_animal_info_documents()
    print(f"생성된 Document 개수: {len(documents)}")

    # TextSplitter 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Document 분할
    split_documents = text_splitter.split_documents(documents)
    print(f"분할된 Document 개수: {len(split_documents)}")

    # FAISS VectorStore 생성
    print("\nFAISS VectorStore 생성 중...")
    vectorstore = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings
    )

    # VectorStore를 Retriever로 변환
    print("VectorStore를 Retriever로 변환 중...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 상위 3개 문서 검색
    )

    print("VectorStore 및 Retriever 생성 완료!")
    print("="*70)

    return vectorstore, retriever


def create_qa_chain(retriever):
    """
    Retriever와 LLM을 연결하는 QA Chain을 생성합니다.

    Args:
        retriever: VectorStore Retriever

    Returns:
        Chain: 질의응답 체인
    """
    print("\n" + "="*70)
    print("QA Chain 생성")
    print("="*70)

    # Gemini LLM 초기화
    print("\nGemini LLM 초기화 중...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )

    # PromptTemplate 생성
    template = """당신은 반려동물 전문가입니다. 주어진 동물 정보를 바탕으로 사용자의 질문에 답변해주세요.

동물 정보:
{context}

질문: {question}

답변: 주어진 동물 정보를 바탕으로 구체적이고 유용한 답변을 제공해주세요."""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    # Chain 생성
    print("Chain 구성 중...")
    print("  - Retriever: 질문과 관련된 동물 정보 검색")
    print("  - Prompt: 검색된 정보를 컨텍스트로 활용")
    print("  - LLM: Gemini로 답변 생성")

    def format_docs(docs):
        """검색된 문서를 포맷팅"""
        return "\n\n".join([doc.page_content for doc in docs])

    # Chain 구성: Retriever -> format -> Prompt -> LLM -> Parser
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("QA Chain 생성 완료!")
    print("="*70)

    return chain


def ask_question(chain, question: str):
    """
    Chain을 통해 질문하고 답변을 받습니다.

    Args:
        chain: QA Chain
        question: 사용자 질문

    Returns:
        str: LLM의 답변
    """
    print("\n" + "="*70)
    print(f"질문: {question}")
    print("="*70)

    print("\n답변 생성 중...\n")

    # Chain 실행
    answer = chain.invoke(question)

    print("-" * 70)
    print(f"\n답변:\n{answer}\n")
    print("="*70)

    return answer


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("Retriever와 Chain을 사용한 동물 정보 질의응답 시스템")
    print("="*70)

    # VectorStore와 Retriever 생성
    vectorstore, retriever = create_vectorstore_and_retriever()

    # QA Chain 생성
    chain = create_qa_chain(retriever)

    # 샘플 질문들
    questions = [
        "애완동물로 키우기 좋은 동물은 무엇인가요?",
        "실내에서 키우기 적합한 동물을 추천해주세요.",
        "초보자가 키우기 쉬운 동물은 무엇인가요?"
    ]

    print("\n" + "="*70)
    print("샘플 질문으로 테스트")
    print("="*70)

    for question in questions:
        ask_question(chain, question)
        print("\n")

    print("\n질의응답 완료!\n")


if __name__ == "__main__":
    main()
