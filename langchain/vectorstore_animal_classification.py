"""
FAISS VectorStore와 Embedding을 사용한 동물 분류 검색
동물 이름을 벡터스토어에 저장하고, 분류명으로 검색하여 해당하는 동물들을 찾습니다.
HuggingFace의 sentence-transformers를 사용하여 로컬에서 임베딩을 생성합니다.
"""
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_animal_documents():
    """
    동물 이름 데이터를 Document 형태로 생성
    각 동물의 이름만 저장합니다.
    """
    # 동물 목록 (각 분류별 5마리씩)
    animals = [
        # 포유류
        "강아지", "고양이", "호랑이", "고래", "코끼리",
        # 양서류
        "개구리", "도롱뇽", "맹꽁이", "두꺼비", "영원",
        # 파충류
        "뱀", "도마뱀", "거북이", "악어", "카멜레온",
        # 조류
        "독수리", "참새", "펭귄", "앵무새", "타조",
        # 어류
        "고등어", "연어", "상어", "금붕어", "참치"
    ]

    documents = []
    for animal in animals:
        doc = Document(
            page_content=animal
        )
        documents.append(doc)

    return documents


def create_vectorstore():
    """
    FAISS VectorStore를 생성하고 동물 이름 데이터를 저장합니다.

    Returns:
        FAISS: 동물 이름 데이터가 저장된 벡터스토어
    """
    print("\n" + "="*70)
    print("VectorStore 생성 중...")
    print("="*70)

    # HuggingFace Embedding 모델 초기화
    # all-MiniLM-L6-v2: 경량 모델로 빠른 임베딩 생성 가능
    print("HuggingFace 임베딩 모델 로딩 중... (최초 실행 시 모델 다운로드)")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Document 생성
    documents = create_animal_documents()

    print(f"\n저장할 동물 개수: {len(documents)}")
    print("동물 목록:")
    for i, doc in enumerate(documents, 1):
        print(f"{i:2d}. {doc.page_content}", end="  ")
        if i % 5 == 0:
            print()

    # FAISS VectorStore 생성
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    print("\nVectorStore 생성 완료!")
    print("="*70)

    return vectorstore


def search_animals_by_category(vectorstore, category: str, k: int = 3):
    """
    분류명으로 동물을 검색합니다.

    Args:
        vectorstore: FAISS VectorStore
        category: 검색할 분류명 (예: 포유류, 양서류)
        k: 반환할 결과 개수 (기본값: 3)

    Returns:
        list: 유사도가 높은 동물 목록
    """
    print(f"\n검색어: {category}")
    print("-" * 70)

    # similarity_search_with_score로 유사한 동물 찾기
    results = vectorstore.similarity_search_with_score(query=category, k=k)

    print(f"\n상위 {k}개 결과 (유사도 점수 포함):")
    for i, (doc, score) in enumerate(results, 1):
        animal_name = doc.page_content
        print(f"{i}. {animal_name:10s} - 유사도 점수: {score:.4f}")

    return results


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("FAISS VectorStore를 사용한 동물 분류 검색")
    print("="*70)

    # VectorStore 생성
    vectorstore = create_vectorstore()

    print("\n" + "="*70)
    print("동물 분류 검색 시작")
    print("="*70)
    print("\n분류명을 입력하세요. (종료: 'quit')")
    print("예시: 포유류, 양서류, 파충류, 조류, 어류")

    while True:
        print("\n" + "="*70)
        user_input = input("\n분류명 입력: ").strip()

        if user_input.lower() == 'quit':
            print("\n프로그램을 종료합니다.\n")
            break

        if not user_input:
            print("분류명을 입력해주세요.")
            continue

        # 동물 검색 (상위 3개)
        search_animals_by_category(vectorstore, user_input, k=3)

        print("\n" + "-"*70)


if __name__ == "__main__":
    main()
