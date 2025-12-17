"""
Document와 TextSplitter를 사용한 동물 정보 검색
동물의 특징을 담은 Document를 텍스트 분할하여 FAISS에 저장하고,
"애완동물로 키우기 좋은 동물"을 검색하여 유사도와 함께 반환합니다.
"""
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def create_vectorstore_with_splitter():
    """
    TextSplitter를 사용하여 문서를 분할하고 FAISS VectorStore를 생성합니다.

    Returns:
        FAISS: 동물 정보가 저장된 벡터스토어
    """
    print("\n" + "="*70)
    print("Document와 TextSplitter를 사용한 VectorStore 생성")
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
    print(f"\n생성된 Document 개수: {len(documents)}")

    # TextSplitter 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 청크 크기
        chunk_overlap=40,    # 청크 간 겹치는 부분
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    print("\nTextSplitter 설정:")
    print(f"  - chunk_size: 200")
    print(f"  - chunk_overlap: 40")

    # Document 분할
    split_documents = text_splitter.split_documents(documents)
    print(f"\n분할된 Document 개수: {len(split_documents)}")

    print("\n분할된 Document 샘플 (처음 3개):")
    print("-" * 70)
    for i, doc in enumerate(split_documents[:3], 1):
        print(f"\n[청크 {i}]")
        print(f"내용: {doc.page_content}")
        print(f"메타데이터: {doc.metadata}")
    print("-" * 70)

    # FAISS VectorStore 생성
    print("\nFAISS VectorStore 생성 중...")
    vectorstore = FAISS.from_documents(
        documents=split_documents,
        embedding=embeddings
    )

    print("VectorStore 생성 완료!")
    print("="*70)

    return vectorstore


def search_pet_friendly_animals(vectorstore):
    """
    "애완동물로 키우기 좋은 동물"을 검색하여 상위 3개를 반환합니다.

    Args:
        vectorstore: FAISS VectorStore

    Returns:
        list: 유사도 점수와 함께 검색된 결과
    """
    query = "애완동물로 키우기 좋은 동물"

    print("\n" + "="*70)
    print(f"검색 쿼리: '{query}'")
    print("="*70)

    # similarity_search_with_score로 상위 3개 검색
    results = vectorstore.similarity_search_with_score(query=query, k=3)

    print(f"\n상위 3개 결과 (유사도 점수 포함):")
    print("-" * 70)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}위]")
        print(f"동물: {doc.metadata.get('animal_name', 'Unknown')}")
        print(f"내용: {doc.page_content}")
        print(f"유사도 점수: {score:.4f}")

    print("\n" + "="*70)

    return results


def main():
    """메인 함수"""
    print("\n" + "="*70)
    print("Document와 TextSplitter를 사용한 동물 정보 검색 시스템")
    print("="*70)

    # VectorStore 생성
    vectorstore = create_vectorstore_with_splitter()

    # 애완동물로 키우기 좋은 동물 검색
    search_pet_friendly_animals(vectorstore)

    print("\n검색 완료!\n")


if __name__ == "__main__":
    main()
