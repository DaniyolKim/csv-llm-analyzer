import os
import argparse
import chromadb
import ollama
from utils import clean_text

def load_chroma_collection(persist_directory="./chroma_db", collection_name="csv_test"):
    """
    저장된 ChromaDB 컬렉션을 로드합니다.
    
    Args:
        persist_directory (str): 데이터베이스 저장 경로
        collection_name (str): 컬렉션 이름
        
    Returns:
        chromadb.Collection: ChromaDB 컬렉션
    """
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"ChromaDB를 찾을 수 없습니다: {persist_directory}")
    
    # ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 컬렉션 로드
    try:
        collection = client.get_collection(collection_name)
        return collection
    except Exception as e:
        raise ValueError(f"컬렉션을 로드할 수 없습니다: {e}")

def get_ollama_models():
    """
    Ollama 라이브러리를 사용하여 설치된 모델 목록을 가져옵니다.
    
    Returns:
        list: 설치된 모델 목록
    """
    try:
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except ImportError:
        return []
    except Exception:
        return []

def rag_query(collection, query, model_name="llama2", n_results=5, similarity_threshold=0.7, metadata_filter=None):
    """
    RAG 시스템에 질의합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query (str): 질의 텍스트
        model_name (str): Ollama 모델 이름
        n_results (int): 검색할 결과 수, 0이면 최대 20개 문서 사용, 최소 3개 문서 사용
        similarity_threshold (float): 유사도 임계값 (0~1 사이, 높을수록 더 유사한 문서만 사용)
        metadata_filter (dict): 메타데이터 필터링 조건 (예: {"source": "특정 출처"})
        
    Returns:
        dict: 질의 결과
    """
    # 쿼리 텍스트 정제
    cleaned_query = clean_text(query)
    
    # n_results가 0이면 최대 20개 문서로 제한
    if n_results == 0:
        n_results = 20  # 일반적인 최대값
    
    # n_results가 None이면 기본값 5 사용
    elif n_results is None:
        n_results = 5
        
    # n_results가 3보다 작으면 최소 3으로 설정
    elif n_results < 3:
        n_results = 3
        
    # n_results가 20보다 크면 최대 20으로 제한
    elif n_results > 20:
        n_results = 20
    
    # 검색 시 메타데이터 필터 적용
    where_clause = metadata_filter if metadata_filter else None
    
    # ChromaDB에서 관련 문서 검색 (메타데이터 필터 적용)
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=n_results * 2,  # 필터링을 위해 더 많은 결과 검색
        where=where_clause
    )
    
    # 유사도 점수 필터링 (거리가 작을수록 더 유사함)
    filtered_docs = []
    filtered_metadatas = []
    filtered_distances = []
    
    # 거리를 유사도로 변환 (1 - 거리)하여 필터링
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - distance  # 거리를 유사도로 변환
        if similarity >= similarity_threshold:
            filtered_docs.append(doc)
            filtered_metadatas.append(metadata)
            filtered_distances.append(distance)
    
    # 필터링된 문서가 없으면 가장 유사한 문서 하나만 사용
    if not filtered_docs and results["documents"][0]:
        best_idx = results["distances"][0].index(min(results["distances"][0]))
        filtered_docs = [results["documents"][0][best_idx]]
        filtered_metadatas = [results["metadatas"][0][best_idx]]
        filtered_distances = [results["distances"][0][best_idx]]
    
    # 필터링된 문서 수가 n_results보다 많으면 상위 n_results개만 사용
    if len(filtered_docs) > n_results:
        filtered_docs = filtered_docs[:n_results]
        filtered_metadatas = filtered_metadatas[:n_results]
        filtered_distances = filtered_distances[:n_results]
    
    # 필터링된 문서로 컨텍스트 구성
    context = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(filtered_docs)])
    
    # 메타데이터 정보 추가
    metadata_info = "\n\n".join([f"문서 {i+1} 메타데이터: {metadata}" for i, metadata in enumerate(filtered_metadatas)])
    
    # Ollama에 질의 (개선된 프롬프트)
    prompt = f"""
    다음 정보를 바탕으로 질문에 답변해주세요. 제공된 정보에 없는 내용은 답변하지 마세요.
    정보에 답이 없다면 "제공된 정보에서 답을 찾을 수 없습니다"라고 솔직하게 답변하세요.
    
    정보:
    {context}
    
    메타데이터 정보:
    {metadata_info}
    
    질문: {query}
    답변:
    """
    
    response = query_ollama(prompt, model_name)
    
    return {
        "query": query,
        "context": filtered_docs,
        "metadatas": filtered_metadatas,
        "distances": filtered_distances,
        "response": response
    }

def interactive_mode(collection, model_name="llama2"):
    """
    대화형 모드로 RAG 시스템을 실행합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        model_name (str): Ollama 모델 이름
    """
    print("\n=== RAG 시스템 대화형 모드 ===")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.\n")
    
    while True:
        query = input("\n질문: ")
        if query.lower() in ["exit", "quit"]:
            print("대화형 모드를 종료합니다.")
            break
        
        print("\n답변 생성 중...")
        result = rag_query(collection, query, model_name)
        
        print("\n=== 답변 ===")
        print(result["response"])
        
        print("\n=== 참조 문서 ===")
        for i, (doc, metadata) in enumerate(zip(result["context"], result["metadatas"])):
            print(f"\n문서 {i+1}:")
            print(f"내용: {doc}")
            print(f"메타데이터: {metadata}")

def interactive_mode_enhanced(collection, model_name="llama2", similarity_threshold=0.7, metadata_filter=None):
    """
    개선된 대화형 모드로 RAG 시스템을 실행합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        model_name (str): Ollama 모델 이름
        similarity_threshold (float): 유사도 임계값
        metadata_filter (dict): 메타데이터 필터
    """
    print("\n=== 개선된 RAG 시스템 대화형 모드 ===")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print(f"유사도 임계값: {similarity_threshold}")
    if metadata_filter:
        print(f"메타데이터 필터: {metadata_filter}")
    print()
    
    while True:
        query = input("\n질문: ")
        if query.lower() in ["exit", "quit"]:
            print("대화형 모드를 종료합니다.")
            break
        
        # 메타데이터 필터 설정
        if query.startswith("!metadata "):
            try:
                import json
                metadata_str = query[10:].strip()
                if metadata_str.lower() == "clear":
                    metadata_filter = None
                    print("메타데이터 필터가 초기화되었습니다.")
                else:
                    metadata_filter = json.loads(metadata_str)
                    print(f"메타데이터 필터가 설정되었습니다: {metadata_filter}")
                continue
            except json.JSONDecodeError:
                print("메타데이터 형식이 잘못되었습니다. 올바른 JSON 형식을 사용하세요.")
                continue
        
        # 유사도 임계값 설정
        if query.startswith("!similarity "):
            try:
                new_threshold = float(query[11:].strip())
                if 0 <= new_threshold <= 1:
                    similarity_threshold = new_threshold
                    print(f"유사도 임계값이 {similarity_threshold}로 설정되었습니다.")
                else:
                    print("유사도 임계값은 0과 1 사이의 값이어야 합니다.")
                continue
            except ValueError:
                print("유효한 숫자를 입력하세요.")
                continue
        
        print("\n답변 생성 중...")
        result = rag_query(
            collection, 
            query, 
            model_name,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter
        )
        
        print("\n=== 답변 ===")
        print(result["response"])
        
        print("\n=== 참조 문서 ===")
        for i, (doc, metadata, distance) in enumerate(zip(result["context"], result["metadatas"], result["distances"])):
            similarity = 1 - distance
            print(f"\n문서 {i+1} (유사도: {similarity:.4f}):")
            print(f"내용: {doc}")
            print(f"메타데이터: {metadata}")
            
        # 참조 문서가 없는 경우
        if not result["context"]:
            print("\n참조 문서가 없습니다. 유사도 임계값을 낮추거나 메타데이터 필터를 조정해보세요.")

def main():
    parser = argparse.ArgumentParser(description="Ollama를 사용한 RAG 시스템")
    parser.add_argument("--db_path", default="./chroma_db", help="ChromaDB 경로")
    parser.add_argument("--collection", default="csv_test", help="ChromaDB 컬렉션 이름")
    parser.add_argument("--model", default="llama2", help="Ollama 모델 이름")
    parser.add_argument("--query", help="실행할 질의 (지정하지 않으면 대화형 모드)")
    parser.add_argument("--similarity", type=float, default=0.7, help="유사도 임계값 (0~1 사이)")
    parser.add_argument("--metadata", help="메타데이터 필터 (JSON 형식, 예: '{\"source\":\"특정 출처\"}')")
    
    args = parser.parse_args()
    
    try:
        # ChromaDB 컬렉션 로드
        print(f"ChromaDB를 로드합니다: {args.db_path}, 컬렉션: {args.collection}")
        collection = load_chroma_collection(args.db_path, args.collection)
        
        # 메타데이터 필터 파싱
        metadata_filter = None
        if args.metadata:
            try:
                import json
                metadata_filter = json.loads(args.metadata)
                print(f"메타데이터 필터 적용: {metadata_filter}")
            except json.JSONDecodeError:
                print(f"메타데이터 필터 형식이 잘못되었습니다: {args.metadata}")
                print("올바른 JSON 형식을 사용하세요. 예: '{\"source\":\"특정 출처\"}'")
        
        try:
            import ollama
            print("Ollama 라이브러리를 사용합니다.")
        except ImportError:
            print("Ollama 라이브러리를 찾을 수 없습니다. 'pip install ollama'를 실행하세요.")
            raise
        
        if args.query:
            # 단일 질의 모드
            print(f"\n질의: {args.query}")
            print(f"유사도 임계값: {args.similarity}")
            print("\n답변 생성 중...")
            result = rag_query(
                collection, 
                args.query, 
                args.model,
                similarity_threshold=args.similarity,
                metadata_filter=metadata_filter
            )
            
            print("\n=== 답변 ===")
            print(result["response"])
            
            print("\n=== 참조 문서 ===")
            for i, (doc, metadata, distance) in enumerate(zip(result["context"], result["metadatas"], result["distances"])):
                similarity = 1 - distance
                print(f"\n문서 {i+1} (유사도: {similarity:.4f}):")
                print(f"내용: {doc}")
                print(f"메타데이터: {metadata}")
        else:
            # 대화형 모드
            print(f"유사도 임계값: {args.similarity}")
            interactive_mode_enhanced(collection, args.model, similarity_threshold=args.similarity, metadata_filter=metadata_filter)
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()