import os
import argparse
import chromadb
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

def query_ollama_api(prompt, model_name="llama2"):
    """
    HTTP 요청을 통해 Ollama 모델에 질의합니다.
    
    Args:
        prompt (str): 프롬프트
        model_name (str): Ollama 모델 이름
        
    Returns:
        str: 모델 응답
    """
    try:
        import requests
        
        # Ollama API 호출
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"오류: {response.status_code} - {response.text}"
    
    except ImportError:
        return "requests 패키지가 설치되어 있지 않습니다. 'pip install requests'를 실행하세요."
    except Exception as e:
        return f"Ollama 서버 연결 오류: {e}"

def query_ollama_lib(prompt, model_name="llama2"):
    """
    Ollama 라이브러리를 사용하여 모델에 질의합니다.
    
    Args:
        prompt (str): 프롬프트
        model_name (str): Ollama 모델 이름
        
    Returns:
        str: 모델 응답
    """
    try:
        import ollama
        
        # Ollama 라이브러리 사용
        response = ollama.generate(model=model_name, prompt=prompt)
        return response['response']
    
    except ImportError:
        return "ollama 패키지가 설치되어 있지 않습니다. 'pip install ollama'를 실행하세요."
    except Exception as e:
        return f"Ollama 라이브러리 오류: {e}"

def query_ollama(prompt, model_name="llama2", use_lib=True):
    """
    Ollama 모델에 질의합니다.
    
    Args:
        prompt (str): 프롬프트
        model_name (str): Ollama 모델 이름
        use_lib (bool): Ollama 라이브러리 사용 여부
        
    Returns:
        str: 모델 응답
    """
    if use_lib:
        try:
            return query_ollama_lib(prompt, model_name)
        except Exception:
            # 라이브러리 방식이 실패하면 API 방식으로 폴백
            return query_ollama_api(prompt, model_name)
    else:
        return query_ollama_api(prompt, model_name)

def get_ollama_models_lib():
    """
    Ollama 라이브러리를 사용하여 설치된 모델 목록을 가져옵니다.
    
    Returns:
        list: 설치된 모델 목록
    """
    try:
        import ollama
        
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except ImportError:
        return []
    except Exception:
        return []

def rag_query(collection, query, model_name="llama2", n_results=5, use_lib=True):
    """
    RAG 시스템에 질의합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query (str): 질의 텍스트
        model_name (str): Ollama 모델 이름
        n_results (int): 검색할 결과 수, 0이면 최대 20개 문서 사용, 최소 3개 문서 사용
        use_lib (bool): Ollama 라이브러리 사용 여부
        
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
    
    # ChromaDB에서 관련 문서 검색
    results = collection.query(
        query_texts=[cleaned_query],
        n_results=n_results
    )
    
    # 검색 결과를 컨텍스트로 사용
    context = "\n".join(results["documents"][0])
    
    # Ollama에 질의
    prompt = f"""
    다음 정보를 바탕으로 질문에 답변해주세요:
    
    {context}
    
    질문: {query}
    답변:
    """
    
    response = query_ollama(prompt, model_name, use_lib)
    
    return {
        "query": query,
        "context": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "response": response
    }

def interactive_mode(collection, model_name="llama2", use_lib=True):
    """
    대화형 모드로 RAG 시스템을 실행합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        model_name (str): Ollama 모델 이름
        use_lib (bool): Ollama 라이브러리 사용 여부
    """
    print("\n=== RAG 시스템 대화형 모드 ===")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.\n")
    
    while True:
        query = input("\n질문: ")
        if query.lower() in ["exit", "quit"]:
            print("대화형 모드를 종료합니다.")
            break
        
        print("\n답변 생성 중...")
        result = rag_query(collection, query, model_name, use_lib=use_lib)
        
        print("\n=== 답변 ===")
        print(result["response"])
        
        print("\n=== 참조 문서 ===")
        for i, (doc, metadata) in enumerate(zip(result["context"], result["metadatas"])):
            print(f"\n문서 {i+1}:")
            print(f"내용: {doc}")
            print(f"메타데이터: {metadata}")

def main():
    parser = argparse.ArgumentParser(description="Ollama를 사용한 RAG 시스템")
    parser.add_argument("--db_path", default="./chroma_db", help="ChromaDB 경로")
    parser.add_argument("--collection", default="csv_test", help="ChromaDB 컬렉션 이름")
    parser.add_argument("--model", default="llama2", help="Ollama 모델 이름")
    parser.add_argument("--query", help="실행할 질의 (지정하지 않으면 대화형 모드)")
    parser.add_argument("--api", action="store_true", help="API 방식 사용 (라이브러리 대신)")
    
    args = parser.parse_args()
    
    try:
        # ChromaDB 컬렉션 로드
        print(f"ChromaDB를 로드합니다: {args.db_path}, 컬렉션: {args.collection}")
        collection = load_chroma_collection(args.db_path, args.collection)
        
        # Ollama 라이브러리 사용 여부
        use_lib = not args.api
        
        if use_lib:
            try:
                import ollama
                print("Ollama 라이브러리를 사용합니다.")
            except ImportError:
                print("Ollama 라이브러리를 찾을 수 없습니다. API 방식으로 전환합니다.")
                print("Ollama 라이브러리를 설치하려면: pip install ollama")
                use_lib = False
        
        if args.query:
            # 단일 질의 모드
            print(f"\n질의: {args.query}")
            print("\n답변 생성 중...")
            result = rag_query(collection, args.query, args.model, use_lib=use_lib)
            
            print("\n=== 답변 ===")
            print(result["response"])
            
            print("\n=== 참조 문서 ===")
            for i, (doc, metadata) in enumerate(zip(result["context"], result["metadatas"])):
                print(f"\n문서 {i+1}:")
                print(f"내용: {doc}")
                print(f"메타데이터: {metadata}")
        else:
            # 대화형 모드
            interactive_mode(collection, args.model, use_lib=use_lib)
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()