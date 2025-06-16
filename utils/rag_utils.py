"""
RAG(Retrieval-Augmented Generation) 시스템 관련 유틸리티 함수 모음
"""
from typing import Dict, Any, Optional, List
from .text_utils import clean_text
from .ollama_utils import query_ollama, chat_with_ollama
from .chroma_utils import hybrid_query_chroma
def rag_query_with_ollama(collection, query: str, model_name: str = "llama2", n_results: int = 5, 
                          similarity_threshold: Optional[float] = None, ollama_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    RAG 시스템에 질의합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query (str): 질의 텍스트 (프롬프트와 질문이 결합된 형태일 수 있음)
        model_name (str): Ollama 모델 이름
        n_results (int): 검색할 결과 수, 0이면 최대 20개 문서 사용, 최소 3개 문서 사용
        similarity_threshold (float, optional): 유사도 임계값 (0~1 사이). 이 값보다 유사도가 낮은 문서는 제외됨
        ollama_options (Optional[Dict[str, Any]]): Ollama API에 전달할 추가 옵션
        
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
    
    # 하이브리드 검색 사용 (임베딩 + 키워드 검색)
    results = hybrid_query_chroma(collection, cleaned_query, n_results=n_results)
    
    # 유사도 임계값이 설정된 경우 필터링 적용
    filtered_docs = []
    filtered_metadatas = []
    filtered_distances = []
    filtered_search_types = []
    
    if similarity_threshold is not None and 0 <= similarity_threshold <= 1:
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        )):
            # ChromaDB의 distance는 거리 개념이므로 1에서 빼서 유사도로 변환 (1에 가까울수록 유사)
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
                if "search_type" in results:
                    filtered_search_types.append(results["search_type"][0][i])
    else:
        # 임계값이 없으면 모든 결과 사용
        filtered_docs = results["documents"][0]
        filtered_metadatas = results["metadatas"][0]
        filtered_distances = results["distances"][0]
        if "search_type" in results:
            filtered_search_types = results["search_type"][0]
    
    # 필터링 후 문서가 없는 경우 처리
    if not filtered_docs:
        return {
            "query": query,
            "context": [],
            "metadatas": [],
            "distances": [],
            "search_types": [],
            "response": f"유사도 임계값({similarity_threshold})을 충족하는 문서를 찾을 수 없습니다. 임계값을 낮추거나 다른 질문을 시도해보세요."
        }
    
    # 검색 결과를 컨텍스트로 사용
    context = "\n".join(filtered_docs)
    
    # 사용자 입력이 프롬프트와 질문으로 구성되어 있는지 확인
    if "질문:" in query or "Question:" in query:
        # 사용자가 직접 프롬프트와 질문을 입력한 경우, 그대로 사용
        prompt = f"""
    다음 정보를 바탕으로 응답해주세요:
    
    {context}
    
    {query}
    """
    else:
        # 기존 방식 - 질문 형식으로 구성
        prompt = f"""
    다음 정보를 바탕으로 질문에 답변해주세요:
    
    {context}
    
    질문: {query}
    답변:
    """
    
    response = query_ollama(prompt, model_name, ollama_options=ollama_options)
    
    return {
        "query": query,
        "context": filtered_docs,
        "metadatas": filtered_metadatas,
        "distances": filtered_distances,
        "search_types": filtered_search_types if filtered_search_types else [],
        "response": response
    }

def rag_chat_with_ollama(collection, query: str, model_name: str = "llama2", n_results: int = 5, 
                          similarity_threshold: Optional[float] = None, system_prompt: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None, ollama_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    역할 기반 대화를 지원하는 RAG 시스템에 질의합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query (str): 사용자 질의 텍스트
        model_name (str): Ollama 모델 이름
        n_results (int): 검색할 결과 수
        similarity_threshold (float, optional): 유사도 임계값 (0~1 사이)
        system_prompt (str, optional): 시스템 프롬프트
        chat_history (list, optional): 이전 대화 기록 [{'role': 'user|assistant', 'content': '내용'}, ...]
        ollama_options (Optional[Dict[str, Any]]): Ollama API에 전달할 추가 옵션
        
    Returns:
        dict: 질의 결과
    """
    # 쿼리 텍스트 정제
    cleaned_query = clean_text(query)
    
    # n_results 처리
    if n_results == 0:
        n_results = 20  # 일반적인 최대값
    elif n_results is None:
        n_results = 5
    elif n_results < 3:
        n_results = 3
    elif n_results > 20:
        n_results = 20
    
    # 하이브리드 검색 사용 (임베딩 + 키워드 검색)
    results = hybrid_query_chroma(collection, cleaned_query, n_results=n_results)
    
    # 유사도 임계값이 설정된 경우 필터링 적용
    filtered_docs = []
    filtered_metadatas = []
    filtered_distances = []
    
    if similarity_threshold is not None and 0 <= similarity_threshold <= 1:
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        )):
            # ChromaDB의 distance는 거리 개념이므로 1에서 빼서 유사도로 변환
            similarity = 1 - distance
            if similarity >= similarity_threshold:
                filtered_docs.append(doc)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)
    else:
        # 임계값이 없으면 모든 결과 사용
        filtered_docs = results["documents"][0]
        filtered_metadatas = results["metadatas"][0]
        filtered_distances = results["distances"][0]
    
    # 필터링 후 문서가 없는 경우 처리
    if not filtered_docs:
        return {
            "query": query,
            "context": [],
            "metadatas": [],
            "distances": [],
            "response": f"유사도 임계값({similarity_threshold})을 충족하는 문서를 찾을 수 없습니다."
        }
    
    # 검색 결과를 컨텍스트로 사용
    context = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(filtered_docs)])
    
    # 메시지 구성
    messages = []
    
    # 시스템 프롬프트 추가
    if system_prompt:
        system_content = system_prompt
    else:
        system_content = f"""다음 정보를 바탕으로 질문에 답변해주세요. 제공된 정보에 없는 내용은 답변하지 마세요.
정보에 답이 없다면 "제공된 정보에서 답을 찾을 수 없습니다"라고 솔직하게 답변하세요.

참조 정보:
{context}"""
    
    messages.append({"role": "system", "content": system_content})
    
    # 이전 대화 기록 추가
    if chat_history:
        messages.extend(chat_history)
    
    # 현재 질문 추가
    messages.append({"role": "user", "content": query})
    
    # 역할 기반 대화 실행
    response = chat_with_ollama(messages, model_name, ollama_options=ollama_options)
    
    return {
        "query": query,
        "context": filtered_docs,
        "metadatas": filtered_metadatas,
        "distances": filtered_distances,
        "response": response,
        "messages": messages
    }

def rag_query_with_metadata_filter(collection, query: str, model_name: str = "llama2", n_results: int = 5, 
                                  similarity_threshold: Optional[float] = None, metadata_filter: Optional[Dict[str, Any]] = None, ollama_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    메타데이터 필터링을 지원하는 RAG 시스템에 질의합니다.
    
    Args:
        collection (chromadb.Collection): ChromaDB 컬렉션
        query (str): 질의 텍스트
        model_name (str): Ollama 모델 이름
        n_results (int): 검색할 결과 수
        similarity_threshold (float, optional): 유사도 임계값 (0~1 사이)
        metadata_filter (dict, optional): 메타데이터 필터 (예: {"source": "row_1"})
        ollama_options (Optional[Dict[str, Any]]): Ollama API에 전달할 추가 옵션
        
    Returns:
        dict: 질의 결과
    """
    # 쿼리 텍스트 정제
    cleaned_query = clean_text(query)
    
    # n_results 처리
    if n_results == 0 or n_results > 20:
        n_results = 20
    elif n_results is None or n_results < 3:
        n_results = 5
    
    # 하이브리드 검색 사용 (임베딩 + 키워드 검색)
    # 현재 하이브리드 검색은 메타데이터 필터를 직접 지원하지 않으므로 결과를 후처리
    results = hybrid_query_chroma(collection, cleaned_query, n_results=n_results * 2)  # 더 많은 결과를 가져와서 필터링
    
    # 결과가 없는 경우 처리
    if not results["documents"][0]:
        return {
            "query": query,
            "context": [],
            "metadatas": [],
            "distances": [],
            "search_types": [],
            "response": "검색 결과가 없습니다."
        }
    
    # 메타데이터 필터링 및 유사도 임계값 적용
    filtered_docs = []
    filtered_metadatas = []
    filtered_distances = []
    filtered_search_types = []
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    )):
        # 메타데이터 필터 적용
        if metadata_filter:
            match = True
            for key, value in metadata_filter.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if not match:
                continue
        
        # 유사도 임계값 적용
        if similarity_threshold is not None and 0 <= similarity_threshold <= 1:
            similarity = 1 - distance
            if similarity < similarity_threshold:
                continue
        
        # 필터를 통과한 결과 추가
        filtered_docs.append(doc)
        filtered_metadatas.append(metadata)
        filtered_distances.append(distance)
        if "search_type" in results:
            filtered_search_types.append(results["search_type"][0][i])
        
        # 최대 결과 수에 도달하면 중단
        if len(filtered_docs) >= n_results:
            break
    
    # 필터링 후 문서가 없는 경우 처리
    if not filtered_docs:
        return {
            "query": query,
            "context": [],
            "metadatas": [],
            "distances": [],
            "search_types": [],
            "response": "메타데이터 필터 조건이나 유사도 임계값을 충족하는 문서를 찾을 수 없습니다."
        }
    
    # 검색 결과를 컨텍스트로 사용
    context = "\n\n".join([f"문서 {i+1}:\n{doc}" for i, doc in enumerate(filtered_docs)])
    
    # 메타데이터 정보 추가
    metadata_info = "\n\n".join([f"문서 {i+1} 메타데이터: {metadata}" for i, metadata in enumerate(filtered_metadatas)])
    
    # 프롬프트 구성
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
    
    response = query_ollama(prompt, model_name, ollama_options=ollama_options)
    
    return {
        "query": query,
        "context": filtered_docs,
        "metadatas": filtered_metadatas,
        "distances": filtered_distances,
        "search_types": filtered_search_types if filtered_search_types else [],
        "response": response
    }