import streamlit as st
import pandas as pd
import numpy as np
from text_utils import KOREAN_STOPWORDS, clean_text

# 데이터 로딩 및 표시 함수
def load_collection_data(collection):
    """컬렉션에서 모든 데이터를 로드하여 DataFrame으로 반환"""
    try:
        all_data = collection.get()
        
        if all_data and all_data["documents"]:
            # 결과를 데이터프레임으로 변환
            result_data = []
            for i, (doc, metadata, id) in enumerate(zip(
                all_data["documents"], 
                all_data["metadatas"],
                all_data["ids"]
            )):
                result_data.append({
                    "ID": id,
                    "출처": metadata.get("source", "알 수 없음"),
                    "청크": metadata.get("chunk", "알 수 없음"),
                    "키워드": metadata.get("keywords", "알 수 없음"),
                    "내용": doc,
                })
            
            # 데이터프레임 생성 및 반환
            return pd.DataFrame(result_data), all_data
        else:
            return None, None
    except Exception as e:
        raise Exception(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")

def display_collection_data(result_df):
    """컬렉션 데이터를 표시하는 함수"""
    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(width="medium"),
            "출처": st.column_config.TextColumn(width="small"),
            "청크": st.column_config.NumberColumn(width="small"),
            "키워드": st.column_config.TextColumn(width="medium", help="문서에서 추출된 주요 키워드입니다."),
            "내용": st.column_config.TextColumn(width="large"),
        }
    )
    
    # 데이터 통계
    st.subheader("데이터 통계")
    st.write(f"총 문서 수: {len(result_df)}")
    
    # 출처별 문서 수 계산
    source_counts = result_df["출처"].value_counts().reset_index()
    source_counts.columns = ["출처", "문서 수"]
    
    # 출처별 문서 수 차트
    st.bar_chart(source_counts.set_index("출처"))

# 검색 함수
def search_collection(collection, query, n_results=10):
    """컬렉션에서 쿼리에 가장 관련 있는 문서를 검색"""
    from chroma_utils import hybrid_query_chroma  # 함수 내에서 import하여 circular import 방지
    
    results = hybrid_query_chroma(collection, query, n_results=n_results)
    
    if results and results["documents"] and results["documents"][0]:
        # 결과를 데이터프레임으로 변환
        result_data = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0], 
            results["metadatas"][0], 
            results["distances"][0]
        )):
            # 유사도 점수 계산 (거리를 유사도로 변환)
            similarity = 1 - distance
            
            # 검색 유형 확인
            search_type = "임베딩"
            if "search_type" in results:
                search_type = "키워드" if results["search_type"][0][i] == "keyword" else "임베딩"
            
            # 모든 결과 표시
            result_data.append({
                "순위": i + 1,
                "유사도": f"{similarity:.4f}",
                "검색 유형": search_type,
                "출처": metadata.get("source", "알 수 없음"),
                "청크": metadata.get("chunk", "알 수 없음"),
                "키워드": metadata.get("keywords", "알 수 없음"),
                "내용": doc,
            })
        
        # 데이터프레임 생성 및 반환
        return pd.DataFrame(result_data)
    else:
        return None

def display_search_results(result_df):
    """검색 결과를 표시하는 함수"""
    if result_df is None or len(result_df) == 0:
        st.info("검색 결과가 없습니다.")
        return None
        
    st.subheader(f"검색 결과: {len(result_df)}개")
    
    # 문서 ID를 저장할 세션 상태 초기화
    if 'selected_docs_to_delete' not in st.session_state:
        st.session_state.selected_docs_to_delete = []
    
    # 선택할 수 있는 체크박스 추가
    result_df['선택'] = False
    
    # '순위' 열이 없는 경우 추가
    if '순위' not in result_df.columns:
        result_df['순위'] = range(1, len(result_df) + 1)
    
    # 모든 문서 선택/해제 체크박스
    select_all = st.checkbox("삭제 항목 전체 선택", key="select_all_docs")
    
    if select_all:
        result_df['선택'] = True
    
    # 유사도 열이 문자열인 경우 숫자로 변환
    if '유사도' in result_df.columns and isinstance(result_df['유사도'].iloc[0], str):
        result_df['유사도'] = result_df['유사도'].astype(float)
    
    # 검색 유형 열이 없는 경우 추가
    if '검색 유형' not in result_df.columns:
        result_df['검색 유형'] = "임베딩"
    
    # 결과 데이터프레임에 체크박스 열을 추가하여 표시
    edited_df = st.data_editor(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "삭제",
                help="삭제할 문서를 선택하세요",
                default=False,
                width="small"
            ),
            "순위": st.column_config.NumberColumn(width="small"),
            "유사도": st.column_config.NumberColumn(
                "유사도",
                help="검색어와 문서 간의 유사도 점수 (0~1)",
                format="%.4f",
                width="small"
            ),
            "검색 유형": st.column_config.TextColumn(width="small"),
            "출처": st.column_config.TextColumn(width="small"),
            "청크": st.column_config.NumberColumn(width="small"),
            "키워드": st.column_config.TextColumn(width="medium", help="문서에서 추출된 주요 키워드입니다."),
            "내용": st.column_config.TextColumn(width="large"),
        }
    )
    
    # 선택된 문서 확인
    selected_rows = edited_df[edited_df['선택'] == True]
    
    if not selected_rows.empty:
        st.session_state.selected_docs_to_delete = selected_rows.index.tolist()
        st.write(f"{len(selected_rows)}개 문서가 삭제를 위해 선택되었습니다.")
        
        if st.button("선택한 문서 삭제", type="primary", key="delete_selected_docs"):
            return selected_rows
    
    # 시각화: 유사도 차트
    if len(result_df) > 1:
        st.subheader("유사도 분포")
        chart_data = pd.DataFrame({
            "순위": result_df["순위"],
            "유사도": result_df["유사도"].astype(float)
        })
        st.bar_chart(chart_data.set_index("순위"))
        
    return None

# 시각화 유틸리티 함수 - 이제 visualization_utils.py 모듈로 이전됨
# 다른 파일에서 import할 수 있도록 visualization_utils 모듈 함수를 재내보냄
from visualization_utils import (
    get_embeddings_data,
    handle_missing_embeddings, 
    prepare_visualization_data,
    create_cluster_visualization,
    display_cluster_documents,
    display_cluster_wordclouds,
    display_cluster_lda,
    find_optimal_clusters,
    plot_elbow_method,
    display_optimal_cluster_info
)

def search_collection_by_similarity(collection, query, similarity_threshold=0.5, include_embeddings=False):
    """유사도 임계값 기반으로 컬렉션에서 검색하는 함수"""
    try:
        # include 매개변수에 embeddings 추가 여부 결정
        include_params = ["metadatas", "documents", "distances"]
        if include_embeddings:
            include_params.append("embeddings")
        
        import streamlit as st
        import pandas as pd
        import numpy as np
        
        # 컬렉션의 전체 문서 수 확인
        total_docs = collection.count()
        
        # 청크 크기 설정 (SQLite 변수 제한을 고려하여 설정)
        chunk_size = 500  # SQLite 기본 제한은 약 999개이므로 그보다 작게 설정
        
        # 결과를 저장할 리스트
        all_documents = []
        all_metadatas = []
        all_distances = []
        all_ids = []
        all_embeddings = []
        
        # 전체 문서 수가 청크 크기보다 작으면 한 번에 처리
        if total_docs <= chunk_size:
            with st.spinner(f"전체 {total_docs}개 문서 검색 중..."):
                results = collection.query(
                    query_texts=[query],
                    include=include_params,
                    n_results=total_docs
                )
                
                all_documents = results.get("documents", [[]])[0]
                all_metadatas = results.get("metadatas", [[]])[0]
                all_distances = results.get("distances", [[]])[0]
                all_ids = results.get("ids", [[]])[0]
                
                if include_embeddings and "embeddings" in results:
                    all_embeddings = results.get("embeddings", [[]])[0]
        else:
            # 청크 단위로 나누어 처리
            with st.spinner(f"총 {total_docs}개 문서를 청크 단위로 검색 중..."):
                # offset 대신 첫 번째 청크만 처리하고 유사도 임계값으로 필터링
                # ChromaDB는 현재 offset 파라미터를 지원하지 않음
                first_chunk_size = min(total_docs, 1000)  # 최대 1000개 문서 처리
                
                # 문서 검색
                results = collection.query(
                    query_texts=[query],
                    include=include_params,
                    n_results=first_chunk_size
                )
                
                all_documents = results.get("documents", [[]])[0]
                all_metadatas = results.get("metadatas", [[]])[0]
                all_distances = results.get("distances", [[]])[0]
                all_ids = results.get("ids", [[]])[0]
                
                if include_embeddings and "embeddings" in results:
                    all_embeddings = results.get("embeddings", [[]])[0]
                
                st.info(f"{len(all_documents)}개 문서 검색 완료. 유사도가 {similarity_threshold} 이상인 문서만 필터링합니다.")
        
        # 거리를 유사도 점수로 변환 (1 - 거리)
        # Chroma의 거리는 코사인 거리 기준이므로 유사도로 변환
        similarity_scores = [1 - distance for distance in all_distances]
        
        # 유사도 임계값 필터링
        filtered_results = []
        filtered_embeddings = []
        
        for i, score in enumerate(similarity_scores):
            if score >= similarity_threshold:
                item = {
                    "id": all_ids[i],
                    "document": all_documents[i],
                    "metadata": all_metadatas[i],
                    "similarity": score
                }
                filtered_results.append(item)
                
                # 수정: NumPy 배열을 불리언 컨텍스트에서 평가하는 대신 길이로 확인
                if include_embeddings and len(all_embeddings) > 0:
                    filtered_embeddings.append(all_embeddings[i])
        
        # 데이터프레임 생성
        if filtered_results:
            df = pd.DataFrame({
                "ID": [item["id"] for item in filtered_results],
                "내용": [item["document"] for item in filtered_results],
                "유사도": [round(item["similarity"], 4) for item in filtered_results]
            })
            
            # 메타데이터 정보 추가
            for item in filtered_results:
                for key, value in item["metadata"].items():
                    if key not in df.columns:
                        df[key] = None
                    df.loc[df["ID"] == item["id"], key] = value
            
            # 유사도 기준으로 정렬
            df = df.sort_values(by="유사도", ascending=False)
            
            if include_embeddings:
                return df, filtered_embeddings
            else:
                return df
                
        else:
            return pd.DataFrame() if not include_embeddings else (pd.DataFrame(), [])
        
    except Exception as e:
        import streamlit as st
        import traceback
        st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
        st.error(traceback.format_exc())  # 상세 오류 정보 표시
        if not include_embeddings:
            return pd.DataFrame()
        else:
            return pd.DataFrame(), []

def search_collection_by_similarity_full(collection, query, similarity_threshold=0.5, include_embeddings=False):
    """
    유사도 임계값 기반으로 컬렉션에서 검색하는 함수 (전체 컬렉션 데이터를 사용)
    이 방법은 ChromaDB의 1000개 제한을 우회하지만, 큰 컬렉션의 경우 메모리 사용량이 많을 수 있습니다.
    """
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        from embedding_utils import get_embedding_function
        
        with st.spinner("전체 문서 데이터를 로드하는 중..."):
            # 컬렉션의 전체 문서 가져오기
            all_data = collection.get(include=["metadatas", "documents", "embeddings"])
            
            documents = all_data["documents"]
            metadatas = all_data["metadatas"]
            ids = all_data["ids"]
            doc_embeddings = all_data.get("embeddings", [])
            
            total_docs = len(documents)
            st.info(f"총 {total_docs}개 문서가 로드되었습니다. 유사도 계산을 시작합니다.")
        
        # 쿼리 임베딩 생성
        with st.spinner("검색 쿼리 임베딩 생성 중..."):
            # 컬렉션에 사용된 임베딩 모델 확인
            embedding_model = "all-MiniLM-L6-v2"  # 기본값
            try:
                if collection.metadata and "embedding_model" in collection.metadata:
                    embedding_model = collection.metadata["embedding_model"]
            except:
                pass
                
            embed_fn = get_embedding_function(embedding_model)
            query_embedding = embed_fn([query])[0]
        
        # 유사도 계산 및 필터링
        with st.spinner(f"전체 {total_docs}개 문서의 유사도 계산 중..."):
            # 진행률 표시 바
            progress_bar = st.progress(0.0)
            
            # 결과를 저장할 리스트
            filtered_results = []
            filtered_embeddings = []
            
            # 계산 효율을 위해 NumPy 배열로 변환
            query_embedding_np = np.array(query_embedding)
            
            # 배치 크기 설정
            batch_size = 1000
            num_batches = (total_docs + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_docs)
                
                # 현재 배치의 문서만 처리
                batch_embeddings = doc_embeddings[start_idx:end_idx]
                
                # 코사인 유사도 계산
                batch_embeddings_np = np.array(batch_embeddings)
                
                # 벡터 정규화
                norms = np.linalg.norm(batch_embeddings_np, axis=1, keepdims=True)
                normalized_embeddings = batch_embeddings_np / norms
                
                # 정규화된 쿼리 벡터
                query_norm = np.linalg.norm(query_embedding_np)
                normalized_query = query_embedding_np / query_norm
                
                # 코사인 유사도 계산 (벡터의 내적)
                similarities = np.dot(normalized_embeddings, normalized_query)
                
                # 유사도 기준으로 필터링
                for i, similarity in enumerate(similarities):
                    idx = start_idx + i
                    if similarity >= similarity_threshold:
                        item = {
                            "id": ids[idx],
                            "document": documents[idx],
                            "metadata": metadatas[idx],
                            "similarity": float(similarity)
                        }
                        filtered_results.append(item)
                        
                        if include_embeddings:
                            filtered_embeddings.append(doc_embeddings[idx])
                
                # 진행 상황 업데이트
                progress = min(1.0, (end_idx) / total_docs)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            st.success(f"유사도 계산 완료. 총 {len(filtered_results)}개 문서가 임계값({similarity_threshold}) 이상입니다.")
        
        # 데이터프레임 생성
        if filtered_results:
            df = pd.DataFrame({
                "ID": [item["id"] for item in filtered_results],
                "내용": [item["document"] for item in filtered_results],
                "유사도": [round(item["similarity"], 4) for item in filtered_results]
            })
            
            # 메타데이터 정보 추가
            for i, item in enumerate(filtered_results):
                for key, value in item["metadata"].items():
                    if key not in df.columns:
                        df[key] = None
                    df.loc[i, key] = value
            
            # 유사도 기준으로 정렬
            df = df.sort_values(by="유사도", ascending=False)
            
            if include_embeddings:
                return df, filtered_embeddings
            else:
                return df
                
        else:
            return pd.DataFrame() if not include_embeddings else (pd.DataFrame(), [])
        
    except Exception as e:
        import streamlit as st
        import traceback
        st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
        st.error(traceback.format_exc())  # 상세 오류 정보 표시
        if not include_embeddings:
            return pd.DataFrame()
        else:
            return pd.DataFrame(), []