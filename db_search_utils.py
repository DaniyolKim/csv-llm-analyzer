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
    
    # 모든 문서 선택/해제 체크박스
    col1, col2 = st.columns([1, 20])
    with col1:
        select_all = st.checkbox("전체 선택", key="select_all_docs")
    
    if select_all:
        result_df['선택'] = True
    
    # 결과 데이터프레임에 체크박스 열을 추가하여 표시
    edited_df = st.data_editor(
        result_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "선택",
                help="삭제할 문서를 선택하세요",
                default=False,
                width="small"
            ),
            "순위": st.column_config.NumberColumn(width="small"),
            "유사도": st.column_config.TextColumn(width="small"),
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