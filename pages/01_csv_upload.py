import streamlit as st
import pandas as pd
import os
import time
from utils import (
    store_data_in_chroma,
    clean_text,
    preprocess_dataframe,
    get_available_collections,
    get_available_embedding_models,
    get_embedding_status
)

st.set_page_config(
    page_title="CSV 업로드 및 처리",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'chroma_collection' not in st.session_state:
    st.session_state.chroma_collection = None
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = []
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = False
if 'chroma_path' not in st.session_state:
    st.session_state.chroma_path = "./chroma_db"
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "csv_test"
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "all-MiniLM-L6-v2"  # 기본 임베딩 모델

# 제목
st.title("CSV 파일 업로드 및 처리")

# 파일 업로드
st.subheader("CSV 파일 선택")

# 파일 입력 방식 선택 (업로드 또는 경로 입력)
file_input_method = st.radio("파일 입력 방법", ["파일 업로드", "파일 경로 입력"])

if file_input_method == "파일 업로드":
    uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])
    file_path = None
else:
    uploaded_file = None
    file_path = st.text_input("CSV 파일 경로 입력 (전체 경로)", placeholder="예: C:/path/to/your/file.csv")
    if file_path and not os.path.isfile(file_path):
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        file_path = None

# 파일이 업로드되었거나 유효한 경로가 입력된 경우
if uploaded_file is not None or (file_path and os.path.isfile(file_path)):
    # 데이터 로드
    try:
        # 인코딩 옵션
        encoding_options = ["utf-8", "cp949", "euc-kr", "latin1"]
        selected_encoding = st.selectbox("인코딩 선택", encoding_options, index=0)
        
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, encoding=selected_encoding)
                st.session_state.df = df
                st.success("파일 업로드 성공!")
            else:
                df = pd.read_csv(file_path, encoding=selected_encoding)
                st.session_state.df = df
                st.success(f"파일 로드 성공: {file_path}")
        except UnicodeDecodeError:
            st.error(f"{selected_encoding} 인코딩으로 파일을 읽을 수 없습니다. 다른 인코딩을 선택해 주세요.")
            st.stop()
        
        # 데이터 미리보기
        st.subheader("데이터 미리보기 (상위 10행)")
        st.dataframe(df.head(10))
        
        # 기본 정보
        st.subheader("기본 정보")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"행 수: {df.shape[0]}")
            st.write(f"열 수: {df.shape[1]}")
        with col2:
            st.write(f"결측치 수: {df.isna().sum().sum()}")
            st.write(f"중복 행 수: {df.duplicated().sum()}")
        
        # 열 정보
        st.subheader("열 정보")
        col_info = pd.DataFrame({
            '데이터 타입': [str(dtype) for dtype in df.dtypes],
            '고유값 수': df.nunique(),
            '결측치 수': df.isna().sum(),
            '결측치 비율(%)': (df.isna().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info)
        

        
        # ChromaDB 설정
        st.subheader("ChromaDB 설정")
        
        # 열 선택
        st.write("ChromaDB에 저장할 열을 선택하세요:")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("열 선택", all_columns, default=st.session_state.selected_columns)
        st.session_state.selected_columns = selected_columns
        
        # 데이터 전처리 옵션
        st.subheader("데이터 전처리 옵션")
        st.info("선택한 열에 결측치가 있는 행은 자동으로 제거되며, 텍스트에서 산술 기호(+, -, *, /, %, =)와 문장 구분 기호(., ?, !, ;, :, ,)를 제외한 특수문자가 제거됩니다. 또한 ', 조합과 '] 조합은 .로 변환됩니다.")
        
        # 행 수 제한 옵션
        max_rows = st.number_input("처리할 최대 행 수 (0 = 제한 없음)", min_value=0, value=100, step=100)
        batch_size = st.number_input("배치 처리 크기", min_value=10, value=100, step=10)
        
        # 전처리 미리보기
        if selected_columns:
            try:
                max_preview_rows = max_rows if max_rows > 0 else None
                processed_df = preprocess_dataframe(df, selected_columns, max_preview_rows)
                
                if max_preview_rows:
                    st.write(f"전처리 후 행 수: {processed_df.shape[0]} (제한: {max_preview_rows}, 원본: {df.shape[0]})")
                else:
                    st.write(f"전처리 후 행 수: {processed_df.shape[0]} (원본: {df.shape[0]})")
                    
                st.write(f"결측치로 인해 제거된 행 수: {df.shape[0] - len(df.dropna(subset=selected_columns))}")
                
                if not processed_df.empty:
                    st.write("전처리된 데이터 미리보기:")
                    st.dataframe(processed_df.head(5))
                else:
                    st.error("선택한 열에 유효한 데이터가 없습니다. 모든 행에 결측치가 있습니다.")
            except Exception as e:
                st.error(f"데이터 전처리 중 오류 발생: {e}")
        
        # ChromaDB 저장 옵션
        st.subheader("ChromaDB 저장 옵션")
        collection_name = st.text_input("컬렉션 이름", value=st.session_state.collection_name)
        persist_directory = st.text_input("저장 경로", value=st.session_state.chroma_path)
        
        # 임베딩 모델 선택 UI
        st.write("임베딩 모델 선택:")
        
        # 사용 가능한 한국어 임베딩 모델 목록 가져오기
        embedding_models = get_available_embedding_models().get("한국어 특화 모델", [])
        
        if not embedding_models:
            st.warning("사용 가능한 한국어 임베딩 모델이 없습니다. 기본 모델을 사용합니다.")
            embedding_models = ["snunlp/KR-SBERT-V40K-klueNLI-augSTS"]  # 기본 모델
        
        # 모델 선택
        selected_category_models = embedding_models
        
        # 모델 선택
        selected_embedding_model = st.selectbox(
            "임베딩 모델",
            selected_category_models,
            index=0
        )
        
        # 선택한 모델 세션 상태에 저장
        st.session_state.embedding_model = selected_embedding_model
        
        # ChromaDB 저장 버튼
        if st.button("ChromaDB에 데이터 저장"):
            if not selected_columns:
                st.error("저장할 열을 하나 이상 선택하세요.")
            else:
                with st.spinner("ChromaDB에 데이터 저장 중..."):
                    try:
                        # 행 수 제한 적용
                        max_process_rows = max_rows if max_rows > 0 else None
                        
                        # 진행 상황 표시
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("ChromaDB에 데이터 저장 준비 중...")
                        
                        client, collection = store_data_in_chroma(
                            df, 
                            selected_columns, 
                            collection_name, 
                            persist_directory,
                            max_rows=max_process_rows,
                            batch_size=batch_size,
                            embedding_model=st.session_state.embedding_model  # 선택한 임베딩 모델 전달
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("ChromaDB에 데이터 저장 완료!")
                        
                        st.session_state.chroma_client = client
                        st.session_state.chroma_collection = collection
                        st.session_state.collection_name = collection_name
                        st.session_state.chroma_path = persist_directory
                        
                        # 임베딩 모델 상태 확인
                        embedding_status = get_embedding_status()
                        if embedding_status["fallback_used"]:
                            st.warning(f"""
                            ⚠️ **임베딩 모델 변경됨**: 요청하신 모델 대신 기본 임베딩 모델이 사용되었습니다.
                            - 요청 모델: {embedding_status["requested_model"]}
                            - 사용된 모델: {embedding_status["actual_model"]}
                            - 원인: {embedding_status["error_message"]}
                            """)
                        
                        if max_process_rows:
                            st.success(f"ChromaDB에 데이터가 성공적으로 저장되었습니다. 컬렉션: {collection_name} (처리된 행 수: {max_process_rows})")
                        else:
                            st.success(f"ChromaDB에 데이터가 성공적으로 저장되었습니다. 컬렉션: {collection_name}")
                            
                        st.session_state.rag_enabled = True
                    except Exception as e:
                        st.error(f"ChromaDB 저장 중 오류 발생: {e}")
    
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")