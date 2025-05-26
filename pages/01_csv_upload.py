import streamlit as st
import pandas as pd
import os
import time
import traceback
import logging
from utils import (
    store_data_in_chroma,
    clean_text,
    preprocess_dataframe,
    get_available_collections,
    get_available_embedding_models,
    get_embedding_status
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("csv_upload.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("csv_uploader")

st.set_page_config(
    page_title="CSV 업로드 및 처리",
    page_icon="📄",
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

# 사이드바에 ChromaDB 로드 옵션 추가
with st.sidebar:
    st.header("DB 설정")
    
    # ChromaDB 경로 설정
    default_db_path = "./chroma_db"
    db_path = st.text_input(
        "ChromaDB 경로",
        value=st.session_state.chroma_path,
        help="ChromaDB가 저장된 경로를 입력하세요. 기본값은 './chroma_db'입니다."
    )
    st.session_state.chroma_path = db_path
    
    # 경로가 존재하는지 확인
    if not os.path.exists(db_path):
        st.warning(f"입력한 경로({db_path})가 존재하지 않습니다. 기본 경로를 사용합니다.")
        db_path = default_db_path
        st.session_state.chroma_path = default_db_path
    
    # 사용 가능한 컬렉션 목록 가져오기
    available_collections = get_available_collections(db_path)
    
    if available_collections:
        st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
        
        # 상태 관리를 위한 세션 상태 추가
        if 'collection_to_delete' not in st.session_state:
            st.session_state.collection_to_delete = None
            
        if 'show_delete_confirm' not in st.session_state:
            st.session_state.show_delete_confirm = False
            
        # 현재 사용중인 컬렉션 이름과 DB 경로 상태 추가
        if 'current_collection_name' not in st.session_state:
            st.session_state.current_collection_name = st.session_state.collection_name
        if 'current_db_path' not in st.session_state:
            st.session_state.current_db_path = st.session_state.chroma_path
        if 'collection_loaded' not in st.session_state:
            st.session_state.collection_loaded = False
            
        # 컬렉션 선택 UI
        selected_collection = st.selectbox(
            "컬렉션 선택",
            options=available_collections,
            index=0 if available_collections and available_collections[0] == st.session_state.collection_name else 0,
            help="검색할 ChromaDB 컬렉션을 선택하세요."
        )
        st.session_state.collection_name = selected_collection
        
        # 컬렉션이나 경로가 변경되면 세션 상태 업데이트
        if (selected_collection != st.session_state.current_collection_name or 
            db_path != st.session_state.current_db_path):
            st.session_state.collection_loaded = False
            st.session_state.current_collection_name = selected_collection
            st.session_state.current_db_path = db_path
            
        # 컬렉션 관리 버튼 행
        col1, col2 = st.columns(2)
                
        with col2:
            # 삭제 버튼
            if st.button("컬렉션 삭제", key="delete_collection_btn", type="secondary"):
                st.session_state.collection_to_delete = selected_collection
                st.session_state.show_delete_confirm = True
                        
        # 삭제 확인 다이얼로그
        if st.session_state.show_delete_confirm and st.session_state.collection_to_delete:
            from utils import delete_collection
            with st.expander(f"'{st.session_state.collection_to_delete}' 컬렉션을 삭제하시겠습니까?", expanded=True):
                st.warning(f"'{st.session_state.collection_to_delete}' 컬렉션의 모든 데이터가 삭제됩니다.")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("확인", key="confirm_delete", type="primary"):
                        # 컬렉션 삭제 수행
                        success = delete_collection(st.session_state.collection_to_delete, db_path)
                        if success:
                            # 현재 로드된 컬렉션이 삭제되었다면 상태 초기화
                            if st.session_state.collection_name == st.session_state.collection_to_delete:
                                st.session_state.chroma_collection = None
                                st.session_state.chroma_client = None
                                st.session_state.rag_enabled = False
                                st.session_state.collection_loaded = False
                            
                            st.success(f"'{st.session_state.collection_to_delete}' 컬렉션이 삭제되었습니다.")
                            # 상태 초기화
                            st.session_state.collection_to_delete = None
                            st.session_state.show_delete_confirm = False
                            
                            # 페이지 새로고침
                            st.rerun()
                        else:
                            st.error(f"'{st.session_state.collection_to_delete}' 컬렉션 삭제 중 오류가 발생했습니다.")
                
                with col2:
                    if st.button("취소", key="cancel_delete"):
                        st.session_state.collection_to_delete = None
                        st.session_state.show_delete_confirm = False
                        st.rerun()
        
        # 컬렉션이 로드된 경우 상태 표시
        if st.session_state.collection_loaded:
            # 컬렉션 정보 표시
            with st.expander("컬렉션 정보"):
                try:
                    # 이미 로드된 컬렉션 사용
                    collection = st.session_state.chroma_collection
                    collection_info = collection.count()
                    
                    # 컬렉션에 저장된 임베딩 모델 정보 확인
                    embedding_model = "알 수 없음"
                    try:
                        if collection.metadata and "embedding_model" in collection.metadata:
                            embedding_model = collection.metadata["embedding_model"]
                    except:
                        pass
                    
                    st.write(f"컬렉션 이름: {selected_collection}")
                    st.write(f"문서 수: {collection_info}")
                    st.write(f"임베딩 모델: {embedding_model}")
                    st.write(f"DB 경로: {db_path}")
                except Exception as e:
                    st.error(f"컬렉션 정보를 가져오는 중 오류가 발생했습니다: {str(e)}")
    else:
        st.error(f"'{db_path}' 경로에 사용 가능한 컬렉션이 없습니다.")
    
    st.markdown("---")

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
                        logger.error(f"ChromaDB 저장 중 오류 발생: {e}")
                        logger.debug(traceback.format_exc())
    
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
        logger.error(f"파일을 처리하는 중 오류 발생: {e}")
        logger.debug(traceback.format_exc())