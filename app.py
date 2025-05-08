import streamlit as st
import pandas as pd
import os
import time
from utils import (
    store_data_in_chroma,
    clean_text,
    preprocess_dataframe,
    is_ollama_installed,
    is_ollama_running,
    is_ollama_lib_available,
    get_ollama_models,
    get_ollama_install_guide,
    rag_query_with_ollama,
    get_available_collections,
    load_chroma_collection
)

# 페이지 설정
st.set_page_config(
    page_title="텍스트 CSV 분석기 & RAG",
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
if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []
if 'ollama_status_checked' not in st.session_state:
    st.session_state.ollama_status_checked = False
if 'ollama_installed' not in st.session_state:
    st.session_state.ollama_installed = False
if 'ollama_running' not in st.session_state:
    st.session_state.ollama_running = False
if 'chroma_path' not in st.session_state:
    st.session_state.chroma_path = "./chroma_db"
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "csv_test"

# 제목
st.title("텍스트 CSV 파일 분석기 & RAG 시스템")
st.markdown("텍스트 위주의 CSV 파일을 업로드하여 RAG 시스템을 구성하세요.")

# 사이드바에 기존 ChromaDB 로드 옵션 추가
with st.sidebar:
    st.header("기존 ChromaDB 로드")
    
    # ChromaDB 경로 입력
    chroma_path = st.text_input("ChromaDB 경로", value=st.session_state.chroma_path)
    st.session_state.chroma_path = chroma_path
    
    # 사용 가능한 컬렉션 목록 가져오기
    available_collections = get_available_collections(chroma_path)
    
    if available_collections:
        st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
        
        # 컬렉션 선택
        selected_collection = st.selectbox(
            "컬렉션 선택", 
            available_collections,
            index=0 if st.session_state.collection_name not in available_collections else available_collections.index(st.session_state.collection_name)
        )
        st.session_state.collection_name = selected_collection
        
        # 컬렉션 로드 버튼
        if st.button("컬렉션 로드"):
            try:
                client, collection = load_chroma_collection(selected_collection, chroma_path)
                st.session_state.chroma_client = client
                st.session_state.chroma_collection = collection
                st.session_state.rag_enabled = True
                st.success(f"컬렉션 '{selected_collection}'을 성공적으로 로드했습니다.")
            except Exception as e:
                st.error(f"컬렉션 로드 중 오류 발생: {e}")
    else:
        st.info(f"'{chroma_path}' 경로에 사용 가능한 컬렉션이 없습니다.")
    
    st.markdown("---")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])

if uploaded_file is not None:
    # 데이터 로드
    try:
        # 인코딩 옵션
        encoding_options = ["utf-8", "cp949", "euc-kr", "latin1"]
        selected_encoding = st.selectbox("인코딩 선택", encoding_options, index=0)
        
        try:
            df = pd.read_csv(uploaded_file, encoding=selected_encoding)
            st.session_state.df = df
            st.success("파일 업로드 성공!")
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
        
        # 텍스트 열 미리보기
        st.subheader("텍스트 열 미리보기")
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        if text_columns:
            selected_text_col = st.selectbox("텍스트 열 선택", text_columns)
            
            # 텍스트 데이터 미리보기
            st.write(f"{selected_text_col} 열 미리보기:")
            
            # 텍스트 길이 계산
            text_lengths = df[selected_text_col].str.len()
            avg_length = text_lengths.mean()
            max_length = text_lengths.max()
            
            st.write(f"평균 텍스트 길이: {avg_length:.1f} 문자")
            st.write(f"최대 텍스트 길이: {max_length} 문자")
            
            # 샘플 텍스트 표시
            st.write("샘플 텍스트:")
            for i, text in enumerate(df[selected_text_col].head(5).fillna("").tolist()):
                st.text_area(f"샘플 {i+1}", text, height=100)
                
            # 정제된 텍스트 샘플 표시
            st.write("정제된 샘플 텍스트 (특수문자 제거):")
            for i, text in enumerate(df[selected_text_col].head(5).fillna("").tolist()):
                cleaned_text = clean_text(text)
                st.text_area(f"정제된 샘플 {i+1}", cleaned_text, height=100)
        else:
            st.info("텍스트 데이터가 없습니다.")
        
        # RAG 시스템 섹션
        st.header("RAG(Retrieval-Augmented Generation) 시스템")
        st.markdown("""
        이 섹션에서는 CSV 데이터를 ChromaDB에 저장하고 Ollama를 통해 RAG 시스템을 구성할 수 있습니다.
        """)
        
        # ChromaDB 설정
        st.subheader("1. ChromaDB 설정")
        
        # 열 선택
        st.write("ChromaDB에 저장할 열을 선택하세요:")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("열 선택", all_columns, default=st.session_state.selected_columns)
        st.session_state.selected_columns = selected_columns
        
        # 데이터 전처리 옵션
        st.subheader("데이터 전처리 옵션")
        st.info("선택한 열에 결측치가 있는 행은 자동으로 제거되며, 텍스트에서 산술 기호(+, -, *, /, %, =)를 제외한 특수문자가 제거됩니다.")
        
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
                            batch_size=batch_size
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("ChromaDB에 데이터 저장 완료!")
                        
                        st.session_state.chroma_client = client
                        st.session_state.chroma_collection = collection
                        st.session_state.collection_name = collection_name
                        st.session_state.chroma_path = persist_directory
                        
                        if max_process_rows:
                            st.success(f"ChromaDB에 데이터가 성공적으로 저장되었습니다. 컬렉션: {collection_name} (처리된 행 수: {max_process_rows})")
                        else:
                            st.success(f"ChromaDB에 데이터가 성공적으로 저장되었습니다. 컬렉션: {collection_name}")
                            
                        st.session_state.rag_enabled = True
                    except Exception as e:
                        st.error(f"ChromaDB 저장 중 오류 발생: {e}")
        
        # 구분선
        st.markdown("---")
    
    except Exception as e:
        st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")

# Ollama 연동 섹션
st.subheader("2. Ollama 연동")

if st.session_state.rag_enabled:
    
    # Ollama 라이브러리 확인
    if not is_ollama_lib_available():
        st.error("❌ Ollama 라이브러리가 설치되어 있지 않습니다.")
        st.markdown("""
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        
        설치 후 애플리케이션을 다시 시작하세요.
        """)
        st.stop()
    
    # Ollama 상태 확인 (처음 한 번만)
    if not st.session_state.ollama_status_checked:
        with st.spinner("Ollama 상태 확인 중..."):
            # Ollama 설치 확인
            st.session_state.ollama_installed = is_ollama_installed()
            
            if st.session_state.ollama_installed:
                # Ollama 서버 실행 확인
                st.session_state.ollama_running = is_ollama_running()
                
                if st.session_state.ollama_running:
                    # 모델 목록 가져오기
                    st.session_state.ollama_models = get_ollama_models()
            
            st.session_state.ollama_status_checked = True
    
    # Ollama 상태에 따른 UI 표시
    if not st.session_state.ollama_installed:
        st.error("❌ Ollama가 설치되어 있지 않습니다.")
        st.markdown(get_ollama_install_guide())
        
        if st.button("Ollama 상태 다시 확인"):
            st.session_state.ollama_status_checked = False
            st.rerun()
    
    elif not st.session_state.ollama_running:
        st.error("❌ Ollama 서버가 실행되고 있지 않습니다.")
        st.markdown("""
        ### Ollama 서버 실행하기
        
        터미널에서 다음 명령어를 실행하여 Ollama 서버를 시작하세요:
        ```
        ollama serve
        ```
        
        서버가 실행되면 '상태 다시 확인' 버튼을 클릭하세요.
        """)
        
        if st.button("Ollama 상태 다시 확인"):
            st.session_state.ollama_status_checked = False
            st.rerun()
    
    elif not st.session_state.ollama_models:
        st.warning("⚠️ 설치된 모델이 없습니다.")
        st.markdown("""
        ### 모델 설치하기
        
        터미널에서 다음 명령어를 실행하여 모델을 설치하세요:
        ```
        ollama pull llama2
        ```
        
        또는 다른 모델을 설치할 수 있습니다:
        ```
        ollama pull mistral
        ollama pull gemma:2b
        ```
        
        모델 설치 후 '상태 다시 확인' 버튼을 클릭하세요.
        """)
        
        if st.button("Ollama 상태 다시 확인"):
            st.session_state.ollama_status_checked = False
            st.rerun()
    
    else:
        # 모든 조건이 충족되면 Ollama 사용 가능
        st.success("✅ Ollama가 준비되었습니다.")
        
        # 모델 선택
        selected_model = st.selectbox(
            "사용할 모델 선택", 
            st.session_state.ollama_models,
            index=0 if "llama2" not in st.session_state.ollama_models else st.session_state.ollama_models.index("llama2")
        )
        
        # 모델 새로고침 버튼
        if st.button("모델 목록 새로고침"):
            with st.spinner("모델 목록을 가져오는 중..."):
                st.session_state.ollama_models = get_ollama_models()
                st.rerun()
        
        # 프롬프트와 질문 입력 (2군데로 나누기)
        st.subheader("입력 설정")
        
        # 프롬프트 입력 (여러 줄 입력 가능)
        prompt = st.text_area(
            "Prompt (지시사항)",
            height=150,
            placeholder="모델에게 전달할 지시사항을 입력하세요. 예: '다음 질문에 한국어로 답변해주세요.'"
        )
        
        # 질문 입력 (여러 줄 입력 가능)
        question = st.text_area(
            "Question (질문)",
            height=100,
            placeholder="실제 질문을 입력하세요. 예: '이 데이터에서 가장 중요한 정보는 무엇인가요?'"
        )
        
        # 참조할 문서 수 설정 (최소 3, 최대 20)
        n_results = st.slider(
                "참조할 문서 수", 
                min_value=3, 
                max_value=20, 
                value=5, 
                step=1,
                help="참조할 문서 수를 선택하세요. 일반적으로 3-5개가 적당합니다."
            )
        
        # Ollama로 질의하기
        if st.button("질의하기") and st.session_state.chroma_collection is not None:
            # 프롬프트와 질문을 합쳐서 query 생성
            combined_query = ""
            
            # 프롬프트가 있으면 먼저 추가
            if prompt:
                combined_query += prompt.strip() + "\n\n"
                
            # 질문 추가
            if question:
                combined_query += question.strip()
            
            if not combined_query.strip():
                st.warning("프롬프트 또는 질문을 입력하세요.")
            else:
                with st.spinner(f"Ollama({selected_model})로 질의 중..."):
                    try:
                        # n_results가 0이면 제한 없음(모든 문서 사용)
                        actual_n_results = n_results
                        
                        # 디버깅용 - 실제 전송되는 쿼리 표시
                        with st.expander("전송되는 쿼리 확인"):
                            st.code(combined_query)
                        
                        result = rag_query_with_ollama(
                            st.session_state.chroma_collection,
                            combined_query,
                            selected_model,
                            actual_n_results
                        )
                        
                        st.subheader("Ollama 응답")
                        st.markdown(result["response"])
                        
                        st.subheader("참조 문서")
                        for i, (doc, metadata, distance) in enumerate(zip(
                            result["context"],
                            result["metadatas"],
                            result["distances"]
                        )):
                            st.markdown(f"**문서 {i+1}** (유사도: {1-distance:.4f})")
                            st.info(doc)
                            st.write(f"메타데이터: {metadata}")
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Ollama 질의 중 오류 발생: {e}")
    
    # Ollama 설명
    with st.expander("Ollama란?"):
        st.markdown("""
        **Ollama**는 로컬 환경에서 대규모 언어 모델(LLM)을 실행할 수 있게 해주는 도구입니다.
        
        주요 특징:
        - 로컬에서 실행되므로 데이터가 외부로 전송되지 않습니다.
        - 다양한 오픈 소스 모델을 지원합니다 (Llama 2, Mistral, Gemma 등).
        - 가볍고 빠르게 실행됩니다.
        - Python 라이브러리를 통해 쉽게 통합할 수 있습니다.
        
        [Ollama 공식 웹사이트](https://ollama.ai/)에서 더 많은 정보를 확인할 수 있습니다.
        """)
else:
    st.info("Ollama 연동을 사용하려면 먼저 데이터를 ChromaDB에 저장하거나 기존 컬렉션을 로드하세요.")

# 사이드바 정보 (하단)
with st.sidebar:
    st.header("텍스트 CSV 분석기 & RAG 정보")
    st.info("""
    이 애플리케이션은 텍스트 위주의 CSV 파일을 분석하고 RAG 시스템을 구성하기 위한 도구입니다.
    
    기능:
    - CSV 파일 미리보기
    - 텍스트 데이터 분석
    - 데이터 전처리 (결측치 제거, 특수문자 제거)
    - ChromaDB에 데이터 저장
    - 기존 ChromaDB 컬렉션 로드
    - Ollama를 통한 RAG 시스템 구성
    """)
    
    st.markdown("---")
    
    # RAG 시스템 설명
    st.subheader("RAG 시스템이란?")
    st.markdown("""
    **RAG(Retrieval-Augmented Generation)**는 대규모 언어 모델(LLM)의 성능을 향상시키는 기술입니다.
    
    작동 방식:
    1. 사용자 질의가 들어오면 관련 정보를 벡터 데이터베이스에서 검색합니다.
    2. 검색된 정보를 LLM의 프롬프트에 추가하여 더 정확한 응답을 생성합니다.
    3. 이를 통해 최신 정보 제공, 환각 현상 감소, 도메인 특화 응답이 가능해집니다.
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")