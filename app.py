import streamlit as st
import pandas as pd
import ollama_utils as ou
from fix_dataframe import safe_display_dataframe
from disable_cache import disable_chromadb_cache, toggle_cache, is_cache_enabled

# ChromaDB 캐싱 비활성화 (오류가 계속 발생하는 경우 주석 해제)
# disable_chromadb_cache()

# 세션 상태 초기화
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0.0
if 'progress_status' not in st.session_state:
    st.session_state.progress_status = ""
if 'available_models' not in st.session_state:
    try:
        models, installation_guide = ou.get_available_models()
        st.session_state.available_models = models
        st.session_state.installation_guide = installation_guide
    except Exception as e:
        st.session_state.available_models = ["gemma3:27b", "llama3", "llama3:8b"]
        st.session_state.installation_guide = None
        st.warning(f"모델 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")

# 텍스트 입력 섹션 추가
st.title("CSV 데이터 분석기")

# Ollama 설치 가이드 표시 (필요한 경우)
if hasattr(st.session_state, 'installation_guide') and st.session_state.installation_guide:
    st.error("Ollama가 설치되어 있지 않거나 모델이 없습니다.")
    st.markdown(st.session_state.installation_guide)
    
    if st.button("모델 목록 다시 확인"):
        try:
            models, installation_guide = ou.get_available_models()
            st.session_state.available_models = models
            st.session_state.installation_guide = installation_guide
            if not installation_guide:
                st.success("Ollama 모델을 성공적으로 찾았습니다!")
                st.rerun()
            else:
                st.warning("아직 Ollama가 설치되어 있지 않거나 모델이 없습니다.")
        except Exception as e:
            st.error(f"모델 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")
else:
    # 파일 업로더 섹션
    uploaded_file = st.file_uploader("Choose a csv file", type=["csv"])
    if uploaded_file is not None:
        try:
            # CSV 파일 읽기 시 다양한 인코딩과 구분자 시도
            try:
                # 기본 설정으로 시도
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # UTF-8 디코딩 오류 시 다른 인코딩 시도
                try:
                    df = pd.read_csv(uploaded_file, encoding='cp949')  # 한국어 Windows 인코딩
                except Exception:
                    try:
                        df = pd.read_csv(uploaded_file, encoding='euc-kr')  # 또 다른 한국어 인코딩
                    except Exception:
                        # 마지막 시도: 오류 무시 모드
                        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            except Exception as csv_error:
                # 구분자 문제일 수 있음
                try:
                    # 탭 구분자 시도
                    df = pd.read_csv(uploaded_file, sep='\t')
                except Exception:
                    # 세미콜론 구분자 시도
                    try:
                        df = pd.read_csv(uploaded_file, sep=';')
                    except Exception:
                        # 마지막 시도: 엔진 변경
                        df = pd.read_csv(uploaded_file, engine='python')
            
            # 특수문자 제거 함수
            def remove_special_chars(text):
                if isinstance(text, str):
                    # 일반적인 특수문자 제거 (알파벳, 숫자, 한글, 공백은 유지)
                    import re
                    return re.sub(r'[^\w\s가-힣]', '', text)
                return text
            
            # 모든 문자열 컬럼에 특수문자 제거 적용
            for col in df.columns:
                if df[col].dtype == 'object':  # 문자열 타입 컬럼인 경우
                    df[col] = df[col].apply(remove_special_chars)
            
            st.text("Describe the file : " + uploaded_file.name)
            
            # 데이터프레임 정보 표시
            try:
                st.write(df.describe())
            except Exception as e:
                st.warning(f"데이터 통계 생성 중 오류 발생: {str(e)}")
                st.write("기본 정보:")
                st.write(f"행 수: {len(df)}, 열 수: {len(df.columns)}")
            
            # 컬럼 선택 및 데이터 표시
            columns_list = st.multiselect('컬럼을 선택하세요.', df.columns)
            
            if columns_list:
                st.write("상위 5개 행")
                try:
                    # 안전한 데이터프레임 표시
                    safe_display_dataframe(df[columns_list], num_rows=5)
                except Exception as e:
                    st.error(f"데이터 표시 중 오류 발생: {str(e)}")
                
                # 모델 선택
                col1, col2 = st.columns(2)
                
                with col1:
                    # 모델 새로고침 버튼
                    if st.button("모델 목록 새로고침"):
                        try:
                            models, installation_guide = ou.get_available_models()
                            st.session_state.available_models = models
                            st.session_state.installation_guide = installation_guide
                            if installation_guide:
                                st.warning("Ollama가 설치되어 있지 않거나 모델이 없습니다.")
                                st.rerun()
                            else:
                                st.success("모델 목록을 업데이트했습니다.")
                        except Exception as e:
                            st.error(f"모델 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")
                
                with col2:
                    # 모델 선택 드롭다운
                    default_model_idx = 0
                    if "gemma3:27b" in st.session_state.available_models:
                        default_model_idx = st.session_state.available_models.index("gemma3:27b")
                    selected_model = st.selectbox(
                        "사용할 모델 선택:",
                        st.session_state.available_models,
                        index=default_model_idx
                    )
                
                # 고급 설정 섹션
                with st.expander("고급 설정"):
                    # 행 수 제한 설정
                    limit_rows = st.toggle("분석 행 수 제한", value=True)
                    
                    if limit_rows:
                        max_rows = st.slider("최대 분석 행 수", min_value=10, max_value=100, value=50)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 배치 크기 선택
                        batch_size = st.slider("배치 크기", min_value=1, max_value=20, value=5)
                    
                    with col2:
                        # 타임아웃 설정
                        timeout = st.slider("API 타임아웃(초)", min_value=10, max_value=120, value=30)
                    
                    with col3:
                        # 재시도 횟수 설정
                        max_retries = st.slider("최대 재시도 횟수", min_value=1, max_value=5, value=3)
                    
                    # 병렬 처리 및 캐싱 설정 추가
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 병렬 처리 작업자 수 설정
                        max_workers = st.slider("병렬 처리 작업자 수", min_value=1, max_value=8, value=2)
                    
                    with col2:
                        # 캐싱 사용 여부 설정
                        use_cache = st.toggle("캐싱 사용", value=is_cache_enabled(), key="cache_toggle")
                        # 토글 상태가 변경되면 세션 상태 업데이트
                        if st.session_state.cache_toggle != is_cache_enabled():
                            toggle_cache(st.session_state.cache_toggle)
                        
                    # 텍스트 압축 설정
                    max_text_length = st.slider("최대 텍스트 길이", min_value=1000, max_value=10000, value=4000)
                
                # Ollama 분석 실행 버튼
                st.write("요청 사항을 입력하세요(ex:입력 된 text의 광고글 여부 알려줘. 광고면 O 아니면 X라고 표시만 해줘. 설명은 필요없어.)")
                prompt_input = st.text_area("분석 요청 내용", height=150, placeholder="여기에 요청 사항을 자세히 입력하세요...")
                if prompt_input and st.button("분석 요청"):
                    # 진행 상태 표시 컴포넌트 생성
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 진행 상태 업데이트 콜백 함수
                    def update_progress(progress, status):
                        st.session_state.progress = progress
                        st.session_state.progress_status = status
                        progress_bar.progress(progress)
                        status_text.text(status)
                    
                    try:
                        # 선택된 컬럼만 포함한 데이터프레임으로 분석 실행
                        filtered_df = df[columns_list].copy()
                        
                        # 데이터프레임 정리 - 문제가 될 수 있는 컬럼 처리
                        for col in filtered_df.columns:
                            if col.startswith('Unnamed:'):
                                # Unnamed 컬럼은 문자열로 변환
                                filtered_df[col] = filtered_df[col].astype(str)
                            
                            # None 값 처리
                            filtered_df[col] = filtered_df[col].fillna("")
                        
                        # 행 수 제한이 켜져 있으면 최대 행 수 적용
                        if limit_rows:
                            total_rows = len(filtered_df)
                            if total_rows > max_rows:
                                filtered_df = filtered_df.head(max_rows)
                                st.info(f"행 수 제한이 켜져 있어 전체 {total_rows}개 행 중 {max_rows}개 행만 분석합니다.")
                        
                        # 진행 상태 콜백과 함께 분석 실행
                        result_df = ou.analyze_text_with_prompt(
                            prompt_input, 
                            filtered_df,
                            model_name=selected_model,
                            progress_callback=update_progress,
                            batch_size=batch_size,
                            max_retries=max_retries,
                            timeout=timeout,
                            max_workers=max_workers,
                            use_cache=use_cache,
                            max_text_length=max_text_length
                        )
                        
                        # 세션 상태에 결과 저장
                        st.session_state.result_df = result_df
                        st.session_state.analysis_done = True
                        
                        st.success("분석 완료!")
                    except Exception as e:
                        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                
                # 분석이 완료되었으면 결과 표시
                if st.session_state.analysis_done and st.session_state.result_df is not None:
                    # 결과 표시
                    st.subheader("분석 결과")
                    st.write("상위 5개 행")
                    
                    # 안전한 데이터프레임 표시 유틸리티 사용
                    safe_display_dataframe(st.session_state.result_df, num_rows=5)

                    # 오류 발생 행 확인
                    try:
                        error_rows = st.session_state.result_df[st.session_state.result_df['response'].str.contains('Error:', na=False)]
                        if not error_rows.empty:
                            st.warning(f"{len(error_rows)}개 행에서 오류가 발생했습니다.")
                            if st.checkbox("오류 발생 행 보기"):
                                safe_display_dataframe(error_rows)
                    except Exception as e:
                        st.error(f"오류 행 확인 중 문제가 발생했습니다: {str(e)}")

                    # CSV 다운로드 버튼 (UTF-8만 제공)
                    try:
                        csv_data = st.session_state.result_df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="분석 결과 CSV 다운로드",
                            data=csv_data,
                            file_name="analysis_result.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"CSV 다운로드 준비 중 오류가 발생했습니다: {str(e)}")
                        # 대체 다운로드 방법 제공
                        try:
                            # 모든 컬럼을 문자열로 변환하여 CSV 생성
                            string_df = st.session_state.result_df.astype(str)
                            csv_data = string_df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="문자열 변환 CSV 다운로드",
                                data=csv_data,
                                file_name="analysis_result_str.csv",
                                mime="text/csv",
                            )
                        except Exception as str_e:
                            st.error(f"대체 CSV 다운로드 준비 중 오류가 발생했습니다: {str(str_e)}")
        except Exception as file_error:
            st.error(f"CSV 파일 로드 중 오류가 발생했습니다: {str(file_error)}")