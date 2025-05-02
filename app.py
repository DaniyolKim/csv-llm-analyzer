import streamlit as st
import pandas as pd
import ollama_utils as ou

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
        # st.write("Filename:", uploaded_file.name)
        df = pd.read_csv(uploaded_file)
        st.text("Describe the file : " + uploaded_file.name)
        st.write(df.describe())
        
        # 컬럼 선택 및 데이터 표시
        columns_list = st.multiselect('컬럼을 선택하세요.', df.columns)
        
        if columns_list:
            st.write("상위 5개 행")
            st.dataframe(df[columns_list].head())
            
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
            
            # Ollama 분석 실행 버튼
            st.write("요청 사항을 입력하세요(ex:입력 된 text의 광고글 여부 알려줘. 광고면 O 아니면 X라고 표시만 해줘. 설명은 필요없어.)")
            prompt_input = st.text_area("", height=150, placeholder="여기에 요청 사항을 자세히 입력하세요...")
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
                    filtered_df = df[columns_list]
                    
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
                        timeout=timeout
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
                st.dataframe(st.session_state.result_df.head())

                # 오류 발생 행 확인
                error_rows = st.session_state.result_df[st.session_state.result_df['response'].str.contains('Error:', na=False)]
                if not error_rows.empty:
                    st.warning(f"{len(error_rows)}개 행에서 오류가 발생했습니다.")
                    if st.checkbox("오류 발생 행 보기"):
                        st.dataframe(error_rows)

                # CSV 다운로드 버튼 (UTF-8만 제공)
                csv_data = st.session_state.result_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="분석 결과 CSV 다운로드",
                    data=csv_data,
                    file_name="analysis_result.csv",
                    mime="text/csv",
                )
