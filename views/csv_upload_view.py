import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any

class CsvUploadView:
    """CSV Upload 페이지의 뷰 컴포넌트"""
    
    def __init__(self):
        self.setup_page_config()
    
    def setup_page_config(self):
        """페이지 설정"""
        st.set_page_config(
            page_title="CSV 업로드 및 처리",
            page_icon="📄",
            layout="wide"
        )
    
    def render_title(self):
        """제목 렌더링"""
        st.title("CSV 파일 업로드 및 처리")
    
    def render_sidebar_db_settings(self, db_path: str) -> Dict[str, Any]:
        """사이드바 DB 설정 렌더링"""
        with st.sidebar:
            st.header("DB 설정")
            
            # ChromaDB 경로 설정
            new_db_path = st.text_input(
                "ChromaDB 경로",
                value=db_path,
                help="ChromaDB가 저장된 경로를 입력하세요. 기본값은 './chroma_db'입니다."
            )
            
            # 경로 검증 메시지
            if db_path != "./chroma_db":
                st.warning(f"입력한 경로({db_path})가 존재하지 않습니다. 기본 경로를 사용합니다.")
            
            return {'db_path': new_db_path}
    
    def render_sidebar_collection_management(self, available_collections: List[str], 
                                           selected_collection: str) -> Dict[str, Any]:
        """사이드바에서 컬렉션 관리 UI 렌더링"""
        with st.sidebar:
            st.header("컬렉션 관리")
            
            if available_collections:
                st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
                
                # 컬렉션 선택
                new_selected_collection = st.selectbox(
                    "컬렉션 선택",
                    options=available_collections,
                    index=0 if available_collections and available_collections[0] == selected_collection else 0,
                    help="관리할 ChromaDB 컬렉션을 선택하세요."
                )
                
                # 컬렉션 관리 버튼
                delete_button = st.button("컬렉션 삭제", key="delete_collection_btn", type="secondary")
                
                return {
                    'has_collections': True,
                    'selected_collection': new_selected_collection,
                    'delete_button_clicked': delete_button
                }
            else:
                st.error("사용 가능한 컬렉션이 없습니다.")
                return {'has_collections': False}
    
    def render_sidebar_collection_info(self, collection_name: str, collection_info: Dict[str, Any], 
                                     db_path: str):
        """사이드바에서 컬렉션 정보 표시"""
        with st.sidebar:
            with st.expander("컬렉션 정보", expanded=True):
                if collection_info['success']:
                    st.write(f"**컬렉션:** {collection_name}")
                    st.write(f"**문서 수:** {collection_info['count']}")
                    st.write(f"**임베딩 모델:** {collection_info['embedding_model']}")
                    st.write(f"**DB 경로:** {db_path}")
                else:
                    st.error(collection_info['error'])
    
    def render_collection_management(self, available_collections: List[str], 
                                   selected_collection: str) -> Dict[str, Any]:
        """컬렉션 관리 UI 렌더링 (메인 컨텐츠용 - 사용 안함)"""
        if available_collections:
            st.success(f"✅ {len(available_collections)}개의 컬렉션을 찾았습니다.")
            
            # 컬렉션 선택
            new_selected_collection = st.selectbox(
                "컬렉션 선택",
                options=available_collections,
                index=0 if available_collections and available_collections[0] == selected_collection else 0,
                help="검색할 ChromaDB 컬렉션을 선택하세요."
            )
            
            # 컬렉션 관리 버튼
            col1, col2 = st.columns(2)
            with col2:
                delete_button = st.button("컬렉션 삭제", key="delete_collection_btn", type="secondary")
            
            return {
                'has_collections': True,
                'selected_collection': new_selected_collection,
                'delete_button_clicked': delete_button
            }
        else:
            st.error("사용 가능한 컬렉션이 없습니다.")
            return {'has_collections': False}
    
    def render_delete_confirmation(self, collection_to_delete: str) -> Dict[str, Any]:
        """삭제 확인 다이얼로그 렌더링"""
        with st.expander(f"'{collection_to_delete}' 컬렉션을 삭제하시겠습니까?", expanded=True):
            st.warning(f"'{collection_to_delete}' 컬렉션의 모든 데이터가 삭제됩니다.")
            col1, col2 = st.columns(2)
            
            with col1:
                confirm_button = st.button("확인", key="confirm_delete", type="primary")
            with col2:
                cancel_button = st.button("취소", key="cancel_delete")
            
            return {
                'confirm_clicked': confirm_button,
                'cancel_clicked': cancel_button
            }
    
    def render_collection_info(self, collection_name: str, collection_info: Dict[str, Any], 
                             db_path: str):
        """컬렉션 정보 표시"""
        with st.expander("컬렉션 정보"):
            if collection_info['success']:
                st.write(f"컬렉션 이름: {collection_name}")
                st.write(f"문서 수: {collection_info['count']}")
                st.write(f"임베딩 모델: {collection_info['embedding_model']}")
                st.write(f"DB 경로: {db_path}")
            else:
                st.error(collection_info['error'])
    
    def render_file_input_section(self) -> Dict[str, Any]:
        """파일 입력 섹션 렌더링"""
        st.subheader("CSV 파일 선택")
        
        # 파일 입력 방식 선택
        file_input_method = st.radio("파일 입력 방법", ["파일 업로드", "파일 경로 입력"])
        
        if file_input_method == "파일 업로드":
            uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])
            return {
                'method': 'upload',
                'file_source': uploaded_file,
                'file_path': None
            }
        else:
            file_path = st.text_input("CSV 파일 경로 입력 (전체 경로)", 
                                    placeholder="예: C:/path/to/your/file.csv")
            
            return {
                'method': 'path',
                'file_source': None,
                'file_path': file_path
            }
    
    def render_encoding_selection(self) -> str:
        """인코딩 선택 UI 렌더링"""
        encoding_options = ["utf-8", "cp949", "euc-kr", "latin1"]
        return st.selectbox("인코딩 선택", encoding_options, index=0)
    
    def render_dataframe_preview(self, df_info: Dict[str, Any]):
        """데이터프레임 미리보기 렌더링"""
        if not df_info['success']:
            st.error(df_info['error'])
            return
        
        # 데이터 미리보기
        st.subheader("데이터 미리보기 (상위 10행)")
        st.dataframe(df_info['preview'])
        
        # 기본 정보
        st.subheader("기본 정보")
        basic_info = df_info['basic_info']
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"행 수: {basic_info['rows']}")
            st.write(f"열 수: {basic_info['columns']}")
        with col2:
            st.write(f"결측치 수: {basic_info['null_values']}")
            st.write(f"중복 행 수: {basic_info['duplicates']}")
        
        # 열 정보
        st.subheader("열 정보")
        st.dataframe(df_info['column_info'])
    
    def render_column_selection(self, all_columns: List[str], 
                              selected_columns: List[str]) -> List[str]:
        """열 선택 UI 렌더링"""
        st.subheader("ChromaDB 설정")
        st.write("ChromaDB에 저장할 열을 선택하세요:")
        return st.multiselect("열 선택", all_columns, default=selected_columns)
    
    def render_preprocessing_options(self) -> Dict[str, Any]:
        """데이터 전처리 옵션 렌더링"""
        st.subheader("데이터 전처리 옵션")
        st.info("선택한 열에 결측치가 있는 행은 자동으로 제거되며, 텍스트에서 산술 기호(+, -, *, /, %, =)와 문장 구분 기호(., ?, !, ;, :, ,)를 제외한 특수문자가 제거됩니다. 또한 ', 조합과 '] 조합은 .로 변환됩니다.")
        
        # 행 수 제한 및 배치 크기 옵션
        col1, col2 = st.columns(2)
        with col1:
            max_rows = st.number_input("처리할 최대 행 수 (0 = 제한 없음)", 
                                     min_value=0, value=100, step=100)
        with col2:
            batch_size = st.number_input(
                "배치 처리 크기", 
                min_value=10, 
                value=100, 
                step=10,
                help="한 번에 ChromaDB에 저장할 문서(청크)의 수입니다. 메모리 사용량과 처리 속도에 영향을 줍니다."
            )
        
        return {
            'max_rows': max_rows,
            'batch_size': batch_size
        }
    
    def render_preprocessing_preview(self, preview_result: Dict[str, Any]):
        """전처리 미리보기 렌더링"""
        if not preview_result['success']:
            st.error(preview_result['error'])
            return
        
        # 전처리 결과 정보
        if preview_result['max_rows_applied']:
            st.write(f"전처리 후 행 수: {preview_result['processed_rows']} "
                    f"(제한: {preview_result['max_rows_applied']}, 원본: {preview_result['original_rows']})")
        else:
            st.write(f"전처리 후 행 수: {preview_result['processed_rows']} "
                    f"(원본: {preview_result['original_rows']})")
        
        st.write(f"결측치로 인해 제거된 행 수: {preview_result['dropped_rows']}")
        
        # 전처리 미리보기
        if preview_result['preview'] is not None:
            st.write("전처리된 데이터 미리보기:")
            st.dataframe(preview_result['preview'])
        else:
            st.error("선택한 열에 유효한 데이터가 없습니다. 모든 행에 결측치가 있습니다.")
    
    def render_chroma_storage_options(self, collection_name: str, 
                                    persist_directory: str) -> Dict[str, str]:
        """ChromaDB 저장 옵션 렌더링"""
        st.subheader("ChromaDB 저장 옵션")
        
        new_collection_name = st.text_input("컬렉션 이름", value=collection_name)
        new_persist_directory = st.text_input("저장 경로", value=persist_directory)
        
        return {
            'collection_name': new_collection_name,
            'persist_directory': new_persist_directory
        }
    
    def render_embedding_model_selection(self, embedding_models: Dict[str, List[str]], 
                                       selected_model: str) -> str:
        """임베딩 모델 선택 UI 렌더링"""
        st.write("임베딩 모델 선택:")
        
        # 한국어 특화 모델 사용
        korean_models = embedding_models.get("한국어 특화 모델", [])
        
        if not korean_models:
            st.warning("사용 가능한 한국어 임베딩 모델이 없습니다. 기본 모델을 사용합니다.")
            korean_models = ["snunlp/KR-SBERT-V40K-klueNLI-augSTS"]
        
        return st.selectbox("임베딩 모델", korean_models, index=0)
    
    def render_hardware_acceleration_settings(self, gpu_info: Dict[str, Any], 
                                             device_options: Dict[str, str],
                                             current_preference: str) -> str:
        """하드웨어 가속 설정 렌더링"""
        st.subheader("하드웨어 가속 설정 (임베딩)")
        
        if gpu_info["available"]:
            st.success(f"✅ GPU 사용 가능: {gpu_info['count']}개의 GPU 감지됨.")
            for i, gpu_device in enumerate(gpu_info["devices"]):
                st.markdown(f"  - GPU {i}: {gpu_device['name']} (메모리: {gpu_device['memory_total']:.2f} GB)")
            
            current_index = list(device_options.values()).index(current_preference) if current_preference in device_options.values() else 0
            
            selected_device_label = st.radio(
                "임베딩 연산 장치 선택",
                options=list(device_options.keys()),
                index=current_index,
                help="임베딩 계산에 사용할 장치를 선택합니다. '자동'은 GPU가 있으면 GPU를, 없으면 CPU를 사용합니다."
            )
            
            return device_options[selected_device_label]
        else:
            st.info("ℹ️ 사용 가능한 GPU가 감지되지 않았습니다. 임베딩 연산은 CPU를 사용합니다.")
            return "cpu"
    
    def render_storage_button(self) -> bool:
        """저장 버튼 렌더링"""
        return st.button("ChromaDB에 데이터 저장")
    
    def render_storage_progress(self) -> Dict[str, Any]:
        """저장 진행 상황 UI 렌더링"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        gpu_status_placeholder = st.empty()
        
        return {
            'progress_bar': progress_bar,
            'status_text': status_text,
            'gpu_status_placeholder': gpu_status_placeholder
        }
    
    def render_storage_messages(self, messages: List[Dict[str, str]]):
        """저장 결과 메시지들 표시"""
        for message in messages:
            if message['type'] == 'success':
                st.success(message['content'])
            elif message['type'] == 'warning':
                st.warning(message['content'])
            elif message['type'] == 'info':
                st.info(message['content'])
            elif message['type'] == 'error':
                st.error(message['content'])
    
    def show_spinner(self, message: str = "처리 중..."):
        """스피너 표시"""
        return st.spinner(message)
    
    def show_error(self, message: str):
        """에러 메시지 표시"""
        st.error(message)
    
    def show_success(self, message: str):
        """성공 메시지 표시"""
        st.success(message)
    
    def show_warning(self, message: str):
        """경고 메시지 표시"""
        st.warning(message)
    
    def add_sidebar_separator(self):
        """사이드바 구분선 추가"""
        with st.sidebar:
            st.markdown("---")