"""
데이터프레임 처리 관련 유틸리티 함수 모음
"""
import pandas as pd
from .text_utils import clean_text

def preprocess_dataframe(df, selected_columns, max_rows=None):
    """
    데이터프레임을 전처리합니다.
    
    Args:
        df (pandas.DataFrame): 원본 데이터프레임
        selected_columns (list): 선택한 열 목록
        max_rows (int, optional): 처리할 최대 행 수
        
    Returns:
        pandas.DataFrame: 전처리된 데이터프레임
    """
    # 선택한 열만 추출
    selected_df = df[selected_columns].copy()
    
    # 결측치가 있는 행 제거
    selected_df = selected_df.dropna()
    
    # 최대 행 수 제한
    if max_rows is not None and max_rows > 0 and max_rows < len(selected_df):
        selected_df = selected_df.head(max_rows)
    
    # 텍스트 정제
    for col in selected_df.columns:
        if selected_df[col].dtype == 'object':
            selected_df[col] = selected_df[col].apply(clean_text)
    
    return selected_df