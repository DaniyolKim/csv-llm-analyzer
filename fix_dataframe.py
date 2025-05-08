import pandas as pd
import streamlit as st

def fix_dataframe_for_streamlit(df):
    """
    Streamlit에서 표시할 수 있도록 데이터프레임을 수정합니다.
    PyArrow 변환 오류를 방지하기 위해 모든 컬럼을 적절한 타입으로 변환합니다.
    
    Args:
        df (pandas.DataFrame): 원본 데이터프레임
        
    Returns:
        pandas.DataFrame: 수정된 데이터프레임
    """
    if df is None or len(df) == 0:
        return df
    
    # 데이터프레임 복사
    fixed_df = df.copy()
    
    # 모든 컬럼에 대해 타입 확인 및 변환
    for col in fixed_df.columns:
        # 'Unnamed:' 컬럼은 문자열로 변환
        if col.startswith('Unnamed:'):
            fixed_df[col] = fixed_df[col].astype(str)
            continue
        
        # 컬럼의 데이터 타입 확인
        try:
            # 숫자형으로 변환 가능한지 확인
            pd.to_numeric(fixed_df[col], errors='raise')
            # 숫자형으로 변환 가능하면 그대로 유지
        except:
            # 숫자형으로 변환 불가능하면 문자열로 변환
            fixed_df[col] = fixed_df[col].astype(str)
    
    return fixed_df

def safe_display_dataframe(df, num_rows=5):
    """
    데이터프레임을 안전하게 Streamlit에 표시합니다.
    
    Args:
        df (pandas.DataFrame): 표시할 데이터프레임
        num_rows (int): 표시할 행 수
    """
    if df is None:
        st.warning("표시할 데이터가 없습니다.")
        return
    
    try:
        # 데이터프레임 수정
        display_df = fix_dataframe_for_streamlit(df)
        
        # 행 수 제한
        if num_rows > 0 and len(display_df) > num_rows:
            display_df = display_df.head(num_rows)
        
        # 데이터프레임 표시
        st.dataframe(display_df)
    except Exception as e:
        st.error(f"데이터프레임 표시 오류: {str(e)}")
        
        # 오류 발생 시 대체 표시 방법
        st.write("데이터프레임을 표 형식으로 표시합니다:")
        st.table(df.head(num_rows).astype(str))