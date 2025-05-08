import pandas as pd
import numpy as np
import os

def generate_sample_data(filename='sample_data.csv', rows=100):
    """
    샘플 CSV 데이터를 생성합니다.
    
    Args:
        filename (str): 생성할 파일 이름
        rows (int): 생성할 행 수
    """
    # 랜덤 시드 설정
    np.random.seed(42)
    
    # 데이터 생성
    data = {
        '나이': np.random.randint(20, 65, size=rows),
        '성별': np.random.choice(['남성', '여성'], size=rows),
        '수입': np.random.normal(5000, 1500, size=rows).round(2),
        '지출': np.random.normal(3000, 1000, size=rows).round(2),
        '만족도': np.random.randint(1, 6, size=rows),
        '지역': np.random.choice(['서울', '부산', '인천', '대구', '광주', '대전', '울산'], size=rows),
        '방문횟수': np.random.poisson(5, size=rows)
    }
    
    # 파생 변수 생성
    data['저축'] = (data['수입'] - data['지출']).round(2)
    
    # 일부 결측치 추가
    for col in ['수입', '지출', '만족도']:
        mask = np.random.random(size=rows) < 0.05  # 5% 결측치
        data[col] = np.where(mask, np.nan, data[col])
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    # CSV 파일로 저장
    df.to_csv(filename, index=False, encoding='utf-8')
    
    print(f"샘플 데이터가 '{filename}'에 생성되었습니다.")
    return df

if __name__ == "__main__":
    # 현재 스크립트 위치에 샘플 데이터 생성
    generate_sample_data()