"""
텍스트 처리 관련 유틸리티 함수 모음
"""
import re

def clean_text(text):
    """
    텍스트에서 불필요한 특수문자를 제거하되, 문장 구분 기호와 필수 문법 부호는 보존합니다.
    URL도 제거합니다.
    
    Args:
        text (str): 정제할 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if not isinstance(text, str):
        return str(text)
    
    # URL 패턴 제거 (http://, https://, www. 로 시작하는 주소)
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # 문장 구분 기호와 필수 문법 부호를 보존하면서 불필요한 특수문자 제거
    # 보존할 문자: 
    # - 영문, 숫자, 한글, 공백
    # - 문장 구분 기호: . ? ! ; : 등
    # - 인용 부호: " ' ` 등
    # - 괄호: () [] {} 등
    # - 산술 기호: + - * / % = 등
    # - 기타 필수 문법 부호: , & @ # 등
    cleaned_text = re.sub(r'[^\w\s\.\?!;:\'\"\\/\(\)\[\]\{\}\+\-\*/%=,&@#]', ' ', cleaned_text)
    
    # 연속된 공백을 하나의 공백으로 치환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text