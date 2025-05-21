"""
Ollama 관련 유틸리티 함수 모음
"""
import platform
import subprocess
import traceback

def is_ollama_installed():
    """
    Ollama가 설치되어 있는지 확인합니다.
    
    Returns:
        bool: Ollama가 설치되어 있으면 True, 아니면 False
    """
    try:
        if platform.system() == "Windows":
            # Windows에서는 where 명령어 사용
            subprocess.run(["where", "ollama"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            # Linux/macOS에서는 which 명령어 사용
            subprocess.run(["which", "ollama"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False

def is_ollama_running():
    """
    Ollama 서버가 실행 중인지 확인합니다.
    
    Returns:
        bool: Ollama 서버가 실행 중이면 True, 아니면 False
    """
    try:
        import ollama
        try:
            ollama.list()
            return True
        except Exception:
            return False
    except ImportError:
        return False

def is_ollama_lib_available():
    """
    Ollama 라이브러리가 설치되어 있는지 확인합니다.
    
    Returns:
        bool: Ollama 라이브러리가 설치되어 있으면 True, 아니면 False
    """
    try:
        import ollama
        return True
    except ImportError:
        return False

def get_ollama_models():
    """
    설치된 Ollama 모델 목록을 가져옵니다.
    
    Returns:
        list: 설치된 모델 목록
    """
    try:
        import ollama
        result = ollama.list()
        
        # 디버깅을 위해 전체 응답 출력
        print("Ollama API 응답:", result)
        
        # 응답 구조 확인
        if hasattr(result, 'models') and isinstance(result.models, list):
            # 새로운 API 구조: result.models는 Model 객체 리스트
            return [model.model for model in result.models]
        elif isinstance(result, dict) and 'models' in result:
            # 이전 API 구조: result['models']는 딕셔너리 리스트
            return [model['name'] for model in result['models']]
        else:
            # 다른 구조인 경우
            print("예상치 못한 응답 구조:", result)
            if isinstance(result, list):
                # 리스트인 경우 각 항목의 model 속성 또는 문자열 변환
                return [getattr(item, 'model', str(item)) for item in result]
            else:
                # 그 외의 경우 빈 리스트 반환
                return []
    except Exception as e:
        print(f"모델 목록을 가져오는 중 오류 발생: {e}")
        traceback.print_exc()
        return []

def query_ollama(prompt, model_name="llama2"):
    """
    Ollama 모델에 질의합니다.
    
    Args:
        prompt (str): 프롬프트
        model_name (str): Ollama 모델 이름
        
    Returns:
        str: 모델 응답
    """
    try:
        import ollama
        
        # Ollama 라이브러리 사용
        response = ollama.generate(model=model_name, prompt=prompt)
        
        # 응답 구조 확인
        if isinstance(response, dict) and 'response' in response:
            # 이전 API 구조: 딕셔너리 형태
            return response['response']
        elif hasattr(response, 'response'):
            # 새로운 API 구조: 객체 형태
            return response.response
        else:
            # 다른 구조인 경우
            print("예상치 못한 응답 구조:", response)
            return str(response)
    except ImportError:
        return "ollama 패키지가 설치되어 있지 않습니다. 'pip install ollama'를 실행하세요."
    except Exception as e:
        print(f"Ollama 질의 중 오류 발생: {e}")
        traceback.print_exc()
        return f"Ollama 오류: {e}"

def chat_with_ollama(messages, model_name="llama2"):
    """
    Ollama 모델과 역할 기반 대화를 수행합니다.
    
    Args:
        messages (list): 대화 메시지 목록. 각 메시지는 {'role': 'system|user|assistant', 'content': '내용'} 형식
        model_name (str): Ollama 모델 이름
        
    Returns:
        str: 모델 응답
    """
    try:
        import ollama
        
        # Ollama 라이브러리의 chat 함수 사용
        response = ollama.chat(model=model_name, messages=messages)
        
        # 응답 구조 확인
        if isinstance(response, dict) and 'message' in response:
            # 이전 API 구조: 딕셔너리 형태
            return response['message']['content']
        elif hasattr(response, 'message') and hasattr(response.message, 'content'):
            # 새로운 API 구조: 객체 형태
            return response.message.content
        else:
            # 다른 구조인 경우
            print("예상치 못한 응답 구조:", response)
            return str(response)
    except ImportError:
        return "ollama 패키지가 설치되어 있지 않습니다. 'pip install ollama'를 실행하세요."
    except Exception as e:
        print(f"Ollama 대화 중 오류 발생: {e}")
        traceback.print_exc()
        return f"Ollama 오류: {e}"


def get_ollama_install_guide():
    """
    Ollama 설치 가이드를 반환합니다.
    
    Returns:
        str: 설치 가이드 마크다운
    """
    system = platform.system()
    
    if system == "Windows":
        return """
        ### Windows에 Ollama 설치하기

        1. [Ollama 웹사이트](https://ollama.ai/download/windows)에서 Windows 설치 파일을 다운로드합니다.
        2. 다운로드한 설치 파일을 실행하고 설치 지침을 따릅니다.
        3. 설치가 완료되면 시스템 트레이에 Ollama 아이콘이 나타납니다.
        4. 모델을 다운로드하려면 명령 프롬프트를 열고 다음 명령어를 실행합니다:
           ```
           ollama pull llama2
           ```
        5. 이 애플리케이션으로 돌아와 새로고침 버튼을 클릭하세요.
        
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 직접 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        """
    elif system == "Darwin":  # macOS
        return """
        ### macOS에 Ollama 설치하기

        1. [Ollama 웹사이트](https://ollama.ai/download/mac)에서 macOS 설치 파일을 다운로드합니다.
        2. 다운로드한 .dmg 파일을 열고 Ollama를 Applications 폴더로 드래그합니다.
        3. Applications 폴더에서 Ollama를 실행합니다.
        4. 모델을 다운로드하려면 터미널을 열고 다음 명령어를 실행합니다:
           ```
           ollama pull llama2
           ```
        5. 이 애플리케이션으로 돌아와 새로고침 버튼을 클릭하세요.
        
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 직접 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        """
    else:  # Linux
        return """
        ### Linux에 Ollama 설치하기

        1. 터미널을 열고 다음 명령어를 실행합니다:
           ```
           curl -fsSL https://ollama.ai/install.sh | sh
           ```
        2. 설치가 완료되면 Ollama 서버를 시작합니다:
           ```
           ollama serve
           ```
        3. 새 터미널 창을 열고 모델을 다운로드합니다:
           ```
           ollama pull llama2
           ```
        4. 이 애플리케이션으로 돌아와 새로고침 버튼을 클릭하세요.
        
        ### Ollama 라이브러리 설치하기
        
        Python에서 Ollama를 직접 사용하려면 다음 명령어로 라이브러리를 설치하세요:
        ```
        pip install ollama
        ```
        """