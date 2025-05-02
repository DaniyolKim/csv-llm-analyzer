import pandas as pd
import ollama
import time
import subprocess
import threading

def get_available_models():
    """
    Get a list of available models from Ollama
    
    Returns:
        tuple: (list of model names, installation guide if needed)
    """
    installation_guide = """
    # Ollama 설치 및 모델 다운로드 가이드
    
    ## Ollama 설치 방법
    1. Windows: https://ollama.com/download/windows 에서 설치 프로그램을 다운로드하여 실행하세요.
    2. macOS: https://ollama.com/download/mac 에서 설치 프로그램을 다운로드하여 실행하세요.
    3. Linux: 터미널에서 다음 명령어를 실행하세요:
       ```
       curl -fsSL https://ollama.com/install.sh | sh
       ```
    
    ## Mistral 모델 다운로드 방법
    Ollama가 설치된 후, 터미널 또는 명령 프롬프트에서 다음 명령어를 실행하세요:
    ```
    ollama pull mistral:latest
    ```
    
    다운로드가 완료되면 이 애플리케이션을 다시 시작하세요.
    """
    
    try:
        # First try using the Python API
        try:
            models = ollama.list()
            if 'models' in models and models['models']:
                return [model['name'] for model in models['models']], None
            else:
                print("No models found or unexpected API response format")
                return [], installation_guide
        except Exception as api_error:
            print(f"API error: {str(api_error)}, falling back to command line")
            
        # Fall back to command line if API fails
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse the command line output
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header line
                    models = []
                    for line in lines[1:]:  # Skip header
                        parts = line.split()
                        if parts:
                            models.append(parts[0])  # First column is the name
                    if models:
                        return models, None
            
            # If we get here, no models were found
            return [], installation_guide
        except Exception as cmd_error:
            print(f"Command line error: {str(cmd_error)}")
            return [], installation_guide
            
    except Exception as e:
        print(f"Error getting models: {str(e)}")
        return [], installation_guide

def call_ollama_with_timeout(model_name, prompt, row_text, timeout=30):
    """
    Call Ollama API with a timeout
    
    Args:
        model_name (str): Name of the model to use
        prompt (str): The prompt to send
        row_text (str): The text data to analyze
        timeout (int): Timeout in seconds
        
    Returns:
        str: Response content or error message
    """
    result = {"content": None, "error": None}
    
    def api_call():
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': f"{prompt}: {row_text}"
                    }
                ]
            )
            result["content"] = response['message']['content']
        except Exception as e:
            result["error"] = str(e)
    
    # Create and start the thread
    thread = threading.Thread(target=api_call)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete or timeout
    thread.join(timeout)
    
    # Check if the thread is still alive (timeout occurred)
    if thread.is_alive():
        return f"Error: API call timed out after {timeout} seconds"
    
    # Return the result or error
    if result["error"]:
        return f"Error: {result['error']}"
    
    return result["content"]

def analyze_text_with_prompt(prompt, df, model_name="exaone3.5:7.8b", progress_callback=None, batch_size=10, max_retries=3, timeout=30):
    """
    Analyze text data in a dataframe using Ollama and add responses as a new column.
    
    Args:
        prompt (str): The prompt to send to Ollama
        df (pandas.DataFrame): The dataframe containing the text data to analyze
        model_name (str): Name of the Ollama model to use
        progress_callback (function): Optional callback function to report progress
        batch_size (int): Number of rows to process in each batch
        max_retries (int): Maximum number of retries for failed API calls
        timeout (int): Timeout in seconds for each API call
        
    Returns:
        pandas.DataFrame: The original dataframe with an additional 'response' column
    """
    result_df = pd.DataFrame()
    total_rows = len(df)
    
    for start in range(0, total_rows, batch_size):
        # Calculate progress percentage
        progress = min(1.0, start / total_rows) if total_rows > 0 else 0
        
        # Update progress if callback is provided
        if progress_callback:
            progress_callback(progress, f"처리 중: {start}/{total_rows} 행")
        
        # Process batch
        batch_df = df.iloc[start:start + batch_size].copy()
        batch_df['response'] = None
        
        for idx, row in batch_df.iterrows():
            # Convert row data to string format - limit to first 500 characters per column
            row_text = ' '.join([f"{col}: {str(row[col])[:500]}" for col in batch_df.columns if col != 'response'])
            
            # Retry logic
            retry_count = 0
            response_content = None
            
            while retry_count < max_retries:
                try:
                    # Add a small delay between API calls to avoid rate limiting
                    if idx > start or retry_count > 0:
                        time.sleep(0.5)
                    
                    # Call Ollama with timeout
                    response_content = call_ollama_with_timeout(model_name, prompt, row_text, timeout)
                    
                    # If the response starts with "Error:", it's an error message
                    if response_content and not response_content.startswith("Error:"):
                        break
                    
                    # If we got an error, increment retry count and try again
                    retry_count += 1
                    if retry_count < max_retries:
                        if progress_callback:
                            progress_callback(progress, f"처리 중: {start + (idx - start)}/{total_rows} 행 (재시도 {retry_count}/{max_retries})")
                        time.sleep(1)  # Wait a bit longer before retrying
                except Exception as e:
                    retry_count += 1
                    response_content = f"Error: {str(e)}"
                    if retry_count < max_retries:
                        if progress_callback:
                            progress_callback(progress, f"처리 중: {start + (idx - start)}/{total_rows} 행 (재시도 {retry_count}/{max_retries})")
                        time.sleep(1)  # Wait a bit longer before retrying
            
            # Store the final response (or error message)
            batch_df.at[idx, 'response'] = response_content if response_content else f"Error: Failed after {max_retries} retries"
            
            # Update progress for each row if callback is provided
            if progress_callback:
                row_progress = min(1.0, (start + (idx - start + 1) / batch_size) / total_rows)
                progress_callback(row_progress, f"처리 중: {start + (idx - start + 1)}/{total_rows} 행")
        
        result_df = pd.concat([result_df, batch_df])
    
    # Final progress update
    if progress_callback:
        progress_callback(1.0, "분석 완료!")
    
    return result_df
