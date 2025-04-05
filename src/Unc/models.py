import os
import time
import re
import numpy as np
from openai import OpenAI
LOCALHOST_API_ADDR = os.getenv("LOCALHOST_API_ADDR", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Time gap settings for rate limiting
time_gap = {"deepseek-chat": 3}

def llama_response(message, model=None, temperature=0, max_tokens=500, target_tokens = None, only_probs=False):
    """
    Get a response from the Llama model using the Together API.
    
    Args:
        message (list): List of message dictionaries with 'role' and 'content'
        model (str, optional): Model name (not used, defaults to Llama-2-70b-chat-hf)
        temperature (float): Sampling temperature
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The model's response text
    """
    llama_client = OpenAI(api_key="EMPTY", base_url=LOCALHOST_API_ADDR)
    try:
        if only_probs:
            chat_completion = llama_client.chat.completions.create(
                messages=message,
                model="/home/tlicj/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080",
                logprobs=True,
                temperature = temperature,
                max_tokens=max_tokens
            )
            first_choice = chat_completion.choices[0]
            chosen_tokens_probs = {}
            # Check if logprobs are available
            
            if hasattr(first_choice, 'logprobs'):
                # Iterate through tokens and get probabilities for A, B, C, D
                count = 0
                for token_index in range(len(first_choice.logprobs.content)):
                    token_logprob = first_choice.logprobs.content[token_index]
                    token = token_logprob.token
                    if token in target_tokens:
                        # Remove any spaces from the token
                        token = token.strip()
                        prob = np.exp(token_logprob.logprob)
                        chosen_tokens_probs[token] = prob
                        count += 1
                        if count >= 2:  # Only check first 2 occurrences
                            break
            else:
                print("Token probabilities are not available in the response.")
            return chosen_tokens_probs
        else:
            chat_completion = llama_client.chat.completions.create(
                messages=message,
                model="/home/tlicj/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080",
                temperature=temperature,
                max_tokens=max_tokens
            )
            return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        time.sleep(1)
        return llama_response(message, model, temperature, max_tokens)

def deepseek_response(message: list, model="deepseek-chat", temperature=0, max_tokens=4000):
    """
    Get a response from the Deepseek model using the PPInfra API.
    
    Args:
        message (list): List of message dictionaries with 'role' and 'content'
        model (str): Model name
        temperature (float): Sampling temperature
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: The model's response text, with any <think> tags removed
    """
    ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url='https://api.deepseek.com')
    try:
        res = ds_client.chat.completions.create(
            model="deepseek-chat", 
            messages=message, 
            temperature=temperature, 
            n=1,
            max_tokens=4000
        )
        content = res.choices[0].message.content
        if '<think>' in content and '</think>' in content:
            content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content_clean
        else:
            raise ValueError("Model length exceeds the maximum limit")
    except Exception as e:
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return deepseek_response(message, model, temperature, max_tokens)

def get_response_method(model):
    """
    Get the appropriate response function based on the model name.
    
    Args:
        model (str): Model name
        
    Returns:
        function: The corresponding response function
    """
    response_methods = {
        "llama": llama_response,
        "deepseek": deepseek_response,
    }
    return response_methods.get(model, lambda _: NotImplementedError()) 