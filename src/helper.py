import torch
import logging
from keys import mykey
import re
import time
import numpy as np
import os
# A dictionary to cache models and tokenizers to avoid reloading

global models
models = {}
LOCALHOST_API_ADDR = os.getenv("LOCALHOST_API_ADDR", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")

class ModelCache:
    def __init__(self, model_name, use_vllm=True, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api == "openai":
            from openai import OpenAI
            self.api_account = self.args.get("api_account", "openai")
            self.client = OpenAI(api_key=mykey[self.api_account])
        elif self.use_api == "llama":
            from openai import OpenAI
            self.client = OpenAI(api_key="EMPTY", base_url=LOCALHOST_API_ADDR)
            #print(f"LOCALHOST_API_ADDR: {LOCALHOST_API_ADDR}")
        elif self.use_api == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url='https://api.deepseek.com')
            #print(f"DEEPSEEK_API_KEY: ****{DEEPSEEK_API_KEY[-4:]}")
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                self.model = LLM(model=self.model_name, enable_prefix_caching=enable_prefix_caching)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            except Exception as e:
                log_info(f"[ERROR] [{self.model_name}]: If using a custom local model, it is not compatible with VLLM, will load using Huggingfcae and you can ignore this error: {str(e)}", mode="error")
                self.use_vllm = False
        '''if not self.use_vllm and self.use_api != "openai":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()  # Set the model to evaluation mode
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]'''
    
    def generate(self, messages):
        log_info(f"[{self.model_name}][INPUT]: {messages}")

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", 256)
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)
        
        if self.use_api == "openai": 
            return self.openai_generate(messages)
        elif self.use_api == "llama": 
            return self.llama_generate(messages)
        elif self.use_api == "deepseek": 
            return self.deepseek_generate(messages)
        elif self.use_vllm: 
            return self.vllm_generate(messages)
        else: 
            return self.huggingface_generate(messages)
    
    def huggingface_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            prompt = "\n\n".join([m['content'] for m in messages])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=self.max_tokens, 
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.terminators
        )
        # TODO: If top_logprobs > 0, return logprobs of generation
        response_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        usage = {"input_tokens": inputs.shape[-1], "output_tokens": outputs.shape[-1]-inputs.shape[-1]}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, None, usage
        
    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        if self.top_logprobs == 0:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
        else:
            response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        logprobs=True, 
                        top_logprobs=self.top_logprobs
                    )
        
        num_input_tokens = response["usage"]["prompt_tokens"]
        num_output_tokens = response["usage"]["completion_tokens"]
        response_text = response.choices[0].text.strip()
        log_probs = response.choices[0].logprobs.top_logprobs if self.top_logprobs > 0 else None
        
        log_info(f"[{self.model_name}][OUTPUT]: {response}")
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}

    def llama_generate(self, messages):
        """Handle LLaMA specific generation with probability support"""
        try:
            if self.top_logprobs > 0 and 'target_tokens' in self.args:
                # Generate with probability tracking
                response = self.client.chat.completions.create(
                    messages=messages,
                    model="/home/tlicj/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080",
                    logprobs=True,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Process logprobs for target tokens
                chosen_tokens_probs = {}
                first_choice = response.choices[0]
                if hasattr(first_choice, 'logprobs'):
                    skip = 0
                    for token_index in range(len(first_choice.logprobs.content)):
                        if skip > 0:
                            skip -= 1
                            continue
                            
                        token_logprob = first_choice.logprobs.content[token_index]
                        token = token_logprob.token
                        
                        if token in self.args['target_tokens']:
                            logprob = token_logprob.logprob
                            # Check for multi-token targets
                            next_token_index = token_index
                            while True:
                                next_token_index += 1
                                if next_token_index >= len(first_choice.logprobs.content):
                                    break
                                next_token = first_choice.logprobs.content[next_token_index].token
                                if (token + next_token) in self.args['target_tokens']:
                                    token += next_token
                                    logprob += first_choice.logprobs.content[next_token_index].logprob
                                    skip += 1
                                else:
                                    break
                            
                            if skip > 0:
                                logprob = logprob / skip
                            chosen_tokens_probs[token] = np.exp(logprob)
                
                return chosen_tokens_probs, None, {"input_tokens": response.usage.prompt_tokens, 
                                                 "output_tokens": response.usage.completion_tokens}
            else:
                # Normal generation
                response = self.client.chat.completions.create(
                    messages=messages,
                    model="/home/tlicj/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return (response.choices[0].message.content, None, 
                        {"input_tokens": response.usage.prompt_tokens, 
                         "output_tokens": response.usage.completion_tokens})
                
        except Exception as e:
            log_info(f"LLaMA generation error: {str(e)}", mode="error")
            time.sleep(1)
            return self.llama_generate(messages)

    def deepseek_generate(self, messages):
        """Handle Deepseek specific generation with response cleaning"""
        try:
            response = self.client.chat.completions.create(
                model='deepseek-chat',
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            content = response.choices[0].message.content
            
            # Clean up response based on patterns
            if '<think>' in content and '</think>' in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            elif '**' in content:
                match = re.search(r'\n\n(.*?)\n\n', content, re.DOTALL)
                if match and 'Answer' in match.group(1).strip():
                    content = match.group(1).strip()
                else:
                    match = re.search(r'\*\*Answer:\*\*(.*?)\n\n', content, re.DOTALL)
                    content = match.group(1).strip() if match else content
            elif '\n\n' in content:
                match = re.search(r'(.*?)\n\n', content, re.DOTALL)
                content = match.group(0).strip() if match else content
                
            return (content, None, 
                    {"input_tokens": response.usage.prompt_tokens, 
                     "output_tokens": response.usage.completion_tokens})
                     
        except Exception as e:
            log_info(f"Deepseek generation error: {str(e)}", mode="error")
            time.sleep(3)
            return self.deepseek_generate(messages)

def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    if 'gpt' in model_name or 'o1' in model_name: use_api = "openai"
    if 'llama' in model_name: use_api = "llama"
    if 'deepseek' in model_name: use_api = "deepseek"
    model_cache = models.get(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[model_name] = model_cache
    
    return model_cache.generate(messages)
