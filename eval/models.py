import os
import torch
from transformers import AutoConfig, Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import re
from typing import *

class Agent:
    def __init__(
        self, 
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=128
    ):
        self.modelPath = path
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        
    def generate(self, sample:Dict[str, str]):
        raise NotImplementedError()
    
    def input_lenth(self, sample:Dict[str, str]):
        text = sample["messages"][0]["content"]
        image = sample["images"][0]
        messages = [{
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":text,
                },
                {
                    "type":"image",
                    "image":image
                }
            ]
        }]
        context = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[context],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        input_length = len(inputs.input_ids[0])
        
        return input_length

class Qwen2VLAgent(Agent):
    def __init__(
        self, 
        path:str, cache_dir='~/.cache', 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=128
    ):
        super().__init__(path=path, cache_dir=cache_dir, device=device, max_new_tokens=max_new_tokens)
        self.model_config = AutoConfig.from_pretrained(path)
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                path, 
                config=self.model_config, 
                device_map="auto", 
                trust_remote_code=True, 
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        except Exception:
            print("disable flash attention")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(path, config=self.model_config, device_map="auto", trust_remote_code=True)
        self.processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(path)
        
    def generate(self, sample:Dict[str, str]):
        text = sample["messages"][0]["content"]
        image = sample["images"][0]
        messages = [{
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":text,
                },
                {
                    "type":"image",
                    "image":image
                }
            ]
        }]
        context = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[context],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generatedIds = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens
            ).to(self.device)
        trueGenIds = [
            genIds[len(inIds):] for inIds, genIds in zip(inputs.input_ids, generatedIds)
        ]
        outputText = self.processor.batch_decode(trueGenIds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputText
    
