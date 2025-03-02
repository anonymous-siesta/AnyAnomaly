import torch
import os
import sys
from PIL import Image
from transformers import AutoModel, AutoTokenizer
        

def make_instruction(cfg, keyword, temporal_context=False):
    # simple
    if cfg.prompt_type == 0:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
        )

    # complex (+ consideration)
    elif cfg.prompt_type == 1:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
        )

    # complex (+ reasoning)
    elif cfg.prompt_type == 2:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )

    # complex (+ reasoning, consideration)
    elif cfg.prompt_type == 3:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )
    
    if temporal_context == False:
        return instruction
    
    # insturction for temporal context
    else:
        # simple
        if cfg.prompt_type == 0:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
            )

        # complex (+ consideration)
        elif cfg.prompt_type == 1:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
            )

        # complex (+ reasoning)
        elif cfg.prompt_type == 2:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
            )

        # complex (+ reasoning, consideration)
        elif cfg.prompt_type == 3:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"    
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
            )

        return instruction, tc_instruction
    

def load_lvlm(model_path, device):
    if model_path == 'MiniCPM-V-2_6':
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    elif 'int4' in model_path:
        model = AutoModel.from_pretrained(f'openbmb/{model_path}', trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(f'openbmb/{model_path}', trust_remote_code=True, torch_dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(f'openbmb/{model_path}', trust_remote_code=True)
    model = model.to(device=device).eval()
    return tokenizer, model


def lvlm_test(tokenizer, model, qs, image_path, image=None):
    if image is None:
        image = Image.open(image_path)
    
    image = image.convert('RGB')
    
    msgs = [{'role': 'user', 'content': qs}]
    
    answer = model.chat(
        image=image,
        msgs=msgs,
        tokenizer = tokenizer,
        sampling=False,
        temperature=0.7,
    )
    
    return answer