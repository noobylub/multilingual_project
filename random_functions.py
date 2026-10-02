import pandas as pd
import torch


# For loading data, we can load multiple languages 
def return_data(language:str):
    return pd.read_csv(f"hf://datasets/mrlbenchmarks/global-piqa-parallel/data/parallel_{language}.tsv", sep="\t")



# generate in batches
def batch_output(model,tokenizer, prompts:list[str], max_new_tokens:int=400, batch_size:int=8):
     # Apply left padding so that they are all equal
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is not None:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for p in prompts
        ]
        add_special_tokens = False  # template already includes BOS/special tokens
    else:
        texts = prompts
        add_special_tokens = True
    outputs = []
    # batch outputs and generate 
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
            add_special_tokens=add_special_tokens
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id)
        # input_ids.shape[1] has the max, because we pad it 
        # and new_tokens contains only the generated tokens beyond the input sequence
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs
   
   
        





# save_dir = "./models/qwq-32b-4bit"

# model.save_pretrained(save_dir, safe_serialization=True)
# tokenizer.save_pretrained(save_dir)

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model = AutoModelForCausalLM.from_pretrained(
#     save_dir,
#     device_map="auto",
#     local_files_only=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(save_dir, local_files_only=True)