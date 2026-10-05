import pandas as pd
import torch


# For loading data from hugging face, we can load multiple languages 
def return_dataPIQA(language:str):
    return pd.read_csv(f"hf://datasets/mrlbenchmarks/global-piqa-parallel/data/parallel_{language}.tsv", sep="\t")

def return_dataMGSM(language:str):
    # Login using   e.g. `huggingface-cli login` to access this dataset
    df = pd.read_parquet(f"hf://datasets/CohereLabs/global-mgsm/{language}/test.parquet")
    return df


# generate texts in batches
def batch_output(model,tokenizer, system_prompt:str, prompts:list[str], max_new_tokens:int=400, batch_size:int=8):
     # Apply left padding so that they are all equal
    tokenizer.padding_side = 'left'

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is not None:
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p}],
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
            generated = model.generate(**inputs,  max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.pad_token_id)
        # input_ids.shape[1] has the max, because we pad it 
        # and new_tokens contains only the generated tokens beyond the input sequence
        new_tokens = generated[:, inputs.input_ids.shape[1]:]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    return outputs
   
   
# unfortunately, their batching api is very unreliable, so here we call it one by one
# Calling API through open router
def call_open_router_api(API_KEY:str, messages:str,system_prompt:str, model:str):
    import requests
    import json

    # First API call with reasoning
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "messages": [
                {
                "role":'system',
                "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": messages,
                }
            ],
            "reasoning": {"enabled": False}
        })
    )

    # # Extract the assistant message with reasoning_details
    response = response.json()
    response = response['choices'][0]['message']
    return response






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


# import time
# import requests

# if "batch_id" not in globals():
#     raise RuntimeError("Run the submit cell first so batch_id is defined.")

# API_URL = "https://openrouter.ai/api/v1"
# HEADERS = {"Authorization": f"Bearer {API_KEY}"}
# terminal_statuses = {"completed", "failed", "expired", "cancelled"}
# deadline = time.monotonic() + 24 * 60 * 60
# not_found_since = None

# while True:
#     poll_response = requests.get(
#         f"{API_URL}/batches/{batch_id}",
#         headers=HEADERS,
#         timeout=30,
#     )
#     if poll_response.status_code == 404:
#         if not_found_since is None:
#             not_found_since = time.monotonic()
#         if time.monotonic() - not_found_since > 120:
#             raise RuntimeError(
#                 f"Batch remained unavailable for 2 minutes: {poll_response.text}"
#             )
#         print("Batch is not visible yet; retrying in 2 seconds.")
#         time.sleep(2)
#         continue
#     if not poll_response.ok:
#         raise RuntimeError(
#             f"Batch polling failed (HTTP {poll_response.status_code}): "
#             f"{poll_response.text}"
#         )

#     not_found_since = None
#     batch = poll_response.json()
#     status = batch.get("status")
#     if status is None:
#         raise RuntimeError(f"Batch API response has no status: {batch}")
#     print(f"Batch status: {status}; requests: {batch.get('request_counts')}")

#     if status in terminal_statuses:
#         break
#     if time.monotonic() >= deadline:
#         raise TimeoutError(f"Batch {batch_id} did not finish within 24 hours")
#     time.sleep(2)

# if batch["status"] == "failed":
#     raise RuntimeError(f"Batch failed: {batch.get('error')}")

# print(batch["status"], batch.get("results"))