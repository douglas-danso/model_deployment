import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

def model_fn(model_dir, *args):
    model_id = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float32
    )
    model.eval()
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(input_data, model_dict, *args):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    prompt = input_data.get("prompt", "")
    max_length = input_data.get("max_length", 50)
    temperature = input_data.get("temperature", 0.7)
    top_p = input_data.get("top_p", 0.9)

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")


if __name__ == "__main__":
    
    model_dict = model_fn("model_dir")
    input_data = {"prompt": "hello world"}
    prediction = predict_fn(input_data, model_dict)
    print(prediction)