import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

def load_image(image):
    if image is None:
        return None
    return image.convert('RGB')

def main(args):
    model_path = "sensenova/SenseNova-SI-1.3-InternVL3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    ).eval()

    dataset = load_dataset(args.dataset, split="test")
    
    output_filepath = f"hle_sensenova_8b.json"
    if os.path.exists(output_filepath):
        with open(output_filepath, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100%}"

    for item in tqdm(dataset):
        unique_id = item['id']
        if unique_id in predictions:
            continue
        question_text = item['question']
        image = item.get('image', None)
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question_text}"
        
        pixel_values = None
        if image is not None:
            pixel_values = model.extract_feature(load_image(image)).to(torch.bfloat16).cuda()

        generation_config = dict(max_new_tokens=1024, do_sample=False)
        
        try:
            with torch.no_grad():
                response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config)
            
            predictions[unique_id] = {
                "model": "SenseNova-SI-1.3-InternVL3-8B",
                "response": response,
                "usage": {}
            }
        except Exception as e:
            print(f"Error at {unique_id}: {e}")
            continue

        if len(predictions) % 10 == 0:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=4, ensure_ascii=False)

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="openai/HLE", help="HLE HF Dataset")
    args = parser.parse_args()
    main(args)