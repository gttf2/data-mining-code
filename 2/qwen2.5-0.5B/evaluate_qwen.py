# evaluate_final.py
from config import Config
from model import SentimentClassifier
from load_data import DataLoader as DataLoaderClass
from transformers import AutoTokenizer
import torch
import os

def test_final_model():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from: {config.model_save_path}")
    
    # 加载模型
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.load_model(config.model_save_path)
    model.to(device)
    model.eval()
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载测试集
    data_loader = DataLoaderClass(config)
    test_texts, test_labels = data_loader.load_csv("dataset/test.csv")
    
    correct = 0
    total = len(test_texts)
    
    print(f"正在测试 {total} 条数据...")
    for i, text in enumerate(test_texts):
        label = test_labels[i]
        
        inputs = tokenizer(text, return_tensors="pt", max_length=config.max_seq_length, padding="max_length", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            pred = torch.argmax(outputs, dim=1).item()
            
        if pred == label:
            correct += 1
            
        if (i+1) % 50 == 0:
            print(f"已处理 {i+1}/{total}...")

    print(f"\n最终测试集准确率: {correct/total*100:.2f}%")

if __name__ == "__main__":
    test_final_model()
