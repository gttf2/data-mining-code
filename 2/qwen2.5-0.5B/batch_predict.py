import torch
from transformers import AutoTokenizer
from config import Config
from model import SentimentClassifier
import os

def load_model_and_predict():
    # 1. 初始化配置和设备
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")

    # 2. 加载 Tokenizer
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. 加载模型权重
    print(f"正在加载模型权重: {config.model_save_path} ...")
    model = SentimentClassifier(config.model_name, config.num_classes)
    
    # 检查权重文件是否存在
    if os.path.exists(config.model_save_path):
        model.load_model(config.model_save_path)
    else:
        # 尝试默认路径容错
        default_path = "saved_models/qwen_sentiment_model.pth"
        if os.path.exists(default_path):
            print(f"配置文件路径未找到，尝试默认路径: {default_path}")
            model.load_model(default_path)
        else:
            print("错误：找不到模型权重文件，请先运行训练！")
            return

    model.to(device)
    model.eval()
    print("模型加载完成！准备开始测试...\n")

    # ================= 4. 定义实测数据 =================
    test_cases = [
        # --- 明显的正面评价 (English) ---
        "I absolutely love this product! It works perfectly and looks great.",
        "The battery life is amazing, lasted 2 days without charging. Highly recommended.",
        "Best purchase I've made this year. Fast shipping and great quality.",
        
        # --- 明显的负面评价 (English) ---
        "This is total garbage. It broke after just one use.",
        "Save your money. The quality is terrible and customer service is rude.",
        "Not worth the price. Very disappointed with the performance.",
        
        # --- 明显的正面评价 (中文 - 测试 Qwen 的多语言能力) ---
        "这个产品太棒了，质量非常好，我很喜欢！",
        "物流很快，包装也很精美，这是一次非常愉快的购物体验。",
        
        # --- 明显的负面评价 (中文) ---
        "简直是浪费钱，东西收到就是坏的，根本没法用。",
        "非常失望，跟描述的完全不一样，千万不要买。",
        
        # --- 困难样本/具有挑战性的评价 (Mixed/Sarcasm) ---
        "It's okay, but a bit expensive for what you get.",  # 中性偏负
        "I wanted to like it, but it just stopped working randomly.", # 转折
        "真不错买到一个破盒子，太感激了",  # 讽刺 (字面是 Thanks，实际是负面)
        "The design is nice, but the material feels very cheap and flimsy." # 混合评价
    ]
    # ===================================================

    print(f"{'测试文本':<60} | {'预测结果':<10}")
    print("-" * 75)

    for text in test_cases:
        # 预处理
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 预测
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # outputs 是 logits，取最大值的索引
            prediction = torch.argmax(outputs, dim=1).item()
            
            # 获取置信度 (softmax)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0][prediction].item()

        # 结果映射 (假设 0:负面, 1:正面)
        label_str = "正面 (Positive)" if prediction == 1 else "负面 (Negative)"
        
        # 打印结果 (截断过长文本以便显示)
        display_text = (text[:55] + '...') if len(text) > 55 else text
        print(f"{display_text:<60} | {label_str} (置信度: {confidence:.2f})")

if __name__ == "__main__":
    load_model_and_predict()
