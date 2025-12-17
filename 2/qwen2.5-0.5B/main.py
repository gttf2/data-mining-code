import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from config import Config
from dataset import SentimentDataset
from load_data import DataLoader as DataLoaderClass
from model import SentimentClassifier
# from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_hf_mirrors():
    """
    设置Hugging Face镜像，加速模型下载
    """
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 可选的其他镜像
    # os.environ['HF_ENDPOINT'] = 'https://huggingface.tuna.tsinghua.edu.cn'
    # os.environ['HF_ENDPOINT'] = 'https://mirror.sjtu.edu.cn/hugging-face'
    
    # 设置模型缓存目录（可选）
    os.environ['HF_HOME'] = './hf_cache'
    
# 设置镜像
set_hf_mirrors()

def evaluate(model, eval_loader, device):
    """
    评估模型性能
    
    参数:
        model: 模型对象
        eval_loader: 评估数据加载器
        device: 计算设备（CPU/GPU）
        
    返回:
        Tuple[float, float]: 平均损失和准确率
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.float(), labels)
            
            _, predictions = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(predictions == labels)
            total_predictions += len(labels)
            total_loss += loss.item()
            
    return total_loss / len(eval_loader), correct_predictions.double() / total_predictions

def train(train_texts, train_labels, val_texts=None, val_labels=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("正在加载模型 (FP16)...")
    model = SentimentClassifier(config.model_name, config.num_classes)
    model.to(device)
    
    print("正在处理数据...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, config.max_seq_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=2,      # 允许 2 个帮厨同时切菜
        pin_memory=True,    # 加速内存到显存的传输
        persistent_workers=True # 保持帮厨随时待命，不要每轮都解散
        )
    
    if val_texts is not None and val_labels is not None:
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, config.max_seq_length)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size,
            num_workers=2, 
            pin_memory=True,
            persistent_workers=True
        )
    
    accumulation_steps = 2
    total_steps = len(train_loader) * config.num_epochs // accumulation_steps
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    criterion = nn.CrossEntropyLoss()
    
        
    best_accuracy = 0
    print(f"开始训练，梯度累积: {accumulation_steps}")

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        print(f"Epoch {epoch + 1}/{config.num_epochs} 开始...")
        
        # ★ 这一段一定要缩进在 epoch 里面
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向：outputs 可能是半精度，我们在算 loss 时转成 float32
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.float(), labels)
            loss = loss / accumulation_steps

            # 直接常规反向传播
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                # 可选：做一次梯度裁剪（FP32 下是安全的）
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            if step % 50 == 0:
                print(f"  Step {step + 1}/{len(train_loader)}", end="\r")

        # ★ 注意：平均 loss 和验证，应该在「一个 epoch 结束后」做一次
        avg_train_loss = total_loss / len(train_loader)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        if val_texts is not None:
            torch.cuda.empty_cache()
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
                print(f"保存最佳模型 ({best_accuracy:.4f}) 到: {config.model_save_path}")
                model.save_model(config.model_save_path)
    
    return model

def predict(text, model_path=None):
    """
    使用训练好的模型进行预测
    
    参数:
        text (str): 待预测的文本
        model_path (str, optional): 模型路径
        
    返回:
        int: 预测的标签（0或1）
    """
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化并加载模型
    model = SentimentClassifier(config.model_name, config.num_classes)
    if model_path:
        model.load_model(model_path)
    model.to(device)
    model.eval()
    
    # 预处理文本
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predictions = torch.max(outputs, dim=1)
    
    return predictions.item()

if __name__ == "__main__":
    # 设置Hugging Face镜像
    set_hf_mirrors()
    
    # 加载配置
    config = Config()
    
    # 加载数据
    data_loader = DataLoaderClass(config)
    
    # 分别加载训练集、验证集和测试集
    print("加载训练集...")
    train_texts, train_labels = data_loader.load_csv("dataset/train.csv")
    print("加载验证集...")
    val_texts, val_labels = data_loader.load_csv("dataset/dev.csv")
    print("加载测试集...")
    test_texts, test_labels = data_loader.load_csv("dataset/test.csv")
    
    # 打印数据集大小
    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")
    
    # 训练模型
    print("开始训练模型...")
    model = train(train_texts, train_labels, val_texts, val_labels)
    
    # 预测示例
    example_text = "这个产品质量非常好，我很满意！"
    prediction = predict(example_text, config.model_save_path)
    sentiment = "正面" if prediction == 1 else "负面"
    print(f"示例文本: '{example_text}'")
    print(f"情感预测: {sentiment} (类别 {prediction})")
