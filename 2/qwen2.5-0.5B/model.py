import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        # 注意这一行：必须是 super(SentimentClassifier, self) 或者直接 super()
        super(SentimentClassifier, self).__init__()
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            trust_remote_code=True,
            attn_implementation="eager",  # <--- 加上这一行，强制使用标准模式
            torch_dtype=torch.bfloat16,  # ★ 新增
                  )
        
        # 补全 pad_token 配置，防止 batch_size > 1 报错
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, input_ids, attention_mask):
        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        # 动态检测当前设备（如果有GPU用GPU，没GPU用CPU）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 【关键修改】添加 map_location 参数
        # 这会强制将保存的权重映射到当前机器可用的设备上
        self.load_state_dict(torch.load(path, map_location=device))
