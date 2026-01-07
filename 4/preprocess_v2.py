import os
import json
import re

def clean_markdown_content(text):
    """
    深度清洗微信公众号转换来的 Markdown 文本
    """
    # 1. 移除图片标签 ![...](...)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 2. 移除链接 [text](url) 但保留文本 (视情况而定，这里建议保留文本)
    # 如果想连文本一起删，用 r'\[.*?\]\(.*?\)'
    # 这里我们只移除链接结构，保留链接文字，例如 [专家] -> 专家
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 3. 移除微信特有的干扰文本 (根据你提供的样本定制)
    noise_patterns = [
        r'预览时标签不可点',
        r'轻触阅读原文',
        r'微信扫一扫',
        r'关注该公众号',
        r'向上滑动看下一个',
        r'轻点两下取消',
        r'在小说阅读器中沉浸阅读',
        r'javascript:void\(0\);',
        r'javascript:;',
        r'阅读原文',
        r'收录于合集',
        r'在线填写',
        r'长按下方图片识别',
        r'无需挂号专家为你解答'
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)

    # 4. 移除多余的空行和特殊字符
    text = text.replace('\xa0', ' ') # 移除不间断空格
    text = re.sub(r'\n\s*\n', '\n\n', text) # 将多重换行合并为双换行（保留段落感）
    
    return text.strip()

def extract_from_markdown(filepath):
    """
    读取 Markdown 文件并提取标题和正文
    """
    try:
        filename = os.path.basename(filepath)
        
        # --- 1. 从文件名提取日期和标题 ---
        # 假设文件名格式: "20251204_文章标题.md"
        date_match = re.match(r'(\d{8})_(.*)\.md', filename)
        if date_match:
            publish_date = date_match.group(1)
            title = date_match.group(2)
        else:
            publish_date = "Unknown"
            title = filename.replace('.md', '')

        # --- 2. 读取内容 ---
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 如果文件里有明确的一级标题 # Title，可以尝试提取覆盖文件名标题
        # header_match = re.search(r'^#\s+(.*)', content, re.MULTILINE)
        # if header_match:
        #     title = header_match.group(1)

        # 清洗文本
        cleaned_text = clean_markdown_content(content)
        
        return {
            "title": title,
            "publish_date": publish_date,
            "content": cleaned_text
        }

    except Exception as e:
        print(f"处理文件 {filepath} 时出错: {e}")
        return None

def smart_split_text(text, chunk_size=512, chunk_overlap=50):
    """
    改进版切分：优先按段落(\n\n)切分，保持语义完整性
    """
    if not text:
        return []
    
    # 简单的按段落分割，如果段落过长再按字符强切
    # 在生产环境中，建议使用 langchain 的 RecursiveCharacterTextSplitter
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果当前块加上新段落还没超限，就合并
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para
        else:
            # 如果当前块已经有内容，先保存
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # 如果单个段落本身就超长，强制切分
            if len(para) > chunk_size:
                # 这里简单处理，长段落直接强切
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i:i + chunk_size])
                current_chunk = "" # 清空
            else:
                current_chunk = para # 新起一块
                
    # 保存最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# --- 配置 ---
# **** 即使你的文件夹里是 .html，但如果内容是 markdown，请确保后缀匹配 ****
source_directory = './data/' 
output_json_path = './data/processed_data.json'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- 主逻辑 ---
all_data = []

print(f"开始处理...")
# 支持 .md 和 .html (假设 html 里其实也是 md 文本)
files = [f for f in os.listdir(source_directory) if f.endswith('.md') or f.endswith('.html')]

for filename in files:
    filepath = os.path.join(source_directory, filename)
    print(f"Processing: {filename}")
    
    # 提取
    data = extract_from_markdown(filepath)
    
    if data and data['content']:
        # 切分
        chunks = smart_split_text(data['content'], CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            entry = {
                "id": f"{filename}_{i}",
                "title": data['title'],
                "publish_date": data['publish_date'], # 新增：日期字段
                "abstract": chunk,
                "source_file": filename,
                "chunk_index": i,
                "metadata": {
                    "source": "wechat_official",
                    "type": "medical_diet"
                }
            }
            all_data.append(entry)

# 保存
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print(f"完成！已生成 {len(all_data)} 条 RAG 数据，保存至 {output_json_path}")
