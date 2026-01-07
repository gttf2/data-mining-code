import os
import json
import re

# ================= 配置区域 =================
SOURCE_DIRECTORY = './data/'          # 你的 .md 文件目录
OUTPUT_JSON_PATH = './data/rag_corpus_final.json'
CHUNK_SIZE = 512                      # 每个块的字符上限
CHUNK_OVERLAP = 50                    # 重叠字符数
# ===========================================

def clean_markdown_content(text):
    """
    V3 强力清洗函数：截断广告、移除干扰符
    """
    if not text:
        return ""

    # 1. 核武器级去噪：从“公益问诊”处截断，丢弃后面所有内容（包括菜单、点赞等）
    if "公益问诊" in text:
        text = text.split("公益问诊")[0]
    
    # 2. 移除图片标签 ![...](...) 
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    
    # 3. 移除链接 [text](url) -> 只保留 text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 4. 移除头部引导语
    text = re.sub(r'点击上方.*?关注更多精彩', '', text)
    text = re.sub(r'点击蓝字\s*关注我们', '', text)
    text = re.sub(r'血液病专家\s+血液病专家;\)', '', text) 

    # 5. 移除残留的干扰词 (兜底)
    noise_patterns = [
        r'预览时标签不可点',
        r'轻触阅读原文',
        r'微信扫一扫',
        r'关注该公众号',
        r'javascript:void\(0\);',
        r'javascript:;',
        r'阅读原文',
        r'收录于合集',
        r'视频\s+小程序\s+赞',
        r'取消;\)\s+允许;\)' 
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text)

    # 6. 移除奇怪的标点堆积行 (如: ， ， ，)
    text = re.sub(r'^[，。：；,\.\s]+$', '', text, flags=re.MULTILINE)

    # 7. 合并多余换行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_from_markdown(filepath):
    """
    读取文件，提取日期和标题，并清洗
    """
    try:
        filename = os.path.basename(filepath)
        
        # 从文件名提取日期 "20251204_标题.md"
        date_match = re.match(r'(\d{8})_(.*)\.md', filename)
        
        if date_match:
            publish_date = date_match.group(1)
            title = date_match.group(2)
        else:
            publish_date = "Unknown"
            title = filename.replace('.md', '').replace('.html', '')

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # 调用清洗
        cleaned_text = clean_markdown_content(raw_content)
        
        # 如果清洗完只剩下极短的内容，视为无效
        if len(cleaned_text) < 20: 
            return None

        return {
            "title": title,
            "publish_date": publish_date,
            "content": cleaned_text
        }
    except Exception as e:
        print(f"读取错误 {filepath}: {e}")
        return None

def smart_split_text(text, chunk_size=512, chunk_overlap=50):
    """
    按段落优先切分
    """
    if not text:
        return []
    
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para: continue
            
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += "\n\n" + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            if len(para) > chunk_size:
                # 长段落强切
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunks.append(para[i:i + chunk_size])
                current_chunk = ""
            else:
                current_chunk = para
                
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

# ================= 主执行逻辑 =================
if __name__ == "__main__":
    all_data = []
    print(f"🚀 开始处理...")
    
    # 兼容 .md 和 .html
    files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith('.md') or f.endswith('.html')]
    
    for filename in files:
        filepath = os.path.join(SOURCE_DIRECTORY, filename)
        
        # 1. 提取与清洗
        data = extract_from_markdown(filepath)
        
        if data and data['content']:
            # 2. 切分
            chunks = smart_split_text(data['content'], CHUNK_SIZE, CHUNK_OVERLAP)
            
            # 3. 构建数据块 (加入主循环过滤)
            for i, chunk in enumerate(chunks):
                
                # --- [新增] 主循环垃圾过滤 ---
                # 过滤掉太短的块（少于10个字通常没有检索价值）
                if len(chunk.strip()) < 10:
                    continue
                # 过滤掉依然残留的导航栏特征
                if "小程序" in chunk and "视频" in chunk:
                    continue
                # ---------------------------

                entry = {
                    "id": f"{filename}_{i}",
                    "title": data['title'],
                    "publish_date": data['publish_date'],
                    "abstract": chunk,
                    "source_file": filename,
                    "chunk_index": i,
                    "metadata": {
                        "source": "wechat_official",
                        "type": "medical_article"
                    }
                }
                all_data.append(entry)

    # 4. 保存
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 处理完成！")
    print(f"📊 共生成 {len(all_data)} 个高质量数据块")
    print(f"📁 结果已保存至: {OUTPUT_JSON_PATH}")
