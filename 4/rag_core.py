import os
import json
from bs4 import BeautifulSoup
import re

def extract_text_and_title_from_html(html_filepath):
    """
    从指定的 HTML 文件中提取标题和正文文本。

    Args:
        html_filepath (str): HTML 文件的路径。

    Returns:
        tuple: (标题, 正文文本) 或 (None, None) 如果失败。
    """
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml') # 或者使用 'html.parser'

        # --- 提取标题 ---
        title_tag = soup.find('title')
        title_string = title_tag.string if title_tag else None
        # 确保 title_string 不为 None 才调用 strip()
        title = title_string.strip() if title_string else os.path.basename(html_filepath)
        title = title.replace('.html', '') # 清理标题

        # --- 定位正文内容 ---
        # 根据之前的讨论，优先查找 <content> 或特定 class
        content_tag = soup.find('content')
        if not content_tag:
            content_tag = soup.find('div', class_='rich_media_content') # 微信文章常见
        if not content_tag:
            content_tag = soup.find('article') # HTML5 语义标签
        if not content_tag:
            content_tag = soup.find('main') # HTML5 语义标签
        if not content_tag:
             content_tag = soup.find('body') # 最后尝试 body

        if content_tag:
