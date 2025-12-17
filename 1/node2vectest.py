import networkx as nx
import pandas as pd
from node2vec import Node2Vec

# 读取文件
df = pd.read_csv('postings.csv')

# 初始化图
G = nx.Graph()

# 将职位和公司作为节点，建立关系
for _, row in df.iterrows():
    job_id = str(row['job_id'])  # 使用职位ID作为节点
    company_name = str(row['company_name']) if pd.notna(row['company_name']) else 'Unknown'  # 公司名称
    
    # 添加节点和边
    G.add_node(job_id, type='job')
    G.add_node(company_name, type='company')
    G.add_edge(job_id, company_name)

# 使用 node2vec 获取节点表示，减少并行化工作进程数
node2vec = Node2Vec(G, dimensions=32, walk_length=10, num_walks=20, workers=1)  # 降低工作进程数
model = node2vec.fit()

# 保存模型到文件
model.wv.save_word2vec_format('node2vec_model.txt')  # 使用 `save_word2vec_format` 保存模型

from gensim.models import KeyedVectors

# 加载保存的模型
model = KeyedVectors.load_word2vec_format('node2vec_model.txt', binary=True)

# 获取某个节点的嵌入表示
job_id_example = '921716'  # 例子中的职位ID
job_embedding = model[job_id_example]
print(f"Embedding for job ID {job_id_example}:")
print(job_embedding)
