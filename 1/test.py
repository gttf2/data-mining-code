import networkx as nx
from node2vec import Node2Vec

# 创建一个简单的图
G = nx.Graph()
print("图成功")

# 添加职位和公司节点
G.add_node("Job1", type='job')
G.add_node("Job2", type='job')
G.add_node("CompanyA", type='company')
G.add_node("CompanyB", type='company')

# 添加边，表示职位与公司之间的关系
G.add_edge("Job1", "CompanyA")
G.add_edge("Job2", "CompanyA")
G.add_edge("Job2", "CompanyB")

# 使用 node2vec 获取节点表示
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=100, workers=1)  # 设置参数
model = node2vec.fit()

# 获取某个节点（如职位）的表示
job_embedding = model.wv["Job1"]
company_embedding = model.wv["CompanyA"]

print("Job embedding:", job_embedding)
print("Company embedding:", company_embedding)
