#coding = 'utf-8'
# 运行程序绘制网络语义分析图
from relationship_process.relationship import Relationship


dictpath = r'new_topic_word.txt'
datapath = r'merge.txt'
pic = r'topic_word_net.png'
Re = Relationship(dictpath, datapath)
relation = Re.relationship()
graph = Re.network_digraph(relation, pic)

