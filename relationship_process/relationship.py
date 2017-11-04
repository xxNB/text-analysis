
import re
import os
import jieba
import random
import networkx as nx
import matplotlib.pyplot as plt
from pylab import mpl

#解决显示中文问题
mpl.rcParams['font.sans-serif'] = ['SimHei']   # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

class Relationship(object):
    """
    Relationship(dictpath, datapath)
            dictionary   词典路径
            data         小说路径

    """
    
    def __init__(self, dictionary, data):
        self.dictionary = dictionary
        self.data = data
        self.currentpath = os.getcwd()
    
    def relationship(self):
        #人名、名词、地名等都可以作为关系网络图中的节点
        with open(self.dictionary, 'r', encoding='utf-8') as node_file:
            node_list = [node.replace('\n', '') for node in node_file.readlines()]
        node_freq    = {}          #节点词频，格式{node：frequency}
        relationships= {}          #关系字典，格式{source_node:{target_node:weight}}
        lineNodes = []             #txt文件中每段内的节点关系
        jieba.load_userdict(self.dictionary)  # 载入节点词典，防止分词时将节点词被再次切分
        data = open(self.data, 'r',encoding='utf-8')  # 载入小说
        for line in data.readlines():
            pseg = jieba.lcut(line)  # 分词，返回词的列表
            lineNodes.append([])  # 为新读入的一段添加节点列表
            for w in pseg:
                if w in node_list:
                    lineNodes[-1].append(w)
                    if node_freq.get(w) is None:
                        node_freq[w] = 0
                        relationships[w] = {}
                    node_freq[w] += 1
        
        for line in lineNodes:  # 对于每一段
            for node1 in line:
                for node2 in line:  # 每段中的任意两个节点
                    if node1 == node2:
                        continue
                    if relationships[node1].get(node2) is None:  # 若两个节点未同时出现，则新建项
                        relationships[node1][node2] = 1
                    else:
                        relationships[node1][node2] = relationships[node1][node2] + 1  # 两节点共同出现次数加1
        if '\\' in self.currentpath:
            node_freq_path = self.currentpath + '\\node_freq.txt'
            node_edge_path = self.currentpath + '\\node_edge.txt'
        else:
            node_freq_path = self.currentpath + '/node_freq.txt'
            node_edge_path = self.currentpath + '/node_edge.txt'

        with open(node_freq_path, 'a+', encoding='utf-8') as node_freq_file:
            node_freq_file.write("Id Label Weight\r\n")
            for node, times in node_freq.items():
                node_freq_file.write(node + ' ' + node + ' ' + str(times) + '\r\n')


        with open(node_edge_path, 'a+', encoding='utf-8') as node_edge_file:
            node_edge_file.write("Source Target Weight\r\n")
            for node, edges in relationships.items():
                for v, w in edges.items():
                    if w > 3:
                        node_edge_file.write(node + " " + v + " " + str(w) + "\r\n")
        return [node_list, relationships]

    # node_list节点列表
    # relationship函数的运行结果
    # pic为输出图片的路径
    def network_digraph(self, relationship, pic):
        node_list = relationship[0]
        relation = relationship[1]
        DG = nx.DiGraph()  # 有向图
        DG.add_nodes_from(node_list)  # 添加节点
        node_num = len(node_list)
        for node1 in node_list:
            for node2 in node_list:
                if node1 == node2:
                    continue
                else:
                    try:
                        weight = relation[node1][node2]
                        for i in range(weight):
                            DG.add_edge(node1, node2)
                    except:
                        continue

        if '\\' in self.currentpath:
            colorpath = self.currentpath + '\\colors.txt'
        else:
            colorpath = self.currentpath + '/colors.txt'
        f = open(colorpath, 'r', encoding='utf-8').readlines()
        Colors = [c.replace('\n', '') for c in f]
        colors = random.sample(Colors, node_num)
        nx.draw(DG, with_labels=True, node_size=550,node_color=colors, )

        plt.savefig(pic)
        plt.show()







