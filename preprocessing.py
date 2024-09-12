import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
import time
from rich.progress import track
from rich.console import Console
from rich.table import Table
import random
import sys
from node2vec import Node2Vec


sys.path.insert(0, os.path.abspath("./submodules/"))
sys.path.append('./submodules/CeSpGRN/src/')
sys.path.append('../')

def convert_adjacencylist2edgelist(adj_list):

    """
    Converts an adjacency list to an edge list.
    邻接表to边列表
    """
    edge_list = []
    for node, neighbors in enumerate(adj_list): #enumerate返回索引以及其对应的值 node当前索引，neighbors邻居列表，即对应的值
        for neighbor in neighbors:
            edge_list.append([node, neighbor])  #按行append，记录了5近邻的所有边
    return np.array(edge_list).T

def convert_adjacencylist2adjacencymatrix(adj_list):

    """
    Converts an adjacency list to an adjacency matrix.
    邻接表to邻接矩阵，5-近邻的细胞被标记为1，邻接矩阵
    """
    num_vertices = len(adj_list)    #1597
    adj_matrix = np.zeros(shape=(num_vertices, num_vertices))

    for i in range(num_vertices):
        for j in adj_list[i]:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    return adj_matrix

def select_LRgenes(data_df, num_genespercell, lr_database = 0):
    """
    Selects LR genes from the data_df.
    Selects LR and relevant background genes to be included for GRN inference.
    :data_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y", "Cell_Type", "Gene 1", ..., "Gene n"]
    :num_genespercell: int : represents the number of genes to be included for each Cell-Specific GRN
    :lr_database: int: 0/1/2 corresponding to LR database (0: CellTalkDB Mouse,1: CellTalkDB Human, 2: scMultiSim Simulated)
    :return : pd.DataFrameq with relevant gene columns preserved and dictionary mapping LR genes to their numerical ids
    """
    #lr_database代表数据集，0-CellTalkDB Mouse
    if lr_database == 0:
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        lr_df = pd.read_csv("./data/celltalk_mouse_lr_pair.txt", sep="\t")
        receptors = set(lr_df["receptor_gene_symbol"].str.upper().to_list())    #receptor转大写
        ligands = set(lr_df["ligand_gene_symbol"].str.upper().to_list())    #ligand转大写

        real2uppercase = {x:x.upper() for x in sample_counts.columns}   #real2uppercase是一个字典，键是列名，值是对应的大写，x在sample_counts上遍历所有的列名，转大写传给x，
        uppercase2real = {upper:real for real, upper in real2uppercase.items()}  #uppercase2real，real遍历real2uppercase所有的键值对，real是键，upper是值
        candidate_genes = set(np.vectorize(real2uppercase.get)(sample_counts.columns.to_numpy()))

        selected_ligands = candidate_genes.intersection(ligands)    #candidate_genes与ligands交集
        selected_receptors = candidate_genes.intersection(receptors)    #candidate_genes与receptors交集
        selected_lrs = selected_ligands | selected_receptors
        print(len(selected_lrs))
       
        print(f"Using {len(selected_ligands)} ligands and {len(selected_receptors)} receptors to be included in the {num_genespercell} selected genes per cell. \n")
        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        candidate_genesleft = candidate_genes - selected_ligands - selected_receptors

        selected_randomgenes = set(random.sample(tuple(candidate_genesleft), num_genesleft))
        selected_genes = list(selected_randomgenes | selected_ligands | selected_receptors)
        selected_columns = ["Cell_ID","X","Y","Cell_Type"] + np.vectorize(uppercase2real.get)(selected_genes).tolist()  #对selected_genes整个数据都执行了uppercase2real，并且将numPY数组转换为python列表
        selected_df = data_df[selected_columns] #data_df数据框中的selected_columns列
        print(f"selected_df.shape:{selected_df.shape}")

        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Selected Ligands", style="cyan")
        table.add_column("Selected Receptors", style="deep_pink3")
        table.add_column("Selected Random Genes", justify="right")
        table.add_row("\n".join(selected_ligands),"\n".join(selected_receptors),"\n".join(selected_randomgenes))
        console.print(table)

        lr2id = {gene:list(selected_df.columns).index(gene)-4 for gene in np.vectorize(uppercase2real.get)(list(selected_ligands|selected_receptors))}
        return selected_df, lr2id

    elif lr_database==2:
        sample_counts = data_df.drop(["Cell_ID","X","Y","Cell_Type"],axis=1)
        candidate_genes = sample_counts.columns.to_numpy()
        scmultisim_lrs = pd.read_csv("../data/scMultiSim/simulated/cci_gt.csv")[["ligand","receptor"]]
        scmultisim_lrs["ligand"] = scmultisim_lrs["ligand"]
        scmultisim_lrs["receptor"] = scmultisim_lrs["receptor"]
        selected_ligands = np.unique(scmultisim_lrs["ligand"])
        selected_receptors = np.unique(scmultisim_lrs["receptor"])
        selected_lrs = np.concatenate((selected_ligands, selected_receptors), axis=0)

        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        indices = np.argwhere(candidate_genes == selected_lrs)##返回索引
        candidate_genesleft = np.delete(candidate_genes, indices)
        selected_randomgenes = random.sample(set(candidate_genesleft),num_genesleft)

        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Selected Ligands", style="cyan")
        table.add_column("selected Receptors", style="deep_pink3")
        table.add_column("Selected Random Genes", justify="right")
        table.add_row("\n".join([str(x) for x in selected_ligands]),"\n".join([str(x) for x in selected_receptors]),"\n".join([str(x) for x in selected_randomgenes]))
        console.print(table)

        selected_genes = np.concatenate((selected_lrs,selected_randomgenes), axis=0)

        new_columns = ["Cell_ID","X","Y","Cell_type"] + list(selected_genes)
        selected_df = data_df[new_columns]
        lr2id = {gene:list(selected_df.columns).index(gene)-4 for gene in selected_genes}
        return selected_df, lr2id

    else:
        raise Exception("Invalid lr_database type")


# def infer_initial_grns(data_df, cespgrn_hyperparams):   #1597*125+4
#     console = Console()
#     from submodules.CeSpGRN.src import kernel
#     from submodules.CeSpGRN.src import g_admm as CeSpGRN

#     with console.status("[cyan] preparing CeSpGRN ...") as status:
#         status.update(spinner="aesthetic", spinner_style="cyan")
#         counts = data_df.drop(["Cell_ID","X","Y","Cell_Type"],axis=1).values
#         print(f"GRNs are dimension ({counts.shape[1]} by {counts.shape[1]}) for each of the {counts.shape[0]} cells\n")
      
#         pca_op = PCA(n_components = 20)
#         X_pca = pca_op.fit_transform(counts)

#         #hyper-parameters
#         bandwidth = cespgrn_hyperparams["bandwidth"]    #高斯核带宽
#         n_neigh = cespgrn_hyperparams["n_neigh"]
#         lamb = cespgrn_hyperparams["lamb"]
#         max_iters = cespgrn_hyperparams["max_iters"]

#         status.update(status="[cyan] Calculating kernel function ...")
#         K,K_trun = kernel.calc_kernel_neigh(X_pca, k=5, bandwidth=bandwidth, truncate=True, truncate_param=n_neigh) 
        
#         status.update(status="[cyan] Estimating covariance matrix ...")

#         empir_cov = CeSpGRN.est_cov(X=counts,K_trun=K_trun,weighted_kt=True)
#         status.update(status="[cyan] Load in CeSpGRN model ...")
     
#         cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize=120) #求解凸优化问题的迭代算法，常用于解决带约束的优化问题。
#         status.update(status="[cyan] Ready to train ...")
#         time.sleep(3)

#     grns=cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)#基因调控网络
#     return grns

def construct_celllevel_graph(data_df, k, get_edges=False):
    adjacency = np.zeros(shape=(len(data_df),k),dtype=int)  #len(data_df)返回data_df的行数 1957*5的零矩阵
    coords = np.vstack([data_df["X"].values, data_df["Y"].values]).T    #np.vstack按行堆叠，转置后是2列

    edges=None
    edge_x=[]
    edge_y=[]

    for i in track(range(len(data_df)),description=f"[cyan]2.Constructing Cell-level Graph from ST Data"):  #0-1596整数数列
        cell_id = data_df["Cell_ID"][i]
        x0,y0=data_df["X"].values[i],data_df["Y"].values[i] #X，Y坐标
        candidate_cell = coords[i]  #i的X，Y
        candidate_neighbors = coords    #总的X，Y
        euclidean_distances = np.linalg.norm(candidate_neighbors-candidate_cell, axis=1)    #欧几里得距离(两个向量之间的差的范数)
        neighbors = np.argsort(euclidean_distances)[1:k+1]  #对输入的数据进行排序，得到从小到大排列的索引数组，取[1:k+1]，默认k=5
        adjacency[i] = neighbors    #adjacency[i] 存储的是欧几里得距离5-近邻的索引，即细胞索引
        assert i not in adjacency[i]
        if get_edges:   #判断get_edges不是 0、空、False 等假值
            for ncell in adjacency[i]:
                x1,y1 = data_df["X"].values[ncell],data_df["Y"].values[ncell]   #返回索引为ncell的X，Y
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
    edges=[edge_x,edge_y]
    return adjacency,edges

# def construct_genelevel_graph(disjoint_grns, celllevel_adj_list,node_type="int",lrgenes=None):
#     numgenes = disjoint_grns[0].shape[0]    #第一个元素的基因矩阵？45
#     numcells = disjoint_grns.shape[0]   #细胞数1597
#     num2gene = {}
#     gene2num = {}

#     assert max(lrgenes) <=numgenes

#     grn_graph_list = []
#     for cellnum,grn in enumerate(track(disjoint_grns, description=f"[cyan]3a.Combining individual GRNs")):
#         G = nx.from_numpy_matrix(grn)   #将一个 NumPy 矩阵表示的邻接矩阵转换为 NetworkX 图
#         grn_graph_list.append(G)
#         for i in range(numgenes):   #45，num2gene-1597*45+1-45
#             num2gene[cellnum*numgenes+i] = f"Cell{cellnum}_Gene{i}"
#             gene2num[f"Cell{cellnum}_Gene{i}"] = cellnum*numgenes+i
#     union_of_grns = nx.disjoint_union_all(grn_graph_list)   #基因调控网络组合成一个联合网络
#     gene_level_graph = nx.relabel_nodes(union_of_grns,num2gene) #重新标记联合网络中的节点，使用“num2gene”中的映射


#     for cell, neighborhood in enumerate(track(celllevel_adj_list, description=f"[cyan]3b.Constructing Gene-Level Graph")):
#         for neighbor_cell in neighborhood:
#             if neighbor_cell != -1:
#                 for lrgene1 in lrgenes:
#                     for lrgene2 in lrgenes:
#                         node1=f"Cell{cell}_Gene{lrgene1}"
#                         node2=f"Cell{neighbor_cell}_Gene{lrgene2}"
#                         if not gene_level_graph.has_node(node1) or not gene_level_graph.has_node(node2):
#                             raise Exception(f"Nodes {node1} or {node2} not found. Debug the Gene-Level Graph creation.")
#                         gene_level_graph.add_edge(node1,node2)
#         if node_type == "str":
#             gene_level_graph = gene_level_graph
#         elif node_type == "int":
#             gene_level_graph = nx.convert_node_labels_to_integers(gene_level_graph) #将图中的节点标签转换为整数，创建新的图
#         assert len(gene_level_graph.nodes()) == numcells * numgenes
#         return gene_level_graph,num2gene,gene2num,union_of_grns

# def get_gene_features(graph, type="node2vec"):  #学习图中节点嵌入的方法，通过随机游走计算节点的语义关系
#     console=Console()
#     console.print(f"[cyan]4.Constructing {type} Gene Level Features")
#     if type=="node2vec":
#         """
#         node2vec是一种学习节点表示的图嵌入算法，旨在将图中的节点映射到低维的向量空间中，以便于后续的机器学习任务
#         随机游走：当前节点出发按照一定的策略选择下一个节点，使用两个参数平衡广度优先搜索（BFS）和深度优先搜索（DFS）
#         节点序列生成：根据多次随机游走，得到大量的节点序列，节点序列作为输入数据训练模型
#         skip-gram：根据得到的节点训练skip-gram模型，是一种常用的词嵌入模型，节点看作是词语，学习词语的分布式表示，训练skip-gram学习节点的向量表示
#         """
#         node2vec=Node2Vec(graph,dimensions=64,walk_length=15,num_walks=100,workers=4)
#         model=node2vec.fit()
#         gene_feature_vectors=model.wv.vectors
#     return gene_feature_vectors, model






















