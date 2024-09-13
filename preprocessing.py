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

    edge_list = []
    for node, neighbors in enumerate(adj_list): 
        for neighbor in neighbors:
            edge_list.append([node, neighbor]) 
    return np.array(edge_list).T

def convert_adjacencylist2adjacencymatrix(adj_list):

    num_vertices = len(adj_list)   
    adj_matrix = np.zeros(shape=(num_vertices, num_vertices))

    for i in range(num_vertices):
        for j in adj_list[i]:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    return adj_matrix

def select_LRgenes(data_df, num_genespercell, lr_database = 0):
    if lr_database == 0:
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        lr_df = pd.read_csv("./data/celltalk_mouse_lr_pair.txt", sep="\t")
        receptors = set(lr_df["receptor_gene_symbol"].str.upper().to_list())    
        ligands = set(lr_df["ligand_gene_symbol"].str.upper().to_list())    

        real2uppercase = {x:x.upper() for x in sample_counts.columns}   
        uppercase2real = {upper:real for real, upper in real2uppercase.items()}  
        candidate_genes = set(np.vectorize(real2uppercase.get)(sample_counts.columns.to_numpy()))

        selected_ligands = candidate_genes.intersection(ligands)    
        selected_receptors = candidate_genes.intersection(receptors)    
        selected_lrs = selected_ligands | selected_receptors
        print(len(selected_lrs))
       
        print(f"Using {len(selected_ligands)} ligands and {len(selected_receptors)} receptors to be included in the {num_genespercell} selected genes per cell. \n")
        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        candidate_genesleft = candidate_genes - selected_ligands - selected_receptors

        selected_randomgenes = set(random.sample(tuple(candidate_genesleft), num_genesleft))
        selected_genes = list(selected_randomgenes | selected_ligands | selected_receptors)
        selected_columns = ["Cell_ID","X","Y","Cell_Type"] + np.vectorize(uppercase2real.get)(selected_genes).tolist()  
        selected_df = data_df[selected_columns] 
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



def construct_celllevel_graph(data_df, k, get_edges=False):
    adjacency = np.zeros(shape=(len(data_df),k),dtype=int)  
    coords = np.vstack([data_df["X"].values, data_df["Y"].values]).T    

    edges=None
    edge_x=[]
    edge_y=[]

    for i in track(range(len(data_df)),description=f"[cyan]2.Constructing Cell-level Graph from ST Data"):  
        cell_id = data_df["Cell_ID"][i]
        x0,y0=data_df["X"].values[i],data_df["Y"].values[i] 
        candidate_cell = coords[i]  
        candidate_neighbors = coords    
        euclidean_distances = np.linalg.norm(candidate_neighbors-candidate_cell, axis=1)   
        neighbors = np.argsort(euclidean_distances)[1:k+1]  
        adjacency[i] = neighbors    
        assert i not in adjacency[i]
        if get_edges:   
            for ncell in adjacency[i]:
                x1,y1 = data_df["X"].values[ncell],data_df["Y"].values[ncell]   
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
    edges=[edge_x,edge_y]
    return adjacency,edges




















