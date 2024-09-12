
import os
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from rich.table import Table
from rich.console import Console
from rich.text import Text

import torch
import time
import preprocessing as preprocessing
import training as training
import models as models
from torch_geometric.nn import GAE

debug = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("using {} device.".format(device))


def parse_arguments():
    parser = argparse.ArgumentParser(description='clarifyGAE arguments')
    parser.add_argument("-m", "--mode", type=str, default="preprocess,train,test", help="clarifyGAE mode:preprocess,train,test")
    parser.add_argument("-i", "--inputdirpath", type=str, default="./data/seqFISH/seqfish_dataframe.csv",
                        help="Input directory path where ST data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str, default="./out/seqfish/ceshi/",
                        help="Output directory path where result will be stored")
    parser.add_argument("-s", "--studyname", type=str, default="seqFISH", help="clarifyGAE study name")
    parser.add_argument("-t", "--split", type=float, default=0.3, help="of test edge [0,1)")
    parser.add_argument("-n", "--numgenespercell", type=int, default=120,
                        help="Number of genes in each gene regulatory network")
    parser.add_argument("-k", "--nearestneighbors", type=int, default=5,
                        help="Number of nearest neighbors for each cell")
    parser.add_argument("-l", "--lrdatabase", type=int, default=0, help="0CellTalkDB Mouse/1Human/2scMultiSim")
    parser.add_argument("-fp", type=float, default=0, help="false positive test edges[0,1),experimentation")
    parser.add_argument("-fn", type=float, default=0, help="false negative test edges[0,1),experimentation")
    parser.add_argument("-a", "--ownadjacencypath", type=str, help="Using your own cell level adjacency(give path)")
    args = parser.parse_args()
    return args


def preprocess(st_data, num_nearestneighbors, ownadjacencypath=None):
    
    # construct Cell-level Graph from ST data
    if ownadjacencypath is not None:
        celllevel_adj = np.load(ownadjacencypath)
    else:
        celllevel_adj, _ = preprocessing.construct_celllevel_graph(st_data, num_nearestneighbors, get_edges=False)

    return celllevel_adj


def build_clarifyGAE_pytorch(data, hyperparams=None):
    _, num_cellfeatures = data[0].x.shape[0], data[0].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] // 2
    cellEncoder = models.GraphTrans_Encoder(num_cellfeatures, hidden_dim)#创建实例对象
    CellTEncoder = models.CellTEncoder(GraphTrans_Encoder=cellEncoder)
    gae = GAE(CellTEncoder)

    return gae


def main():
    console = Console()
    with console.status("Clarify booting up...") as status:
        status.update(spinner="aesthetic", spinner_style="cyan")
        time.sleep(4)
        status.update(status="[cyan] Parsing arguments...")
        args = parse_arguments()

        mode = args.mode
        input_dir_path = args.inputdirpath
        output_dir_path = args.outputdirpath
        num_nearestneighbors = args.nearestneighbors
        num_genespercell = args.numgenespercell
        LR_database = args.lrdatabase
        studyname = args.studyname
        ownadjacencypath = args.ownadjacencypath

        preprocess_output_path = os.path.join(output_dir_path, "1_preprocessing_output")
        training_output_path = os.path.join(output_dir_path, "2_training_output")
        evaluation_output_path = os.path.join(output_dir_path, "3_evaluation_output")
        embedding_output_path = os.path.join(output_dir_path, "4_embedding_output")

        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)
        time.sleep(4)
        status.update(status="Done")
        time.sleep(2)

    if "preprocess" in mode:
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        start_time = time.time()
        print("\n#------------------------------------ Loading in data ---------------------------------#\n")
        st_data = pd.read_csv(input_dir_path, index_col=None)
        assert {"Cell_ID", "X", "Y", "Cell_Type"}.issubset(set(st_data.columns.to_list()))

        numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4
        print(f"{numcells} Cell & {totalnumgenes} Total Genes\n")

        print(
            f"Hyperparameter:\n # of Nearest Neighbors: {num_nearestneighbors}\n # of Genes per Cell: {num_genespercell}\n")
        selected_st_data, _ = preprocessing.select_LRgenes(st_data, num_genespercell, LR_database)
        print("\n#------------------------------------ Preprocessing ----------------------------------#\n")
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        celllevel_features = st_data.drop(["Cell_ID", "Cell_Type", "X", "Y"], axis=1).values    #删除4列后的值1597*125
        celllevel_adj = preprocess(selected_st_data, num_nearestneighbors, ownadjacencypath)
        celllevel_edgelist = preprocessing.convert_adjacencylist2edgelist(celllevel_adj)

        assert celllevel_edgelist.shape == (2, celllevel_adj.shape[0] * celllevel_adj.shape[1])

        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencylist.npy"), celllevel_adj)
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencymatrix.npy"),
                preprocessing.convert_adjacencylist2adjacencymatrix(celllevel_adj))
        np.save(os.path.join(preprocess_output_path, "celllevel_edgelist.npy"), celllevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "celllevel_features.npy"), celllevel_features)


        print(f"Finished preprocessing in {(time.time() - start_time) / 60} mins.\n")

    if "train" in mode:
        hyperparameters = {
            "num_genespercell": num_genespercell,
            "concat_hidden_dim": 128,
            "optimizer": "adam",
            "criterion": torch.nn.BCELoss(),
            "num_epochs": 1500,
            "split": args.split,
        }

        false_edges = None if args.fp == 0 and args.fn == 0 else {"fp": args.fp, "fn": args.fn}

        print("\n#------------------------------ Creating PyG Datasets ----------------------------#\n")

        celllevel_data = training.create_pyg_data(preprocess_output_path, hyperparameters["split"],
                                                                 false_edges)
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Cell Level PyG Data")

        celllevel_str = str(celllevel_data)
        table.add_row(celllevel_str)
        console.print(table)

        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)
        if not os.path.exists(evaluation_output_path):
            os.mkdir(evaluation_output_path)
        if not os.path.exists(embedding_output_path):
            os.mkdir(embedding_output_path)

        print("\n#------------------------------- ClarifyGAE Training -----------------------------#\n")
        data = (celllevel_data)
        model = build_clarifyGAE_pytorch(data, hyperparameters).to(device)
        if hyperparameters["optimizer"] == "adam":
            hyperparameters["optimizer"] = torch.optim.Adam(model.parameters(), lr=0.001),
        split = hyperparameters["split"]

        trained_model, metrics_df = training.train_gae(model=model, data=data, hyperparameters=hyperparameters)

        torch.save(trained_model.state_dict(), os.path.join(training_output_path, f'{studyname}fp0fn0_trained_gae_model-split{split}-3.pth'))

        if not os.path.exists(evaluation_output_path):
            os.mkdir(evaluation_output_path)
        metrics_df.to_csv(os.path.join(evaluation_output_path, f"{studyname}fp0fn0_train_threshold_split{split}-3.csv"))


    return


if __name__ == "__main__":
    main()
