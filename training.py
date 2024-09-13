import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from torch_geometric.data import Data
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, classification_report
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import pandas as pd
from tqdm import trange
import torch_geometric.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tb_writer = SummaryWriter(log_dir="./out/MERFISH/logs")

def add_fake_edges(num_vertices, num_old_edges, fp):
    # add fake edges for self-loops
    print(f"-------------add_fake_edges-------------")
    num_add_edges = int((fp) * num_old_edges)
    add_edges = torch.from_numpy(np.random.randint(0, high=num_vertices, size=(2, num_add_edges)))
    print(f"Added {fp*100}% false edges")
    print(f"New edge index dimension: {add_edges.size()}")

    return add_edges


def remove_real_edges(old_edge_indices, fn):
    new_edge_indices = None
    print(f"-------------remove_real_edges-------------")
    print(f"old_edge_position:{old_edge_indices.shape}")
    num_remove_edges = int(fn * old_edge_indices.shape[0])
    new_edge_indices = torch.from_numpy(np.random.choice(old_edge_indices, size=num_remove_edges, replace=False)).int()
    print(f"Removed {fn*100}% real edges")
    print(f"New edge index dimension:{new_edge_indices.size()}")
    return new_edge_indices


def create_pyg_data(preprocessing_output_folderpath, split=0.1, false_edges=None):
   
    if not os.path.exists(preprocessing_output_folderpath) or not{"celllevel_adjacencylist.npy","celllevel_adjacencymatrix.npy","celllevel_edgelist.npy","celllevel_features.npy"}.issubset(set(os.listdir(preprocessing_output_folderpath))):
        raise Exception("Proper preprocessing files not found. Please run the 'preprocessing' step.")


    celllevel_adjacencymatrix = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath,"celllevel_adjacencymatrix.npy"))).type(torch.LongTensor)
    celllevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath,"celllevel_features.npy")))).type(torch.float32)
    celllevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath,"celllevel_edgelist.npy"))).type(torch.LongTensor)
    cell_level_data = Data(x=celllevel_features, edge_index=celllevel_edgelist, y=celllevel_adjacencymatrix)
   
    if split is not None:
        print(f"{1-split} training edges | {split} testing edges")

        transform = T.RandomLinkSplit(
            num_test=split,
            num_val=0,
            is_undirected=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=1.0,
            key="edge_label",
            disjoint_train_ratio=0,
        )
        train_cell_level_data, _, test_cell_level_data = transform(cell_level_data)
        cell_level_data = (train_cell_level_data, test_cell_level_data)

        if false_edges is not None:
            fp = false_edges["fp"]
            fn = false_edges["fn"]

            if fn != 0:
                new_indices = train_cell_level_data.edge_label.clone()
                old_edge_indices = np.argwhere(train_cell_level_data.edge_label == 1).squeeze()
                print(f"old_edge_position2:{old_edge_indices.shape}")
                new_neg_edge_indices = remove_real_edges(old_edge_indices, fn).long()  
                new_indices[new_neg_edge_indices] = 0
                train_cell_level_data.edge_label = new_indices

            if fp != 0:
                posmask = train_cell_level_data.edge_label == 1
                newedges = add_fake_edges(train_cell_level_data.x.size()[0],train_cell_level_data.edge_label_index[:, posmask].shape[1], fp)
                train_cell_level_data.edge_label = torch.cat([train_cell_level_data.edge_label, torch.ones(newedges.shape[1])])
                train_cell_level_data.edge_label_index = torch.cat([train_cell_level_data.edge_label_index, newedges], dim=1)

        return cell_level_data




def create_intracellular_gene_mask(num_cells, num_genespercell):
    I = np.ones(shape=(num_genespercell, num_genespercell))
    block_list= [I for _ in range(num_cells)]
    return block_diag(*block_list).astype(bool)


def train_gae(data, model, hyperparameters):

    num_epochs = hyperparameters["num_epochs"]
    optimizer = hyperparameters["optimizer"][0]
    criterion = hyperparameters["criterion"]
    split = hyperparameters["split"]


    if split is not None:
        cell_train_data = data[0]
        cell_test_data = data[1]
    else:
        cell_train_data =data[0]

    gene_train_data = data[1]

    num_cells = cell_train_data.x.shape[0]


    test_roc_scores = []
    test_ap_scores = []
    test_auprc_scores = []
    train_auroc_scores = []
    train_ap_scores = []
    test_f1_scores = []
    test_ACC_scores = []
    test_precision_scores = []
    test_recall_scores = []
  

    with trange(num_epochs, desc="") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            model.train()
            optimizer.zero_grad()
            cell_train_data.to(device)
            gene_train_data.to(device)
            posmask = cell_train_data.edge_label == 1
            z, _ = model.encode(cell_train_data.x, cell_train_data.edge_label_index[:, posmask])


            recon_loss = model.recon_loss(z, cell_train_data.edge_label_index[:, posmask])
            loss = recon_loss          

            loss.backward()
            optimizer.step()
            model.eval()
            auroc, ap, best_threshold = traintrain(model, z, cell_train_data.edge_label_index[:, posmask],cell_train_data.edge_label_index[:, ~posmask])

            posmask = cell_test_data.edge_label == 1
            test_recon_loss = model.recon_loss(z, cell_test_data.edge_label_index[:, posmask])
            test_rocauc, test_ap = model.test(z, cell_test_data.edge_label_index[:, posmask], cell_test_data.edge_label_index[:, ~posmask])
            test_precision, test_recall,_ = precision_recall(model,z,cell_test_data.edge_label_index[:, posmask], cell_test_data.edge_label_index[:, ~posmask])
            test_auprc = auc(test_recall, test_precision)

            test_acc, test_precision, test_recall, test_f1_score = class_report(model, z, cell_test_data.edge_label_index[:, posmask], cell_test_data.edge_label_index[:, ~posmask], best_threshold)

            train_auroc_scores.append(auroc)
            train_ap_scores.append(ap)
            test_roc_scores.append(test_rocauc)
            test_ap_scores.append(test_ap)
            test_auprc_scores.append(test_auprc)
            test_f1_scores.append(test_f1_score)
            test_ACC_scores.append(test_acc)
            test_recall_scores.append(test_recall)
            test_precision_scores.append(test_precision)


            pbar.set_postfix(train_loss=loss.item(), train_recon_loss=recon_loss.item(), test_recon_loss = test_recon_loss.item())
            train_loss = loss,
            test_recon_loss = test_recon_loss
            tags = ["train_loss",  "test_recon_loss", "test_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], recon_loss, epoch)
            tb_writer.add_scalar(tags[1], test_recon_loss, epoch)
            tb_writer.add_scalar(tags[2], test_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)
    metrics_df = pd.DataFrame({"Epoch":range(num_epochs), f"Train AUROC":train_auroc_scores, f"Train AP":train_ap_scores,
                               f"Test AP":test_ap_scores, f"Test ROC": test_roc_scores, f"Test AUPRC": test_auprc_scores,
                               f"Test ACC":test_ACC_scores, f"Test Recall":test_recall_scores, 
                               f"Test f1-score":test_f1_scores, f"Test Precision":test_precision_scores})

    return model, metrics_df



def traintrain(model, z, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decoder(z, pos_edge_index,  sigmoid=True)
    neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)



    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(y, pred)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    print(f"\ny_train_len:{len(y)}")

    return roc_auc_score(y, pred), average_precision_score(y, pred), best_threshold 


def precision_recall(model, z, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decoder(z, pos_edge_index,  sigmoid=True)
    neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return precision_recall_curve(y, pred)


def class_report(model, z, pos_edge_index, neg_edge_index, best_threshold):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decoder(z, pos_edge_index,  sigmoid=True)
    neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    pred_labels = (pred >= best_threshold).astype(int)
    Acc = metrics.accuracy_score(y,pred_labels)
    precision = metrics.precision_score(y,pred_labels)
    recall = metrics.recall_score(y,pred_labels)
    f1_score = metrics.f1_score(y,pred_labels)


    return Acc, precision, recall, f1_score







