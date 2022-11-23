import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import argparse
from models import DGI, LogReg
from utils import process

def pre_train(args, lr, l2_coef):    
    adj, features = process.load_data(args.dataset)
    features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]  # number of nodes
    ft_size = features.shape[1]   # size of features    
    #adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    
    features = torch.FloatTensor(features[np.newaxis])
 
    model = DGI(ft_size, args.hid_units, nn.ReLU())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    device = torch.device('cuda:' + args.device)
    model.to(device)
    features = features.to(device)
    sp_adj = sp_adj.to(device)    
    bce_loss = nn.BCEWithLogitsLoss()
   
    cnt_wait = 0
    best = 1e9
    best_t = 0
    # Train
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(args.batch_size, nb_nodes)
        lbl_2 = torch.zeros(args.batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        shuf_fts = torch.tensor(shuf_fts, device=device)
        lbl = torch.tensor(lbl, device=device)
        logits = model(features, shuf_fts, sp_adj, True, None, None, None) 
        loss = bce_loss(logits, lbl)

        print('Loss:', loss)
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.gcn.fc.state_dict(), f'output/{args.dataset}/lr{lr}_wd{l2_coef}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        loss.backward()
        optimizer.step()
        output_file = f'output/{args.dataset}/lr{lr}_wd{l2_coef}.txt'
        with open(output_file, 'a') as f:
            f.write(f'epoch: {epoch}, loss: {loss.item()}\n')
            print(f'epoch: {epoch}, loss: {loss.item()}')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pretraining with dgl')
    parser.add_argument('--dataset', '-d', type=str,
                       default="cora", help="the name of dataset")
    parser.add_argument('--batch-size', '-b', type=int, 
                        default=1, help='the size of batches')
    parser.add_argument('--epochs', '-e', type=int, 
                        default=10000, help='the number of epochs')
    parser.add_argument('--patience', '-p', type=int,
                        default=20, help='the number of patience')
    parser.add_argument('--hid-units', '-hu', type=int, 
                        default=512, help='hidden units for GCN')
    parser.add_argument('--device', '-de', type=str, 
                        default="0", help='device')
    
    args = parser.parse_args()
    for lr in [0.001]:
        for lr_coef in [5e-6]:            
            pre_train(args, lr, lr_coef)            
    # training params