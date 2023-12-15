import os.path as osp
import torch
import torch.nn.functional as F

from tqdm import tqdm

from torch_geometric.nn import SAGEConv
from torch.nn import Linear
from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
import torch.utils.data as data_utils

from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.loader import NeighborSampler


from mlp_implementation import MLP, GAT_like_MLP
from gat_implementation import GAT, GATConv





def main():


    dataset_name = "ogbn-products"
    dataset_dir = "./data"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = osp.join( dataset_dir, dataset_name)

    dataset = PygNodePropPredDataset(dataset_name, root)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name=dataset_name)
    data = dataset[0]
    train_idx = split_idx['train']

    X_train = data.x[ split_idx["train"] ]
    y_train = data.y[ split_idx["train"] ].reshape(-1).type(torch.long)


    x = data.x
    y = data.y.squeeze()



    print( "data.x.shape:", data.x.shape )
    print( "data.y.shape:", data.y.shape )
    print( "data.x.type:", x.dtype )
    print( "data.y.type:", y.dtype )
    print( "X_train.shape:", X_train.shape )
    print( "y_train.shape:", y_train.shape )

    y = data.y.squeeze().type(torch.long)
    print( "data.y.type:", y.dtype )


    X_y_train_mlpinit = data_utils.TensorDataset(X_train, y_train)
    X_y_all_mlpinit = data_utils.TensorDataset(x, y)

    train_mlpinit_loader = data_utils.DataLoader(X_y_train_mlpinit, batch_size=4096, shuffle=True, num_workers=4)
    all_mlpinit_loader = data_utils.DataLoader(X_y_all_mlpinit, batch_size=4096, shuffle=False, num_workers=4)







    model_mlpinit = MLP(dataset.num_features, 512, dataset.num_classes, num_layers=4)
    model_mlpinit = model_mlpinit.to(device)
    optimizer_model_mlpinit = torch.optim.Adam(model_mlpinit.parameters(), lr=0.001, weight_decay = 0.0)




    def train_mlpinit():
        model_mlpinit.train()
        total_loss = total_correct = 0
        for x, y in train_mlpinit_loader:

            x = x.to( device )
            y = y.to( device )

            optimizer_model_mlpinit.zero_grad()
            out = model_mlpinit(x)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer_model_mlpinit.step()

            total_loss += float(loss)

        loss = total_loss / len(train_mlpinit_loader)
        approx_acc = total_correct / train_idx.size(0)

        return loss, approx_acc


    @torch.no_grad()
    def test_mlpinit():
        model_mlpinit.eval()
        out_list = []
        y_list = []

        for x, y in  all_mlpinit_loader:
            x = x.to( device )
            y = y.to( device )
            out = model_mlpinit(x)
            out_list.append( out )
            y_list.append( y )


        out = torch.cat(out_list, dim=0)
        y_true = torch.cat(y_list, dim=0).cpu().unsqueeze(-1)


        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

        return train_acc, val_acc, test_acc







    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                sizes=[25, 10, 5, 5], batch_size=1024,
                                shuffle=True, num_workers=4)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                    batch_size=4096, shuffle=False,
                                    num_workers=4)


    model = GAT(dataset.num_features, 512, dataset.num_classes, num_layers=4, num_heads=4)

    model = model.to(device)






    def train(epoch):
        
        
        model.train()
 
    
        pbar = tqdm(total=train_idx.size(0))
        pbar.set_description(f'Epoch {epoch:02d}')

        total_loss = total_correct = 0



        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.

            for i, adj in enumerate(adjs):
                edge_index, _, size = adj

                
            adjs = [adj.to(device) for adj in adjs]

            
            optimizer.zero_grad()
            
            x_input = x[n_id].to(device)



            out = model(x_input, adjs)



            
            loss = F.nll_loss( out, y[n_id[:batch_size]].to(device) )

            loss.backward()

            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq( y[n_id[:batch_size]].to(device) ).sum())
            pbar.update(batch_size)

        pbar.close()

        loss = total_loss / len(train_loader)
        approx_acc = total_correct / train_idx.size(0)

        return loss, approx_acc




    @torch.no_grad()
    def test():
        model.eval()

        #out = model.inference(x)
        out = model.inference(x, subgraph_loader)

        y_true = y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': y_true[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        val_acc = evaluator.eval({
            'y_true': y_true[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': y_true[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']

        return train_acc, val_acc, test_acc




#Train wiithout mlp initialization
#    random_losses = []
#    random_test_accs = []

#    model.reset_parameters()

#    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0)

#    best_val_acc = final_test_acc = 0
#    for epoch in range(1, 21):
#        loss, acc = train(epoch)
#        train_acc, val_acc, test_acc = test()
#        print(f'Epoch {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
#        
#        random_losses.append(loss)
#        random_test_accs.append(test_acc)
        
        
  
    print("training mlp")

    model_mlpinit.reset_parameters()

    for epoch in range(1, 20):
        loss, acc = train_mlpinit()
        
    torch.save(model_mlpinit.state_dict(), f'./model_mlpinit.pt' )
    train_acc_init, val_acc_init, test_acc_init = test_mlpinit()
    print(  "train_acc_init, val_acc_init, test_acc_init:", train_acc_init, val_acc_init, test_acc_init )



    mlpinit_losses = []
    mlpinit_test_accs = []

    #model.load_state_dict(torch.load( f'./model_mlpinit.pt'  ))

    print("transfering weights...")
    for mlp_layer, gat_layer in zip(model_mlpinit.convs, model.convs):
        if isinstance(mlp_layer, GAT_like_MLP) and isinstance(gat_layer, GATConv):
            gat_layer.lin.weight.data = mlp_layer.lin_l.weight.data
            
            if mlp_layer.lin_l.bias is not None and gat_layer.lin.bias is not None:
                gat_layer.lin.bias.data = mlp_layer.lin_l.bias.data
            elif mlp_layer.lin_l.bias is None and gat_layer.lin.bias is not None:
                gat_layer.lin.bias.data.zero_()
    print("weighr transfer done")



    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 21):

        loss, acc = train(epoch)
        train_acc, val_acc, test_acc = test()
        print(f'Epoch {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, 'f'Test: {test_acc:.4f}')
        
        mlpinit_losses.append(loss)
        mlpinit_test_accs.append(test_acc)
        
if __name__ == "__main__":
    main()