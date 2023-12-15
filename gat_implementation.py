import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F
from tqdm import tqdm


class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.6, leaky_relu_slope=0.2):
        super(GATConv, self).__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)

        # Expanded attention parameters
        self.att_src_expanded = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.att_dst_expanded = torch.nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.leaky_relu = torch.nn.LeakyReLU(leaky_relu_slope)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_src_expanded)
        torch.nn.init.xavier_uniform_(self.att_dst_expanded)

    def forward(self, x_tuple, edge_index):

        x, x_target = x_tuple  # Unpack the tuple
        x = self.lin(x)
        print("Shape of x before view operation:", x.shape)

        x = x.view(-1, self.heads, self.out_channels)
        print("Shape of x after view operation:", x.shape)


        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(edge_index, x=x, size=None)

        if self.concat:
            return out.view(-1, self.heads * self.out_channels)
        else:
            return out.mean(dim=1)
        
            
    def message(self, edge_index_i, x_i, x_j, size_i):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha_input = torch.cat([x_i, x_j], dim=-1)

        alpha = (alpha_input * self.att_src_expanded).sum(dim=-1)
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)



    def __repr__(self):
        return f'{self.__class__.__name__} ({self.in_channels} -> {self.out_channels})'


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        out_channels_per_head = hidden_channels // num_heads

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, out_channels_per_head, heads=num_heads, concat=True))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, out_channels_per_head, heads=num_heads, concat=True))

        final_out_channels_per_head = out_channels // num_heads
        self.convs.append(GATConv(hidden_channels * num_heads, final_out_channels_per_head, heads=num_heads, concat=False))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []

            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all.log_softmax(dim=-1)
