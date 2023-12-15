import torch
from torch.nn import Linear
import torch.nn.functional as F


class GAT_like_MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()

    def forward(self, x):
        out = self.lin_l(x)  
        return out 

    
    
    
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GAT_like_MLP(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GAT_like_MLP(hidden_channels, hidden_channels))
        self.convs.append(GAT_like_MLP(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.convs[i](x)  
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)
