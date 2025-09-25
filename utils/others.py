import torch.nn as nn
import torch

class Variablefc(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU()):
        super(Variablefc, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation)
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def hypergraphSyn(vectors):
    '''
    usage: fused_vector = hypergraphSyn([vectors])
    '''
    x = torch.stack(vectors, dim=0)
    n_vectors = x.shape[0]
    H = 1 - torch.nn.functional.cosine_similarity(
        x.unsqueeze(1), x.unsqueeze(0), dim=2
    )
    H = torch.clamp(H, 0, 1)
    A = H @ H.T # Adjacency matrix
    fused_x = A @ x
    fused_x = torch.sum(fused_x, dim=0, keepdim=True).T
    return fused_x
        
