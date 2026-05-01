import torch
from torch import nn
from torch.distributions import Normal, Independent
import numpy as np

class QNet(nn.Module):
    def __init__(self, state_size, action_size, network_sizes, output_size):
        super(QNet, self).__init__()
        self.action_size = action_size
        self.layers = nn.ModuleList()
        if len(network_sizes) == 0:
            raise ValueError("Network sizes list cannot be empty.")
        self.layers.append(nn.Linear(state_size, network_sizes[0]))
        self.layers.append(nn.ReLU())
        for i in range(1, len(network_sizes)):
            self.layers.append(nn.Linear(network_sizes[i-1], network_sizes[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(network_sizes[-1], output_size))
        self._init_weights()
        with torch.no_grad():
            self.layers[-1].bias[self.action_size:].fill_(0.0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        

def get_action_prob(actor_mean, actor_log_std, action):
    actor_std = torch.exp(actor_log_std)
    dist = Independent(Normal(actor_mean, actor_std), 1)
    log_prob = dist.log_prob(action)
    return log_prob

def get_action_sample(actor_mean, actor_log_std):
    actor_std = torch.exp(actor_log_std)
    dist = Independent(Normal(actor_mean, actor_std), 1)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.detach(), log_prob.detach()

def action_forward(actor, x):
    squeezed = False
    if x.dim() == 1:
        squeezed = True
        x = x.unsqueeze(0)  # (D,) -> (1, D)
    x = actor(x)
    mean = torch.tanh(x[:, :actor.action_size]) * 1.0  
    log_std = torch.clamp(x[:, actor.action_size:], -2, 1)
    out = torch.cat([mean, log_std], dim=-1)
    return out.squeeze(0) if squeezed else out

class ObsNormalizer:
    def __init__(self, dim, eps=1e-8):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps
        self.eps = eps

    def update(self, x_np):
        # x_np: numpy array of shape (N, dim)
        batch_mean = np.mean(x_np, axis=0)
        batch_var = np.var(x_np, axis=0)
        batch_count = x_np.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        
        self.var = m2 / tot_count
        self.mean = new_mean
        self.count = tot_count

    def normalize(self, tensor):
        tensor = tensor.detach()
        with torch.no_grad():
            mean_t = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
            std_t = torch.tensor(np.sqrt(self.var + self.eps), dtype=tensor.dtype, device=tensor.device)
            return (tensor - mean_t) / std_t