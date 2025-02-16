import torch
from torch import nn
from torch.nn import functional as F
from .transformer import Encoder as TF_Encoder
from .transformer import Decoder as TF_Decoder

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        # self._embedding.weight.data.uniform_(-1, 1)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCN -> BNC
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encodings
    

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.contiguous(), perplexity, encodings
    

class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cluster_number: int,
        **kwargs
    ):
        super().__init__()

        self.node_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.downsampling = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, cluster_number)
        )

    def forward(self, edge_mtx):
        node_feat = self.node_proj(edge_mtx)

        latent = self.downsampling(node_feat.permute(0, 2, 1))
        latent = latent.permute(0, 2, 1)

        return latent
    

class Decoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        cluster_number: int,
        **kwargs
    ):
        super().__init__()

        self.upsampling = nn.Sequential(
            nn.Linear(cluster_number, cluster_number),
            nn.ReLU(),
            nn.Linear(cluster_number, in_dim)
        )
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
        self.out_scale = nn.Parameter(torch.ones((in_dim * in_dim)))
        self.out_shift = nn.Parameter(torch.zeros((in_dim * in_dim)))

    def modulate(self, x):
        b, n, _ = x.shape
        x = x.reshape(b, -1)
        x = (x + self.out_shift.unsqueeze(0)) * self.out_scale
        x = x.reshape(b, n, n)
        return x
    
    def set_diag(self, x):
        diag = torch.diag(torch.ones(x.shape[1])).type(torch.bool).to(x.device)
        diag_mask = ~diag

        x = x * diag_mask.unsqueeze(0) + diag.unsqueeze(0)
        return x

    def forward(self, cluster_mtx):
        node_repr = self.upsampling(cluster_mtx.permute(0, 2, 1))
        node_repr = node_repr.permute(0, 2, 1)
        node_repr = self.node_proj(node_repr)

        reconstruct = torch.bmm(node_repr, node_repr.permute(0, 2, 1))
        # reconstruct = self.modulate(reconstruct)
        # reconstruct = self.set_diag(reconstruct)

        # reconstruct = node_repr
        return node_repr, reconstruct
    

class VQ_GAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, cluster_number, num_embeddings,
                 embedding_dim, commitment_cost, decay=0, layer_num=4):
        super(VQ_GAE, self).__init__()
        
        self._encoder = TF_Encoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            cluster_number=cluster_number,
            layer_num=layer_num
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = TF_Decoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            cluster_number=cluster_number,
        )

        self.recon_loss = nn.L1Loss(reduce="mean")

    def get_flattened_edge(self, edge_mtx: torch.Tensor):
        # Flatten the edge matrix
        b, n, _ = edge_mtx.shape
        mask = torch.tril(torch.ones(n, n), diagonal=-1).bool()
        edge = edge_mtx[:, mask].reshape(b, -1)

        return edge
    
    def encode_feat(self, edge_mtx: torch.Tensor):
        z = self._encoder(edge_mtx)

        return z

    def quantize(self, latent: torch.Tensor):
        _, quantized, _, _ = self._vq_vae(latent)
        return quantized
    
    def forward(self, edge_mtx: torch.Tensor, **kwargs):
        z = self._encoder(edge_mtx)
        vq_loss, quantized, perplexity, _ = self._vq_vae(z)

        _, x_recon = self._decoder(quantized)

        # print(torch.sum(torch.abs([p for p in self._encoder.parameters()][0])))
        # print(torch.mean(torch.abs([p for p in self._vq_vae._embedding.parameters()][0])))
        # print(f"{torch.std(z).item():.3f}, {torch.std(quantized).item():.3f}")
        
        recon_loss = self.recon_loss(x_recon, edge_mtx)
        loss = {
            "loss": recon_loss + vq_loss,
            "rec_loss": recon_loss,
            "kl_loss": vq_loss
        }

        return loss, x_recon, z
    
    def predict(self, edge_mtx: torch.Tensor, **kwargs):
        return self.forward(edge_mtx)