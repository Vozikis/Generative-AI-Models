import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from intelligraphs.data_loaders.loaders import IntelliGraphsDataLoader
import torch

data_load = IntelliGraphsDataLoader('syn-paths')
train_data, val_data, test_data = data_load.load_torch()
print("Data loaded successfully.")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from intelligraphs.data_loaders.loaders import IntelliGraphsDataLoader

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.layer = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
        self.transformer = TransformerEncoder(self.layer, num_layers=3)  # 3 Transformer Blocks

    def forward(self, x):
        return self.transformer(x)

# Variational Encoder
class VariationalEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, latent_dim, num_entities, num_relations):
        super(VariationalEncoder, self).__init__()
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        self.relation_embedding = nn.Embedding(num_relations, embed_dim)

        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.mean_proj = nn.Linear(embed_dim, latent_dim)  # Projection for μ
        self.logvar_proj = nn.Linear(embed_dim, latent_dim)  # Projection for log(σ)

    def forward(self, x):
        # x shape: (batch_size, 3, 3) -> entity, relation, entity
        head, rel, tail = x[:, 0], x[:, 1], x[:, 2]  # Extract triplets

        # Convert indices to embeddings
        head_emb = self.entity_embedding(head)  # (batch_size, embed_dim)
        rel_emb = self.relation_embedding(rel)  # (batch_size, embed_dim)
        tail_emb = self.entity_embedding(tail)  # (batch_size, embed_dim)

        # Stack instead of concatenate to match expected Transformer shape (batch_size, seq_len=3, embed_dim)
        x = torch.stack([head_emb, rel_emb, tail_emb], dim=1)  # Shape: (batch_size, 3, embed_dim)

        # Transformer Encoder
        x = self.transformer(x)  # Output shape: (batch_size, 3, embed_dim)
        x = torch.mean(x, dim=1)  # Mean pooling over sequence dimension -> (batch_size, embed_dim)

        # Compute Variational Parameters
        mu = self.mean_proj(x)  # (batch_size, latent_dim)
        logvar = self.logvar_proj(x)  # (batch_size, latent_dim)

        return mu, logvar


# Reparameterization Trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Variational Decoder
class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim, embed_dim, num_heads, ff_dim, output_dim):
        super(VariationalDecoder, self).__init__()
        self.linear_proj = nn.Linear(latent_dim, embed_dim)  # Linear projection to initialize input
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)  # Output projection

    def forward(self, z):
        z = self.linear_proj(z).unsqueeze(1)  # Expand to sequence dimension
        z = self.transformer(z)  # Transformer blocks
        z = self.output_proj(z)  # Final LP layer
        return z.squeeze(1)

# Structure Decoder with Tensor Factorization
class StructureDecoder(nn.Module):
    def __init__(self, embed_dim, relation_dim, num_entities):
        super(StructureDecoder, self).__init__()
        self.entity_embedding = nn.Linear(embed_dim, embed_dim)  # Keep embedding dim
        self.relation_embedding = nn.Linear(embed_dim, relation_dim)

    def forward(self, decoded_output):
        entity_recon = self.entity_embedding(decoded_output)  # (batch_size, embed_dim)
        entity_recon = entity_recon.unsqueeze(1).expand(-1, num_entities, -1)  # Expand to (batch_size, num_entities, embed_dim)

        adjacency_matrix = torch.matmul(entity_recon, entity_recon.transpose(1, 2))  # (batch_size, num_entities, num_entities)

        relation_recon = self.relation_embedding(decoded_output)  # (batch_size, relation_dim)
        return adjacency_matrix, relation_recon



# Full FG-VAE Model
class FG_VAE(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, latent_dim, output_dim, relation_dim, num_entities, num_relations):
        super(FG_VAE, self).__init__()
        self.encoder = VariationalEncoder(embed_dim, num_heads, ff_dim, latent_dim, num_entities, num_relations)
        self.decoder = VariationalDecoder(latent_dim, embed_dim, num_heads, ff_dim, output_dim)
        self.structure_decoder = StructureDecoder(embed_dim, relation_dim, num_entities)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        decoded_output = self.decoder(z)
        adjacency_matrix, relations = self.structure_decoder(decoded_output)
        return adjacency_matrix, relations, mu, logvar

# Model Instantiation
embed_dim = 64
num_heads = 4
ff_dim = 128
latent_dim = 32
output_dim = 64  # Same as embed_dim
relation_dim = 16
num_entities = 49  # Example: Number of unique entities
num_relations = 3  # Example: Number of unique relations

model = FG_VAE(embed_dim, num_heads, ff_dim, latent_dim, output_dim, relation_dim, num_entities, num_relations)

# Load dataset
# Load dataset
data_load = IntelliGraphsDataLoader('syn-paths')
train_loader, val_loader, test_loader = data_load.load_torch()
print("Data loaded successfully.")

# Get one batch from train_loader
batch = next(iter(train_loader))  # Get the first batch

# Ensure input is a tensor
x = batch[0]  # Assuming the first element is the input data

# Convert input to integer tensor if it's not already
x = x.long()  # Ensure it's integer-based for embedding lookup

print("Input Shape:", x.shape)  # Expected: (batch_size, 3, 3)

# Forward Pass
adj_matrix, relations, mu, logvar = model(x)

# Print Output Shapes
print("Adjacency Matrix Shape:", adj_matrix.shape)
print("Relations Shape:", relations.shape)
print("Mean Shape:", mu.shape)
print("Log Variance Shape:", logvar.shape)



import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# KL Divergence Loss
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Training Function
def train(model, train_loader, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch[0].long()  # Ensure integer inputs for embedding
            adj_matrix, relations, mu, logvar = model(x)
            
            # Reconstruction Loss (Adjacency Matrix)
            identity_matrix = torch.eye(num_entities, device=adj_matrix.device).expand(adj_matrix.shape[0], -1, -1)
            recon_loss = F.mse_loss(adj_matrix, identity_matrix)


            # KL Divergence
            kl_loss = kl_divergence(mu, logvar)

            # Total VAE Loss
            loss = recon_loss + 1 * kl_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Evaluation Function: Mean Reciprocal Rank (MRR)
def evaluate_mrr(model, test_loader):
    model.eval()
    ranks = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].long()
            adj_matrix, relations, mu, logvar = model(x)

            print(f"adj_matrix shape: {adj_matrix.shape}")  # Debugging
            print(f"x shape: {x.shape}")  # Debugging

            for i in range(x.shape[0]):  # Iterate over batch
                head, rel, tail = x[i, 0].item(), x[i, 1].item(), x[i, 2].item()
                print(f"Head: {head}, Relation: {rel}, Tail: {tail}")  # Debugging

                # Check if head index is within bounds
                if head >= num_entities:
                    print(f"Error: head index {head} is out of bounds for num_entities {num_entities}")
                    continue  # Skip this sample to prevent crashing

                # Compute scores for all possible tails
                scores = adj_matrix[i, head]  # Use batch index i
                sorted_scores = torch.argsort(scores, descending=True)

                # Find rank of the correct tail
                if tail >= num_entities:  # Ensure valid tail index
                    print(f"Error: tail index {tail} is out of bounds for num_entities {num_entities}")
                    continue

                rank_pos = (sorted_scores == tail).nonzero(as_tuple=True)
                if len(rank_pos[0]) > 0:  # Ensure there is a match
                    rank = rank_pos[0].item() + 1  # Convert to 1-based index
                    ranks.append(1 / rank)
                else:
                    print(f"Warning: Tail {tail} not found in sorted scores.")

    if len(ranks) == 0:
        print("Warning: No valid ranks computed.")
        return 0.0

    mrr = torch.tensor(ranks).mean().item()
    print(f"MRR: {mrr:.4f}")
    return mrr


# Ensure `train_loader`, `val_loader`, and `test_loader` are passed from your existing dataset
# If they are already defined, you can directly use them below:

# Initialize Model
embed_dim = 64
num_heads = 4
ff_dim = 128
latent_dim = 32
output_dim = 64  # Same as embed_dim
relation_dim = 16
num_entities = 49  # Example: Number of unique entities
num_relations = 3  # Example: Number of unique relations

# Initialize FG_VAE model (make sure the class definition exists)
model = FG_VAE(embed_dim, num_heads, ff_dim, latent_dim, output_dim, relation_dim, num_entities, num_relations)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Ensure dataset entity range before training
for batch in train_loader:
    x = batch[0].long()
    max_entity_index = x.max().item()
    print(f"Max entity index in dataset: {max_entity_index}, Model num_entities: {num_entities}")
    if max_entity_index >= num_entities:
        print(f"Error: max entity index {max_entity_index} exceeds num_entities {num_entities}.")
    break  # Only check one batch

# Train Model with the existing dataset
train(model, train_loader, optimizer, num_epochs=10)

# Evaluate Model with the existing dataset
evaluate_mrr(model, test_loader)
