from torch import nn
import torch
import Utils

class NeuMFModel(nn.Module):
    def __init__(self, num_users, num_items, num_ratings):
        super(NeuMFModel, self).__init__()
        self.embedding_dim = Utils.embedding_dim
        self.num_ratings = num_ratings
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=self.embedding_dim)
        
        self.gmf_layer = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=32),
            nn.ReLU(),
        )
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim*2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
        )
        
        self.neumf_layer = nn.Linear(in_features=64, out_features=self.num_ratings)
        self.neumf_act = nn.Softmax(dim=1)
        
    def forward(self, user, movie):
        user_latent_vector = self.user_embedding(user)
        movie_latent_vector = self.movie_embedding(movie)
        
        # gmf
        gmf_input = torch.mul(user_latent_vector, movie_latent_vector)
        gmf_output = self.gmf_layer(gmf_input)
        
        # mlp
        mlp_input = torch.cat([user_latent_vector, movie_latent_vector], dim=1)
        mlp_output = self.mlp_layer(mlp_input)

        # neural MF
        neumf_input = torch.cat([gmf_output, mlp_output], dim=1)
        out = self.neumf_layer(neumf_input)
        out = self.neumf_act(out)
        
        return out
    
