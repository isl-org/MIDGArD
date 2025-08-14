import torch
import numpy as np
import logging
from core.utils.data_utils import map_upper_triangle_to_list
from core.models.utils.common import BatchNormLastDim, MLP


class GraphAttentionLayer(torch.nn.Module):
    """
    Original implementation of the NAP graph denoiser
    Inputs:
        in_node_features - Dimensionality of input node features
        in_edge_features - Dimensionality of input edge features
        out_node_features - Dimensionality of output node features
        out_edge_features - Dimensionality of output edge features
        p_emb_dim - Dimensionality of positional embeddings
        t_emb_dim - Dimensionality of time embeddings
        num_attention_heads - Number of attention heads, i.e. attention mechanisms to apply in parallel.
        dropout - The dropout probability
        use_batch_normalization - Use batch normalization in the graph layers
    """

    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int,
        out_edge_features: int,
        p_emb_dim: int,
        t_emb_dim: int,
        num_attention_heads: int = 1,
        dropout: float = 0.0,
        use_batch_normalization=False,
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads

        # Node & edge embedding
        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(in_node_features + p_emb_dim + t_emb_dim, in_node_features),
            (
                BatchNormLastDim(in_node_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.edge_embed = torch.nn.Sequential(
            torch.nn.Linear(in_edge_features + t_emb_dim, in_edge_features),
            (
                BatchNormLastDim(in_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )

        # Graph convolution layers
        self.node_update = torch.nn.Sequential(
            torch.nn.Linear(in_node_features, out_edge_features),
            (
                BatchNormLastDim(out_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.node_output = torch.nn.Sequential(
            torch.nn.Linear(2 * out_edge_features, out_node_features),
            (
                BatchNormLastDim(out_node_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )

        self.edge_update = torch.nn.Sequential(
            torch.nn.Linear(2 * in_node_features + in_edge_features, out_edge_features),
            (
                BatchNormLastDim(in_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(out_edge_features, out_edge_features),
        )

        # Transformer attention mechanism
        self.W_q = torch.nn.Linear(in_node_features, out_edge_features)
        self.W_k = torch.nn.Linear(in_node_features, out_edge_features)

        # Softmax to compute attention αij
        self.softmax = torch.nn.Softmax(dim=2)

        # Dropout layer to be applied for attention
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features_flat: torch.Tensor,
        p_emb: torch.Tensor,
        t_emb: torch.Tensor,
    ):
        """
        Inputs:
            node_features - Input node embeddings of shape [B, N, in_features]
            edge_features_flat - Input features of the edge. Shape: [B, N^2, edge_features], where edge_features is the edge feature vector size
            adj_matrix - Adjacency matrix including self-connections. Shape: [B, N, N]
            p_emb - Position encoding
            t_emb - Time embeddings
        """
        # Dimensionality:
        # node_features: [B, N, F]
        # edge_features_flat: [B, N^2, F]
        # p_emb: [B, N, F]
        # t_emb: [B, F]
        B, N, _ = node_features.shape

        # 1) Embed node & edge features with positional/time info
        if p_emb is not None:
            node_in = torch.cat(
                [node_features, p_emb, t_emb[:, None, :].expand(-1, N, -1)], dim=-1
            )
        else:
            node_in = torch.cat(
                [node_features, t_emb[:, None, :].expand(-1, N, -1)], dim=-1
            )
        node_embed = self.node_embed(node_in)

        edge_in = torch.cat(
            [
                edge_features_flat,
                t_emb[:, None, :].expand(-1, N * N, -1),
            ],
            dim=-1,
        )
        edge_embed = self.edge_embed(edge_in)

        # 2) Edge update
        # Create [f_i || f_j] for all pairs of i,j.

        # node_i: [B, N, N, in_node_features]
        node_i = node_embed[:, :, None, :].expand(-1, -1, N, -1)
        # node_j: [B, N, N, in_node_features]
        node_j = node_embed[:, None, :, :].expand(-1, N, -1, -1)
        # pairwise: [B, N, N, 2*in_node_features]
        pairwise = torch.cat([node_i, node_j], dim=-1)
        # pairwise_flat: [B, N*N, 2*in_node_features]
        pairwise_flat = pairwise.reshape(B, N * N, -1)

        # e_ij = f( f_i||f_j|| old_edge ): [B, N*N, out_edge_features]
        updated_edge_flat = self.edge_update(
            torch.cat([pairwise_flat, edge_embed], dim=-1)
        )

        # Calculate the corresponding attention scores
        e = updated_edge_flat.reshape(B, N, N, -1)

        # 3) Node-level attention
        # Build node Q, K, self-attention and aggregation mask
        q = self.W_q(node_embed)[:, :, None, :].expand(-1, -1, N, -1).contiguous()
        k = self.W_k(node_embed)[:, None, :, :].expand(-1, N, -1, -1).contiguous()
        attention_matrix = q * k
        attention_heads = attention_matrix.reshape(B, N, N, -1, self.num_heads).sum(
            -1, keepdim=True
        ) / np.sqrt(self.num_heads * 3.0)

        # We then normalize attention scores (or coefficients)
        a = self.softmax(attention_heads)
        a = a.expand(-1, -1, -1, -1, self.num_heads)
        a = a.reshape(B, N, N, -1)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        attn_output = (a * e).sum(2)

        # 4) Node update: combine attn_output with a transformed node_embed

        # node_updated: [B, N, out_edge_features]
        node_updated = self.node_update(node_embed)
        node_updated = node_updated + attn_output

        # 5) Global pooling or aggregator

        # node_pooling: [B, N, out_node_features]
        node_pooling = node_updated.max(dim=1, keepdim=True)[0].expand(-1, N, -1)

        # final_node_out: [B, N, out_node_features]
        final_node_out = self.node_output(
            torch.cat([node_updated, node_pooling], dim=-1)
        )

        return final_node_out, updated_edge_flat


class GraphAttentionLayerV2(torch.nn.Module):
    """
    Original implementation of the NAP graph denoiser
    Inputs:
        in_node_features - Dimensionality of input node features
        in_edge_features - Dimensionality of input edge features
        out_node_features - Dimensionality of output node features
        out_edge_features - Dimensionality of output edge features
        p_emb_dim - Dimensionality of positional embeddings
        t_emb_dim - Dimensionality of time embeddings
        num_attention_heads - Number of attention heads, i.e. attention mechanisms to apply in parallel.
        dropout - The dropout probability
        use_batch_normalization - Use batch normalization in the graph layers
    """

    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int,
        out_edge_features: int,
        p_emb_dim: int,
        t_emb_dim: int,
        num_attention_heads: int = 1,
        dropout: float = 0.0,
        use_batch_normalization=False,
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads

        # Node & edge embedding
        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(in_node_features + p_emb_dim + t_emb_dim, in_node_features),
            (
                BatchNormLastDim(in_node_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.edge_embed = torch.nn.Sequential(
            torch.nn.Linear(in_edge_features + t_emb_dim, in_edge_features),
            (
                BatchNormLastDim(in_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )

        # Graph convolution layers
        self.node_update = torch.nn.Sequential(
            torch.nn.Linear(in_node_features, out_edge_features),
            (
                BatchNormLastDim(out_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.edge_update = torch.nn.Sequential(
            torch.nn.Linear(2 * in_node_features + in_edge_features, out_edge_features),
            (
                BatchNormLastDim(in_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
            torch.nn.Linear(out_edge_features, out_edge_features),
        )
        self.node_output = torch.nn.Sequential(
            torch.nn.Linear(2 * out_edge_features, out_node_features),
            (
                BatchNormLastDim(out_node_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )

        # Transformer attention mechanism
        self.W_q = torch.nn.Linear(in_node_features, out_edge_features, bias=False)
        self.W_k = torch.nn.Linear(in_node_features, out_edge_features, bias=False)

        # Dropout probability introduced in the attention mechanism
        self.dropout_p = dropout

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features_flat: torch.Tensor,
        p_emb: torch.Tensor,
        t_emb: torch.Tensor,
    ):
        """
        Inputs:
            node_features - Input node embeddings of shape [B, N, in_features]
            edge_features_flat - Input features of the edge. Shape: [B, N^2, edge_features], where edge_features is the edge feature vector size
            p_emb - Position encoding
            t_emb - Time embeddings
        """
        # Dimensionality:
        # node_features: [B, N, F]
        # edge_features_flat: [B, N*N, F]
        # p_emb: [B, N, F]
        # t_emb: [B, F]
        B, N, _ = node_features.shape

        # 1) Embed node & edge features with positional/time info
        if p_emb is not None:
            node_in = torch.cat(
                [node_features, p_emb, t_emb[:, None, :].expand(-1, N, -1)], dim=-1
            )
        else:
            node_in = torch.cat(
                [node_features, t_emb[:, None, :].expand(-1, N, -1)], dim=-1
            )
        node_embed = self.node_embed(node_in)

        edge_in = torch.cat(
            [
                edge_features_flat,
                t_emb[:, None, :].expand(-1, N * N, -1),
            ],
            dim=-1,
        )
        edge_embed = self.edge_embed(edge_in)

        # 2) Edge update
        # Create [f_i || f_j] for all pairs of i,j.

        # node_i: [B, N, N, in_node_features]
        node_i = node_embed[:, :, None, :].expand(-1, -1, N, -1)
        # node_j: [B, N, N, in_node_features]
        node_j = node_embed[:, None, :, :].expand(-1, N, -1, -1)
        # pairwise: [B, N, N, 2*in_node_features]
        pairwise = torch.cat([node_i, node_j], dim=-1)
        # pairwise_flat: [B, N*N, 2*in_node_features]
        pairwise_flat = pairwise.reshape(B, N * N, -1)

        # e_ij = f( f_i||f_j|| old_edge ): [B, N*N, out_edge_features]
        updated_edge_flat = self.edge_update(
            torch.cat([pairwise_flat, edge_embed], dim=-1)
        )

        # 3) Node-level attention
        # q, k, v: [B, N, out_edge_features] -> reshape -> [B, num_heads, N, head_dim]
        out_dim = updated_edge_flat.shape[-1]  # = out_edge_features
        head_dim = out_dim // self.num_heads

        # Build Q, K from nodes and reshape to multi-head
        q = (
            self.W_q(node_embed)
            .reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            self.W_k(node_embed)
            .reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # Build V from edges
        E = updated_edge_flat.reshape(B, N, N, out_dim)
        E_agg = E.mean(dim=2)
        v = (
            E_agg.reshape(B, N, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # attn_output: [B, num_heads, N, head_dim]
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p
        )

        # [B, N, num_heads*head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, -1)

        # 4) Node update: combine attn_output with a transformed node_embed
        # node_updated: [B, N, out_edge_features]
        node_updated = self.node_update(node_embed)
        node_updated = node_updated + attn_output  # simple skip/res connection

        # 5) Global pooling or aggregator
        # node_pooling: [B, N, out_node_features]
        node_pooling = node_updated.max(dim=1, keepdim=True)[0].expand(-1, N, -1)

        # final_node_out: [B, N, out_node_features]
        final_node_out = self.node_output(
            torch.cat([node_updated, node_pooling], dim=-1)
        )

        return final_node_out, updated_edge_flat


class DenoiserMLP(torch.nn.Module):
    """
    MLP denoiser implementation proposed in NAP for their ablation study
    Inputs:
        in_node_features - Dimensionality of input node features
        in_edge_features - Dimensionality of input edge features
        out_node_features - Dimensionality of output node features
        out_edge_features - Dimensionality of output edge features
        p_emb_dim - Dimensionality of positional embeddings
        t_emb_dim - Dimensionality of time embeddings
        use_batch_normalization -
    """

    def __init__(
        self,
        in_node_features: int,
        in_edge_features: int,
        out_node_features: int,
        out_edge_features: int,
        p_emb_dim: int,
        t_emb_dim: int,
        use_batch_normalization: bool = False,
    ) -> None:
        super().__init__()

        # Node & edge embedding
        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(in_node_features + p_emb_dim + t_emb_dim, in_node_features),
            (
                BatchNormLastDim(in_node_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.edge_embed = torch.nn.Sequential(
            torch.nn.Linear(in_edge_features + t_emb_dim, in_edge_features),
            (
                BatchNormLastDim(in_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )

        # MLP layers
        self.node_update = torch.nn.Sequential(
            torch.nn.Linear(in_node_features, out_node_features),
            (
                BatchNormLastDim(out_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.edge_update = torch.nn.Sequential(
            torch.nn.Linear(in_node_features, out_node_features),
            (
                BatchNormLastDim(out_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.node_update_out = torch.nn.Sequential(
            torch.nn.Linear(
                out_node_features * 2 + out_edge_features, out_node_features
            ),
            (
                BatchNormLastDim(out_node_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )
        self.edge_update_out = torch.nn.Sequential(
            torch.nn.Linear(
                out_edge_features * 2 + out_node_features, out_edge_features
            ),
            (
                BatchNormLastDim(out_edge_features)
                if use_batch_normalization
                else torch.nn.Identity()
            ),
            torch.nn.SiLU(),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features_flat: torch.Tensor,
        p_emb: torch.Tensor = None,
        t_emb: torch.Tensor = None,
    ):
        """
        Inputs:
            node_features - Input node embeddings of shape [B, N, in_features]
            edge_features_flat - Input features of the edge. Shape: [B, N^2, edge_features], where edge_features is the edge feature vector size
            p_emb - Position encoding
            t_emb - Time embeddings
        """
        # Dimensionality:
        # node_features: [B, N, F]
        # edge_features_flat: [B, N*N, F]
        # p_emb: [B, N, F]
        # t_emb: [B, F]
        B, N, _ = node_features.shape

        # 1) Embed node & edge features with positional/time info
        if p_emb is not None:
            node_in = torch.cat(
                [node_features, p_emb, t_emb[:, None, :].expand(-1, N, -1)], dim=-1
            )
        else:
            node_in = torch.cat(
                [node_features, t_emb[:, None, :].expand(-1, N, -1)], dim=-1
            )
        node_embed = self.node_embed(node_in)

        edge_in = torch.cat(
            [
                edge_features_flat,
                t_emb[:, None, :].expand(-1, N * N, -1),
            ],
            dim=-1,
        )
        edge_embed = self.edge_embed(edge_in)

        # For the ablation study purposes: use a MLP in place of a GAT
        updated_node_features = self.node_update(node_embed)
        projected_edge_features = self.edge_update(edge_embed)
        node_max_pool = updated_node_features.max(1)[0]
        edge_max_pool = updated_node_features.max(1)[0]
        pooled = torch.cat([node_max_pool, edge_max_pool], -1)[:, None, :]
        updated_node_features = self.node_update_out(
            torch.cat(
                [
                    updated_node_features,
                    pooled.expand(-1, updated_node_features.shape[1], -1),
                ],
                dim=-1,
            )
        )
        updated_edge_features = self.edge_update_out(
            torch.cat(
                [
                    projected_edge_features,
                    pooled.expand(-1, projected_edge_features.shape[1], -1),
                ],
                dim=-1,
            )
        )

        return updated_node_features, updated_edge_features


class GraphTransformer(torch.nn.Module):
    """
    Graph Transformer denoiser for fully connected graphs
    """

    def __init__(
        self,
        node_features_struct: list = [1, 3, 3, 128],
        edge_features_struct: list = [3, 6, 4],
        hidden_node_features_dim: list = [256, 256 + 16, 256 + 32, 256],
        hidden_edge_features_dim: list = [512, 512 + 32, 512 + 16, 512],
        out_node_features_dim: list = [128, 128],
        out_edge_features_dim: list = [128, 64],
        p_emb_dim: int = 200,
        t_emb_dim: int = 100,
        num_attention_heads: int = 16,
        max_nodes: int = 8,
        diffusion_time_steps: int = 10,
        use_batch_normalization: bool = False,
        type_graph_layers: str = "gt",
        dir_handling=True,
        sym_sync=False,
    ) -> None:
        super(GraphTransformer, self).__init__()

        # Sanity checks:
        assert len(hidden_node_features_dim) == len(hidden_edge_features_dim)
        assert max_nodes > 2
        assert diffusion_time_steps > 0

        # Initialization of the member variables
        self.node_features_struct = node_features_struct
        self.edge_features_struct = edge_features_struct
        self.tri_ind_to_full_ind, self.src_dst_ind = self.tri_index_to_full_index(
            max_nodes
        )
        self.sym_sync = sym_sync  # Flag to enforce symmetry of the Edge feature matrix
        self.dir_handling = dir_handling
        self.N = max_nodes
        self.p_emb_dim = p_emb_dim
        self.t_emb_dim = t_emb_dim

        # Time and position embedings
        self.register_buffer(
            "t_emb", self.sinusoidal_embedding(diffusion_time_steps, self.t_emb_dim)
        )
        if self.p_emb_dim > 0:
            self.register_buffer(
                "p_emb", self.sinusoidal_embedding(max_nodes, self.p_emb_dim)
            )

        # Input layers
        node_input_dim = hidden_node_features_dim[0]
        edge_input_dim = hidden_edge_features_dim[0]

        # Each feature class (e.g. bounding box, Plücker coordinates, chirality, and so on...) is encoded by its own fully connected layer
        self.node_input_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(node_feature_dim, node_input_dim), torch.nn.SiLU()
                )
                for node_feature_dim in node_features_struct
            ]
        )
        self.edge_input_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(edge_feature_dim, edge_input_dim), torch.nn.SiLU()
                )
                for edge_feature_dim in edge_features_struct
            ]
        )

        # GAT layers
        node_concat_dim = node_input_dim
        edge_concat_dim = edge_input_dim
        self.hidden_graph_layers = torch.nn.ModuleList()
        for i, (node_hidden_dim, edge_hidden_dim) in enumerate(
            zip(hidden_node_features_dim[1:], hidden_edge_features_dim[1:])
        ):
            # Determine if it's the last iteration
            is_last_iteration = i == len(hidden_node_features_dim) - 2

            # Create the graph layers
            if type_graph_layers == "gt":
                self.hidden_graph_layers.append(
                    GraphAttentionLayer(
                        node_input_dim,
                        edge_input_dim,
                        node_hidden_dim,
                        edge_hidden_dim,
                        p_emb_dim,
                        t_emb_dim,
                        num_attention_heads,
                        use_batch_normalization=use_batch_normalization,
                    )
                )
            elif type_graph_layers == "gt2":
                self.hidden_graph_layers.append(
                    GraphAttentionLayerV2(
                        node_input_dim,
                        edge_input_dim,
                        node_hidden_dim,
                        edge_hidden_dim,
                        p_emb_dim,
                        t_emb_dim,
                        num_attention_heads,
                        use_batch_normalization=use_batch_normalization,
                    )
                )
            elif type_graph_layers == "mlp":
                self.hidden_graph_layers.append(
                    DenoiserMLP(
                        node_input_dim,
                        edge_input_dim,
                        node_hidden_dim,
                        edge_hidden_dim,
                        p_emb_dim,
                        t_emb_dim,
                    )
                )
            else:
                NotImplementedError(type_graph_layers)
            node_input_dim = node_hidden_dim
            edge_input_dim = edge_hidden_dim
            node_concat_dim += node_input_dim
            edge_concat_dim += edge_input_dim

        # Concatenation of the inputs
        self.node_concat_layer = torch.nn.Linear(
            node_concat_dim, hidden_node_features_dim[-1]
        )
        self.edge_concat_layer = torch.nn.Linear(
            edge_concat_dim, hidden_edge_features_dim[-1]
        )
        if self.sym_sync:
            self.edge_concat_layer_symsync = torch.nn.Linear(
                edge_concat_dim, hidden_edge_features_dim[-1]
            )

        # Output layers
        self.node_output_layers = torch.nn.ModuleList()
        for c_out in self.node_features_struct:
            self.node_output_layers.append(
                MLP(
                    hidden_node_features_dim[-1] + c_out,
                    c_out,
                    out_node_features_dim,
                    use_batch_normalization=use_batch_normalization,
                )
            )

        self.edge_output_layers = torch.nn.ModuleList()
        for c_out in self.edge_features_struct:
            self.edge_output_layers.append(
                MLP(
                    hidden_edge_features_dim[-1] + c_out,
                    c_out,
                    out_edge_features_dim,
                    use_batch_normalization=use_batch_normalization,
                )
            )

        return

    @staticmethod
    def sinusoidal_embedding(n, d) -> torch.Tensor:
        """
        Returns the standard positional embedding
        """
        embedding = torch.zeros(n, d)
        wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
        wk = wk.reshape((1, d))
        t = torch.arange(n).reshape((n, 1))
        embedding[:, ::2] = torch.sin(t * wk[:, ::2])
        embedding[:, 1::2] = torch.cos(t * wk[:, ::2])
        return embedding

    @staticmethod
    def tri_index_to_full_index(N: int):
        """
        Convert triangle indices to full matrix indices for an upper-triangular matrix.

        This function creates a mapping from the indices of an upper-triangular part of a matrix
        to the corresponding full matrix indices considering only the upper part without the diagonal.

        Parameters:
        N (int): The number of nodes (i.e. dimension of the square matrix)

        Returns:
        tuple: A tuple containing two tensors. The first tensor contains the flattened upper triangular indices.
            The second tensor is a 2xN tensor of the original row and column indices where N is the number of
            elements in the upper triangle of the matrix, excluding the diagonal.
        """
        # Initialize lists to store the triangle indices and full matrix indices
        tri_ind = []
        full_ind = []

        # Initialize a list to store the source and destination indices
        src_dst_ind = []

        # Iterate through the rows of the matrix
        for i in range(N):
            # Iterate through the columns of the matrix
            for j in range(N):
                # Check if the index is in the upper triangle (excluding the diagonal)
                if i < j:
                    # Append the source-destination index pair
                    src_dst_ind.append([i, j])
                    # Append the triangle index after mapping
                    tri_ind.append(map_upper_triangle_to_list(i, j, N))
                    # Append the corresponding index in the flattened full matrix
                    full_ind.append(i * N + j)

        # Check that the triangular indices list is a sequence from 0 to N-1
        assert tri_ind == [
            i for i in range(len(tri_ind))
        ], "Triangular indices are not sequential"

        # Convert source-destination indices to a numpy array and transpose
        src_dst_ind = np.array(src_dst_ind).T.astype(np.longlong)

        # Return a tuple of tensors: the full indices and the transposed source-destination indices
        return torch.Tensor(full_ind).long(), torch.from_numpy(src_dst_ind)

    def scatter_trilist_to_matrix(self, buffer) -> torch.Tensor:
        """
        Parameters:
        buffer: A 3D tensor with dimensions [B, |num_edges|, F], where |num_edges| is the number of edges
        in the lower triangular part of the matrix (excluding diagonal), and F is the feature dimension.
        """

        B, num_edges, F = buffer.shape
        # Ensure the second dimension of the buffer corresponds to a lower triangular number of elements.
        assert num_edges == self.N * (self.N - 1) // 2, "num_edges should be N*(N-1)/2"

        # Convert the indices for a triangular matrix to indices for a full matrix.
        ind = self.tri_ind_to_full_ind.to(buffer.device)[None, :, None]

        # Expand indices to match the batch size and feature dimension for scatter operation.
        ind = ind.expand(B, -1, F)

        # Initialize a zero tensor with dimensions for the full matrix including all pairs of nodes.
        ret = torch.zeros(B, self.N * self.N, F, device=buffer.device)

        # Use scatter to map the lower triangular elements (buffer) to the full matrix (ret).
        ret = torch.scatter(ret, 1, ind, buffer)

        # Reshape to a 4D tensor to get separate matrices for each feature dimension.
        ret = ret.reshape(B, self.N, self.N, F)

        # Return the full matrix with zero padding for upper triangular elements not present in buffer.
        return ret

    def get_edge_mask(self, node_mask, return_trilist=True):
        """
        Parameters:
        node_mask: A 2D tensor with dimensions [B, N].

        Returns:
        edge_mask: A 3D tensor with dimensions [B, N, N].
        """

        B = node_mask.shape[0]
        # Create an edge mask from the node mask; an edge is present if either of its vertices is present.
        edge_mask = ((node_mask[:, :, None] + node_mask[:, None, :]) > 0.0).float()

        # If a list of edges in the lower triangular representation is required,
        # convert the full adjacency mask to a lower triangular list.
        if return_trilist:
            # Clone the index tensor to match with the device of edge_mask.
            gather_ind = self.tri_ind_to_full_ind.clone()

            # Reshape the edge mask to a 2D tensor for gather operation.
            edge_mask = edge_mask.reshape(B, -1)

            # Gather the lower triangular part of the mask using the previously calculated indices.
            edge_mask = torch.gather(
                edge_mask,
                1,
                gather_ind[None, :].expand(B, -1).to(edge_mask.device),
            )

        # Return the ground truth edge mask, either as a full matrix or a lower triangular list.
        return edge_mask

    def forward(
        self,
        nodes_features_noisy: torch.Tensor,
        edges_features_noisy: torch.Tensor,
        diffusion_timesteps: list,
        nodes_features_cond: torch.Tensor = None,
        edges_features_cond: torch.Tensor = None,
        node_mask=None,
    ):
        """
        Defines the forward pass of the GraphTransformer.

        Args:
            nodes_features_noisy (torch.Tensor): The input tensor of size [B, N, F] containing node features.
            edges_features_noisy (torch.Tensor): The input tensor of size [B, N*N, F] containing edge features.
            diffusion_timesteps: List of diffusion timesteps
            node_mask (torch.Tensor): Node features mask

        Returns:
            node_feature_pred (torch.Tensor): The output tensor of size [B, N, F] containing node features.
            edge_feature_pred (torch.Tensor): The output tensor of size [B, N*N, F] containing edge features.
        """
        B, N, _ = nodes_features_noisy.shape

        # Given a specific node mask, generate a suitable edge mask...
        if node_mask is not None:
            nodes_features_noisy = nodes_features_noisy * node_mask[..., None]
            edge_mask = self.get_edge_mask(node_mask, return_trilist=True)
            edges_features_noisy = edges_features_noisy * edge_mask[..., None]

        nodes_features = nodes_features_noisy

        # Get a symmetric matrix representation out of the upper triangular representation of the symmetric edge features
        edges_features_sym = edges_features_noisy[..., 3:]  # Symmetric edge features
        edges_features_sym = self.scatter_trilist_to_matrix(edges_features_sym)
        edges_features_sym = edges_features_sym + edges_features_sym.permute(0, 2, 1, 3)

        # Get a matrix representation out of the upper triangular representation of the chirality edge features
        edges_features_dir = edges_features_noisy[
            ..., :3
        ]  # Chirality-related (i.e. non-symmetric) edge features
        edges_features_dir = self.scatter_trilist_to_matrix(edges_features_dir)

        if self.dir_handling:
            edges_features_dir_negative = edges_features_dir[..., [0, 2, 1]]
            edges_features_dir = (
                edges_features_dir + edges_features_dir_negative.permute(0, 2, 1, 3)
            )

        # Flatten the matrix representation for use in the neural processing layers
        edges_features_sym_flat = edges_features_sym.reshape(B, N**2, -1)
        edges_features_dir_flat = edges_features_dir.reshape(B, N**2, -1)

        if edges_features_cond is not None:
            edges_features_sym_cond = edges_features_cond[
                ..., 3:
            ]  # Symmetric edge features
            edges_features_sym_cond = self.scatter_trilist_to_matrix(
                edges_features_sym_cond
            )
            edges_features_sym_cond = (
                edges_features_sym_cond + edges_features_sym_cond.permute(0, 2, 1, 3)
            )

            # Get a matrix representation out of the upper triangular representation of the chirality edge features
            edges_features_dir_cond = edges_features_cond[
                ..., :3
            ]  # Chirality-related (i.e. non-symmetric) edge features
            edges_features_dir_cond = self.scatter_trilist_to_matrix(
                edges_features_dir_cond
            )
            if self.dir_handling:
                edges_features_dir_negative_cond = edges_features_dir_cond[
                    ..., [0, 2, 1]
                ]
                edges_features_dir_cond = (
                    edges_features_dir_cond
                    + edges_features_dir_negative_cond.permute(0, 2, 1, 3)
                )

            # Flatten the matrix representation for use in the neural processing layers
            edges_features_sym_flat_cond = edges_features_sym_cond.reshape(B, N**2, -1)
            edges_features_dir_flat_cond = edges_features_dir_cond.reshape(B, N**2, -1)

        # Prepare Potitional and Temporal Embeddings
        t_emb = self.t_emb[diffusion_timesteps]  # [B, embedding_dim]
        if self.p_emb_dim > 0:
            p_emb = (
                self.p_emb.clone().unsqueeze(0).expand(B, -1, -1)
            )  # [B, N, embedding_dim]
        else:
            p_emb = None

        # Node feature preprocessing
        feature_index = 0
        updated_node_features = None  # [B, N, F]
        updated_node_features_cond = None  # [B, N, F]
        for i, node_feature_index in enumerate(self.node_features_struct):
            input_node_features = nodes_features_noisy[
                ..., feature_index : feature_index + node_feature_index
            ]
            if updated_node_features is None:
                updated_node_features = self.node_input_layers[i](input_node_features)
            else:
                updated_node_features += self.node_input_layers[i](input_node_features)

            if nodes_features_cond is not None:
                input_node_features_cond = nodes_features_cond[
                    ..., feature_index : feature_index + node_feature_index
                ]
                if updated_node_features_cond is None:
                    updated_node_features_cond = self.node_input_layers[i](
                        input_node_features_cond
                    )
                else:
                    updated_node_features_cond += self.node_input_layers[i](
                        input_node_features_cond
                    )

            feature_index += node_feature_index

        # Edge feature preprocessing
        feature_index = 0
        updated_edge_features_dir_flat = self.edge_input_layers[0](
            edges_features_dir_flat
        )  # [B, N^2, F]
        updated_edge_features_sym_flat = torch.zeros_like(
            updated_edge_features_dir_flat
        )  # [B, N^2, F]

        if edges_features_cond is not None:
            updated_edge_features_dir_flat_cond = self.edge_input_layers[0](
                edges_features_dir_flat_cond
            )  # [B, N^2, F]
            updated_edge_features_sym_flat_cond = torch.zeros_like(
                updated_edge_features_dir_flat_cond
            )  # [B, N^2, F]

        for i, edge_feature_index in enumerate(self.edge_features_struct[1:]):
            input_edge_features = edges_features_sym_flat[
                ..., feature_index : feature_index + edge_feature_index
            ]
            updated_edge_features_sym_flat += self.edge_input_layers[i + 1](
                input_edge_features
            )

            if edges_features_cond is not None:
                input_edge_features_cond = edges_features_sym_flat_cond[
                    ..., feature_index : feature_index + edge_feature_index
                ]
                updated_edge_features_sym_flat_cond += self.edge_input_layers[i + 1](
                    input_edge_features_cond
                )

            feature_index += edge_feature_index

        updated_edge_features_flat = (
            updated_edge_features_sym_flat + updated_edge_features_dir_flat
        )  # [B, N^2, F]
        updated_edges_features_flat_cond = None
        if edges_features_cond is not None:
            updated_edges_features_flat_cond = (
                updated_edge_features_sym_flat_cond
                + updated_edge_features_dir_flat_cond
            )  # [B, N^2, F]
        edge_feature_sym_list = [updated_edge_features_sym_flat]

        node_feature_list = [updated_node_features]
        edge_feature_list = [updated_edge_features_flat]

        # Noisy features propagation in the denoising layers
        for i in range(len(self.hidden_graph_layers)):
            # Hidden graph layers
            (
                updated_node_features,
                updated_edge_features_flat,
            ) = self.hidden_graph_layers[i](
                updated_node_features,
                updated_edge_features_flat,
                p_emb=p_emb,
                t_emb=t_emb,
                # node_features_cond=updated_node_features_cond,
                # edge_features_flat_cond=updated_edges_features_flat_cond,
            )
            node_feature_list.append(updated_node_features)
            edge_feature_list.append(updated_edge_features_flat)

            # Make sure that edge features matrix is symmetric (i.e. complete graph with diagonal adjacency matrix)
            if self.sym_sync:
                updated_edge_features = updated_edge_features_flat.reshape(B, N, N, -1)
                updated_edge_features = 0.5 * (
                    updated_edge_features + updated_edge_features.permute(0, 2, 1, 3)
                )
                updated_edge_features_flat = updated_edge_features.reshape(B, N**2, -1)
                edge_feature_sym_list.append(updated_edge_features_flat)

        # Hidden concatenation layers
        updated_node_features = torch.cat(node_feature_list, dim=-1)
        edge_feature_list = torch.cat(edge_feature_list, dim=-1)  # B, N, N,ALL EF
        updated_node_features = self.node_concat_layer(updated_node_features)
        edge_feature_list = self.edge_concat_layer(edge_feature_list).reshape(
            B, N, N, -1
        )
        edge_feature_dir = edge_feature_list
        if self.dir_handling:
            if self.sym_sync:
                edge_feature_sym = torch.cat(edge_feature_sym_list, dim=-1)
                edge_feature_sym = self.edge_concat_layer_symsync(
                    edge_feature_sym
                ).reshape(B, N, N, -1)
            else:
                edge_feature_sym = (
                    edge_feature_list + edge_feature_list.permute(0, 2, 1, 3)
                ) / 2.0
        else:
            edge_feature_sym = edge_feature_list

        gather_ind = self.tri_ind_to_full_ind.to(nodes_features.device)[None, :, None]
        edge_feature_dir = torch.gather(
            edge_feature_dir.reshape(B, N**2, -1),
            1,
            gather_ind.expand(B, -1, edge_feature_dir.shape[-1]),
        )
        edge_feature_sym = torch.gather(
            edge_feature_sym.reshape(B, N**2, -1),
            1,
            gather_ind.expand(B, -1, edge_feature_sym.shape[-1]),
        )

        # Output layers
        feature_index = 0
        node_feature_pred = []
        for i in range(len(self.node_features_struct)):
            node_feature_pred.append(
                self.node_output_layers[i](
                    torch.cat(
                        [
                            updated_node_features,
                            nodes_features[
                                ...,
                                feature_index : feature_index
                                + self.node_features_struct[i],
                            ],
                        ],
                        dim=-1,
                    )
                )
            )
            feature_index += self.node_features_struct[i]

        node_feature_pred = torch.cat(node_feature_pred, -1)
        edge_feature_pred = [
            self.edge_output_layers[0](
                torch.cat(
                    [
                        edge_feature_dir,
                        edges_features_noisy[..., : self.edge_features_struct[0]],
                    ],
                    dim=-1,
                )
            )
        ]

        feature_index = self.edge_features_struct[0]
        for i in range(1, len(self.edge_features_struct)):
            edge_feature_pred.append(
                self.edge_output_layers[i](
                    torch.cat(
                        [
                            edge_feature_sym,
                            edges_features_noisy[
                                ...,
                                feature_index : feature_index
                                + self.edge_features_struct[i],
                            ],
                        ],
                        dim=-1,
                    )
                )
            )
            feature_index += self.edge_features_struct[i]

        edge_feature_pred = torch.cat(edge_feature_pred, dim=-1)

        return node_feature_pred, edge_feature_pred
