import numpy as np
import torch
from torch_geometric.nn import GATConv
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from torch_geometric.data import Data


def obb_str_to_arr(size_npz):
    return np.array(size_npz.strip().split(" ")).astype(float)


def get_graph_bb(
    graph,
    part_id: int,
    opt_bb_mode: str,
    mode_global_graph: bool,
    bb_attr_name: str = "bbox",
):
    """
    creates a torch geometric Data input from a dictionary with nodes and edges
    as well as the id of the current part
    """
    # get edges - (global: all, local: the ones that are connected to the part node)
    pluckers, edge_idx, node_list = [], [], [part_id]
    node_idx = 0
    for src, dst, e_data in graph.edges(data=True):
        # for both directions
        if mode_global_graph:
            # global method: append all of the edges (bidir) & pluckers as edge attr
            pluckers.append(e_data["plucker"])
            edge_idx.append([src, dst])

            pluckers.append(e_data["plucker"] * (-1))
            edge_idx.append([dst, src])
        elif src == part_id:
            # local method: only append edges adjacent to this part.
            pluckers.append(e_data["plucker"])
            edge_idx.append(
                [0, node_idx + 1]
            )  # current part is always at zero, other nodes are appended
            node_list.append(dst)
            node_idx += 1
        elif dst == part_id:
            pluckers.append(e_data["plucker"] * (-1))
            edge_idx.append([0, node_idx + 1])
            node_list.append(src)
            node_idx += 1

    pluckers = np.array(pluckers)
    edge_index = torch.from_numpy(np.swapaxes(np.array(edge_idx), 1, 0)).long()

    # get node features - by default, node feats are bb and edge feats are pluckers
    if "plucker" in opt_bb_mode:  # new mode option, e.g. graph_local_plucker
        assert (
            not mode_global_graph
        ), "pluckers can only be used as node attributes with local method"
        # if everything is centered and scaled: condition only on the pluckers and take them as node attr
        node_attr = torch.from_numpy(pluckers).float()
        edge_attr = None
    else:
        if mode_global_graph:
            # overwrite node list for global graph method:
            node_list = np.arange(graph.number_of_nodes())
        node_attr = torch.from_numpy(
            np.array([graph.nodes[node][bb_attr_name] for node in node_list])
        ).float()
        edge_attr = torch.from_numpy(pluckers).float()

    if mode_global_graph:
        # add indicator vector to node attributes --> current part has 1, else 0
        node_ind_vec = torch.zeros(node_attr.size()[0])
        node_ind_vec[part_id] = 1
        node_attr = torch.cat((node_ind_vec.unsqueeze(1), node_attr), dim=1).float()

    # print("edge index", edge_index)
    # print("node attr", node_attr)
    # print(edge_index.size(), edge_attr.size(), node_attr.size())
    # print()
    bb = Data(x=node_attr.float(), edge_index=edge_index, edge_attr=edge_attr)
    return bb


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_classes):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6
        )
        self.lin = Linear(out_channels, num_classes)

    def forward(self, data):
        # extract data
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)

        # mean pooling over nodes
        x = global_mean_pool(x, batch)

        # apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x
