import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from InOutGGNN import InOutGGNN
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv

PADDED_LENGTH = 8

class Embedding2Score(nn.Module):
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, node_embedding, embedding_table, batch, sequence, itemset_len, sequence_len, cue, device):
        sections = torch.bincount(batch)
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i, tuple(sections * hidden_size)
        v_i_zeros = tuple(torch.cat((nodes, torch.zeros(1, self.hidden_size).to(device)), dim=0) for nodes in v_i) # append row of zeros to each graph, tuple((sections + 1) * hidden_size)
        seq_i = torch.split(sequence, tuple(sequence_len.cpu().numpy())) # tuple(sequence_len * 3)
        itemset_len_i = torch.split(itemset_len, tuple(sequence_len.cpu().numpy())) # tuple(sequence_len,)
        # represent session as mean of items in each itemset of the sequence
        # TODO 看看这里reshape是为什么
        session_i = tuple(torch.sum(nodes[seq.reshape(-1)].reshape(-1, PADDED_LENGTH, self.hidden_size), dim=1) / itemset_len.view(-1, 1).repeat(1, self.hidden_size) \
                    for nodes, seq, itemset_len in zip(v_i_zeros, seq_i, itemset_len_i)) # tuple(sequence_len * hidden_size)

        # # represent session as mean of items in each itemset of the sequence
        # session_i = tuple(torch.cat(tuple(torch.mean(nodes[itemset], 0).unsqueeze(0) for itemset in seq), dim=0) for nodes, seq in zip(v_i, sequence))
        itemset_node_embedding = torch.cat(session_i, dim=0) # sum(sequence_len * hidden_size)
        v_n_repeat = tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in session_i)    # repeat |V|_i times for the last itemset node embedding

        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(itemset_node_embedding)))    # |V|_i * 1
        s_g_whole = alpha * itemset_node_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sequence_len.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in session_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        # Eq(8)
        y_hat = torch.sum(s_h * embedding_table(cue), dim=1)
        all_scores = torch.mm(s_h, embedding_table.weight.transpose(1, 0))
        
        return y_hat, all_scores

class Residual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.2)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out

class Embedding2ScoreSAN(nn.Module):
    def __init__(self, hidden_size, san_blocks=3):
        super(Embedding2ScoreSAN, self).__init__()
        self.hidden_size = hidden_size
        # self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.rn = Residual(self.hidden_size)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, 1).cuda()
        self.san_blocks = san_blocks

    def forward(self, node_embedding, embedding_table, batch, sequence, itemset_len, sequence_len, cue, device):
        sections = torch.bincount(batch)
        v_i = torch.split(node_embedding, tuple(sections.cpu().numpy()))    # split whole x back into graphs G_i, tuple(sections * hidden_size)
        v_i_zeros = tuple(torch.cat((nodes, torch.zeros(1, self.hidden_size).to(device)), dim=0) for nodes in v_i) # append row of zeros to each graph, tuple((sections + 1) * hidden_size)
        seq_i = torch.split(sequence, tuple(sequence_len.cpu().numpy())) # tuple(sequence_len * 3)
        itemset_len_i = torch.split(itemset_len, tuple(sequence_len.cpu().numpy())) # tuple(sequence_len,)
        # represent session as mean of items in each itemset of the sequence
        session_i = tuple(torch.sum(nodes[seq.reshape(-1)].reshape(-1, 3, self.hidden_size), dim=1) / itemset_len.view(-1, 1).repeat(1, self.hidden_size) \
                    for nodes, seq, itemset_len in zip(v_i_zeros, seq_i, itemset_len_i)) # tuple(sequence_len * hidden_size)

        # # represent session as mean of items in each itemset of the sequence
        # session_i = tuple(torch.cat(tuple(torch.mean(nodes[itemset], 0).unsqueeze(0) for itemset in seq), dim=0) for nodes, seq in zip(v_i, sequence))

        s_g = []
        for node_embs in session_i:
            attn_output = node_embs.unsqueeze(0)
            for k in range(self.san_blocks):
                attn_output, attn_output_weights = self.multihead_attn(attn_output, attn_output, attn_output)
                attn_output = self.rn(attn_output)
            s_g.append(attn_output[0: 1, -1])

        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in session_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        # Eq(8)
        y_hat = torch.sum(s_h * embedding_table(cue), dim=1)
        all_scores = torch.mm(s_h, embedding_table.weight.transpose(1, 0))
        
        return y_hat, all_scores


class GNNModel(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node, num_layers, use_san=False, use_gat=False):
        super(GNNModel, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.num_layers = num_layers
        self.use_san = use_san
        self.use_gat = use_gat
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        if not use_gat:
            self.gated = InOutGGNN(self.hidden_size, num_layers=self.num_layers)
        elif num_layers == 1:
            self.gat1 = GATConv(self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)
        elif num_layers == 2:
            self.gat1 = GATConv(self.hidden_size, self.hidden_size, heads=3, negative_slope=0.2)
            self.gat2 = GATConv(3 * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)
        if not use_san:
            self.e2s = Embedding2Score(self.hidden_size)
        else:
            self.e2s = Embedding2ScoreSAN(self.hidden_size)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        
    def forward(self, data, cue, device):
        x, edge_index, in_degree_inv, out_degree_inv, batch, sequence, itemset_len, sequence_len = \
            data.x, data.edge_index, data.in_degree_inv, data.out_degree_inv, data.batch, data.sequence, data.itemset_len, data.sequence_len
     
        embedding = self.embedding(x).squeeze()
        if not self.use_gat:
            hidden = self.gated(embedding, edge_index, [in_degree_inv, out_degree_inv])
        elif self.num_layers == 1:
            hidden = F.relu(self.gat1(embedding, edge_index))
        elif self.num_layers == 2:
            hidden = F.relu(self.gat1(embedding, edge_index))
            hidden = self.gat2(hidden, edge_index)
        
        return self.e2s(hidden, self.embedding, batch, sequence, itemset_len, sequence_len, cue, device)