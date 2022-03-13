import pickle
from tqdm import tqdm
from rich import print
import shutil
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
from torch_geometric.data import InMemoryDataset, Data

PADDED_LENGTH = 25

class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None, **kwargs):
        """
        Args:
            root: ''
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
        self.kwargs = kwargs
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass

    def _process(self):
        if not self.kwargs.get('skip_preprocess', False) and self.phrase=='train':
            print(f'[bold magenta]REMOVING PROCESSED_DIR[/bold magenta]: {self.processed_dir}')
            shutil.rmtree(self.processed_dir, ignore_errors=True)  # remove the processed dir, reprocess the graph data everytime.
            
        super(MultiSessionsGraph, self)._process()
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        
        for user, sequence, cue_l, y_l in (zip(tqdm(data[0]), data[1], data[2], data[3])):
            count = Counter([item for itemset in sequence for item in itemset])
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}  # ReId nodes for each session graph
            senders = []  # same as `sequence_t[1:]`
            sequence_t = [torch.tensor([0] * PADDED_LENGTH, dtype=torch.long)]  # list of item sets. Each element is a set of ReIded node sharing same timestamp. First 0s for padding, deleted later.
            itemset_len = []  # length of each itemset
            itemset_len_simplify = [len(itemset) for itemset in sequence]
            x = []  # unique nodes in temporal order. Origin node ID.
            for itemset in sequence:
                temp = [] # set of nodes in same timestamp, ReId
                if itemset != []:
                    for node in itemset:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            i += 1
                        temp.append(nodes[node])
                    assert len(temp) == len(itemset)
                    senders.append(temp)
                    itemset_len.append(len(temp))
                    sequence_t.append(torch.tensor(temp, dtype=torch.long))
            assert itemset_len_simplify == itemset_len

            sequence = pad_sequence(sequence_t, batch_first=True, padding_value=i)[1:]
            # # XXX 是否删除timestamp太少的序列？
            # # exclude sessions with length of 1 or less
            # if len(sequence) < 2:
            #     print(user, sequence) 
            #     continue
            receivers = senders[1:] # the first itemset is not a receiver
            senders = senders[:-1] # the last itemset is not a sender
            #num_count = [count[i[0]] for i in x]

            pair = {}  # occurrence of the edge
            send_idx, receive_idx = [], [] # to contain the unique node pairs of senders and receivers  # ReIded sender and receiver for all unique edges
            send_flatten, receive_flatten = [], [] # flattened version of senders and receivers  # ReIded sender and receiver for all edges including duplications

            for sender_list, receiver_list in zip(senders, receivers):
                for sender in sender_list:
                    for receiver in receiver_list:
                        if str(sender) + '-' + str(receiver) in pair:
                            pair[str(sender) + '-' + str(receiver)] += 1
                        else:
                            pair[str(sender) + '-' + str(receiver)] = 1
                            send_idx.append(sender)
                            receive_idx.append(receiver)
                        send_flatten.append(sender)
                        receive_flatten.append(receiver)

            edge_index = torch.tensor([send_idx, receive_idx], dtype=torch.long)
            edge_count = [pair[str(send_idx[i]) + '-' + str(receive_idx[i])] for i in range(len(send_idx))]
            edge_count = torch.tensor(edge_count, dtype=torch.float)  # occurrence of each unique edge

            count = Counter(send_flatten)
            out_degree_inv = list(np.array(edge_count)/np.array([count[i] for i in send_idx]))

            count = Counter(receive_flatten)
            in_degree_inv = list(np.array(edge_count)/np.array([count[i] for i in receive_idx]))

            in_degree_inv = torch.tensor(in_degree_inv, dtype=torch.float)
            out_degree_inv = torch.tensor(out_degree_inv, dtype=torch.float)

            x = torch.tensor(x, dtype=torch.long) - 67
            #num_count = torch.tensor(num_count, dtype=torch.float)
            itemset_len = torch.tensor(itemset_len, dtype=torch.long)
            sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
            for i in range(len(cue_l)):
                cue = torch.tensor([cue_l[i]], dtype=torch.long) - 67
                y = torch.tensor([y_l[i]], dtype=torch.long)
                session_graph = Data(user=user, x=x, cue=cue, y=y, edge_index=edge_index,
                                     sequence=sequence, itemset_len=itemset_len, sequence_len=sequence_len,
                                     in_degree_inv=in_degree_inv, out_degree_inv=out_degree_inv)
                data_list.append(session_graph)
 
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
