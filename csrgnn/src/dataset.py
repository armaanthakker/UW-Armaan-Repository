import pickle
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
from torch_geometric.data import InMemoryDataset, Data


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        """
        Args:
            root: ''
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'test']
        self.phrase = phrase
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
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        
        for user, sequence, cue_l, y_l in tqdm(zip(data[0], data[1], data[2], data[3])):
            count = Counter([item for itemset in sequence for item in itemset])
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            # TODO 看看sequence_t是怎么用的
            sequence_t = [torch.tensor([0] * 7, dtype=torch.long)]
            itemset_len = []
            x = []
            for itemset in sequence:
                temp = []
                if itemset != []:
                    for node in itemset:
                        if node not in nodes:
                            nodes[node] = i
                            x.append([node])
                            i += 1
                        temp.append(nodes[node])
                    senders.append(temp)
                    itemset_len.append(len(temp))
                    sequence_t.append(torch.tensor(temp, dtype=torch.long))

            sequence = pad_sequence(sequence_t, batch_first=True, padding_value=i)[1:]
            # exclude sessions with length of 1 or less
            if len(sequence) < 2:
                print(user, sequence) 
                continue
            receivers = senders[1:] # the first itemset is not a receiver
            senders = senders[:-1] # the last itemset is not a sender
            #num_count = [count[i[0]] for i in x]

            pair = {}
            send_idx, receive_idx = [], [] # to contain the unique node pairs of senders and receivers
            send_flatten, receive_flatten = [], [] # flattened version of senders and receivers

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
            edge_count = torch.tensor(edge_count, dtype=torch.float)

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
