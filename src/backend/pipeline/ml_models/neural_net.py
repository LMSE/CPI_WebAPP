#!/usr/bin/env python
# coding: utf-8

from typing import List, Tuple, Dict, Any

# Numpy
import numpy as np

# PyTorch imports
import torch
from torch import nn
from torch.utils import data


class NeuralNetDataset(data.Dataset):
    def __init__(self, embedding, substrate, target, max_len):
        super().__init__()
        self.embedding = embedding
        self.substrate = substrate
        self.target = target
        self.max_len = max_len

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        return self.embedding[idx], self.substrate[idx], self.target[idx]

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        embedding, substrate, target = zip(*batch)
        batch_size = len(embedding)
        emb_dim = embedding[0].shape[1]
        arra = np.full([batch_size, self.max_len, emb_dim], 0.0)
        for arr, seq in zip(arra, embedding):
            arrslice = tuple(slice(dim) for dim in seq.shape)
            arr[arrslice] = seq
        return {'seqs_embeddings': torch.from_numpy(arra), 'subs_encodings': torch.tensor(list(substrate)),
                'y_property': torch.tensor(list(target))}


def generate_cnn_loader(train_seqs, train_subs, y_train,
                        valid_seqs, valid_subs, y_valid,
                        test_seqs, test_subs, y_test,
                        seqs_max_len, batch_size):
    """
    Generate a data loader for the model.
    """
    train_set = NeuralNetDataset(list(train_seqs), list(train_subs), y_train, seqs_max_len)
    valid_set = NeuralNetDataset(list(valid_seqs), list(valid_subs), y_valid, seqs_max_len)
    test_set = NeuralNetDataset(list(test_seqs), list(test_subs), y_test, seqs_max_len)
    train_loader = []  # train_loader = data.DataLoader(X_y_tr, batch_size, True,  collate_fn=X_y_tr.collate_fn)
    valid_loader = []  # valid_loader = data.DataLoader(X_y_va, batch_size, False, collate_fn=X_y_va.collate_fn)
    test_loader = data.DataLoader(test_set, batch_size, False, collate_fn=test_set.collate_fn)
    return train_loader, valid_loader, test_loader


class LoaderClass(data.Dataset):
    """
    Class for loading data to torch
    """
    def __init__(self, seqs_embeddings, subs_encodings, y_property):
        super(LoaderClass, self).__init__()
        self.seqs_embeddings = seqs_embeddings
        self.subs_encodings = subs_encodings
        self.y_property = y_property

    def __len__(self):
        return self.seqs_embeddings.shape[0]

    def __getitem__(self, idx):
        return self.seqs_embeddings[idx], self.subs_encodings[idx], self.y_property[idx]

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        seqs_embeddings, subs_encodings, y_property = zip(*batch)
        batch_size = len(seqs_embeddings)
        seqs_embeddings_dim = seqs_embeddings[0].shape[1]
        return {'seqs_embeddings': torch.tensor(seqs_embeddings), 'subs_encodings': torch.tensor(list(subs_encodings)),
                'y_property': torch.tensor(list(y_property))}


class NeuralNet(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 kernel_1: int,
                 out_dim: int,
                 kernel_2: int,
                 max_len: int,
                 sub_dim: int,
                 last_hid: int,
                 dropout: float = 0.
                 ):
        super().__init__()

        self.norm1 = nn.BatchNorm1d(max_len)
        self.conv1 = nn.Conv1d(max_len, hid_dim, kernel_1, padding=int((kernel_1 - 1) / 2))
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        # --------------------------------------------------#
        self.norm2 = nn.BatchNorm1d(in_dim)
        self.conv2 = nn.Conv1d(in_dim, hid_dim, kernel_1, padding=int((kernel_1 - 1) / 2))
        self.dropout2 = nn.Dropout(dropout, inplace=True)
        # --------------------------------------------------#
        self.conv2_1 = nn.Conv1d(hid_dim, out_dim, kernel_2, padding=int((kernel_2 - 1) / 2))
        self.dropout2_1 = nn.Dropout(dropout, inplace=True)
        # --------------------------------------------------#
        self.conv2_2 = nn.Conv1d(hid_dim, hid_dim, kernel_2, padding=int((kernel_2 - 1) / 2))
        self.dropout2_2 = nn.Dropout(dropout, inplace=True)
        # --------------------------------------------------#
        self.fc_early = nn.Linear(max_len * hid_dim + sub_dim, 1)
        # --------------------------------------------------#
        self.conv3 = nn.Conv1d(hid_dim, out_dim, kernel_2, padding=int((kernel_2 - 1) / 2))
        self.dropout3 = nn.Dropout(dropout, inplace=True)
        # self.pooling = nn.MaxPool1d(3, stride=3,padding=1)
        # --------------------------------------------------#
        self.fc_1 = nn.Linear(int((in_dim + max_len) * out_dim + sub_dim), last_hid)
        self.fc_2 = nn.Linear(last_hid, last_hid)
        self.fc_3 = nn.Linear(last_hid, 1)
        self.cls = nn.Sigmoid()

    def forward(self, enc_inputs, substrate):
        output_1 = enc_inputs
        output_1 = nn.functional.relu(self.conv1(self.norm1(output_1)))
        output_1 = self.dropout1(output_1)

        output_1 = nn.functional.relu(self.conv2_1(output_1))
        output_1 = self.dropout2_1(output_1)

        output_2 = enc_inputs.transpose(1, 2)
        output_2 = nn.functional.relu(self.conv2(self.norm2(output_2)))
        output_2 = self.dropout2(output_2)

        output_2 = nn.functional.relu(self.conv2_2(output_2)) + output_2
        output_2 = self.dropout2_2(output_2)

        single_conv = torch.cat((torch.flatten(output_2, 1), substrate), 1)
        single_conv = self.cls(self.fc_early(single_conv))

        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)

        output = torch.cat((torch.flatten(output_1, 1), torch.flatten(output_2, 1)), 1)

        output = torch.cat((torch.flatten(output, 1), substrate), 1)

        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)
        return output, single_conv

