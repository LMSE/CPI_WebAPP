#!/usr/bin/env python
# coding: utf-8
import numpy as np
from sklearn.model_selection import train_test_split


def get_represented_xy_data(x_seqs_all_hidden_list, subs_properties_list, x_seqs_encodings_dict, screen_bool,
                            classification_threshold_type):
    """
    Get the x-y data from the representation
    """
    # new: x_seqs_all_hidden_list, old: seq_all_hidden_list
    # subs_properties_list: [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    x_seqs_all_hidden = []
    x_seqs_encodings = []
    y_data = []
    seqs_subs_idx_book = []
    print("len(x_seqs_all_hidden_list): ", len(x_seqs_all_hidden_list))
    for i in range(len(subs_properties_list)):
        for j in range(len(x_seqs_all_hidden_list)):
            x_smiles_rep = subs_properties_list[i][-1]  # list of SMILES
            x_one_all_hidden = x_seqs_all_hidden_list[j]
            if not (screen_bool == True and subs_properties_list[i][classification_threshold_type][j] == False):
                # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
                # subs_properties_list[i] ----> [one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
                # subs_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
                # subs_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
                # subs_properties_list[i][1][j] ----> y_prpty_reg[j]
                x_seqs_all_hidden.append(x_one_all_hidden)
                x_seqs_encodings.append(
                    x_seqs_encodings_dict[x_smiles_rep])  # x_seqs_encodings.append(x_seqs_encodings_dict[x_smiles_rep])
                y_data.append(subs_properties_list[i][1][j])
                seqs_subs_idx_book.append([j, i])  # [seqs_idx, subs_idx]
    return x_seqs_all_hidden, x_seqs_encodings, y_data, seqs_subs_idx_book


def get_represented_xy_data_clf(x_seqs_all_hidden_list, subs_properties_list, x_seqs_encodings_dict, screen_bool,
                                classification_threshold_type):
    """
    Get the x-y data from the representation for classification
    """
    # new: x_seqs_all_hidden_list, old: seq_all_hidden_list
    # subs_properties_list: [[one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES], [], [], ...[] ]
    x_seqs_all_hidden = []
    x_seqs_encodings = []
    y_data = []
    seqs_subs_idx_book = []
    print("len(x_seqs_all_hidden_list): ", len(x_seqs_all_hidden_list))
    for i in range(len(subs_properties_list)):
        for j in range(len(x_seqs_all_hidden_list)):
            x_smiles_rep = subs_properties_list[i][-1]  # list of SMILES
            x_one_all_hidden = x_seqs_all_hidden_list[j]
            # classification_threshold_type ----> # 2: Threshold = 1e-5 | 3: Threshold = 1e-2 (from Original CSV File)
            # subs_properties_list[i] ----> [one_substrate, y_prpty_reg, y_prpty_cls, y_prpty_cls2, SMILES]
            # subs_properties_list[i][classification_threshold_type] ----> y_prpty_cls OR y_prpty_cls2
            # subs_properties_list[i][classification_threshold_type][j] ----> y_prpty_cls[j] OR y_prpty_cls2[j]
            x_seqs_all_hidden.append(x_one_all_hidden)
            x_seqs_encodings.append(x_seqs_encodings_dict[x_smiles_rep])
            y_data.append(subs_properties_list[i][classification_threshold_type][j])
            seqs_subs_idx_book.append([j, i])  # [seqs_idx, subs_idx]
    return x_seqs_all_hidden, x_seqs_encodings, y_data, seqs_subs_idx_book


# ====================================================================================================#
def split_idx(x_num, train_split, test_split, random_state=42):
    # x_seqs_idx = y_seqs_idx = list(range(len(x_seqs_all_hidden_list)))
    # x_seqs_idx = y_subs_idx = list(range(len(subs_properties_list)))
    x_idx = y_idx = list(range(x_num))
    x_tr_idx, x_ts_idx, y_tr_idx, y_ts_idx = train_test_split(x_idx, y_idx, test_size=(1 - train_split),
                                                              random_state=42)
    x_va_idx, x_ts_idx, y_va_idx, y_ts_idx = train_test_split(x_ts_idx, y_ts_idx,
                                                              test_size=(test_split / (1.0 - train_split)),
                                                              random_state=42)
    return x_tr_idx, x_ts_idx, x_va_idx


# ====================================================================================================#
def split_seqs_idx_custom(x_num, customized_idx_list, valid_test_split=0.5, random_state=42):
    x_idx = y_idx = list(range(x_num))
    x_tr_idx = [idx for idx in x_idx if (idx not in customized_idx_list)]
    # --------------------------------------------------#
    x_ts_idx = y_ts_idx = customized_idx_list
    x_va_idx, x_ts_idx, y_va_idx, y_ts_idx = train_test_split(x_ts_idx, y_ts_idx, test_size=valid_test_split,
                                                              random_state=42)
    return x_tr_idx, x_ts_idx, x_va_idx


# ====================================================================================================#
def split_seqs_subs_idx_book(tr_idx_seqs, ts_idx_seqs, va_idx_seqs,
                             tr_idx_subs, ts_idx_subs, va_idx_subs,
                             y_data, seqs_subs_idx_book, split_type):
    # --------------------------------------------------#
    # split_type = 0, train/test/split completely randomly selected
    # split_type = 1, train/test/split looks at different seq-subs pairs
    # split_type = 2, train/test/split looks at different seqs
    # split_type = 3, train/test/split looks at different subs
    # --------------------------------------------------#
    tr_idx, ts_idx, va_idx = [], [], []
    if split_type == 0:
        dataset_size = len(seqs_subs_idx_book)
        x_data_idx = np.array(list(range(dataset_size)))
        tr_idx, ts_idx, y_train, y_test = train_test_split(x_data_idx, y_data, test_size=0.3, random_state=42)
        va_idx, ts_idx, y_valid, y_test = train_test_split(ts_idx, y_test, test_size=0.6667, random_state=42)
    for one_pair_idx in seqs_subs_idx_book:
        if split_type == 1:
            if one_pair_idx[0] in tr_idx_seqs and one_pair_idx[1] in tr_idx_seqs:
                tr_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in va_idx_seqs and one_pair_idx[1] in va_idx_seqs:
                va_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in ts_idx_seqs and one_pair_idx[1] in ts_idx_seqs:
                ts_idx.append(seqs_subs_idx_book.index(one_pair_idx))
        if split_type == 2:
            if one_pair_idx[0] in tr_idx_seqs:
                tr_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in va_idx_seqs:
                va_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[0] in ts_idx_seqs:
                ts_idx.append(seqs_subs_idx_book.index(one_pair_idx))
        if split_type == 3:
            if one_pair_idx[1] in tr_idx_seqs:
                tr_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[1] in va_idx_seqs:
                va_idx.append(seqs_subs_idx_book.index(one_pair_idx))
            if one_pair_idx[1] in ts_idx_seqs:
                ts_idx.append(seqs_subs_idx_book.index(one_pair_idx))
    return tr_idx, ts_idx, va_idx


# ====================================================================================================#
def get_selected_xy_data(x_idx, x_seqs_all_hidden, x_subs_encodings, y_data, log_value):
    x_seqs_selected = []
    x_subs_selected = []
    y_data_selected = []
    for idx in x_idx:
        # print(y_data[idx])
        if y_data[idx] is None:
            continue
        # print(y_data[idx])
        x_seqs_selected.append(x_seqs_all_hidden[idx])
        x_subs_selected.append(x_subs_encodings[idx])
        y_data_selected.append(y_data[idx])
    x_seqs_selected = x_seqs_selected
    x_subs_selected = x_subs_selected
    y_data_selected = np.array(y_data_selected)
    if log_value:
        y_data_selected = np.log10(y_data_selected)
    return x_seqs_selected, x_subs_selected, y_data_selected
