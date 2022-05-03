#!/usr/bin/env python
# coding: utf-8
import os
import os.path
import pickle
import random
import sys
from io import BytesIO

from datetime import datetime

from pathlib import Path

from pipeline.ml_models.neural_net import *
from pipeline.ml_models.split_data import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def create_temp_folder(results_folder,
                       encoding_file,
                       embedding_file,
                       dataset_nme,
                       job_id,
                       job_type="Reg",
                       split_type=1,
                       screen_bool=False,
                       log_value=True,
                       classification_threshold_type=2):
    """
    Create temp folder for saving results

    Parameters
    ----------
    results_folder : str
        Path to the folder where results will be saved
    encoding_file : str
        Path to the file containing the substrate encoding
    embedding_file : str
        Path to the file containing the sequence embedding
    dataset_nme : str
        Name of the dataset
    job_id : str
        Job id
    job_type : str
        Type of job (Reg or Clf)
    split_type : int
        Type of split (0 = train/test/split completely randomly selected;
        1 = train/test/split looks at different seq-subs pairs;
        2 = train/test/split looks at different seqs;
        train/test/split looks at different subs)
    screen_bool : bool
        Default is False
    log_value : bool
        Default is True
    classification_threshold_type : int
        Type of classification threshold (2 = 1e-5, 3 = 1e-2)
    """
    print(">>>>> Creating temporary subfolder and clear past empty folders! <<<<<")
    now = datetime.now()
    d_t_string = now.strftime("%m%d-%H%M%S")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_folder_contents = os.listdir(results_folder)
    for item in results_folder_contents:
        if os.path.isdir(results_folder / item):
            try:
                os.rmdir(results_folder / item)
                print("Remove empty folder " + item + "!")
            except:
                print("Found Non-empty folder " + item + "!")
    # ====================================================================================================#
    encoding_code = encoding_file.replace("X02_" + dataset_nme + "_", "")
    encoding_code = encoding_code.replace("_encodings_dict.p", "")
    # ====================================================================================================#
    embedding_code = embedding_file.replace("X03_" + dataset_nme + "_embedding_", "")
    embedding_code = embedding_code.replace(".p", "")
    # ====================================================================================================#
    temp_folder_name = job_id
    temp_folder_name += d_t_string + "_"
    temp_folder_name += dataset_nme + "_"
    temp_folder_name += embedding_code.replace("_", "") + "_"
    temp_folder_name += encoding_code + "_"
    temp_folder_name += job_type.upper() + "_"
    temp_folder_name += "split" + str(split_type) + "_"
    temp_folder_name += "scrn" + str(screen_bool)[0] + "_"
    temp_folder_name += "log" + str(log_value)[0] + "_"
    temp_folder_name += "thrhld" + str(classification_threshold_type)
    # ====================================================================================================#
    results_sub_folder = Path("X_DataProcessing/" + job_id + "intermediate_results/" + temp_folder_name + "/")
    if not os.path.exists(results_sub_folder):
        os.makedirs(results_sub_folder)
    print(">>>>> Temporary subfolder created! <<<<<")
    return results_sub_folder


# Modify the print function AND print all interested info.
class Tee(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


def output_print(results_sub_folder,
                 embedding_file,
                 encoding_file,
                 log_value,
                 screen_bool,
                 classification_threshold_type,
                 split_type,
                 epoch_num,
                 batch_size,
                 learning_rate,
                 nn_type,
                 hid_dim, kernel_1, out_dim, kernel_2, last_hid, dropout):
    orig_stdout = sys.stdout
    f = open(results_sub_folder / 'print_out.txt', 'w')
    sys.stdout = Tee(sys.stdout, f)

    print("=" * 50)

    print("embedding_file                : ", embedding_file)
    print("encoding_file                 : ", encoding_file)

    print("log_value                     : ", log_value, " (Whether to use log values of Y.)")
    print("screen_bool                   : ", screen_bool, " (Whether to remove zeroes.)")
    print("classification_threshold_type : ", classification_threshold_type, " (2: 1e-5, 3: 1e-2)")
    # --------------------------------------------------#
    print("split_type                    : ", split_type)
    # --------------------------------------------------#
    print("epoch_num                     : ", epoch_num)
    print("batch_size                    : ", batch_size)
    print("learning_rate                 : ", learning_rate)
    print("NN_type                       : ", nn_type)
    # --------------------------------------------------#
    print("-" * 50)
    for i in ["hid_dim", "kernel_1", "out_dim", "kernel_2", "last_hid", "dropout"]:
        print(i, ": ", locals()[i])
    print("-" * 50)
    return


def prep_neural_net_from_byte_input(encoding_file: bytes, embedding_file, screen_bool, log_value, split_type, nn_type,
                                    classification_threshold_type):
    x_subs_dict = pickle.load(BytesIO(encoding_file))
    x_seqs_dict = pickle.load(BytesIO(embedding_file))
    return prep_neural_net(x_subs_dict, x_seqs_dict, screen_bool, log_value, split_type, nn_type,
                           classification_threshold_type)


def prep_neural_net_from_local_file(data_folder: Path, encoding_filename: str, embedding_filename, screen_bool,
                                    log_value, split_type, nn_type, classification_threshold_type):
    subs = open(data_folder / encoding_filename, "rb")
    seqs = open(data_folder / embedding_filename, "rb")
    subs_dict = pickle.load(subs)
    seqs_dict = pickle.load(seqs)
    X_tr_seqs, X_tr_subs, y_tr, X_ts_seqs, X_ts_subs, y_ts, X_va_seqs, X_va_subs, y_va, \
        X_seqs_all_hiddens_dim, X_subs_encodings_dim, seqs_max_len, NN_input_dim, X_seqs_num, X_subs_num = \
        prep_neural_net(subs_dict, seqs_dict, screen_bool, log_value, split_type, nn_type, classification_threshold_type)
    subs.close()
    seqs.close()
    return X_tr_seqs, X_tr_subs, y_tr, X_ts_seqs, X_ts_subs, y_ts, X_va_seqs, X_va_subs, y_va, \
        X_seqs_all_hiddens_dim, X_subs_encodings_dim, seqs_max_len, NN_input_dim, X_seqs_num, X_subs_num


def prep_neural_net(x_subs_dict: Dict,
                    x_seqs_dict: Dict,
                    screen_bool,
                    log_value,
                    split_type,
                    nn_type,
                    classification_threshold_type):
    """
    Prepare Train/Test/Validation Dataset for NN model.
    """
    x_seqs_all_hidden_list = x_seqs_dict['seq_all_hiddens']

    X_seqs_num = len(x_seqs_all_hidden_list)
    X_subs_num = len(x_subs_dict)
    subs_properties_list = []
    subs_num = 0
    for one_smiles in x_subs_dict.keys():
        subs_properties_list.append(
            ["substrate_name", np.array([subs_num * 100 + seq_num for seq_num in range(X_seqs_num)]),
             np.zeros(X_seqs_num), np.zeros(X_seqs_num), one_smiles])
        subs_num += 1

    # Get embeddings and encodings
    x_seqs_all_hidden, x_subs_encoding, y_data, seqs_subs_idx_book = None, None, None, None
    if nn_type == "Reg":
        x_seqs_all_hidden, x_subs_encoding, y_data, seqs_subs_idx_book = \
            get_represented_xy_data(x_seqs_all_hidden_list,
                                    subs_properties_list,
                                    x_subs_dict,
                                    screen_bool,
                                    classification_threshold_type)
    elif nn_type == "Clf":
        x_seqs_all_hidden, x_subs_encoding, y_data, seqs_subs_idx_book = \
            get_represented_xy_data_clf(x_seqs_all_hidden_list,
                                        subs_properties_list,
                                        x_subs_dict,
                                        screen_bool,
                                        classification_threshold_type)

    # Save seqs_embeddings and subs_encodings

    print("len(X_seqs_all_hiddens): ", len(x_seqs_all_hidden), ", len(X_subs_encodings): ", len(x_subs_encoding),
          ", len(y_data): ", len(y_data))

    # Get size of some interested parameters.
    X_seqs_all_hiddens_dim = [max([x_seqs_all_hidden_list[i].shape[0] for i in range(len(x_seqs_all_hidden_list))]),
                              x_seqs_all_hidden_list[0].shape[1], ]
    X_subs_encodings_dim = len(x_subs_encoding[0])
    X_seqs_num = len(x_seqs_all_hidden_list)
    X_subs_num = len(x_subs_dict)
    print("seqs, subs dimensions: ", X_seqs_all_hiddens_dim, ", ", X_subs_encodings_dim)
    print("seqs, subs counts: ", X_seqs_num, ", ", X_subs_num)

    seqs_max_len = max([x_seqs_all_hidden_list[i].shape[0] for i in range(len(x_seqs_all_hidden_list))])
    print("seqs_max_len: ", seqs_max_len)

    NN_input_dim = X_seqs_all_hiddens_dim[1]
    print("NN_input_dim: ", NN_input_dim)

    # Get Separate SEQS index and SUBS index.
    tr_idx_seqs, ts_idx_seqs, va_idx_seqs = [], [i for i in range(X_seqs_num)], []
    tr_idx_subs, ts_idx_subs, va_idx_subs = [], [i for i in range(X_subs_num)], []
    print("len(tr_idx_seqs): ", len(tr_idx_seqs))
    print("len(ts_idx_seqs): ", len(ts_idx_seqs))
    print("len(va_idx_seqs): ", len(va_idx_seqs))

    # Get splitted index of the entire combined dataset.
    X_train_idx, X_test_idx, X_valid_idx = split_seqs_subs_idx_book(tr_idx_seqs, ts_idx_seqs, va_idx_seqs, tr_idx_subs,
                                                                    ts_idx_subs, va_idx_subs, y_data,
                                                                    seqs_subs_idx_book, split_type)
    dataset_size = len(seqs_subs_idx_book)
    print("dataset_size: ", dataset_size)
    # Get splitted data of the combined dataset using the splitted index.
    X_tr_seqs, X_tr_subs, y_tr = get_selected_xy_data(X_train_idx, x_seqs_all_hidden, x_subs_encoding, y_data,
                                                      log_value)
    X_ts_seqs, X_ts_subs, y_ts = get_selected_xy_data(X_test_idx, x_seqs_all_hidden, x_subs_encoding, y_data,
                                                      log_value)
    X_va_seqs, X_va_subs, y_va = get_selected_xy_data(X_valid_idx, x_seqs_all_hidden, x_subs_encoding, y_data,
                                                      log_value)

    print("X_tr_seqs_dimension: ", len(X_tr_seqs), ", X_tr_subs_dimension: ", len(X_tr_subs), ", y_tr_dimension: ",
          y_tr.shape)
    print("X_ts_seqs_dimension: ", len(X_ts_seqs), ", X_ts_subs_dimension: ", len(X_ts_subs), ", y_ts_dimension: ",
          y_ts.shape)
    print("X_va_seqs_dimension: ", len(X_va_seqs), ", X_va_subs_dimension: ", len(X_va_subs), ", y_va_dimension: ",
          y_va.shape)

    return X_tr_seqs, X_tr_subs, y_tr, X_ts_seqs, X_ts_subs, y_ts, X_va_seqs, X_va_subs, y_va, \
        X_seqs_all_hiddens_dim, X_subs_encodings_dim, seqs_max_len, NN_input_dim, X_seqs_num, X_subs_num

