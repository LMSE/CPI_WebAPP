from pipeline.ml_models.seqsub_neuralnet import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def predict(trained_model, seqs, subs, params):
    hid_dim = 256
    kernel_1 = 3
    out_dim = 1
    kernel_2 = 3
    last_hid = 1024

    train_seqs, train_subs, y_train, test_seqs, test_subs, y_test, valid_seqs, valid_subs, y_valid, \
        x_seqs_all_hidden_dim, x_subs_encoding_dim, _, nn_input_dim, x_seqs_num, x_subs_num = \
        prep_neural_net_from_local_file(data_folder=Path("../"),
                                        encoding_filename=subs,
                                        embedding_filename=seqs,
                                        screen_bool=False,
                                        log_value=True,
                                        split_type=1,
                                        nn_type='Reg',
                                        classification_threshold_type=2)
    seq_max_len = 1009
    nn_input_dim = 1280
    subs_dim = 731
    if "seq_max_len" in params:
        seq_max_len = params["seq_max_len"]
    if "nn_input_dim" in params:
        nn_input_dim = params["nn_input_dim"]
    if "subs_dim" in params:
        subs_dim = params["subs_dim"]

    train_loader, valid_loader, test_loader = \
        generate_cnn_loader(train_seqs,
                            train_subs,
                            y_train,
                            valid_seqs,
                            valid_subs,
                            y_valid,
                            test_seqs,
                            test_subs,
                            y_test,
                            seq_max_len,
                            64)
    model = NeuralNet(
        in_dim=nn_input_dim,
        hid_dim=hid_dim,
        kernel_1=kernel_1,
        out_dim=out_dim,  # 2
        kernel_2=kernel_2,
        max_len=seq_max_len,
        sub_dim=subs_dim,
        last_hid=last_hid,  # 256
        dropout=0.
    )
    model.double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model)

    checkpoint = torch.load(trained_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    y_pred = []
    for one_seqsubs_ppt_group in test_loader:
        seq_rep, subs_rep, target = one_seqsubs_ppt_group["seqs_embeddings"], one_seqsubs_ppt_group["subs_encodings"], \
                                    one_seqsubs_ppt_group["y_property"]
        if torch.cuda.is_available():
            seq_rep, subs_rep = seq_rep.double().cuda(), subs_rep.double().cuda()
        else:
            seq_rep, subs_rep = seq_rep.double(), subs_rep.double()
        output, _ = model(seq_rep, subs_rep)
        output = output.cpu().detach().numpy().reshape(-1)
        target = target.numpy()
        y_pred.append(output)
    y_pred = np.concatenate(y_pred)

    return y_pred


def predict_from_byte_input(trained_model, seqs, subs, params):
    hid_dim = 256
    kernel_1 = 3
    out_dim = 1
    kernel_2 = 3
    last_hid = 1024

    train_seqs, train_subs, y_train, test_seqs, test_subs, y_test, valid_seqs, valid_subs, y_valid, \
    x_seqs_all_hidden_dim, x_subs_encoding_dim, _, nn_input_dim, x_seqs_num, x_subs_num = \
        prep_neural_net_from_byte_input(encoding_file=subs,
                                        embedding_file=seqs,
                                        screen_bool=False,
                                        log_value=True,
                                        split_type=1,
                                        nn_type='Reg',
                                        classification_threshold_type=2)
    seq_max_len = 1009
    nn_input_dim = 1280
    subs_dim = 731
    if "seq_max_len" in params:
        seq_max_len = params["seq_max_len"]
    if "nn_input_dim" in params:
        nn_input_dim = params["nn_input_dim"]
    if "subs_dim" in params:
        subs_dim = params["subs_dim"]

    train_loader, valid_loader, test_loader = \
        generate_cnn_loader(train_seqs,
                            train_subs,
                            y_train,
                            valid_seqs,
                            valid_subs,
                            y_valid,
                            test_seqs,
                            test_subs,
                            y_test,
                            seq_max_len,
                            64)
    model = NeuralNet(
        in_dim=nn_input_dim,
        hid_dim=hid_dim,
        kernel_1=kernel_1,
        out_dim=out_dim,  # 2
        kernel_2=kernel_2,
        max_len=seq_max_len,
        sub_dim=subs_dim,
        last_hid=last_hid,  # 256
        dropout=0.
    )
    model.double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(model)

    checkpoint = torch.load(trained_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    y_pred = []
    for one_seqsubs_ppt_group in test_loader:
        seq_rep, subs_rep, target = one_seqsubs_ppt_group["seqs_embeddings"], one_seqsubs_ppt_group["subs_encodings"], \
                                    one_seqsubs_ppt_group["y_property"]
        if torch.cuda.is_available():
            seq_rep, subs_rep = seq_rep.double().cuda(), subs_rep.double().cuda()
        else:
            seq_rep, subs_rep = seq_rep.double(), subs_rep.double()
        output, _ = model(seq_rep, subs_rep)
        output = output.cpu().detach().numpy().reshape(-1)
        target = target.numpy()
        y_pred.append(output)
    y_pred = np.concatenate(y_pred)
    print("Trying to save job results")

    if "job_id" in params.keys():
        write_result(f"results/{params['job_id']}/predictions.csv", y_pred, x_subs_num, x_seqs_num)
        import pandas as pd
        import matplotlib.pyplot as plt
        import pipeline.figure as figure
        if os.path.exists(f"results/{params['job_id']}/predictions.csv"):
            print("Creating heatmap")
            pairs = pd.read_csv(f"results/{params['job_id']}/predictions.csv", header=None)
            df = pd.DataFrame(pairs)
            df.columns = ['sub', 'seq', 'val']
            pivot_tab = df.pivot_table(index='sub', columns='seq', values='val', sort=False)
            plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots(figsize=(8, 6))

            im, cbar = figure.heatmap(pivot_tab, pivot_tab.columns, pivot_tab.index, ax=ax,
                                      cmap="bwr", cbarlabel="activity")
            texts = figure.annotate_heatmap(im, valfmt="{x:.1f}")

            fig.tight_layout()
            plt.savefig(f"results/{params['job_id']}/heatmap.svg", dpi=200, format="svg")
    return y_pred, x_subs_num, x_seqs_num


def write_result(file_name, y_pred, n_subs, n_seqs):
    print("Writing result to file: ", file_name)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w+') as f:
        for i in range(n_subs):
            for j in range(n_seqs):
                f.write("sub" + str(i + 1) + "," + "seq" + str(j + 1) + "," + str(
                    y_pred[i * n_seqs + j]) + "\n")
