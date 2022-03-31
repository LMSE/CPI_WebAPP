import sys
import os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from pipeline.data_process_new import *


def test_read_seqs():
    test_input = os.path.dirname(os.path.abspath(__file__)) + "/phosphatase_chiral.csv"
    test_output = os.path.dirname(os.path.abspath(__file__)) + "/phosphatase_chiral.out"
    read_seqs(test_input, test_output, max_seq_len=10000)


def test_read_subs():
    test_input = os.path.dirname(os.path.abspath(__file__)) + "/phosphatase_chiral.csv"
    test_output = os.path.dirname(os.path.abspath(__file__)) + "/phosphatase_chiral_substrates.out"
    with open(test_output, "w") as f:
        for sub in read_substrates(test_input):
            f.write(sub + "\n")


def test_draw_susbtrates():
    test_input = os.path.dirname(os.path.abspath(__file__)) + "/phosphatase_chiral.csv"
    test_output_dir = os.path.dirname(os.path.abspath(__file__)) + "/phosphatase_sub_struct"
    sub_list = read_substrates(test_input)
    draw_all_substrates(sub_list, test_output_dir)


def test_sub_mapping():
    test_input = os.path.dirname(os.path.abspath(__file__)) + "/esterase_smiles.csv"
    print(get_substrate_mapping(test_input))
