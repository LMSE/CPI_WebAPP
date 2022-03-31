# rdkit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import Draw

# np imports
import numpy as np
import pandas as pd

# utils imports
from utils.chemconvert import *
from utils.chemconvert import unique_canonical_smiles_zx as unis

# system imports
import os
import io


MAX_SEQ_LEN = 300


def read_seqs(input_filename: str, output_filename: str,
              max_seq_len: int = 300) -> pd.array:
    """
    Write a fasta file including all sequences and return a seq_list
    containing all sequences
    """
    sub_df = pd.read_csv(input_filename, index_col=0, header=0)
    sub_df["seq_length"] = sub_df["seq"].str.len()
    sub_df = sub_df[sub_df["seq_length"] <= max_seq_len]
    sub_df.reset_index(drop=True, inplace=True)

    seq_list = []

    with open(output_filename, "w") as f:
        for index, row in sub_df.iterrows():
            f.write(f">SEQ{str(index)}\n")
            f.write(row["seq"] + "\n")
            seq_list.append(row["seq"])

    return pd.Series(data=seq_list, dtype=str)


def read_seqs_from_df(input_df: pd.DataFrame, output: str,
                      max_seq_len: int = 300) -> pd.array:
    """
    Write a fasta file including all sequences and return a seq_list
    containing all sequences
    """
    sub_df = input_df
    sub_df["seq_length"] = sub_df["seq"].str.len()
    sub_df = sub_df[sub_df["seq_length"] <= max_seq_len]
    sub_df.reset_index(drop=True, inplace=True)

    seq_list = []

    with open(output, "w") as f:
        for index, row in sub_df.iterrows():
            f.write(f">SEQ{str(index)}\n")
            f.write(row["seq"] + "\n")
            seq_list.append(row["seq"])

    return pd.array(data=seq_list, dtype=str)


def read_substrates(input_filename: str, max_seq_len: int = 300) -> pd.array:
    """
    Get a list of all substrates in SMILES format
    """
    sub_df = pd.read_csv(input_filename, index_col=0, header=0)
    sub_list = []

    for index, row in sub_df.iterrows():
        sub = sub_df.loc[index, "substrate"]
        if sub not in sub_list and \
           len(sub_df.loc[index, "seq"]) <= max_seq_len:
            smiles = unis(sub)
            sub_list.append(smiles)
    return pd.array(data=sub_list, dtype=str)


def draw_molecule(molecule: str, output_filename: str):
    """
    Draw a molecule
    """
    mol = Chem.MolFromSmiles(molecule)
    img = Draw.MolsToGridImage([mol, ], molsPerRow=1, subImgSize=(200, 200))
    img.save(output_filename)


def draw_all_substrates(sub_list: list, output_dir: str):
    """
    Draw all substrates in a list
    """
    for i, sub in enumerate(sub_list):
        draw_molecule(sub, os.path.join(output_dir, f"substrate{i}.png"))


def morgan_fingerprint(smiles: str, radius: int = 2, num_bits: int = 1024,
                       use_counts: bool = False) -> np.ndarray:
    """
    Generates a morgan fingerprint for a smiles string.

    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the
    fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if type(smiles) == str:
        mol = Chem.MolFromSmiles(smiles)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius,
                                                     nBits=num_bits)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius,
                                                        nBits=num_bits)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)
    return fp


def get_substrate_mapping(mapping_filename: str):
    """
    Get a mapping of all substrates to their names
    """
    sub_dict = {}
    sub_df = pd.read_csv(mapping_filename, index_col=0, header=0)
    for index, row in sub_df.iterrows():
        sub_dict[row["name"]] = row["smiles"]
    return sub_dict


def get_substrate_mapping_from_df(df: pd.DataFrame):
    """
    Get a mapping of all substrates to their names
    """
    sub_dict = {}
    for index, row in df.iterrows():
        sub_dict[row["name"]] = row["smiles"]
    return sub_dict


def get_substrate_properties(input_filename: str):
    pass
