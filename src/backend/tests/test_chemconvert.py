""" from utils.chemconvert import *

def test_chemconvert():
    # All of a,b and c are CoA.
    a="CC(C)(COP(=O)(O)OP(=O)(O)OCC1C(C(C(O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)C(C(=O)NCCC(=O)NCCS)O"
    b="O=C(NCCS)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3OP(=O)(O)O"
    c="CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C2N=CN=C3N)O)OP(=O)(O)O)[C@H](C(=O)NCCC(=O)NCCS)O"
    d="[H]N=c1c(c([H])nc(n1[H])C([H])([H])[H])C([H])([H])[n+]1cc(CC(N)C(=O)O)c2ccccc21"
    print(unique_canonical_smiles_zx(a))
    print(unique_canonical_smiles_zx(b))
    print(unique_canonical_smiles_zx(c))
    print(unique_input_smiles_zx(a))
    print(unique_input_smiles_zx(b))
    print(unique_input_smiles_zx(c))
    bad_ss_dict=dict([])
    print(MolToSmiles_zx(MolFromSmiles_zx(a,bad_ss_dict),bad_ss_dict))
    print(MolToSmiles_zx(MolFromSmiles_zx(b,bad_ss_dict),bad_ss_dict))
    print(MolToSmiles_zx(MolFromSmiles_zx(c,bad_ss_dict),bad_ss_dict))
    MolFromSmiles_zx
    print(unique_canonical_smiles_list_zx([a,b,c]))
    #####
    bkgd=['O','CC(C)(COP(=O)(O)OP(=O)(O)OCC1OC(n2cnc3c(N)ncnc32)C(O)C1OP(=O)(O)O)C(O)C(=O)NCCC(=O)NCCS','O=P(O)(O)O','O=C=O','N','CCCC(O)CC=O']
    bkgd_cmpd_list=[]
    for i in bkgd:
        bkgd_cmpd_list.append(unique_input_smiles_zx(i))
        print(unique_input_smiles_zx(i))
    print(bkgd_cmpd_list)
    print(unique_canonical_smiles_zx(unique_canonical_smiles_zx(c)))
 """