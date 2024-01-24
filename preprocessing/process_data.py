
# need test_coo

from http import cookiejar
import json, pickle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import random, sys, os

from rdkit import Chem
from rdkit.Chem import AllChem as Chem

from dpu_utils.utils.richpath import RichPath
from fsmol_dataset import DataFold, FSMolDataset
from pathlib import Path
# from test_utils import set_up_test_run


def morgan_fp(smiles, r=3, bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = Chem.GetMorganFingerprintAsBitVect(mol, r, nBits=1024)
    fp = np.array(fp)
    return fp

path = './FS-Mol/'

def setup_dset(tasks, fold, smilesmap={}, targetsmap={}):
    print("STARTING SETUP DSET FUNCTION"*100)
        # get data in to form for sklearn

    mols_list = []
    tasks_list = []
    labels_list = []

    dictTaskidActivemolecules = {}
    dictTaskidInactivemolecules = {}

    if len(smilesmap.keys()) > 0:
        n_mol_counter = max(smilesmap.keys())
    else:
        n_mol_counter = 0

    if len(targetsmap.keys()) > 0:
        i = max(targetsmap.keys())
    else:
        i = 0

    for task in tasks:
        mols = task.samples
        targetsmap[i] = mols[0].task_name
        dictTaskidActivemolecules[i] = []
        dictTaskidInactivemolecules[i] = []
        for mol in mols:
            smiles = mol.smiles
            if smiles in smilesmap.values():
                for key, value in smilesmap.items():
                    if value == smiles:
                        n_mol = key
                        break
            else:
                n_mol_counter +=1
                n_mol = n_mol_counter
                smilesmap[n_mol] = smiles
            pic50 = mol.numeric_label
            print("bool label", type(mol.bool_label))
            sys.exit()
            if mol.bool_label:
                dictTaskidActivemolecules[i].append(n_mol)
            else:
                dictTaskidInactivemolecules[i].append(n_mol)

            tasks_list.append(i)
            mols_list.append(n_mol)
            labels_list.append(pic50)
        i+=1


    mols_np  = np.array(mols_list)
    tasks_np  = np.array(tasks_list)
    labels_np = np.array(labels_list)

    print("mols", mols_list)
    print("tasks", tasks_list)
    print("labels", labels_list)

    output_directory = Path(path, "processed_data", fold)
    output_directory.mkdir(parents=True, exist_ok=True)

    np.save(output_directory / 'mol_ids.npy', mols_np)

    np.save(Path(path, "processed_data", fold, 'mol_ids.npy'), mols_np)
    np.save(Path(path, "processed_data", fold, 'target_ids.npy'), tasks_np)
    np.save(Path(path, "processed_data", fold, 'labels.npy'), labels_np)

    # fingerprints = {}
    # for n_mol, smiles in smilesmap.items():
    #     fingerprint = morgan_fp(smiles)
    #     fingerprints[n_mol] = np.array(fingerprint)

    max_n_mol = max(smilesmap.keys())
    # Create an array to store fingerprints
    fingerprints_array = np.zeros((max_n_mol + 1, len(morgan_fp(smilesmap[0]))))

    # Fill the array based on the dictionary
    for n_mol, smiles in smilesmap.items():
        fingerprints_array[n_mol] = np.array(morgan_fp(smiles))

    # feats = np.array(fingerprints)
    np.save(Path(path, "processed_data", fold, 'fingerprints.npy'), fingerprints_array)

    with open(Path(path, "processed_data", fold, 'smilesmap.pickle'), 'wb') as fp:
        pickle.dump(smilesmap, fp)

    with open(Path(path, "processed_data", fold, 'targetsmap.pickle'), 'wb') as fp:
        pickle.dump(targetsmap, fp)
    
    with open(Path(path, "processed_data", fold, 'dictTaskidActivemolecules.pickle'), 'wb') as fp:
        pickle.dump(dictTaskidActivemolecules, fp)
    
    with open(Path(path, "processed_data", fold, 'dictTaskidInactivemolecules.pickle'), 'wb') as fp:
        pickle.dump(dictTaskidInactivemolecules, fp)
    

def set_up_dataset(DATA_PATH, task_list_file):

    if RichPath.create(DATA_PATH).is_dir():
        assert (
            RichPath.create(DATA_PATH).join("test").exists()
        ), "If DATA_PATH is a directory it must contain test/ sub-directory for evaluation."

        return FSMolDataset.from_directory(
            DATA_PATH, task_list_file=RichPath.create(task_list_file))
    else:
        return FSMolDataset(test_data_paths=[RichPath.create(p) for p in DATA_PATH])


if __name__ == '__main__':

    DATA_PATH = '/Users/pippaduckett/EXS_GNN/FS-Mol-dataset/fs-mol/'
    task_list_file = "/Users/pippaduckett/MHNfs/preprocessing/FS-Mol/fsmol-0.1.json"

    dataset = set_up_dataset(DATA_PATH, task_list_file)

    task_reading_kwargs = {}

    for fold in [DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST]:
        dataset_iterable_valid = dataset.get_task_reading_iterable(fold, **task_reading_kwargs)
        smilesmap, targetsmap = setup_dset(dataset_iterable_valid, fold.name.lower())
        sys.exit()


    # path = '/Users/pippaduckett/EXS_GNN/MetaDTA/data/FSMol/'

    # fingerprints = []
    # for smiles in smilesmap.keys():
    #     fingerprint = morgan_fp(smiles)
    #     fingerprints.append(np.array(fingerprint))

    # feats = np.array(fingerprints)
    # np.save(path + 'total_ecfp.npy', feats)

    # with open(path + 'smilesmap.pickle', 'wb') as fp:
    #     pickle.dump(smilesmap, fp)

    # with open(path + 'targetsmap.pickle', 'wb') as fp:
    #     pickle.dump(targetsmap, fp)