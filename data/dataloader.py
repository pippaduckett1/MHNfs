
import numpy as np
import pickle

fold = "training"
name_mol_ids = ""
name_target_ids = ""
name_labels = ""
name_mol_inputs = ""
name_dict_mol_smiles_id = ""
name_dict_target_id_activeMolecules = ""
name_dict_target_id_inactiveMolecules = ""
name_dict_target_names_id = ""

path = "/Users/pippaduckett/MHNfs/data/" + fold + "/"

# "Data triplet": (molecule index, task index, label)
molIds = np.load(path + name_mol_ids)  # molecule indices for triplets
taskIds = np.load(path + name_target_ids)  # target indices for triplets
labels = np.load(path + name_labels).astype('float32')  # labels for triplets
molInputs = np.load(path + name_mol_inputs).astype('float32')  # molecule data base (fingerprints, descriptors)
dictMolSmilesid = pickle.load(open(path + name_dict_mol_smiles_id, 'rb'))  # connects molecule index wuth original SMILES
dictTaskidActivemolecules = pickle.load(
    open(path + name_dict_target_id_activeMolecules, 'rb'))  # stores molecule indices of active mols for each target
dictTaskidInactivemolecules = pickle.load(
    open(path + name_dict_target_id_inactiveMolecules, 'rb'))  # stores molecule indices of inactive mols for each target
dictTasknamesId = pickle.load(open(path + name_dict_target_names_id, 'rb'))  # connects target indices with original target names

dataDict = {'molIds': molIds,
                'taskIds': taskIds,
                'labels': labels ,
                'molInputs': molInputs,
                'dictMolSmilesid': dictMolSmilesid,
                'dictTaskidActivemolecules': dictTaskidActivemolecules,
                'dictTaskidInactivemolecules': dictTaskidInactivemolecules,
                'dictTasknamesId': dictTasknamesId
                }