from dataclasses import dataclass
from typing import List, Optional, Tuple
from aiohttp import Fingerprint
import math

import numpy as np
import pandas as pd
from more_itertools import partition, partitions
from dpu_utils.utils import RichPath
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Descriptors
import sys
from chembl_webresource_client.new_client import new_client
assay_chembl_client = new_client.assay
target_chembl_client = new_client.target

def get_task_name_from_path(path: RichPath) -> str:
    # Use filename as task name:
    name = path.basename()
    if name.endswith(".jsonl.gz"):
        name = name[: -len(".jsonl.gz")]
    return name

# def get_chembl_description(list_of_chembl_ids):
#     ress = {}
#     for assay_chembl_id in tqdm(list_of_chembl_ids):
#         # try:
#         res = assay_chembl_client.filter(assay_chembl_id=assay_chembl_id)
#         #     # ress[assay_chembl_id] = res[0]
#         # except:
#         #     ress[assay_chembl_id] = {}
#         #     print('error with', assay_chembl_id)
#     return ress


@dataclass
class GraphData:
    """Data structure holding information about a graph with typed edges.

    Args:
        node_features: Initial node features as ndarray of shape [V, ...]
        adjacency_lists: Adjacency information by edge type as list of ndarrays of shape [E, 2]
        edge_features: Edge features by edge type as list of ndarrays of shape [E, edge_feat_dim].
            If not present, all edge_feat_dim=0.
    """

    node_features: np.ndarray
    adjacency_lists: List[np.ndarray]
    edge_features: List[np.ndarray]


@dataclass(frozen=True)
class MoleculeDatapoint:
    """Data structure holding information for a single molecule.

    Args:
        task_name: String describing the task this datapoint is taken from.
        smiles: SMILES string describing the molecule this datapoint corresponds to.
        graph: GraphData object containing information about the molecule in graph representation
            form, according to featurization chosen in preprocessing.
        numeric_label: numerical label (e.g., activity), usually measured in the lab
        bool_label: bool classification label, usually derived from the numeric label using a
            threshold.
        fingerprint: optional ECFP (Extended-Connectivity Fingerprint) for the molecule.
        descriptors: optional phys-chem descriptors for the molecule.
    """

    task_name: str
    smiles: str
    numeric_label: float
    bool_label: bool
    bin_label: Optional[int]
    graph: Optional[GraphData]
    fingerprint: Optional[np.ndarray]
    descriptors: Optional[np.ndarray]

    def get_fingerprint(self) -> np.ndarray:
        if self.fingerprint is not None:
            return self.fingerprint
        else:
            # TODO(mabrocks): It would be much faster if these would be computed in preprocessing and just passed through
            mol = Chem.MolFromSmiles(self.smiles)
            fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
                [mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fingerprint = np.zeros((0,), np.float32)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(fingerprints_vect, fingerprint)
            return fingerprint
    def get_descriptors(self) -> np.ndarray:
        if self.descriptors is not None:
            return self.descriptors
        else:
            # TODO(mabrocks): It would be much faster if these would be computed in preprocessing and just passed through
            mol = Chem.MolFromSmiles(self.smiles)
            descriptors = []
            for _, descr_calc_fn in Descriptors._descList:
                descriptors.append(descr_calc_fn(mol))
            return np.array(descriptors)


@dataclass(frozen=True)
class FSMolTask:
    """Data structure holding information for a single task.

    Args:
        name: String describing the task's name eg. "CHEMBL1000114".
        samples: List of MoleculeDatapoint samples associated with this task.
    """

    name: str
    assay_type : str
    assay_description : str
    target_chembl_id : str
    protein_fam: str
    samples: List[MoleculeDatapoint]

    def get_pos_neg_separated(self) -> Tuple[List[MoleculeDatapoint], List[MoleculeDatapoint]]:
        pos_samples, neg_samples = partition(pred=lambda s: s.bool_label, iterable=self.samples)
        return list(pos_samples), list(neg_samples)

    def get_mutliclass_separated(self, bins) -> Tuple[List[MoleculeDatapoint], List[MoleculeDatapoint]]:

        # pos_samples, neg_samples = partition(pred=lambda s: s.bin_label[0], iterable=self.samples)

        samples_list = []

        for bin in range(bins):
            print("BIN: ", bin)
            bin_samples = list(partition(lambda x: x.bin_label[0] == bin, iterable=self.samples)[1])
            samples_list.append(bin_samples)

        return samples_list


    @staticmethod
    def load_from_file(path: RichPath) -> "FSMolTask":

        print('GOT TO LOAD FROM FILE')

        taskname = get_task_name_from_path(path)

        res = assay_chembl_client.filter(assay_chembl_id=taskname).only(['assay_type', 'assay_type_description', 'target_chembl_id'])

        assay_type = res[0]['assay_type']
        assay_description = res[0]['assay_type_description']
        target_chembl_id = res[0]['target_chembl_id']

        # targetinfo = target_chembl_client.filter(target_chembl_id=target_chembl_id).only(['organism', 'pref_name'])

        # df = pd.read_csv("/Users/pippaduckett/EXS_GNN/Benchmarks/fsmol_me/data/fsmol/fs-mol/target_info.csv")
        # protein_fam = [str(x) for x in df[df.chembl_id == taskname]['protein_family']][0]

        protein_fam = 'none'
        # 'assay_type': 'B'
        # 'assay_type_description': 'Binding'
        # 'bao_format': 'BAO_0000357', 
        # 'description': 'Inhibition of recombinant PI3Kdelta by radioactive phosphotransfer assay in presence of 10 uM ATP'
        # 'document_chembl_id': 'CHEMBL1240340'
        # 'relationship_description': 'Homologous protein target assigned' 
        # 'relationship_type': 'H'
        # 'target_chembl_id': 'CHEMBL3130'

        samples = []
        for raw_sample in path.read_by_file_suffix():
            graph_data = raw_sample.get("graph")

            fingerprint_raw = raw_sample.get("fingerprints")
            if fingerprint_raw is not None:
                fingerprint: Optional[np.ndarray] = np.array(fingerprint_raw, dtype=np.int32)
            else:
                fingerprint = None

            descriptors_raw = raw_sample.get("descriptors")
            if descriptors_raw is not None:
                descriptors: Optional[np.ndarray] = np.array(descriptors_raw, dtype=np.float32)
            else:
                descriptors = None

            adjacency_lists = []
            for adj_list in graph_data["adjacency_lists"]:
                if len(adj_list) > 0:
                    adjacency_lists.append(np.array(adj_list, dtype=np.int64))
                else:
                    adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))

            print("GOT TO PRE BINS")
            end=11
            start=0
            bin_size=1
            num_bins = int((end - start) / bin_size)  # Determine the number of bins needed
            bin_label = int((float((raw_sample.get("LogRegressionProperty"))) - start) / bin_size)

            print("got to post bins")

            samples.append(
                MoleculeDatapoint(
                    task_name=get_task_name_from_path(path),
                    smiles=raw_sample["SMILES"],
                    bool_label=bool(float(raw_sample["Property"])),
                    bin_label=[bin_label, num_bins],
                    numeric_label=float(raw_sample.get("LogRegressionProperty")),
                    fingerprint=fingerprint,
                    descriptors=descriptors,
                    graph=GraphData(
                        node_features=np.array(graph_data["node_features"], dtype=np.float32),
                        adjacency_lists=adjacency_lists,
                        edge_features=[
                            np.array(edge_feats, dtype=np.float32)
                            for edge_feats in graph_data.get("edge_features") or []
                        ],
                    ),
                )
            )

        print("MADE SAMPLE")
 
        # return FSMolTask(get_task_name_from_path(path), samples)
        return FSMolTask(taskname, assay_type, assay_description, target_chembl_id, protein_fam, samples)


@dataclass(frozen=True)
class FSMolTaskSample:
    """Data structure output of a Task Sampler.

    Args:
        name: String describing the task's name eg. "CHEMBL1000114".
        train_samples: List of MoleculeDatapoint samples drawn as the support set.
        valid_samples: List of MoleculeDatapoint samples drawn as the validation set.
            This may be empty, dependent on the nature of the Task Sampler.
        test_samples: List of MoleculeDatapoint samples drawn as the query set.
    """

    name: str
    train_samples: List[MoleculeDatapoint]
    valid_samples: List[MoleculeDatapoint]
    test_samples: List[MoleculeDatapoint]

    @staticmethod
    def __compute_positive_fraction(samples: List[MoleculeDatapoint]) -> float:
        num_pos_samples = sum(s.bool_label for s in samples)
        return num_pos_samples / len(samples)

    @property
    def train_pos_label_ratio(self) -> float:
        return self.__compute_positive_fraction(self.train_samples)

    @property
    def test_pos_label_ratio(self) -> float:
        return self.__compute_positive_fraction(self.test_samples)