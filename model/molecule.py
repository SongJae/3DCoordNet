import os
import pickle
import copy
import torch
import numpy as np
import os.path as osp
import glob
import random

from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm.auto import tqdm
RDLogger.DisableLog('rdApp.*')

from model.util import mol2graph
from model.utils import get_dihedral_pairs
from torch_geometric.data import Batch

from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit.Chem.rdchem import BondType as BT

BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BT.names.keys())}
dihedral_pattern = Chem.MolFromSmarts('[*]~[*]~[*]~[*]')


def rdmol_to_data(mol:Mol):

    assert mol.GetNumConformers() == 1
    data_list=[]
    N = mol.GetNumAtoms()
    canonical_smi = Chem.MolToSmiles(mol)

    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    

    # skip mols with atoms with more than 4 neighbors for now
   
    #pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
    #pos_mask[k] = 1
   
    
    correct_mol = mol
    
    
        
    graph = mol2graph(correct_mol)
    data = Data()
    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
    data.smiles = Chem.MolToSmiles(correct_mol)
    data.xyz = pos
    
    #data = Data(__num_nodes = data.__num_nodes__, x=data.x, xyz=data.xyz, edge_index=data.edge_index, edge_attr=data.edge_attr)
   
    
   
    return data


def enumerate_conformers(mol):
    num_confs = mol.GetNumConformers()
    if num_confs == 1:
        yield mol
        return
    mol_templ = copy.deepcopy(mol)
    mol_templ.RemoveAllConformers()
    for conf_id in tqdm(range(num_confs), desc='Conformer'):
        conf = mol.GetConformer(conf_id)
        conf.SetId(0)
        mol_conf = copy.deepcopy(mol_templ)
        conf_id = mol_conf.AddConformer(conf)
        yield mol_conf


class MoleculeDataset(Dataset):

    def __init__(self, raw_path, force_reload=False, transform=None):
        super().__init__()
        """
        if mode=='train':
            self.processed_path = '/workspace/Molecule3D/data/QM9_train.processed'
        elif mode=='val':
            self.processed_path = '/workspace/Molecule3D/data/QM9_val.processed'
        else:
            self.processed_path = '/workspace/Molecule3D/data/QM9_test.processed'

        self.transform = transform

        self.root = root
        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(split_path, allow_pickle=True)[self.split_idx]
        self.bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

       
        self.dihedral_pairs = {} # for memoization
        all_files = sorted(glob.glob(osp.join(self.root, '*.pickle')))
        self.pickle_files = [f for i, f in enumerate(all_files) if i in self.split]
        self.max_confs = max_confs

        """
        self.raw_path = raw_path
        self.processed_path = raw_path + '.processed'
        self.transform = transform

        _, extname = os.path.splitext(raw_path)
        assert extname in ('.sdf', '.pkl'), 'Only supports .sdf and .pkl files'
        
        self.dataset = None
        if force_reload or not os.path.exists(self.processed_path):
            self.process_pickle()
        else:
            self.load_processed()

    def load_processed(self):
        self.dataset = torch.load(self.processed_path)

    def process_pickle(self):
        self.dataset = []
        with open(self.raw_path, 'rb') as f:
            mols = pickle.load(f)
            for mol in tqdm(mols):
                for conf in enumerate_conformers(mol):
                    self.dataset.append(rdmol_to_data(conf))
            torch.save(self.dataset, self.processed_path)
        """
        pickle_files = self.pickle_files #해당 split에 있는 전체 pickle file 가져옴
        pickle_file = random.choice(self.pickle_files)
    
        for pickle_file in tqdm(pickle_files):
            with open(pickle_file, 'rb') as f:
                mol_dic = pickle.load(f)
                confs = mol_dic['conformers']
                name = mol_dic["smiles"]
                # filter mols rdkit can't intrinsically handle
                mol_ = Chem.MolFromSmiles(name)
                if mol_:
                    canonical_smi = Chem.MolToSmiles(mol_)
                else:
                    continue
               
                # skip conformers with fragments
                if '.' in name:
                    continue

                # skip conformers without dihedrals
                N = confs[0]['rd_mol'].GetNumAtoms()
                if N < 4:
                    continue
                if confs[0]['rd_mol'].GetNumBonds() < 4:
                    continue
                if not confs[0]['rd_mol'].HasSubstructMatch(dihedral_pattern):
                    continue
                
                for mol in tqdm(mols):
                    for conf in enumerate_conformers(mol):
                        self.dataset.append(rdmol_to_data(conf))
                
                k=0
                for conf in confs:
                    mol = conf['rd_mol']
                    # skip mols with atoms with more than 4 neighbors for now
                    n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
                    if np.max(n_neighbors) > 4:
                        continue

                    # filter for conformers that may have reacted
                    try:
                        conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
                    except Exception as e:
                        continue

                    if conf_canonical_smi != canonical_smi:
                        continue
                    k += 1
                    self.dataset.append(rdmol_to_data(mol))
                    if k == self.max_confs:
                        break

            
                
                if k < self.max_confs:
                    for i in range(self.max_confs-k):
                        self.dataset.append(rdmol_to_data(mol))

                """
       

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
    
        data = self.dataset[idx].clone()
        #data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data