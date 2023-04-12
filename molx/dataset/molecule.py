import os, json, ast, glob, ssl
import torch
import os.path as osp
import numpy as np
import pandas as pd

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import repeat
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url
from glob import glob

from .utils import mol2graph

    
class Molecule3DGEnergy(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for 
        datasets used in molecule generation.
        
        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`. 
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property 
            label when the processed dataset is used. You can change the augment :obj:`processed_filename` 
            to re-process the dataset with intended property.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            split (string, optional): If :obj:`"train"`, loads the training dataset.
                If :obj:`"val"`, loads the validation dataset.
                If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
            split_mode (string, optional): Mode of split chosen from :obj:`"random"` and :obj:`"scaffold"`.
                (default: :obj:`penalized_logp`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            process_dir_base (string, optional): target directory to store your processed data. Should use 
                different dir when using different :obj:`pre_transform' functions.
            test_pt_dir (string, optional): If you already called :obj:`Molecule3D' and have raw data 
                pre-processed, set :obj:`test_pt_dir` to the folder name where test data file is stored.
                Usually stored in "processed".
        """
    
    def __init__(self,
                 root=None,
                 split='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 process_dir_base='processed_downstream_energy',
                 target_tasks: list = None
                 ):
        self.split = split
        self.name = 'data'
        self.process_dir_base = process_dir_base
        if split =="train":
            self.property = "data/train_set.ReorgE.csv"
            self.geo = "data/mol_files/train_set/*.mol"
        else:
            self.property = "data/test_set.csv"
            self.geo = "data/mol_files/test_set/*.mol"

        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        # data in the csv file is in Hartree units.
        self.unit_conversion = {'g_energy': 1.0,
                                'ex_energy': 1.0,
                                }
        if target_tasks == None or target_tasks == []:  # set default
            self.target_tasks = ['g_energy', 'ex_energy']
        else:
            self.target_tasks: list = target_tasks
        for target_task in self.target_tasks:
            assert target_task in self.unit_conversion.keys()
        super(Molecule3DGEnergy, self).__init__(root, transform, pre_transform, pre_filter)
        if split =="train":
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            # indices of the tasks that should be retrieved
            self.task_indices = torch.tensor([list(self.unit_conversion.keys()).index(task) for task in self.target_tasks])
            # select targets in the order specified by the target_tasks argument

            self.targets = self.data.targets.index_select(dim=1, index=self.task_indices)  # [130831, n_tasks]
            self.targets_mean = self.targets.mean(dim=0)
            self.targets_std = self.targets.std(dim=0)
            # if self.normalize:
            #     self.targets = ((self.targets - self.targets_mean) / self.targets_std)
            self.targets_mean = self.targets_mean.to(self.device)
            self.targets_std = self.targets_std.to(self.device)
            # get a tensor that is 1000 for all targets that are energies and 1.0 for all other ones
            self.eV2meV = torch.tensor(
                [1.0 if list(self.unit_conversion.values())[task_index] == 1.0 else 1000 for task_index in
                self.task_indices]).to(self.device)  # [n_tasks]
    
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])
            
        
       
    
    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0    
    
    @property
    def raw_dir(self):
        return osp.join(self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['mol_Genergy_train.pt', 'mol_Genergy_test.pt']
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    
    def pre_process(self):
        data_list = []
        train = pd.read_csv(self.property)
        if "Reorg_g" in train.columns:
            train.columns = ["index", "SMILES", "Reorg_g", "Reorg_ex"]
        else:
            train.columns = ["index", "SMILES"]

        if os.path.isdir("ex_file"):
            pass
        else:
            os.mkdir("ex_file")

        if os.path.isdir("g_file"):
            pass
        else:
            os.mkdir("g_file")

        train_mol = sorted(glob(self.geo))
        
       
        if "Reorg_g" in train.columns:
            for i in tqdm(range(0, len(train_mol), 2)):
                mol_ex = Chem.MolFromMolFile(train_mol[i], removeHs = False)
                mol_g = Chem.MolFromMolFile(train_mol[i+1], removeHs = False)
                index = int(train_mol[i].split('\\')[-1].split('_')[3])
                smiles = train.iloc[index, 1]
                z = [atom.GetAtomicNum() for atom in mol_g.GetAtoms()]

                
                ex_coords = mol_ex.GetConformer().GetPositions()
                ex_energy = train.iloc[index, 3]
                
                g_coords = mol_g.GetConformer().GetPositions()
                g_energy = train.iloc[index, 2]

                target = torch.tensor([g_energy, ex_energy], dtype=torch.float)
                targets = target * torch.tensor(list(self.unit_conversion.values()))[None, :]



                graph = mol2graph(mol_g)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                
                # Required by GNNs
                data.z = torch.tensor(z, dtype=torch.int64)
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.energy = g_energy
                data.smiles = smiles

                data.target = target
                data.targets = targets
                
                # Required by egnn
                data.g_xyz = torch.tensor(g_coords, dtype=torch.float32)
                data.ex_xyz = torch.tensor(ex_coords, dtype=torch.float32)
                data_list.append(data)
        else:
            for i in tqdm(range(0, len(train_mol), 2)):
                mol_ex = Chem.MolFromMolFile(train_mol[i], removeHs = False)   
                mol_g = Chem.MolFromMolFile(train_mol[i+1], removeHs = False)
                index = int(train_mol[i].split('\\')[-1].split('_')[3])
                smiles = train.iloc[index, 1]
                z = [atom.GetAtomicNum() for atom in mol_g.GetAtoms()]

                ex_coords = mol_ex.GetConformer().GetPositions()
                g_coords = mol_g.GetConformer().GetPositions()
               
                graph = mol2graph(mol_g)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                
                # Required by GNNs
                data.z = torch.tensor(z, dtype=torch.int64)
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.smiles = smiles
                data.index = index
                
                # Required by egnn
                data.g_xyz = torch.tensor(g_coords, dtype=torch.float32)
                data.ex_xyz = torch.tensor(ex_coords, dtype=torch.float32)
                data_list.append(data)
                
        return data_list
    
    
    def process(self):
        print("here to start processing")
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
      
        
        self.data = self.pre_process()

        #ind_path = osp.join(self.raw_dir, '{}_test_split_inds.json').format(self.split_mode)
        #with open(ind_path, 'r') as f:
        #     inds = json.load(f)
                
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
       
            
        
        data_list = [self.get_data_prop(self.data, idx) for idx in range(len(self.data))]
        #data_list = [self.get(idx) for idx in range(len(self.data))]
        
        if self.split == "train":
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            torch.save(self.collate(data_list), self.processed_paths[1])
        
       
    
    def get_data_prop(self, full_list, abs_idx):
        data = full_list[abs_idx]
        #if split == 'test':
        #    data.props = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values)
        return data
        
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
    
    def get(self, idx):
        r"""Gets the data object at index :idx:.
        
        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data

class Molecule3DExEnergy(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for 
        datasets used in molecule generation.
        
        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`. 
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property 
            label when the processed dataset is used. You can change the augment :obj:`processed_filename` 
            to re-process the dataset with intended property.
        
        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            split (string, optional): If :obj:`"train"`, loads the training dataset.
                If :obj:`"val"`, loads the validation dataset.
                If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
            split_mode (string, optional): Mode of split chosen from :obj:`"random"` and :obj:`"scaffold"`.
                (default: :obj:`penalized_logp`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            process_dir_base (string, optional): target directory to store your processed data. Should use 
                different dir when using different :obj:`pre_transform' functions.
            test_pt_dir (string, optional): If you already called :obj:`Molecule3D' and have raw data 
                pre-processed, set :obj:`test_pt_dir` to the folder name where test data file is stored.
                Usually stored in "processed".
        """
    
    def __init__(self,
                 root=None,
                 split='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 process_dir_base='processed_downstream_energy',
                 target_tasks: list = None
                 ):
        self.split = split
        self.name = 'data'
        self.process_dir_base = process_dir_base
        if split =="train":
            self.property = "data/train_set.ReorgE.csv"
            self.geo = "data/mol_files/train_set/*.mol"
        else:
            self.property = "data/test_set.csv"
            self.geo = "data/mol_files/test_set/*.mol"

        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        self.unit_conversion = {'g_energy': 1.0,
                                'ex_energy': 1.0,
                                }
        if target_tasks == None or target_tasks == []:  # set default
            self.target_tasks = ['g_energy', 'ex_energy']
        else:
            self.target_tasks: list = target_tasks
        for target_task in self.target_tasks:
            assert target_task in self.unit_conversion.keys()
        super(Molecule3DExEnergy, self).__init__(root, transform, pre_transform, pre_filter)
        if split =="train":
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            # indices of the tasks that should be retrieved
            self.task_indices = torch.tensor([list(self.unit_conversion.keys()).index(task) for task in self.target_tasks])
            # select targets in the order specified by the target_tasks argument

            self.targets = self.data.targets.index_select(dim=1, index=self.task_indices)  # [130831, n_tasks]
            self.targets_mean = self.targets.mean(dim=0)
            self.targets_std = self.targets.std(dim=0)
            # if self.normalize:
            #     self.targets = ((self.targets - self.targets_mean) / self.targets_std)
            self.targets_mean = self.targets_mean.to(self.device)
            self.targets_std = self.targets_std.to(self.device)
            # get a tensor that is 1000 for all targets that are energies and 1.0 for all other ones
            self.eV2meV = torch.tensor(
                [1.0 if list(self.unit_conversion.values())[task_index] == 1.0 else 1000 for task_index in
                self.task_indices]).to(self.device)  # [n_tasks]
    
        else:
            self.data, self.slices = torch.load(self.processed_paths[1])
            
        
       
    
    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0    
    
    @property
    def raw_dir(self):
        return osp.join(self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['mol_Exenergy_train.pt', 'mol_Exenergy_test.pt']
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    
    def pre_process(self):
        data_list = []
        train = pd.read_csv(self.property)
        if "Reorg_g" in train.columns:
            train.columns = ["index", "SMILES", "Reorg_g", "Reorg_ex"]
        else:
            train.columns = ["index", "SMILES"]

        if os.path.isdir("ex_file"):
            pass
        else:
            os.mkdir("ex_file")

        if os.path.isdir("g_file"):
            pass
        else:
            os.mkdir("g_file")

        train_mol = sorted(glob(self.geo))
        if "Reorg_g" in train.columns:
            for i in tqdm(range(0, len(train_mol), 2)):   
                mol_ex = Chem.MolFromMolFile(train_mol[i], removeHs = False)
                mol_g = Chem.MolFromMolFile(train_mol[i+1], removeHs = False)
               
                index = int(train_mol[i].split('\\')[-1].split('_')[3])
            
                ex_coords = mol_ex.GetConformer().GetPositions()
                ex_energy = train.iloc[index, 3]
                
                g_coords = mol_g.GetConformer().GetPositions()
                g_energy = train.iloc[index, 2]

                target = torch.tensor([g_energy, ex_energy], dtype=torch.float)
                targets = target * torch.tensor(list(self.unit_conversion.values()))[None, :]
                
                smiles = train.iloc[index, 1]
                z = [atom.GetAtomicNum() for atom in mol_ex.GetAtoms()]
                
                graph = mol2graph(mol_ex)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                
                # Required by GNNs
                data.z = torch.tensor(z, dtype=torch.int64)
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.energy = ex_energy
                data.smiles = smiles

                data.target = target
                data.targets = targets
                
                # Required by egnn
                data.g_xyz = torch.tensor(g_coords, dtype=torch.float32)
                data.ex_xyz = torch.tensor(ex_coords, dtype=torch.float32)
                data_list.append(data)
        else:
            for i in tqdm(range(0, len(train_mol), 2)):   
                mol_ex = Chem.MolFromMolFile(train_mol[i], removeHs = False)
                mol_g = Chem.MolFromMolFile(train_mol[i+1], removeHs = False)

                index = int(train_mol[i].split('\\')[-1].split('_')[3])
                smiles = train.iloc[index, 1]
                z = [atom.GetAtomicNum() for atom in mol_ex.GetAtoms()]
              
                ex_coords = mol_ex.GetConformer().GetPositions()
                g_coords = mol_g.GetConformer().GetPositions()
              
               
                graph = mol2graph(mol_ex)
                data = Data()
                data.__num_nodes__ = int(graph['num_nodes'])
                
                # Required by GNNs
                data.z = torch.tensor(z, dtype=torch.int64)
                data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
                data.smiles = smiles
                data.index = index
                
                # Required by egnn
                data.g_xyz = torch.tensor(g_coords, dtype=torch.float32)
                data.ex_xyz = torch.tensor(ex_coords, dtype=torch.float32)
                data_list.append(data)
                
        return data_list
    
    
    def process(self):
        print("here to start processing")
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
      
        
        self.data = self.pre_process()

        #ind_path = osp.join(self.raw_dir, '{}_test_split_inds.json').format(self.split_mode)
        #with open(ind_path, 'r') as f:
        #     inds = json.load(f)
                
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
       
            
        
        data_list = [self.get_data_prop(self.data, idx) for idx in range(len(self.data))]
        #data_list = [self.get(idx) for idx in range(len(self.data))]
        
        if self.split == "train":
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            torch.save(self.collate(data_list), self.processed_paths[1])
        
       
    
    def get_data_prop(self, full_list, abs_idx):
        data = full_list[abs_idx]
        #if split == 'test':
        #    data.props = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values)
        return data
        
    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
    
    
    def get(self, idx):
        r"""Gets the data object at index :idx:.
        
        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        return data