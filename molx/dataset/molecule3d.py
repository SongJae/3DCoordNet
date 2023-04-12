import os, json, ast, glob, ssl
import torch
import os.path as osp
import numpy as np
import pandas as pd
# import atom3d.util.formats as fo
# import atom3d.datasets as da

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import repeat
from six.moves import urllib
from torch_geometric.data import Data, InMemoryDataset, download_url
# from atom3d.datasets import LMDBDataset
from scipy.constants import physical_constants
from .utils import mol2graph, mol2graph_prop

hartree2eV = physical_constants['hartree-electron volt relationship'][0]

class Molecule3D(InMemoryDataset):
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
        """
    
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        
        
        
        self.root = root
        self.name = 'data'
        #self.target_df = pd.read_csv(osp.join(self.raw_dir, 'properties.csv'))
        self.qm9_directory = 'dataset/QM9'
        self.processed_file = 'qm9_processed.pt'
        self.distances_file = 'qm9_distances.pt'
        self.raw_qm9_file = 'qm9.csv'
        self.raw_spatial_data = 'qm9_eV.npz'
        self.atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        
        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
      
        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)
        
        
        self.data, self.slices = torch.load(
            osp.join(self.processed_dir, 'qm9_processed.pt'))
        
    
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
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['qm9_processed.pt']
        #return ['random_train.pt', 'random_val.pt', 'random_test.pt']
    
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    

    def pre_process(self):
        data_list = []
        #load qm9 data with spatial coordinates
        data_qm9 = dict(np.load(os.path.join(self.qm9_directory, self.raw_spatial_data), allow_pickle=True))
        coordinates = torch.tensor(data_qm9['R'], dtype=torch.float)
        # Read the QM9 data with SMILES information
        molecules_df = pd.read_csv(os.path.join(self.qm9_directory, self.raw_qm9_file))
        atom_idx = 0
        
        # go through all molecules in the npz file
        for mol_idx, n_atoms in tqdm(enumerate(data_qm9['N'])):         
          
          mol = Chem.MolFromSmiles(molecules_df['smiles'][data_qm9['id'][mol_idx]])
          mol = Chem.AddHs(mol)
          smiles = molecules_df['smiles'][data_qm9['id'][mol_idx]]
          coords = torch.tensor(data_qm9['R'][atom_idx:atom_idx+n_atoms], dtype=torch.float)
          atom_idx = atom_idx + n_atoms
          z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        
          graph = mol2graph(mol)
          data = Data()
          data.__num_nodes__ = int(graph['num_nodes'])
          data.smiles = smiles
          data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
          data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
          data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
          data.xyz = torch.tensor(coords, dtype=torch.float32)
          data_list.append(data)
                    
              
        return data_list
    
    
    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
        full_list = self.pre_process()
        print("full_list length: ", len(full_list))
                
        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        """
        for m, split_mode in enumerate(['random']):
            ind_path = osp.join(self.raw_dir, '{}_split_inds.json').format(split_mode)
            with open(ind_path, 'r') as f:
                 inds = json.load(f)
        """    
        
        data_list = [self.get_data_prop(full_list, idx) for idx in range(len(full_list))]
        torch.save(self.collate(data_list), self.processed_paths[0])
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        # if self.split == 'train':
        #   print("split: ", self.split)
        #   torch.save(self.collate(data_list), self.processed_paths[0])
        # elif self.split == 'val':
        #   print("split: ", self.split)
        #   torch.save(self.collate(data_list), self.processed_paths[1])
        # else:
        #   print("split: ", self.split)
        #   torch.save(self.collate(data_list), self.processed_paths[2])
            
            
    def get_data_prop(self, full_list, abs_idx):
        data = full_list[abs_idx]
        # if split == 'test':
        #     data.props = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values)
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

    
class Molecule3DProps(InMemoryDataset):
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
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 process_dir_base='processed_downstream',
                 test_pt_dir=None,
                 target_tasks: list = None,
                 normalize: bool = True # 원래는 True였음
                 ):
        print("here")
        self.root = root
        self.name = 'data'
        self.qm9_directory = 'dataset/QM9'
        self.processed_file = 'qm9_property_processed.pt'
        self.distances_file = 'qm9_distances.pt'
        self.raw_qm9_file = 'qm9.csv'
        self.raw_spatial_data = 'qm9_eV.npz'
        self.atom_types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        self.symbols = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
        self.normalize = normalize
        self.process_dir_base = process_dir_base
        self.test_pt_dir = test_pt_dir

        # data in the csv file is in Hartree units.
        self.unit_conversion = {'A': 1.0,
                                'B': 1.0,
                                'C': 1.0,
                                'mu': 1.0,
                                'alpha': 1.0,
                                'homo': hartree2eV,
                                'lumo': hartree2eV,
                                'gap': hartree2eV,
                                'r2': 1.0,
                                'zpve': hartree2eV,
                                'u0': hartree2eV,
                                'u298': hartree2eV,
                                'h298': hartree2eV,
                                'g298': hartree2eV,
                                'cv': 1.0,
                                'u0_atom': hartree2eV,
                                'u298_atom': hartree2eV,
                                'h298_atom': hartree2eV,
                                'g298_atom': hartree2eV}

        if target_tasks == None or target_tasks == []:  # set default
            self.target_tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        else:
            self.target_tasks: list = target_tasks
        for target_task in self.target_tasks:
            assert target_task in self.unit_conversion.keys()

        print("target_tasks: ", target_tasks)
        # if not osp.exists(self.raw_paths[0]):
        #     self.download()
        super(Molecule3DProps, self).__init__(root, transform, pre_transform, pre_filter)
        
         
        #self.data, self.slices = torch.load(
        #    osp.join(self.processed_dir, '{}.pt'.format(split)))
        
        self.data, self.slices = torch.load(
        osp.join(self.processed_dir, 'qm9_property_processed_gap.pt'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("data: ", self.data)
        # indices of the tasks that should be retrieved
        self.task_indices = torch.tensor([list(self.unit_conversion.keys()).index(task) for task in self.target_tasks])
        # select targets in the order specified by the target_tasks argument
        
        self.targets = self.data.targets.index_select(dim=1, index=self.task_indices)  # [130831, n_tasks]
        print("targets: ", self.targets, self.targets.shape)
        self.targets_mean = self.targets.mean(dim=0)
        print("self.targets_mean: ", self.targets_mean)
        self.targets_std = self.targets.std(dim=0)
        if self.normalize:
            self.targets = ((self.targets - self.targets_mean) / self.targets_std)
        print("targets after normalize: ", self.targets, self.targets.shape)
        self.targets_mean = self.targets_mean.to(self.device)
        self.targets_std = self.targets_std.to(self.device)
        print("mean: ", self.targets_mean)
        print("std: ", self.targets_std)
        self.data.y = self.targets
        # get a tensor that is 1000 for all targets that are energies and 1.0 for all other ones
        self.eV2meV = torch.tensor(
            [1.0 if list(self.unit_conversion.values())[task_index] == 1.0 else 1000 for task_index in
             self.task_indices]).to(self.device)  # [n_tasks]
        print("eV2meV here: ", self.eV2meV)
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
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.process_dir_base)

    @property
    def raw_file_names(self):
        name = self.name + '.csv'
        return name

    @property
    def processed_file_names(self):
        return ['qm9_property_processed_gap.pt']
        #return ['random_train.pt', 'random_val.pt', 'random_test.pt']
        
   
    def download(self):
        # print('making raw files:', self.raw_dir)
        # if not osp.exists(self.raw_dir):
        #     os.makedirs(self.raw_dir)
        # url = self.url
        # path = download_url(url, self.raw_dir)
        pass
    
    def pre_process(self):
        data_list = []

        #load qm9 data with spatial coordinates
        data_qm9 = dict(np.load(os.path.join(self.qm9_directory, self.raw_spatial_data), allow_pickle=True))
        coordinates = torch.tensor(data_qm9['R'], dtype=torch.float)
        # Read the QM9 data with SMILES information
        molecules_df = pd.read_csv(os.path.join(self.qm9_directory, self.raw_qm9_file))
        atom_idx = 0
        
        # go through all molecules in the npz file
        for mol_idx, n_atoms in tqdm(enumerate(data_qm9['N'])):         
          
          mol = Chem.MolFromSmiles(molecules_df['smiles'][data_qm9['id'][mol_idx]])
          mol = Chem.AddHs(mol)
          smiles = molecules_df['smiles'][data_qm9['id'][mol_idx]]
          coords = torch.tensor(data_qm9['R'][atom_idx:atom_idx+n_atoms], dtype=torch.float)
          atom_idx = atom_idx + n_atoms
          z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
          target = torch.tensor(molecules_df.iloc[data_qm9['id'][mol_idx]][2:], dtype=torch.float)
          targets = target * torch.tensor(list(self.unit_conversion.values()))[None, :]
          
        
          graph = mol2graph(mol)
          graph_prop = mol2graph_prop(mol)
          data = Data()
          data.__num_nodes__ = int(graph['num_nodes'])
        
          # Required by GNNs
          data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
          data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
          data.x_coord = torch.from_numpy(graph['node_feat']).to(torch.int64)
          data.target = target
          data.targets = targets
          data.smiles = smiles
        
          # Required by EGNN
          data.x = torch.from_numpy(graph_prop['node_feat']).to(torch.int64)
          data.z = torch.tensor(z, dtype=torch.int64)
          data_list.append(data)
       

      
       
        return data_list

    def process(self):
        print("here to start processing")
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.
        
            If one-hot format is required, the processed data type will include an extra dimension 
            of virtual node and edge feature.
        """
      
        
        """
        if self.train_pt_dir is not None:
            train_path = osp.join(self.root, self.name, self.train_pt_dir,
                                 '{}.pt'.format(self.split))
            print('Loading pre-processed data from: {}...'.format(train_path))
            self.data, self.slices = torch.load(train_path)
            print("loaded data: ", len(self.data))

        elif self.val_pt_dir is not None:
            val_path = osp.join(self.root, self.name, self.val_pt_dir,
                                 '{}.pt'.format(self.split))
            print('Loading pre-processed data from: {}...'.format(val_path))
            self.data, self.slices = torch.load(val_path)
            print("loaded data: ", len(self.data))

        elif self.test_pt_dir is not None:
            test_path = osp.join(self.root, self.name, self.test_pt_dir,
                                 '{}.pt'.format(self.split))
            print('Loading pre-processed data from: {}...'.format(test_path))
            self.data, self.slices = torch.load(test_path)
            print("loaded data: ", len(self.data))
           
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
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            print("here to pre_transform")
            data_list = [self.pre_transform(data) for data in data_list]
        torch.save(self.collate(data_list), self.processed_paths[0])
        
       
    
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
