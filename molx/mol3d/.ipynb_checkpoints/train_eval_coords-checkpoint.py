import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader
from .utils import generate_xyz
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from TorchProteinLibrary import RMSD


class Mol3DTrainer_coordinate():
    def __init__(self, train_dataset, val_dataset, configs, device):
        self.configs = configs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        #self.train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=4)
        #self.val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=4)
        self.train_loader = train_dataset
        self.val_loader = val_dataset
        self.out_path = configs['out_path']
        self.start_epoch = 1
        self.device = device
        self.batch_size = configs['batch_size']


    def _get_loss(self):
        if self.configs['criterion'] == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif self.configs['criterion'] == 'mae':
            return torch.nn.L1Loss(reduction='sum')


    def _train_loss(self, model, optimizer, criterion):
        model.train()
        loss_total = 0
        i = 0
        rmsd = RMSD.Coords2RMSD().cuda()
        #for batch_data in tqdm(self.train_loader, total=len(self.train_loader)):
        for i, batch_data in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            optimizer.zero_grad()

            batch_data = batch_data.to(self.device)
            batch = batch_data.batch
            coords = batch_data.xyz
           
            pred_coords = model(batch_data, self.batch_size, train=True)
               
      
            output, counts = torch.unique(batch, sorted = True, return_counts = True)
            counts = counts.type(torch.int)
          
            
                      
            counts_list = counts.tolist()
          
            coords_batch_split = torch.split(coords, counts_list, dim=0)
            
            pred_coords_batch_split = torch.split(pred_coords, counts_list, dim=0)
            
            coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
            pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)
            
            
            coords_tensor = torch.reshape(coords_tensor,(coords_tensor.shape[0],coords_tensor.shape[1]*3))
            pred_coords_tensor = torch.reshape(pred_coords_tensor,(pred_coords_tensor.shape[0],pred_coords_tensor.shape[1]*3))
            
            """
            print("coords_tensor: ", coords_tensor, coords_tensor.shape)
            print("pred_coords_tensor: ", pred_coords_tensor, pred_coords_tensor.shape)
            """
            """
            coords_list = []
            pred_coords_list = []
            counts_list = []
            
            for i in range(10):
                indices = torch.tensor([i+10*j for j in range(self.batch_size)], device=self.device)
                coords_list.append(torch.index_select(coords_tensor, 0, indices))
                pred_coords_list.append(torch.index_select(pred_coords_tensor, 0, indices))
                counts_list.append(torch.index_select(counts, 0, indices))
                
                
            print("coords_list here: ", coords_list, len(coords_list))
            print("pred_coords_list here: ", pred_coords_list, len(pred_coords_list))
            """
            """
            local_loss = torch.ones([100], dtype=torch.float64, device=self.device)
            ind = 0
            for tc in coords_list:
                index = 0
                for mc in pred_coords_list:
                    local_loss[ind] = rmsd(mc, tc, counts_list[index]).mean()
                    ind = ind + 1
                index = index + 1
            """
            loss = rmsd(pred_coords_tensor, coords_tensor, counts).mean()
            #loss = local_loss.mean()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            i += 1
        return loss_total / i


    def save_ckpt(self, epoch, model, optimizer, best_valid=False):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        if best_valid:
            torch.save(checkpoint, os.path.join(self.out_path, 'ckpt_best_val.pth'.format(epoch)))
        else:
            torch.save(checkpoint, os.path.join(self.out_path, 'ckpt_{}.pth'.format(epoch)))


    def train(self, model):
        if self.configs['out_path'] is not None:
            try:
                os.makedirs(self.configs['out_path'])
            except OSError:
                pass
            
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'])
        criterion = self._get_loss()
        if 'load_pth' in self.configs and self.configs['load_pth'] is not None:
            checkpoint = torch.load(self.configs['load_pth'])
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
        
        best_val_rmsd = 10000
        epoch_bvl = 0
        for i in range(self.start_epoch, self.configs['epochs']+1):
            loss_dist = self._train_loss(model, optimizer, criterion)
            rmsd = eval3d_coordinate(model, self.val_dataset, self.batch_size)

            # One possible way to selection model: do testing when val metric is best
            if self.configs['save_ckpt'] == "best_valid":
                if rmsd < best_val_rmsd:
                    epoch_bvl = i
                    best_val_rmsd = rmsd
                    if self.out_path is not None:
                        self.save_ckpt(i, model, optimizer, best_valid=True)

            # Or we can save model at each epoch
            elif i % self.configs['save_ckpt'] == 0:
                if self.out_path is not None:
                    self.save_ckpt(i, model, optimizer, best_valid=False)

            if i % self.configs['lr_decay_step_size'] == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.configs['lr_decay_factor'] * param_group['lr']

            writer = SummaryWriter("coords_testing_v1")
            writer.add_scalar('rmsd/train', loss_dist, i)
            writer.add_scalar('rmsd/validation', rmsd, i)
            writer.close()

            print('epoch: {}; Train -- loss: {:.3f}'.format(i, loss_dist))
            print('epoch: {}; Valid -- val_rmsd: {:.3f}'.format(i, rmsd))
            print('============================================================================================================')

        print('Best valid epoch is {}; Best val_rmsd: {:.3f}'.format(epoch_bvl, best_val_rmsd))
        print('============================================================================================================')

        return model


def eval3d_coordinate(model, dataset, batch_size):
    #dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    i = 0
    loss_total = 0
    rmsd = RMSD.Coords2RMSD().cuda()
    #for batch_data in tqdm(dataloader, total=len(dataloader), ncols=80):
    for i, batch_data in tqdm(enumerate(dataset), total=len(dataset)):
        batch_data = batch_data.to(device)
        
        with torch.no_grad():
          pred_coords = model(batch_data, batch_size, train=False)
        
        coords = batch_data.xyz
    
        batch = batch_data.batch
                 
        output, counts = torch.unique(batch, sorted = True, return_counts = True)
        counts = counts.type(torch.int)
            
        counts_list = counts.tolist()
        coords_batch_split = torch.split(coords, counts_list)
        pred_coords_batch_split = torch.split(pred_coords, counts_list)
        
        coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
        pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)

        coords_tensor = torch.reshape(coords_tensor,(coords_tensor.shape[0],coords_tensor.shape[1]*3))
        pred_coords_tensor = torch.reshape(pred_coords_tensor,(pred_coords_tensor.shape[0],pred_coords_tensor.shape[1]*3))
       
        loss = rmsd(pred_coords_tensor, coords_tensor, counts).mean()
        
        loss_total += loss.item()
        i += 1

    
    return loss_total / i
