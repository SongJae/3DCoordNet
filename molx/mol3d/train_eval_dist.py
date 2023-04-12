import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader
from .utils import generate_xyz
from .rmsd import rmsd_batch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence


class Mol3DTrainer():
    def __init__(self, train_dataset, val_dataset, configs, device):
        self.configs = configs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_loader = DataLoader(train_dataset, num_workers=4, batch_size=configs['batch_size'], shuffle=True)
        self.val_loader = DataLoader(val_dataset, num_workers=4, batch_size=configs['batch_size'])
        self.out_path = configs['out_path']
        self.start_epoch = 1
        self.device = device


    def _get_loss(self):
        if self.configs['criterion'] == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif self.configs['criterion'] == 'mae':
            return torch.nn.L1Loss(reduction='sum')


    def _train_loss(self, model, optimizer, criterion):
        model.train()
        loss_total = 0
        i = 0
        for batch_data in tqdm(self.train_loader, total=len(self.train_loader)):
            optimizer.zero_grad()

            batch_data = batch_data.to(self.device)
            mask_d_pred, mask, dist_count, pred_coords = model(batch_data, train=True)
            coords = batch_data.xyz
            d_target = torch.cdist(coords, coords).float().to(self.device)
            mask_d_target = d_target * mask


            loss = criterion(mask_d_pred, mask_d_target) / dist_count  # MAE or MSE

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
            rmsd = eval3d(model, self.val_dataset)


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

            writer = SummaryWriter()

            print('epoch: {}; Train -- loss: {:.3f}'.format(i, loss_dist))
            writer.add_scalar('Loss/train', loss_dist, i)
            print('epoch: {}; Valid -- val_RMSD: {:.3f}'.format(i, rmsd))
            writer.add_scalar('rmsd/validation', rmsd, i)
            
            print('============================================================================================================')

        print('Best valid epoch is {}; Best val_RMSD: {:.3f}'.format(epoch_bvl, best_val_rmsd))
        writer.add_scalar('best rmsd/validation', best_val_rmsd, epoch_bvl)
        writer.close()

        print('============================================================================================================')

        return model


def eval3d(model, dataset):
    dataloader = DataLoader(dataset, num_workers=4, batch_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    mses, maes, rmses = 0., 0., 0.
    count = 0
    loss_total = 0
    dist_loss_total = 0
    for batch_data in tqdm(dataloader, total=len(dataloader), ncols=80):

        batch_data = batch_data.to(device)
        with torch.no_grad():
            mask_d_pred, mask, dist_count, coords_pred = model(batch_data, train=False)

        coords = batch_data.xyz

        batch = batch_data.batch

        coords = coords.to(torch.float64)
        pred_coords = coords_pred.to(torch.float64)

        torch.set_default_dtype(torch.float64)

        
    
        output, counts = torch.unique(batch, sorted = True, return_counts = True)
         
            
        counts_list = counts.tolist()
        coords_batch_split = torch.split(coords, counts_list)
        pred_coords_batch_split = torch.split(pred_coords, counts_list)
        
        coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
        pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)




        coords_tensor = coords_tensor - rmsd_batch.centroid(coords_tensor, counts)
        pred_coords_tensor = pred_coords_tensor - rmsd_batch.centroid(pred_coords_tensor, counts)

        counts += torch.arange(coords_tensor.shape[0], device=device) * coords_tensor.shape[1]
        b = torch.arange(coords_tensor.shape[0]*coords_tensor.shape[1], device=device).view(coords_tensor.shape[0], coords_tensor.shape[1])
        b = b < counts.view(coords_tensor.shape[0], 1)
        
        
        coords_tensor = coords_tensor * b.view(coords_tensor.shape[0], coords_tensor.shape[1], 1).float()
        pred_coords_tensor = pred_coords_tensor * b.view(coords_tensor.shape[0], coords_tensor.shape[1], 1).float()
        
           
        
      
        U_batch = rmsd_batch.kabsch(coords_tensor, pred_coords_tensor)
        coords_tensor = torch.matmul(coords_tensor, U_batch)

        loss = rmsd_batch.rmsd(coords_tensor, pred_coords_tensor)
        
        

          
        
        loss_total += loss.item()
    
        
        count += 1
    
    return loss_total / count
