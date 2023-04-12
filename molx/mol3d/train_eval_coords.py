import os
import torch
import numpy as np
import shutil
from tqdm import tqdm
from torch_geometric.data import DataLoader
from .utils import generate_xyz
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from TorchProteinLibrary import RMSD
from trainer.lr_schedulers import WarmUpWrapper  # do not remove
from torch.optim.lr_scheduler import *  # For loading optimizer specified in config
from datetime import datetime
from commons.utils import flatten_dict, tensorboard_gradient_magnitude, move_to_device

class Mol3DTrainer_coordinate():
    def __init__(self, model, args, train_loader, val_loader, configs, device, optim=None, scheduler_step_per_batch: bool = True ):
        self.model = model
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        #self.train_loader = train_dataset
        #self.val_loader = val_dataset
        self.out_path = configs['out_path']
        self.start_epoch = 1
        self.device = device
        self.batch_size = configs['batch_size']
        self.args = args
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.initialize_optimizer(optim)
        self.initialize_scheduler()
        self.optim_steps = 0
        self.writer = SummaryWriter(
        '{}/{}_{}_{}_{}_{}'.format(args.logdir, args.model_type, args.dataset, args.experiment_name, args.seed,
                                datetime.now().strftime('%d-%m_%H-%M-%S')))
        shutil.copyfile(self.args.config.name,
                        os.path.join(self.writer.log_dir, os.path.basename(self.args.config.name)))
        print('Log directory: ', self.writer.log_dir)

    def after_optim_step(self):
        if self.optim_steps % self.args.log_iterations == 0:
            tensorboard_gradient_magnitude(self.optim, self.writer, self.optim_steps)
        if self.lr_scheduler != None and (self.scheduler_step_per_batch or (isinstance(self.lr_scheduler,
                                                                                        WarmUpWrapper) and self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
            self.step_schedulers()

    def initialize_optimizer(self, optim):
        transferred_keys = [k for k in self.model.state_dict().keys() if
                            any(transfer_layer in k for transfer_layer in self.args.transfer_layers) and not any(
                                to_exclude in k for to_exclude in self.args.exclude_from_transfer)]
        frozen_keys = [k for k in self.model.state_dict().keys() if any(to_freeze in k for to_freeze in self.args.frozen_layers)]
        frozen_params = [v for k, v in self.model.named_parameters() if k in frozen_keys]
        transferred_params = [v for k, v in self.model.named_parameters() if k in transferred_keys]
        new_params = [v for k, v in self.model.named_parameters() if
                      k not in transferred_keys and 'batch_norm' not in k and k not in frozen_keys]
        batch_norm_params = [v for k, v in self.model.named_parameters() if
                             'batch_norm' in k and k not in transferred_keys and k not in frozen_keys]

        transfer_lr = self.args.optimizer_params['lr'] if self.args.transferred_lr == None else self.args.transferred_lr
        # the order of the params here determines in which order they will start being updated during warmup when using ordered warmup in the warmupwrapper
        param_groups = []
        if batch_norm_params != []:
            param_groups.append({'params': batch_norm_params, 'weight_decay': 0})
        param_groups.append({'params': new_params})
        if transferred_params != []:
            param_groups.append({'params': transferred_params, 'lr': transfer_lr})
        if frozen_params != []:
            param_groups.append({'params': frozen_params, 'lr': 0})
       
        self.optim = optim(param_groups, **self.args.optimizer_params)

    def step_schedulers(self, metrics=None):
        try:
            self.lr_scheduler.step(metrics=metrics)
        except:
            self.lr_scheduler.step()

    def initialize_scheduler(self):
        if self.args.lr_scheduler:  # Needs "from torch.optim.lr_scheduler import *" to work
            self.lr_scheduler = globals()[self.args.lr_scheduler](self.optim, **self.args.lr_scheduler_params)
        else:
            self.lr_scheduler = None

    def _get_loss(self):
        if self.configs['criterion'] == 'mse':
            return torch.nn.MSELoss(reduction='sum')
        elif self.configs['criterion'] == 'mae':
            return torch.nn.L1Loss(reduction='sum')


    def _train_loss(self, model, criterion):
        model.train()
        loss_total = 0
        i = 0
        rmsd = RMSD.Coords2RMSD().cuda()
        #for batch_data in tqdm(self.train_loader, total=len(self.train_loader)):
        for batch_data in tqdm(self.train_loader, total=len(self.train_loader)):
            #optimizer.zero_grad()
           

            batch_data = batch_data.to(self.device)
            batch = batch_data.batch
            coords = batch_data.xyz
            #print("coords: ", coords, coords.shape)
           
            pred_coords = model(batch_data, self.batch_size, train=True)
            #print("pred_coords: ", pred_coords, pred_coords.shape)
           
      
            output, counts = torch.unique(batch, sorted = True, return_counts = True)
            counts = counts.type(torch.int)
          
            
                      
            # counts_list = counts.tolist()

            # counts_list.insert(0, 49)
            # a = torch.ones((49, 3), device=self.device)
            # coords = torch.cat((a, coords), 0)
            # coords_batch_split = torch.split(coords, counts_list, dim=0)
            # coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
            # coords_tensor = coords_tensor[1:]
           

            # # coords_batch_split = torch.split(coords, counts_list, dim=0)
            # # pred_coords_batch_split = torch.split(pred_coords, counts_list, dim=0)
            
            # # coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
            # # pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)
            
            
            # coords_tensor = torch.reshape(coords_tensor,(coords_tensor.shape[0],coords_tensor.shape[1]*3))
            # pred_coords_tensor = torch.reshape(pred_coords_tensor,(pred_coords_tensor.shape[0],pred_coords_tensor.shape[1]*3))
            # #print("shape coords_tensor: ", coords_tensor.shape)
            # #print("shape pred_coords_tensor: ", pred_coords_tensor.shape)

            counts_list = counts.tolist()
            coords_batch_split = torch.split(coords, counts_list)
            pred_coords_batch_split = torch.split(pred_coords, counts_list)
            
            coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
            pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)
            #print("coords_tensor: ", coords_tensor, coords_tensor.shape)
            
            
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
            #print("pred_coords_tensor before rmsd: ", pred_coords_tensor, pred_coords_tensor.shape)
            loss = rmsd(pred_coords_tensor, coords_tensor, counts).mean()
            
            #loss = local_loss.mean()
            if self.optim != None:
                loss.backward()
                self.optim.step()
                self.after_optim_step()
                self.optim.zero_grad()
                self.optim_steps += 1
            #optimizer.step()
           
            loss_total += loss.item()
            i += 1
        return loss_total / i


    def save_ckpt(self, epoch, model, best_valid=False):
        # checkpoint = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict(),
        #     'epoch': epoch,
        # }
        for param_group in self.optim.param_groups:
            print("learning rate: ", param_group['lr'])
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'epoch': epoch,
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
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
            
        #optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['lr'], weight_decay=self.configs['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=self.configs['lr']/100)
        criterion = self._get_loss()
        if 'load_pth' in self.configs and self.configs['load_pth'] is not None:
            checkpoint = torch.load(self.configs['load_pth'])
            model.load_state_dict(checkpoint['model'])
            self.optim.load_state_dict(checkpoint['optimizer'])
            #scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
        else:
          print("Do not load path")
        
        best_val_rmsd = 10000
        epoch_bvl = 0
        for i in range(self.start_epoch, self.args.num_epochs + 1):
            loss_dist = self._train_loss(model, criterion)
            rmsd = eval3d_coordinate(model, self.val_loader, self.batch_size)

            # One possible way to selection model: do testing when val metric is best
            if self.configs['save_ckpt'] == "best_valid":
                if rmsd < best_val_rmsd:
                    epoch_bvl = i
                    best_val_rmsd = rmsd
                    if self.out_path is not None:
                        self.save_ckpt(i, model, best_valid=True)
                        #self.save_ckpt(i, model, optimizer, scheduler, best_valid=True)

            # Or we can save model at each epoch
            elif i % self.configs['save_ckpt'] == 0:
                if self.out_path is not None:
                    self.save_ckpt(i, model, best_valid=False)
                    #self.save_ckpt(i, model, optimizer, scheduler, best_valid=False)
            
            # if i % self.configs['lr_decay_step_size'] == 0:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = self.configs['lr_decay_factor'] * param_group['lr']
        
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


def eval3d_coordinate(model, test_loader, batch_size):
    dataloader = test_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    i = 0
    loss_total = 0
    rmsd = RMSD.Coords2RMSD().cuda()
    #for batch_data in tqdm(dataloader, total=len(dataloader), ncols=80):
    for batch_data in tqdm(dataloader, total=len(dataloader)):
        batch_data = batch_data.to(device)
        with torch.no_grad():
          pred_coords = model(batch_data, batch_size, train=False)
        
        coords = batch_data.xyz
    
        batch = batch_data.batch
                 
        output, counts = torch.unique(batch, sorted = True, return_counts = True)
        counts = counts.type(torch.int)
            
        counts_list = counts.tolist()
        # coords_batch_split = torch.split(coords, counts_list)
        # pred_coords_batch_split = torch.split(pred_coords, counts_list)
        
        # coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
        # pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)

        
        # counts_list.insert(0, 49)
        # a = torch.ones((49, 3), device=device)
        # coords = torch.cat((a, coords), 0)
        # coords_batch_split = torch.split(coords, counts_list, dim=0)
    
    
        # coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
        # coords_tensor = coords_tensor[1:]
        
        # """
        # print("coords_tensor: ", coords_tensor)
        # print("pred_coords_tensor: ", pred_coords_tensor)
        # print("batch: ", batch)
        # print("counts: ", counts)
        # """
        # coords_tensor = torch.reshape(coords_tensor,(coords_tensor.shape[0],coords_tensor.shape[1]*3))
        # pred_coords_tensor = torch.reshape(pred_coords_tensor,(pred_coords_tensor.shape[0],pred_coords_tensor.shape[1]*3))
        coords_batch_split = torch.split(coords, counts_list)
        pred_coords_batch_split = torch.split(pred_coords, counts_list)
        
        coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
        pred_coords_tensor = pad_sequence(pred_coords_batch_split, batch_first = True)

        coords_tensor = torch.reshape(coords_tensor,(coords_tensor.shape[0],coords_tensor.shape[1]*3))
        pred_coords_tensor = torch.reshape(pred_coords_tensor,(pred_coords_tensor.shape[0],pred_coords_tensor.shape[1]*3))
        
       
        loss = rmsd(pred_coords_tensor, coords_tensor, counts).mean()
       
        #print("val smiles:", batch_data.smiles)
        #print("val loss: ", loss)
        
        loss_total += loss.item()
        i += 1

    
    return loss_total / i, i, coords_tensor, pred_coords_tensor
