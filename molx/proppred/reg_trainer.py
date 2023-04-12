import os
import torch
import numpy as np
import shutil
import copy
import pyaml
import inspect
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch.nn.modules.loss import L1Loss
from typing import Tuple, Dict, Callable, Union
from torch.utils.tensorboard import SummaryWriter
from commons.utils import flatten_dict, tensorboard_gradient_magnitude, move_to_device
from torch.utils.tensorboard.summary import hparams
from trainer.lr_schedulers import WarmUpWrapper  # do not remove
from torch.optim.lr_scheduler import *  # For loading optimizer specified in config
from datetime import datetime
from .models import EGNN
from torch.nn.utils.rnn import pad_sequence
from qm9 import utils as qm9_utils

dtype = torch.float32
class RegTrainer():
    def __init__(self, model, args, train_loader, val_loader, configs, device,  metrics: Dict[str, Callable], main_metric: str,  tensorboard_functions: Dict[str, Callable], optim=None, loss_func=torch.nn.MSELoss(), scheduler_step_per_batch: bool = True):
        self.model = model
        self.configs = configs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target = configs['target']
        self.geoinput = configs['geoinput']
        self.out_path = configs['out_path']
        self.device = device
        self.step = 0
        self.start_epoch = 1
        self.args = args
        self.loss_func = loss_func
        self.tensorboard_functions = tensorboard_functions
        self.metrics = metrics
        self.val_per_batch = args.val_per_batch
        self.main_metric = type(self.loss_func).__name__ if main_metric == 'loss' else main_metric
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.initialize_optimizer(optim)
        self.initialize_scheduler()

        self.start_epoch = 1
        self.optim_steps = 0
        self.best_val_score = np.inf  # running score to decide whether or not a new model should be saved
        self.writer = SummaryWriter(
            '{}/{}_{}_{}_{}_{}'.format(args.logdir, args.model_type, args.dataset, args.experiment_name, args.seed,
                                    datetime.now().strftime('%d-%m_%H-%M-%S')))
        shutil.copyfile(self.args.config.name,
                        os.path.join(self.writer.log_dir, os.path.basename(self.args.config.name)))
        print('Log directory: ', self.writer.log_dir)
        self.hparams = copy.copy(args).__dict__
        for key, value in flatten_dict(self.hparams).items():
            print(f'{key}: {value}')

    def run_per_epoch_evaluations(self, loader):
        pass

    def train(self):
        if self.configs['out_path'] is not None:
            try:
                os.makedirs(self.configs['out_path'])
            except OSError:
                pass
        epochs_no_improve = 0

        for epoch in range(self.start_epoch, self.args.num_epochs + 1):
            self.model.train()
            self.predict(self.train_loader, epoch, optim=self.optim)
            
            self.model.eval()
            with torch.no_grad():
                metrics = self.predict(self.val_loader, epoch)
                val_score = metrics[self.main_metric]

                if self.lr_scheduler != None and not self.scheduler_step_per_batch:
                    self.step_schedulers(metrics=val_score)

                if self.args.eval_per_epochs > 0 and epoch % self.args.eval_per_epochs == 0:
                    self.run_per_epoch_evaluations(self.val_loader)

                self.tensorboard_log(metrics, data_split='val', epoch=epoch, log_hparam=True, step=self.optim_steps)
                val_loss = metrics[type(self.loss_func).__name__]
                print('[Epoch %d] %s: %.6f val loss: %.6f' % (epoch, self.main_metric, val_score, val_loss))

                # save the model with the best main_metric depending on wether we want to maximize or minimize the main metric
                if val_score <= self.best_val_score:
                    epochs_no_improve = 0
                    self.best_val_score = val_score
                    self.save_checkpoint(epoch, checkpoint_name='best_checkpoint.pt')
                else:
                    epochs_no_improve += 1
                self.save_checkpoint(epoch, checkpoint_name='last_checkpoint.pt')

                if epochs_no_improve >= self.args.patience and epoch >= self.args.minimum_epochs:  # stopping criterion
                    print(
                        f'Early stopping reached after {epoch} epochs. Best model checkpoint was in epoch {epoch - epochs_no_improve}.')
                    break
                if epoch in self.args.models_to_save:
                    shutil.copyfile(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'), os.path.join(self.writer.log_dir, f'best_checkpoint_{epoch}epochs.pt'))

        # evaluate on best checkpoint
        checkpoint = torch.load(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.evaluation(self.val_loader, data_split='val_best_checkpoint')


      
    def forward_pass(self, nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, batch, optim):
        # targets = batch[-1]  # the last entry of the batch tuple is always the targets
        # predictions = self.model(*batch[0])  # foward the rest of the batch to the model
       
        targets =  torch.squeeze(batch.y)
        if self.geoinput in ['gt', 'rdkit', 'coordinate']:
            # Use ground truth position
            #predictions = model(batch, dist_index=None, dist_weight=None)
            predictions = self.model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes, batch=batch.batch)
        

        elif self.geoinput == 'pred':
            # Use predicted position
            # xs = self.EDMModel(batch_data)
            # dist_index_pred, dist_weight_pred = self.from_xs_to_edm(xs, batch_data.batch)
            predictions = self.model(batch, dist_index=batch.dist_index, dist_weight=batch.dist_weight)
        
        elif self.geoinput == '2d':
            # Used molecular graph
            predictions = self.model(batch)

        else:
            raise NameError('Must use gt, rdkit, coordinate, 2d or pred for edm in arguments!')
        predictions = torch.squeeze(predictions)
        # print("predictions: ", predictions, predictions.shape)
        # print("targets: ", targets, targets.shape)
        return self.loss_func(predictions, targets), predictions, targets

    def process_batch(self, nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, batch, optim):
        loss, predictions, targets = self.forward_pass(nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, batch, optim)
        if optim != None:  # run backpropagation if an optimizer is provided
            #optim.zero_grad()
            loss.backward()
            #optim.step()
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss, predictions.detach(), targets.detach()

    def predict(self, data_loader: DataLoader, epoch: int, optim: torch.optim.Optimizer = None,
                return_predictions: bool = False) -> Union[
        Dict, Tuple[float, Union[torch.Tensor, None], Union[torch.Tensor, None]]]:
        total_metrics = {k: 0 for k in
                         list(self.metrics.keys()) + [type(self.loss_func).__name__, 'mean_pred', 'std_pred',
                                                      'mean_targets', 'std_targets']}
        epoch_targets = torch.tensor([]).to(self.device)
        epoch_predictions = torch.tensor([]).to(self.device)
        epoch_loss = 0
        i = 0
        for i, batch in enumerate(data_loader):
           
            batch = batch.to(self.device)
            
            coords = batch.xyz
            batches = batch.batch
            output, counts = torch.unique(batches, sorted = True, return_counts = True)
            counts = counts.type(torch.int)
            counts_list = counts.tolist()
            
            coords_batch_split = torch.split(coords, counts_list)
            nodes_batch_split = torch.split(batch.x, counts_list)
            
            coords_tensor = pad_sequence(coords_batch_split, batch_first = True)
            nodes_tensor = pad_sequence(nodes_batch_split, batch_first = True)

            batch_size, n_nodes, _ = coords_tensor.size()
            atom_mask = torch.ones(coords.shape[0], 1, device=self.device)
            atom_mask_split = torch.split(atom_mask, counts_list)
            atom_mask = pad_sequence(atom_mask_split, batch_first = True)
            atom_mask = atom_mask.reshape(batch_size, -1)
            edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=self.device).unsqueeze(0)
            edge_mask *= diag_mask
            
            atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(self.device, dtype)
            atom_positions = coords_tensor.view(batch_size * n_nodes, -1).to(self.device, dtype)
            edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).to(self.device, dtype)
            nodes = nodes_tensor.view(batch_size * n_nodes, -1).to(self.device, dtype)
            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)
            
            # nodes = batch.x.to(self.device, dtype)
            # atom_positions = batch.xyz.to(self.device, dtype)
            # edges = batch.edge_index
            

            
            loss, predictions, targets = self.process_batch(nodes, atom_positions, edges, atom_mask, edge_mask, n_nodes, batch, optim)
            i += 1
            self.step += 1

            with torch.no_grad():
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='train')
                    self.tensorboard_log(metrics, data_split='train', step=self.optim_steps, epoch=epoch)
                    print('[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' % (
                        epoch, i + 1, len(data_loader), 'train', loss.item()))
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics(predictions, targets, val=True)
                    metrics_results[type(self.loss_func).__name__] = loss.item()
                    if i ==0 and epoch in self.args.models_to_save:
                        self.run_tensorboard_functions(predictions, targets, step=self.optim_steps, data_split='val')
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += loss.item()
                    epoch_targets = torch.cat((targets, epoch_targets), 0)
                    epoch_predictions = torch.cat((predictions, epoch_predictions), 0)

        if optim == None:
            if self.val_per_batch:
                total_metrics = {k: v / len(data_loader) for k, v in total_metrics.items()}
            else:
                total_metrics = self.evaluate_metrics(epoch_predictions, epoch_targets, val=True)
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(data_loader)
            return total_metrics

    def after_optim_step(self):
        if self.optim_steps % self.args.log_iterations == 0:
            tensorboard_gradient_magnitude(self.optim, self.writer, self.optim_steps)
        if self.lr_scheduler != None and (self.scheduler_step_per_batch or (isinstance(self.lr_scheduler,
                                                                                       WarmUpWrapper) and self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
            self.step_schedulers()

    def evaluate_metrics(self, predictions, targets, batch=None, val=False) -> Dict[str, float]:
        metrics = {}
        metrics[f'mean_pred'] = torch.mean(predictions).item()
        metrics[f'std_pred'] = torch.std(predictions).item()
        metrics[f'mean_targets'] = torch.mean(targets).item()
        metrics[f'std_targets'] = torch.std(targets).item()
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metrics[key] = metric(predictions, targets).item()
        return metrics

    def tensorboard_log(self, metrics, data_split: str, epoch: int, step: int, log_hparam: bool = False):
        metrics['epoch'] = epoch
        for i, param_group in enumerate(self.optim.param_groups):
            metrics[f'lr_param_group_{i}'] = param_group['lr']
        logs = {}
        for key, metric in metrics.items():
            metric_name = f'{key}/{data_split}'
            logs[metric_name] = metric
            self.writer.add_scalar(metric_name, metric, step)

        if log_hparam:  # write hyperparameters to tensorboard
            exp, ssi, sei = hparams(flatten_dict(self.hparams), flatten_dict(logs))
            self.writer.file_writer.add_summary(exp)
            self.writer.file_writer.add_summary(ssi)
            self.writer.file_writer.add_summary(sei)

    def run_tensorboard_functions(self, predictions, targets, step, data_split):
        for key, tensorboard_function in self.tensorboard_functions.items():
            tensorboard_function(predictions, targets, self.writer, step, data_split=data_split)
   
    def evaluation(self, data_loader: DataLoader, data_split: str = ''):
        self.model.eval()
        metrics = self.predict(data_loader, epoch=2)

        with open(os.path.join(self.writer.log_dir, 'evaluation_' + data_split + '.txt'), 'w') as file:
            print("came here")
            print('Statistics on ', data_split)
            for key, value in metrics.items():
                file.write(f'{key}: {value}\n')
                print(f'{key}: {value}')
        return metrics

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

    def save_checkpoint(self, epoch: int, checkpoint_name: str):
        """
        Saves checkpoint of model in the logdir of the summarywriter in the used rundi
        """
        run_dir = self.writer.log_dir
        self.save_model_state(epoch, checkpoint_name)
        train_args = copy.copy(self.args)
        # when loading from a checkpoint the config entry is a string. Otherwise it is a file object
        config_path = self.args.config if isinstance(self.args.config, str) else self.args.config.name
        train_args.config = os.path.join(run_dir, os.path.basename(config_path))
        with open(os.path.join(run_dir, 'train_arguments.yaml'), 'w') as yaml_path:
            pyaml.dump(train_args.__dict__, yaml_path)

        # Get the class of the used model (works because of the "from models import *" calling the init.py in the models dir)
        model_class = globals()[type(self.model).__name__]
        source_code = inspect.getsource(model_class)  # Get the sourcecode of the class of the model.
        file_name = os.path.basename(inspect.getfile(model_class))
        with open(os.path.join(run_dir, file_name), "w") as f:
            f.write(source_code)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
