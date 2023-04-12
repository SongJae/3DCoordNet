import torch
import os
import numpy as np
import random
import yaml
import argparse

from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from molx.dataset import Molecule3D
from molx.model import Deepergcn_dagnn_coordinate
from molx.mol3d import Mol3DTrainer_coordinate, eval3d_coordinate
from model.featurization import construct_loader
from model.parsing import parse_train_args, set_hyperparams
from utils import create_logger, dict_to_str, plot_train_val_loss, save_yaml_file, get_optimizer_and_scheduler
from commons.utils import seed_all, get_random_indices, TENSORBOARD_FUNCTIONS

from torch.utils.tensorboard import SummaryWriter

from trainer.metrics import QM9DenormalizedL1, QM9DenormalizedL2, \
    QM9SingleTargetDenormalizedL1, Rsquared, NegativeSimilarity, MeanPredictorLoss, \
    PositiveSimilarity, ContrastiveAccuracy, TrueNegativeRate, TruePositiveRate, Alignment, Uniformity, \
    BatchVariance, DimensionCovariance, MAE, PositiveSimilarityMultiplePositivesSeparate2d, \
    NegativeSimilarityMultiplePositivesSeparate2d, OGBEvaluator, PearsonR, PositiveProb, NegativeProb, \
    Conformer2DVariance, Conformer3DVariance, PCQM4MEvaluatorWrapper
from torch.optim import *  # do not remove

def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/pna.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='qm9', help='[qm9, zinc, drugs, geom_qm9, molhiv]')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--critic_loss', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--critic_loss_params', type=dict, default={},
                   help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--num_conformers', type=int, default=3,
                   help='number of conformers to use if we are using multiple conformers on the 3d side')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--frozen_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--required_data', default=[],
                   help='what will be included in a batch like [dgl_graph, targets, dgl_graph3d]')
    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--use_e_features', default=True, type=bool, help='ignore edge features if set to False')
    p.add_argument('--targets', default=[], help='properties that should be predicted')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--dist_embedding', type=bool, default=False, help='add dist embedding to complete graphs edges')
    p.add_argument('--num_radial', type=int, default=6, help='number of frequencies for distance embedding')
    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--model3d_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--model3d_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--critic_type', type=str, default=None, help='Classname of one of the models in the models dir')
    p.add_argument('--critic_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--force_random_split', type=bool, default=False, help='use random split for ogb')
    p.add_argument('--reuse_pre_train_data', type=bool, default=False, help='use all data instead of ignoring that used during pre-training')
    p.add_argument('--transfer_3d', type=bool, default=False, help='set true to load the 3d network instead of the 2d network')

    p.add_argument('--node_attr', type=int, default=0, metavar='N', help='node_attr or not')
    p.add_argument('--attention', type=int, default=1, metavar='N', help='attention in the ae model')
    p.add_argument('--nf', type=int, default=128, metavar='N', help='learning rate')
    p.add_argument('--n_layers', type=int, default=7, metavar='N', help='number of layers for the autoencoder')
    return p.parse_args()

def get_arguments():
    args = parse_arguments()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}

    return args


args = get_arguments()
conf = {}
conf['epochs'] = 150
conf['early_stopping'] = 200
conf['lr'] = 0.0001 #0.0001
conf['lr_decay_factor'] = 0.8
conf['lr_decay_step_size'] = 50
conf['dropout'] = 0
conf['weight_decay'] = 0
conf['depth'] = 6 #원래 3
conf['hidden'] = 256 #원래 256
conf['batch_size'] = 32
conf['save_ckpt'] = 'best_valid'
conf['out_path'] = 'results/without_index' #'results/coord_result_1/' 
conf['split'] = 'random' #'scaffold'
conf['criterion'] = 'mae'


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
root_dir = os.getcwd()


all_data = Molecule3D(root=root_dir, transform=None)
all_idx = get_random_indices(len(all_data), args.seed_data)
model_idx = all_idx[:100000]
test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(all_data))]
val_idx = all_idx[len(model_idx) + len(test_idx):]
train_idx = model_idx[:args.num_train]

if args.num_val != None:
      train_idx = all_idx[:args.num_train]
      val_idx = all_idx[len(train_idx): len(train_idx) + args.num_val]
      test_idx = all_idx[len(train_idx) + args.num_val: len(train_idx) + 2*args.num_val]

train_loader = DataLoader(Subset(all_data, train_idx), batch_size=conf['batch_size'], shuffle=True)
#train_loader = DataLoader(all_data[:2], batch_size=conf['batch_size'], shuffle=False)
val_loader = DataLoader(Subset(all_data, val_idx), batch_size=conf['batch_size'])
#val_loader = DataLoader(all_data[2:4], batch_size=conf['batch_size'], shuffle=False)
test_loader = DataLoader(Subset(all_data, test_idx), batch_size=conf['batch_size'])

model = Deepergcn_dagnn_coordinate(num_layers=conf['depth'], emb_dim=conf['hidden'], drop_ratio=conf['dropout'], JK="last", aggr='softmax', norm='batch').to(device)

trainer = Mol3DTrainer_coordinate(model, args, train_loader, val_loader, conf, device=device, optim=globals()[args.optimizer], scheduler_step_per_batch=args.scheduler_step_per_batch)
model = trainer.train(model)

best_model_path = os.path.join(conf['out_path'], 'ckpt_best_val.pth')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model'])

rmsd = eval3d_coordinate(model, test_loader, conf['batch_size'])
writer = SummaryWriter("coords_testing2")
print('epoch: {}; Test -- test_rmsd: {:.3f}'
      .format(checkpoint['epoch'], rmsd))
writer.add_scalar('rmsd/test', rmsd, checkpoint['epoch'])
writer.close()

