import torch
import os
import argparse
import yaml
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from molx.dataset import Molecule3DProps
from molx.model import Deepergcn_dagnn_coordinate, SchNet
from molx.mol3d import TransformPred3D, TransformCoord3D, TransformGT3D, TransformRDKit3D
from molx.proppred import RegTrainer
from molx.proppred import EGNN
from commons.utils import seed_all, get_random_indices, TENSORBOARD_FUNCTIONS
from trainer.metrics import QM9DenormalizedL1, QM9DenormalizedL2, \
    QM9SingleTargetDenormalizedL1, Rsquared, NegativeSimilarity, MeanPredictorLoss, \
    PositiveSimilarity, ContrastiveAccuracy, TrueNegativeRate, TruePositiveRate, Alignment, Uniformity, \
    BatchVariance, DimensionCovariance, MAE, PositiveSimilarityMultiplePositivesSeparate2d, \
    NegativeSimilarityMultiplePositivesSeparate2d, OGBEvaluator, PearsonR, PositiveProb, NegativeProb, \
    Conformer2DVariance, Conformer3DVariance, PCQM4MEvaluatorWrapper
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

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
conf['epochs'] = 2#1000
conf['early_stopping'] = 200
conf['lr'] = 0.001 #0.0001
conf['lr_decay_factor'] = 0.8
conf['lr_decay_step_size'] = 100
conf['weight_decay'] = 0
conf['batch_size'] = 128
conf['save_ckpt'] = 'best_valid'
conf['target'] = 7
conf['geoinput'] = 'coordinate'
conf['geo_model_path'] = 'results/coord_result/ckpt_best_val.pth'
conf['metric'] = 'mae'
conf['out_path'] = 'pred_prop_results'
conf['split'] = 'random' #'scaffold'

conf['hidden_channels'] = 128
conf['num_filters'] = 128
conf['num_interactions'] = 6
conf['num_gaussians'] = 50
conf['cutoff'] = 10.0
conf['readout'] = 'add'
conf['dipole'] = False
conf['mean'] = None
conf['std'] = None
conf['atomref'] = None

conf['depth'] = 6
conf['emb_dim'] = 256
conf['dropout'] = 0
conf['norm'] = 'batch'
conf['JK'] = 'last'
conf['aggr'] = 'softmax'

metrics_dict = {'rsquared': Rsquared(),
                'mae': MAE(),
                'pearsonr': PearsonR(),
                'ogbg-molhiv': OGBEvaluator(d_name='ogbg-molhiv', metric='rocauc'),
                'ogbg-molpcba': OGBEvaluator(d_name='ogbg-molpcba', metric='ap'),
                'ogbg-molbace': OGBEvaluator(d_name='ogbg-molbace', metric='rocauc'),
                'ogbg-molbbbp': OGBEvaluator(d_name='ogbg-molbbbp', metric='rocauc'),
                'ogbg-molclintox': OGBEvaluator(d_name='ogbg-molclintox', metric='rocauc'),
                'ogbg-moltoxcast': OGBEvaluator(d_name='ogbg-moltoxcast', metric='rocauc'),
                'ogbg-moltox21': OGBEvaluator(d_name='ogbg-moltox21', metric='rocauc'),
                'ogbg-mollipo': OGBEvaluator(d_name='ogbg-mollipo', metric='rmse'),
                'ogbg-molmuv': OGBEvaluator(d_name='ogbg-molmuv', metric='ap'),
                'ogbg-molsider': OGBEvaluator(d_name='ogbg-molsider', metric='rocauc'),
                'ogbg-molfreesolv': OGBEvaluator(d_name='ogbg-molfreesolv', metric='rmse'),
                'ogbg-molesol': OGBEvaluator(d_name='ogbg-molesol', metric='rmse'),
                'pcqm4m': PCQM4MEvaluatorWrapper(),
                'conformer_3d_variance': Conformer3DVariance(),
                'conformer_2d_variance': Conformer2DVariance(),
                'positive_similarity': PositiveSimilarity(),
                'positive_similarity_multiple_positives_separate2d': PositiveSimilarityMultiplePositivesSeparate2d(),
                'positive_prob': PositiveProb(),
                'negative_prob': NegativeProb(),
                'negative_similarity': NegativeSimilarity(),
                'negative_similarity_multiple_positives_separate2d': NegativeSimilarityMultiplePositivesSeparate2d(),
                'contrastive_accuracy': ContrastiveAccuracy(threshold=0.5009),
                'true_negative_rate': TrueNegativeRate(threshold=0.5009),
                'true_positive_rate': TruePositiveRate(threshold=0.5009),
                'uniformity': Uniformity(t=2),
                'alignment': Alignment(alpha=2),
                'batch_variance': BatchVariance(),
                'dimension_covariance': DimensionCovariance()
                }
metrics = ['mae_denormalized', 'pearsonr', 'rsquared', 'qm9_properties']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
root_dir = os.getcwd()

if conf['geoinput'] == 'pred':
    geo_model = Deepergcn_dagnn_dist(num_layers=conf['depth'], emb_dim=conf['emb_dim'], drop_ratio=conf['dropout'], 
        JK=conf['JK'], aggr=conf['aggr'], norm=conf['norm']).to(device)
    geo_model_ckpt = torch.load(conf['geo_model_path'])
    geo_model.load_state_dict(geo_model_ckpt['model'])
    transform=TransformPred3D(geo_model, target_id=conf['target'], device=device)
elif conf['geoinput'] == 'coordinate':
    geo_model = Deepergcn_dagnn_coordinate(num_layers=conf['depth'], emb_dim=conf['emb_dim'], drop_ratio=conf['dropout'], JK=conf['JK'], aggr=conf['aggr'], norm=conf['norm']).to(device)
    geo_model_ckpt = torch.load(conf['geo_model_path'], map_location=device)
    geo_model.load_state_dict(geo_model_ckpt['model'])
    transform = TransformCoord3D(geo_model, target_id=conf['target'], device=device, batch_size=conf['batch_size'] , train=False)
elif conf['geoinput'] == 'gt':
    transform = TransformGT3D(target_id=conf['target'])
elif conf['geoinput'] == 'rdkit':
    transform = TransformRDKit3D(target_id=conf['target'])

all_data = Molecule3DProps(root=root_dir, pre_transform=transform, target_tasks = args.targets)
print("all data length: ", len(all_data))
all_idx = get_random_indices(len(all_data), args.seed_data)
model_idx = all_idx[:100000]
test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(all_data))]
val_idx = all_idx[len(model_idx) + len(test_idx):]
train_idx = model_idx[:args.num_train]

#transfer from same dataset
num_pretrain = 50000
train_idx = model_idx[num_pretrain: num_pretrain + args.num_train]

if args.num_val != None:
      train_idx = all_idx[:args.num_train]
      val_idx = all_idx[len(train_idx): len(train_idx) + args.num_val]
      test_idx = all_idx[len(train_idx) + args.num_val: len(train_idx) + 2*args.num_val]

train_loader = DataLoader(Subset(all_data, train_idx), batch_size=conf['batch_size'], shuffle=True)
print("train_loader: ", train_loader, len(Subset(all_data, train_idx)))
val_loader = DataLoader(Subset(all_data, val_idx), batch_size=conf['batch_size'])
print("val_loader: ", val_loader, len(Subset(all_data, val_idx)))
test_loader = DataLoader(Subset(all_data, test_idx), batch_size=conf['batch_size'])
print("test_loader: ", test_loader, len(Subset(all_data, test_idx)))

metrics_dict.update({'mae_denormalized': QM9DenormalizedL1(dataset=all_data),
                        'mse_denormalized': QM9DenormalizedL2(dataset=all_data)})
metrics = {metric: metrics_dict[metric] for metric in metrics if metric != 'qm9_properties'}
if 'qm9_properties' in metrics:
    metrics.update(
        {task: QM9SingleTargetDenormalizedL1(dataset=all_data, task=task) for task in all_data.target_tasks})
print("metrics: ", metrics)
print("args.loss_func: ", args.loss_func)
print("args.loss_params: ", args.loss_params)
model = EGNN(in_node_nf=9, in_edge_nf=0, hidden_nf=args.nf, device=device, n_layers=args.n_layers, coords_weight=1.0,
             attention=args.attention, node_attr=args.node_attr).to(device)
tensorboard_functions = {function: TENSORBOARD_FUNCTIONS[function] for function in args.tensorboard_functions}
trainer = RegTrainer(model, args, train_loader, val_loader, conf, device, metrics, main_metric=args.main_metric, tensorboard_functions=tensorboard_functions, optim=globals()[args.optimizer], loss_func=globals()[args.loss_func](**args.loss_params), scheduler_step_per_batch=args.scheduler_step_per_batch)
# model = SchNet(hidden_channels=conf['hidden_channels'], num_filters=conf['num_filters'], num_interactions=conf['num_interactions'],
#     num_gaussians=conf['num_gaussians'], cutoff=conf['cutoff'], readout=conf['readout'], dipole=conf['dipole'],
#     mean=conf['mean'], std=conf['std'], atomref=conf['atomref']).to(device)

#model = trainer.train(model)
val_metrics = trainer.train()
if args.eval_on_test:
    checkpoint = torch.load(os.path.join(trainer.writer.log_dir, 'best_checkpoint.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = trainer.evaluation(test_loader, data_split='test')


# best_model_path = os.path.join(conf['out_path'], 'ckpt_best_val.pth')
# checkpoint = torch.load(best_model_path)
# model.load_state_dict(checkpoint['model'])

# rmse = eval_reg(model, test_loader, conf['metric'], conf['geoinput'])
# print('epoch: {}; Test -- test_RMSE: {:.3f};'
#       .format(checkpoint['epoch'], rmse))