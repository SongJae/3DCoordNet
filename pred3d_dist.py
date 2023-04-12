import torch
import os
from molx.dataset import Molecule3D
from molx.model import Deepergcn_dagnn_dist, Deepergcn_dagnn_coords
from molx.mol3d import Mol3DTrainer, eval3d
from torch.utils.tensorboard import SummaryWriter


conf = {}
conf['epochs'] = 100
conf['early_stopping'] = 200
conf['lr'] = 0.0001
conf['lr_decay_factor'] = 0.8
conf['lr_decay_step_size'] = 10
conf['dropout'] = 0
conf['weight_decay'] = 0
conf['depth'] = 3
conf['hidden'] = 256
conf['batch_size'] = 128
conf['save_ckpt'] = 'best_valid'
conf['out_path'] = 'results/dist_testing/'
conf['split'] = 'random' #'scaffold'
conf['criterion'] = 'mae'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
root_dir = os.getcwd()

train_dataset = Molecule3D(root=root_dir, transform=None, split='train', split_mode=conf['split'])

val_dataset = Molecule3D(root=root_dir, transform=None, split='val', split_mode=conf['split'])

test_dataset = Molecule3D(root=root_dir, transform=None, split='test', split_mode=conf['split'])


model = Deepergcn_dagnn_coords(num_layers=conf['depth'], emb_dim=conf['hidden'], drop_ratio=conf['dropout'], JK="last", aggr='softmax', norm='batch').to(device)
trainer = Mol3DTrainer(train_dataset, val_dataset, conf, device=device)
model = trainer.train(model)



best_model_path = os.path.join(conf['out_path'], 'ckpt_best_val.pth')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model'])

rmsd = eval3d(model, test_dataset)
print('epoch: {}; Test -- test_rmsd: {:.3f}'.format(checkpoint['epoch'], rmsd))
writer = SummaryWriter()
writer.add_scalar('rmsd/test', rmsd, checkpoint['epoch'])
writer.close()

