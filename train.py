import os
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from models.fragen import Fragen
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.train import *
from time import time
from utils.train import get_model_loss
from utils.datasets.dataset import SurfLigandPairDataset
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/train.yml')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--logdir', type=str, default='./logs')

args = parser.parse_args()
config = load_config(args.config)
config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
seed_all(config.train.seed)

log_dir = get_new_log_dir(args.logdir, prefix=config_name)
ckpt_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
logger = get_logger('train', log_dir)
logger.info(args)
logger.info(config)
shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
shutil.copytree('./models', os.path.join(log_dir, 'models'))

protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom()
masking = get_mask(config.train.transform.mask)
composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, config.model.encoder.knn)

edge_sampler = EdgeSample(config.train.transform.edgesampler)
cfg_ctr = config.train.transform.contrastive
contrastive_sampler = ContrastiveSample(cfg_ctr.num_real, cfg_ctr.num_fake, cfg_ctr.pos_real_std, cfg_ctr.pos_fake_std, config.model.field.knn)
transform = Compose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    Geodesic_builder(),
    masking,
    composer,
    FocalBuilder(),
    edge_sampler,
    contrastive_sampler,
])

dataset, subsets = get_dataset(
    config = config.dataset,
    transform = transform,
)
train_set, val_set = subsets['train'], subsets['test']
follow_batch = []
collate_exclude_keys = ['ligand_nbh_list']
train_iterator = inf_iterator(DataLoader(
    train_set,  # type: ignore
    batch_size = config.train.batch_size,  # type: ignore
    shuffle = True,
    num_workers = config.train.num_workers, # type: ignore
    pin_memory = config.train.pin_memory, # type: ignore
    follow_batch = follow_batch,
    exclude_keys = collate_exclude_keys,
))

# train_set = list(train_set)
# random.shuffle(train_set)
# train_set = train_set[:len(train_set)//2]
train_loader = DataLoader(train_set, config.train.batch_size , shuffle=True,  exclude_keys = collate_exclude_keys)
val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False, follow_batch=follow_batch, exclude_keys = collate_exclude_keys,)


model = SurfGen(
    config.model, 
    num_classes = contrastive_sampler.num_elements,
    num_bond_types = edge_sampler.num_bond_types,
    protein_atom_feature_dim = protein_featurizer.feature_dim,
    ligand_atom_feature_dim = ligand_featurizer.feature_dim,
).to(args.device)

print('Num of parameters is {0:.4}M'.format(np.sum([p.numel() for p in model.parameters()]) /100000 ))

optimizer = get_optimizer(config.train.optimizer, model)
scheduler = get_scheduler(config.train.scheduler, optimizer)

def update_losses(eval_loss, loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf):
    eval_loss['total'].append(loss)
    eval_loss['frontier'].append(loss_frontier)
    eval_loss['pos'].append(loss_pos)
    eval_loss['cls'].append(loss_cls)
    eval_loss['edge'].append(loss_edge)
    eval_loss['real'].append(loss_real)
    eval_loss['fake'].append(loss_fake)
    eval_loss['surf'].append(loss_surf)
    return eval_loss
def evaluate(epoch, verbose=1):
    model.eval()
    eval_start = time()
    #eval_losses = {'total':[], 'frontier':[], 'pos':[], 'cls':[], 'edge':[], 'real':[], 'fake':[], 'surf':[] }
    eval_losses = []
    for batch in val_loader:
        batch = batch.to(args.device)  
        loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = get_model_loss(model, batch, config )
        eval_losses.append(loss.item())    
    average_loss = sum(eval_losses) / len(eval_losses)
    if verbose:
        logger.info('Evaluate Epoch %d | Average_Loss %.5f | Single Batch Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f | Loss(Surf) %.6f  ' % (
                epoch, average_loss,  loss.item(), loss_frontier.item(), loss_pos.item(), loss_cls.item(), loss_edge.item(), loss_real.item(), loss_fake.item(), loss_surf.item()
                ))
    return average_loss

def load(checkpoint, epoch=None, load_optimizer=False, load_scheduler=False):
    
    epoch = str(epoch) if epoch is not None else ''
    checkpoint = os.path.join(checkpoint,epoch)
    logger.info("Load checkpoint from %s" % checkpoint)

    state = torch.load(checkpoint, map_location=args.device)   
    model.load_state_dict(state["model"])
    #self._model.load_state_dict(state["model"], strict=False)
    #best_loss = state['best_loss']
    #start_epoch = state['cur_epoch'] + 1

    if load_scheduler:
        scheduler.load_state_dict(state["scheduler"])
        
    if load_optimizer:
        optimizer.load_state_dict(state["optimizer"])
        if args.device == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(args.devicedevice)
    return state['best_loss']


def train(verbose=1, num_epoches=300):
    train_start = time()
    train_losses = []
    val_losses = []
    start_epoch = 0
    best_loss = 1000
    if config.train.resume_train:  # type: ignore
        ckpt_name = config.train.ckpt_name  # type: ignore
        start_epoch = int(config.train.start_epoch)  # type: ignore
        best_loss = load(osp.join(config.train.checkpoint_path, ckpt_name))  # type: ignore
    logger.info('start training...')

    for epoch in range(num_epoches):
        model.train()
        epoch_start = time()
        batch_losses = []
        batch_cnt = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+start_epoch}", unit="batch") as t:
            for batch in t:
                batch_cnt += 1
                batch = batch.to(args.device)
                loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = get_model_loss(
                    model, batch, config)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                t.set_postfix(
                    loss=loss.item(),
                    loss_frontier=loss_frontier.item(),
                    loss_pos=loss_pos.item(),
                    loss_cls=loss_cls.item(),
                    loss_edge=loss_edge.item(),
                    loss_real=loss_real.item(),
                    loss_fake=loss_fake.item(),
                    loss_surf=loss_surf.item()
                )
                t.update()

        average_loss = sum(batch_losses) / (len(batch_losses) + 1)
        train_losses.append(average_loss)
        if verbose:
            logger.info('Training Epoch %d | Average_Loss %.5f | Loss %.6f | Loss(Fron) %.6f | Loss(Pos) %.6f | Loss(Cls) %.6f | Loss(Edge) %.6f | Loss(Real) %.6f | Loss(Fake) %.6f | Loss(Surf) %.6f  ' % (
                epoch + start_epoch, average_loss, loss.item(), loss_frontier.item(), loss_pos.item(), loss_cls.item(),
                loss_edge.item(), loss_real.item(), loss_fake.item(), loss_surf.item()
            ))
        average_eval_loss = evaluate(epoch + start_epoch, verbose=1)
        val_losses.append(average_eval_loss)

        if config.train.scheduler.type == "plateau":  # type: ignore
            scheduler.step(average_eval_loss)
        else:
            scheduler.step()  # type: ignore
        if val_losses[-1] < best_loss:
            best_loss = val_losses[-1]
            if config.train.save:  # type: ignore
                ckpt_path = os.path.join(ckpt_dir, 'val_%d.pt' % int(epoch + start_epoch))
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': start_epoch + epoch,
                    'best_loss': best_loss
                }, ckpt_path)
        else:
            if len(train_losses) > 20:
                if (train_losses[-1] < train_losses[-2]):
                    if config.train.save:  # type: ignore
                        ckpt_path = os.path.join(ckpt_dir, 'train_%d.pt' % int(epoch + start_epoch))
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': start_epoch + epoch,
                            'best_loss': best_loss
                        }, ckpt_path)
        torch.cuda.empty_cache()

train()
