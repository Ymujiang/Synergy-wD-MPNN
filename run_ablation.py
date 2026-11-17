"""Controlled ablation runner and variant registry for Synergy-wD-MPNN."""

import argparse
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, get_data, MoleculeDataLoader
from chemprop.features import set_polymer
from chemprop.models import MoleculeModel
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func


VARIANTS = {
    'canonical':        dict(readout_mode='canonical',     mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'canonical_pm':     dict(readout_mode='canonical',     mp_dropout_p=0.0, use_residual_head=False, hidden_size=600),
    'readout_only':     dict(readout_mode='synergy6',      mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'readout_drop':     dict(readout_mode='synergy6',      mp_dropout_p=0.1, use_residual_head=False, hidden_size=300),
    'synergy_full':     dict(readout_mode='synergy6',      mp_dropout_p=0.1, use_residual_head=True,  hidden_size=300),
    'pruned_attn4':     dict(readout_mode='attn4',         mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'pruned_attn4_res': dict(readout_mode='attn4',         mp_dropout_p=0.0, use_residual_head=True,  hidden_size=300),
    'attn_only':        dict(readout_mode='attn_only',     mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'frontier_only':    dict(readout_mode='frontier_only', mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'synergy_only':     dict(readout_mode='synergy_only',  mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'no_attn':          dict(readout_mode='no_attn',       mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
    'no_frontier':      dict(readout_mode='no_frontier',   mp_dropout_p=0.0, use_residual_head=False, hidden_size=300),
}


def build_variant_model(variant_name: str, data_path: str, epochs: int = 30, gpu: int = 0):
    cfg = VARIANTS[variant_name]
    use_gpu = torch.cuda.is_available() and gpu >= 0
    args = TrainArgs().parse_args([
        '--data_path', data_path,
        '--dataset_type', 'regression',
        '--save_dir', '/tmp/synergy_ablation',
        '--polymer',
        '--hidden_size', str(cfg['hidden_size']),
        '--epochs', str(epochs),
        '--quiet',
    ] + (['--gpu', str(gpu)] if use_gpu else ['--no_cuda']))
    args.readout_mode = cfg['readout_mode']
    args.mp_dropout_p = cfg['mp_dropout_p']
    args.use_residual_head = cfg['use_residual_head']
    set_polymer(True)
    args.task_names = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
    return args, MoleculeModel(args).to(args.device)


def main():
    parser = argparse.ArgumentParser(description='Run or inspect Synergy-wD-MPNN ablations.')
    parser.add_argument('--variant', choices=list(VARIANTS.keys()), default='synergy_full')
    parser.add_argument('--train_csv', default=os.path.join(REPO_ROOT, 'data', 'sample_copolymers.csv'))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--list_params', action='store_true')
    cli = parser.parse_args()

    if cli.list_params:
        for name in VARIANTS:
            _, model = build_variant_model(name, cli.train_csv, epochs=1, gpu=-1)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{name:18s}: {n_params:,} trainable parameters')
        return

    args, model = build_variant_model(cli.variant, cli.train_csv, epochs=cli.epochs, gpu=cli.gpu)
    data = get_data(path=cli.train_csv, args=args)
    scaler = StandardScaler().fit(data.targets())
    data.set_targets(scaler.transform(data.targets()).tolist())
    args.train_data_size = len(data)

    loader = MoleculeDataLoader(dataset=data, batch_size=min(16, len(data)), num_workers=0, shuffle=True, seed=0)
    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(optimizer, args)
    loss_fn = get_loss_func(args)

    model.train()
    for ep in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            y = torch.tensor([[0.0 if v is None else float(v) for v in t] for t in batch.targets()],
                             dtype=torch.float32, device=args.device)
            model.zero_grad()
            pred = model(batch.batch_graph())
            loss = loss_fn(pred, y).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
        print(f'[{cli.variant}] Epoch {ep + 1}/{args.epochs} | Loss={total_loss / len(loader):.4f}')


if __name__ == '__main__':
    main()
