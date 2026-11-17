"""Zero-leakage evaluation script for Synergy-wD-MPNN checkpoints."""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chemprop.args import TrainArgs
from chemprop.data import StandardScaler, get_data, MoleculeDataLoader
from chemprop.features import set_polymer
from chemprop.models import MoleculeModel


def evaluate_checkpoint(checkpoint_path: str, test_csv: str, batch_size: int = 128, gpu: int = 0):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    args = TrainArgs().parse_args([
        '--data_path', test_csv,
        '--dataset_type', 'regression',
        '--save_dir', '/tmp/synergy_eval',
        '--polymer',
        '--quiet',
    ] + (['--no_cuda'] if device.type == 'cpu' else ['--gpu', str(gpu)]))

    set_polymer(True)
    test_data = get_data(path=test_csv, args=args)
    args.task_names = test_data.task_names if hasattr(test_data, 'task_names') else ['EA vs SHE (eV)', 'IP vs SHE (eV)']

    scaler = StandardScaler(means=ckpt['scaler_means'], stds=ckpt['scaler_stds'])
    model = MoleculeModel(args).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()}, strict=True)
    model.eval()

    loader = MoleculeDataLoader(dataset=test_data, batch_size=batch_size, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch.batch_graph()).cpu().numpy())

    preds = np.asarray(scaler.inverse_transform(np.vstack(preds)), dtype=np.float32)
    targets = np.asarray(test_data.targets(), dtype=np.float32)
    rmses = np.sqrt(np.mean((targets - preds) ** 2, axis=0))
    return rmses, preds, targets


def main():
    parser = argparse.ArgumentParser(description='Evaluate a Synergy-wD-MPNN checkpoint without target leakage.')
    parser.add_argument('--checkpoint', default=os.path.join(REPO_ROOT, 'checkpoints', 'synergy_wdmpnn_fold0_seed0.pt'))
    parser.add_argument('--test_csv', default=os.path.join(REPO_ROOT, 'data', 'sample_copolymers.csv'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    rmses, preds, _ = evaluate_checkpoint(args.checkpoint, args.test_csv, args.batch_size, args.gpu)
    print(f'Evaluated {len(preds)} copolymers from {args.test_csv}:')
    print(f'  EA_RMSE = {rmses[0]:.4f} eV | IP_RMSE = {rmses[1]:.4f} eV | Mean_RMSE = {np.mean(rmses):.4f} eV')


if __name__ == '__main__':
    main()
