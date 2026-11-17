import os, sys, time, json, pickle
import numpy as np
import pandas as pd
import torch

BASELINE_REPO = '/home/ymujdb/wdmpnn_journal_exp/baseline_repo'
EVOLVED_REPO = '/home/ymujdb/wdmpnn_journal_exp/evolved_repo'
CACHE_PATH = '/home/ymujdb/repos/polymer-chemprop-data/vipea_fast_graphs.pkl'
TEST_CSV = '/home/ymujdb/repos/polymer-chemprop/data/vipea_fold0/input_test_0.csv'


def clear_chemprop():
    for k in list(sys.modules.keys()):
        if k == 'chemprop' or k.startswith('chemprop.'):
            del sys.modules[k]


def profile_model(name, repo_path, hidden_size=300, batch_size=50, warmup=20, iters=100):
    clear_chemprop()
    sys.path.insert(0, repo_path)
    from chemprop.args import TrainArgs
    from chemprop.features import set_polymer, BatchMolGraph, get_atom_fdim, get_bond_fdim
    from chemprop.models import MoleculeModel

    set_polymer(True)
    df = pd.read_csv(TEST_CSV)
    smiles = df.iloc[:batch_size, 0].tolist()

    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    atom_fdim, bond_fdim = get_atom_fdim(), get_bond_fdim()

    n_atoms, n_bonds = 1, 1
    fa = [torch.zeros((1, atom_fdim), dtype=torch.float32)]
    fb = [torch.zeros((1, bond_fdim), dtype=torch.float32)]
    wa = [torch.zeros((1,), dtype=torch.float32)]
    wb = [torch.zeros((1,), dtype=torch.float32)]
    b2a = [torch.zeros((1,), dtype=torch.long)]
    b2revb = [torch.zeros((1,), dtype=torch.long)]
    a_scope, b_scope, dop = [], [], []
    a2b_rows = [[]]
    for s in smiles:
        mg = cache[s]
        fa.append(torch.from_numpy(mg['f_atoms']))
        fb.append(torch.from_numpy(mg['f_bonds']))
        wa.append(torch.from_numpy(mg['w_atoms']))
        wb.append(torch.from_numpy(mg['w_bonds']))
        b2a.append(torch.from_numpy(mg['b2a']) + n_atoms)
        b2revb.append(torch.from_numpy(mg['b2revb']) + n_bonds)
        for row in mg['a2b']:
            a2b_rows.append([b + n_bonds for b in row])
        a_scope.append((n_atoms, mg['n_atoms']))
        b_scope.append((n_bonds, mg['n_bonds']))
        n_atoms += mg['n_atoms']
        n_bonds += mg['n_bonds']
        dop.append(mg['degree_of_polym'])
    max_nb = max(1, max(len(r) for r in a2b_rows))
    a2b = torch.zeros((n_atoms, max_nb), dtype=torch.long)
    for idx, r in enumerate(a2b_rows):
        if r:
            a2b[idx, :len(r)] = torch.tensor(r, dtype=torch.long)

    bm = BatchMolGraph.__new__(BatchMolGraph)
    bm.overwrite_default_atom_features = False
    bm.overwrite_default_bond_features = False
    bm.atom_fdim, bm.bond_fdim = atom_fdim, bond_fdim
    bm.n_atoms, bm.n_bonds = n_atoms, n_bonds
    bm.a_scope, bm.b_scope, bm.degree_of_polym, bm.max_num_bonds = a_scope, b_scope, dop, max_nb
    bm.f_atoms = torch.cat(fa, dim=0)
    bm.f_bonds = torch.cat(fb, dim=0)
    bm.w_atoms = torch.cat(wa, dim=0)
    bm.w_bonds = torch.cat(wb, dim=0)
    bm.a2b = a2b
    bm.b2a = torch.cat(b2a, dim=0)
    bm.b2revb = torch.cat(b2revb, dim=0)
    bm.b2b = None
    bm.a2a = None

    args = TrainArgs().parse_args([
        '--data_path', TEST_CSV, '--dataset_type', 'regression', '--save_dir', '/tmp/p',
        '--polymer', '--hidden_size', str(hidden_size), '--gpu', '0', '--quiet'
    ])
    args.task_names = ['EA vs SHE (eV)', 'IP vs SHE (eV)']
    model = MoleculeModel(args).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target = torch.randn(batch_size, 2, device='cuda:0')

    # Forward eval latency
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model([bm])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model([bm])
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) / iters * 1000.0
    fwd_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Forward + Backward train latency
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model.zero_grad()
        loss = torch.nn.functional.mse_loss(model([bm]), target)
        loss.backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.zero_grad()
        loss = torch.nn.functional.mse_loss(model([bm]), target)
        loss.backward()
    torch.cuda.synchronize()
    train_ms = (time.perf_counter() - t0) / iters * 1000.0
    train_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    sys.path.pop(0)
    return {
        'name': name,
        'hidden_size': hidden_size,
        'trainable_params': trainable,
        'batch_size': batch_size,
        'eval_fwd_ms': round(fwd_ms, 3),
        'train_fwdbwd_ms': round(train_ms, 3),
        'eval_throughput_polymers_per_s': round(batch_size / (fwd_ms / 1000.0), 1),
        'train_throughput_polymers_per_s': round(batch_size / (train_ms / 1000.0), 1),
        'eval_peak_mem_mb': round(fwd_mem_mb, 2),
        'train_peak_mem_mb': round(train_mem_mb, 2),
    }


if __name__ == '__main__':
    res = [
        profile_model('Canonical_wD-MPNN', BASELINE_REPO, 300),
        profile_model('Canonical_wD-MPNN_ParamMatched', BASELINE_REPO, 600),
        profile_model('Evolved_wD-MPNN', EVOLVED_REPO, 300),
    ]
    with open('/home/ymujdb/wdmpnn_journal_exp/results/profiling_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    for r in res:
        print(r)
