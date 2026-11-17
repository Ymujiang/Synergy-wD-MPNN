import math
from typing import List, Union, Tuple
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from chemprop.nn_utils import index_select_ND, get_activation_function


class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule or polymer ensemble."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = args.device
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        # Configurable architectural flags (defaulting to full Synergy-wD-MPNN)
        self.readout_mode = getattr(args, 'readout_mode', 'synergy6')
        mp_dropout_p = getattr(args, 'mp_dropout_p', 0.0 if self.readout_mode == 'canonical' else 0.1)

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.mp_dropout = nn.Dropout(p=mp_dropout_p)

        # Activation
        self.act_func = get_activation_function(args.activation)

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # Stoichiometry-conditioned attention + non-linear synergy readout.
        if self.readout_mode != 'canonical':
            self.readout_heads = 4
            self.readout_head_dim = self.hidden_size // self.readout_heads
            # 3D query parameter prevents initialize_weights() from zeroing 1D tensors
            self.readout_query = nn.Parameter(torch.empty(1, self.readout_heads, self.readout_head_dim))
            self.readout_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.readout_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            if self.readout_mode in ('synergy6', 'synergy_only', 'no_attn', 'no_frontier'):
                self.readout_b1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
                self.readout_b2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            in_mult = {
                'synergy6': 6,
                'attn4': 4,
                'attn_only': 2,
                'frontier_only': 3,
                'synergy_only': 3,
                'no_attn': 5,
                'no_frontier': 4,
            }[self.readout_mode]
            self.readout_out = nn.Linear(in_mult * self.hidden_size, self.hidden_size, bias=True)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular/polymer graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, \
        a_scope, b_scope, degree_of_polym = mol_graph.get_components(atom_messages=self.atom_messages)

        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device), f_bonds.to(self.device),
            w_atoms.to(self.device), w_bonds.to(self.device),
            a2b.to(self.device), b2a.to(self.device), b2revb.to(self.device)
        )

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
        message = self.act_func(input)

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            else:
                nei_a_message = index_select_ND(message, a2b)
                nei_a_weight = index_select_ND(w_bonds, a2b)
                nei_a_message = nei_a_message * nei_a_weight[..., None]
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.mp_dropout(message) if self.readout_mode != 'canonical' else self.dropout_layer(message)

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)
        nei_a_weight = index_select_ND(w_bonds, a2x)
        nei_a_message = nei_a_message * nei_a_weight[..., None]
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.mp_dropout(atom_hiddens) if self.readout_mode != 'canonical' else self.dropout_layer(atom_hiddens)

        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError('The number of atoms is different from the length of the extra atom features')
            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            atom_hiddens = self.dropout_layer(atom_hiddens)

        # Canonical linear readout
        if self.readout_mode == 'canonical':
            mol_vecs = []
            for i, (a_start, a_size) in enumerate(a_scope):
                if a_size == 0:
                    mol_vecs.append(self.cached_zero_vector)
                else:
                    cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                    w_atom_vec = w_atoms.narrow(0, a_start, a_size)
                    mol_vec = w_atom_vec[..., None] * cur_hiddens
                    if self.aggregation == 'mean':
                        mol_vec = mol_vec.sum(dim=0) / w_atom_vec.sum(dim=0)
                    elif self.aggregation == 'sum':
                        mol_vec = mol_vec.sum(dim=0)
                    elif self.aggregation == 'norm':
                        mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                    mol_vec = degree_of_polym[i] * mol_vec
                    mol_vecs.append(mol_vec)
            return torch.stack(mol_vecs, dim=0)

        # Vectorized scatter-based non-linear stoichiometry-conditioned readout
        num_atoms_t = atom_hiddens.size(0)
        num_mols = len(a_scope)
        mol_idx = torch.zeros(num_atoms_t, dtype=torch.long, device=self.device)
        for i, (a_start, a_size) in enumerate(a_scope):
            mol_idx[a_start:a_start + a_size] = i
        mol_idx2d = mol_idx[:, None].expand(-1, self.hidden_size)

        wsum = torch.zeros(num_mols, device=self.device).index_add_(0, mol_idx, w_atoms)
        wsum_safe = wsum.clamp(min=1e-6)

        if self.aggregation == 'mean':
            D = self.hidden_size
            H = self.readout_heads
            d = self.readout_head_dim
            wlog = torch.log(w_atoms + 1e-6)
            wlog[0] = float('-inf')
            wh = w_atoms[:, None] * atom_hiddens

            mean_pool = torch.zeros(num_mols, D, device=self.device).index_add_(0, mol_idx, wh)
            mean_pool = mean_pool / wsum_safe[:, None]

            parts = []
            if self.readout_mode in ('synergy6', 'attn4', 'attn_only', 'no_frontier'):
                k_all = self.readout_k(atom_hiddens).view(-1, H, d)
                v_all = self.readout_v(atom_hiddens).view(-1, H, d)
                logits = torch.einsum('hd,nhd->nh', self.readout_query[0], k_all) / math.sqrt(d) + wlog[:, None]
                mx = torch.scatter_reduce(
                    torch.full((num_mols, H), float('-inf'), device=self.device),
                    0, mol_idx[:, None].expand(-1, H), logits, reduce='amax'
                )
                s = torch.exp(logits - mx[mol_idx])
                denom = torch.zeros(num_mols, H, device=self.device).index_add_(0, mol_idx, s)
                attn = s / denom[mol_idx].clamp(min=1e-6)
                attn_pool = torch.zeros(num_mols, D, device=self.device).index_add_(
                    0, mol_idx, (attn[:, :, None] * v_all).reshape(-1, D)
                )
                parts.append(attn_pool)

            parts.append(mean_pool)

            lse_pool = None
            if self.readout_mode in ('synergy6', 'attn4', 'frontier_only', 'no_attn', 'synergy_only'):
                hl = atom_hiddens + wlog[:, None]
                lse_max = torch.scatter_reduce(
                    torch.full((num_mols, D), float('-inf'), device=self.device),
                    0, mol_idx2d, hl, reduce='amax'
                )
                lse_sum = torch.zeros(num_mols, D, device=self.device).index_add_(
                    0, mol_idx, torch.exp(hl - lse_max[mol_idx])
                )
                lse_pool = torch.log(lse_sum.clamp(min=1e-6)) + lse_max

            if self.readout_mode in ('synergy6', 'attn4', 'frontier_only', 'no_attn'):
                wh_max = wh.clone()
                wh_max[0] = float('-inf')
                max_pool = torch.scatter_reduce(
                    torch.full((num_mols, D), float('-inf'), device=self.device),
                    0, mol_idx2d, wh_max, reduce='amax'
                )
                parts.extend([lse_pool, max_pool])

            if self.readout_mode in ('synergy6', 'synergy_only', 'no_attn', 'no_frontier'):
                var_pool = torch.zeros(num_mols, D, device=self.device).index_add_(
                    0, mol_idx, w_atoms[:, None] * (atom_hiddens - mean_pool[mol_idx]) ** 2
                )
                var_pool = var_pool / wsum_safe[:, None]
                if lse_pool is None:
                    hl = atom_hiddens + wlog[:, None]
                    lse_max = torch.scatter_reduce(
                        torch.full((num_mols, D), float('-inf'), device=self.device),
                        0, mol_idx2d, hl, reduce='amax'
                    )
                    lse_sum = torch.zeros(num_mols, D, device=self.device).index_add_(
                        0, mol_idx, torch.exp(hl - lse_max[mol_idx])
                    )
                    lse_pool = torch.log(lse_sum.clamp(min=1e-6)) + lse_max
                bilinear = self.readout_b1(mean_pool) * self.readout_b2(lse_pool)
                parts.extend([var_pool, bilinear])

            feat = torch.cat(parts, dim=1)
            mol_vecs = self.readout_out(feat)
        else:
            wh = w_atoms[:, None] * atom_hiddens
            mol_vecs = torch.zeros(num_mols, self.hidden_size, device=self.device).index_add_(0, mol_idx, wh)
            if self.aggregation == 'norm':
                mol_vecs = mol_vecs / self.aggregation_norm

        dops = torch.tensor(degree_of_polym, dtype=torch.float32, device=self.device)
        mol_vecs = mol_vecs * dops[:, None]
        mol_vecs = torch.where(wsum[:, None] > 0, mol_vecs, torch.zeros_like(mol_vecs))

        return mol_vecs


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder` which featurizes input as needed."""

    def __init__(self,
                 args: TrainArgs,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim(overwrite_default_atom=args.overwrite_default_atom_features)
        self.bond_fdim = bond_fdim or get_bond_fdim(
            overwrite_default_atom=args.overwrite_default_atom_features,
            overwrite_default_bond=args.overwrite_default_bond_features,
            atom_messages=args.atom_messages
        )

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors
        self.overwrite_default_atom_features = args.overwrite_default_atom_features
        self.overwrite_default_bond_features = args.overwrite_default_bond_features

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                                          for _ in range(args.number_of_molecules)])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        if type(batch[0]) != BatchMolGraph:
            batch = [[mols[i] for mols in batch] for i in range(len(batch[0]))]

            if self.atom_descriptors == 'feature':
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')
                batch = [
                    mol2graph(
                        mols=b,
                        atom_features_batch=atom_features_batch,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            elif bond_features_batch is not None:
                if len(batch) > 1:
                    raise NotImplementedError('Atom/bond descriptors are currently only supported with one molecule '
                                              'per input (i.e., number_of_molecules = 1).')
                batch = [
                    mol2graph(
                        mols=b,
                        bond_features_batch=bond_features_batch,
                        overwrite_default_atom_features=self.overwrite_default_atom_features,
                        overwrite_default_bond_features=self.overwrite_default_bond_features
                    )
                    for b in batch
                ]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float().to(self.device)
            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError('Atom descriptors are currently only supported with one molecule '
                                          'per input (i.e., number_of_molecules = 1).')
            encodings = [enc(ba, atom_descriptors_batch) for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)
            output = torch.cat([output, features_batch], dim=1)

        return output
