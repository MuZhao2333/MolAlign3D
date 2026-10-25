# Original training module from Cls_vae commit a47090b56f7f8db07b6d002640891807f8c96512.
import copy
import numpy as np
import torch
from types import SimpleNamespace
from time import time
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_sum
import os
from molalign3d.models.bfn4sbdd import BFN4SBDDScoreModel
from molalign3d.models.encoder import Encoder
from molalign3d.utils import atom_num
import datetime
import json
from molalign3d.utils.train import get_optimizer, get_scheduler
import wandb
from molalign3d.utils.build_mol import MoleculeBuilder
from rdkit import Chem
from rdkit.Chem import rdShapeHelpers
import torch.nn as nn
import random
import math
from molalign3d.vendor.unimol.models.unimol import UniMolModel
from molalign3d.vendor.unimol.data.conformer import coords2unimol

#ligand居中
def center_pos(ligand_pos,batch_ligand, mode=True):
    if mode == False:
        offset = 0.0
        pass
    elif mode == True:
        # print(ligand_pos.shape)
        # print(batch_ligand.shape)
        # print(f"in_centerpos:")
        # print(f"ligand_pos:{len(ligand_pos)},len_batch_ligand:{len(batch_ligand)}")
        offset = scatter_mean(ligand_pos, batch_ligand, dim=0)
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return ligand_pos, offset
# config数据类型转换
def dict_to_namespace(d):
        """递归地将字典转换为 SimpleNamespace 对象"""
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})

class TrainLoop(pl.LightningModule):
    # def __init__(self, config: Config):
    #     super().__init__()
    #     self.cfg = config
    #     self.decoder = BFN4SBDDScoreModel(**self.cfg.decoder.todict())
    #     # [ time, h_t, pos_t, edge_index]
    #     self.train_losses = []
    #     self.save_hyperparameters(self.cfg.todict())
    #     self.time_records = np.zeros(6)
    #     self.log_time = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        print(self.cfg)
        #网络
        print("encoder")
        print(self.cfg['encoder_config'])
        self.encoder = Encoder(**self.cfg['encoder_config'])

        # For KL divergence（正则化层）
        self.Wh_mu = nn.Linear(self.cfg['encoder_config']['hidden_dim'], self.cfg['optimal_layer_config']['latent_dim'])
        self.Wh_log_var = nn.Linear(self.cfg['encoder_config']['hidden_dim'], self.cfg['optimal_layer_config']['latent_dim'])
        self.Wx_log_var = nn.Linear(self.cfg['encoder_config']['hidden_dim'], 1) # has to be isotropic gaussian to maintain equivariance
        # with torch.no_grad():
        #     self.Wx_log_var.weight.fill_(0.0)
        #     bias=torch.log(torch.tensor(self.cfg['train']['kl_loss']['sigma2']**2))
        #     self.Wx_log_var.bias.fill_(bias)
        self.decoder = BFN4SBDDScoreModel(**self.cfg['decoder_config'])
        self.time_records = np.zeros(6)
        # predictor for atom num
        latent_dim = self.cfg['decoder_config']['protein_atom_feature_dim']
        num_global_nodes= self.cfg['encoder_config']['global_node_num']
        atom_predictor_hidden_dim = self.cfg['atom_num_predictor']['hidden_dim']
        self.atom_num_predictor = nn.Sequential(
            nn.Linear((latent_dim)*num_global_nodes, atom_predictor_hidden_dim),  # Zh的维度是latent_dim，Zx的维度是3
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(atom_predictor_hidden_dim, atom_predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(atom_predictor_hidden_dim, 1)  
        )

        # 保存配置参数
        self.encoder_config = self.cfg['encoder_config']
        self.decoder_config = self.cfg['decoder_config']
        
        self.train_losses = []

        #一些参数
        self.log_time = False
        if self.cfg['train']['contrastive_loss']['type']=='infonce':
            self.logit_scale=nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.print_model_params()
        
        self.save_dir = cfg['evaluation']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.test_result=[]
        if(self.cfg['data']['require_H']):
            self.MAP_ATOM_TYPE_ONLY_TO_INDEX = {
                1: 0,
                6: 1,
                7: 2,
                8: 3,
                9: 4,
                15: 5,
                16: 6,
                17: 7,
                35: 8,
                53: 9,
            }
            self.MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in self.MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
        else:
            self.MAP_ATOM_TYPE_ONLY_TO_INDEX = {
                6: 0,
                7: 1,
                8: 2,
                9: 3,
                15: 4,
                16: 5,
                17: 6,
                35: 7,
                53: 8,
            }
            self.MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in self.MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
        
        # token classifier for masked token prediction
        self.num_atom_types = len(self.MAP_ATOM_TYPE_ONLY_TO_INDEX)
        self.token_classifier = nn.Linear(self.encoder_config['hidden_dim'], self.num_atom_types)
        # ---- Uni-Mol integration -------------------------------------------------
        self.unimol_cfg = self.cfg.get('unimol', {})
        self.unimol_enabled = bool(self.unimol_cfg.get('enabled', False))
        self.unimol_model = None
        self.unimol_proj = None
        if self.unimol_enabled:
            # Use UniMolModel (v1 wrapper)
            self.unimol_model = UniMolModel(output_dim=2, data_type='molecule', remove_hs=True)
            ckpt = self.unimol_cfg.get('ckpt_path')
            state = torch.load(ckpt, map_location='cpu')
            self.unimol_model.load_state_dict(state['model'], strict=False)
            if self.unimol_cfg.get('freeze'):
                for p in self.unimol_model.parameters():
                    p.requires_grad = False

            self.unimol_cls_dim = self.unimol_model.args.encoder_embed_dim
            if bool(self.unimol_cfg.get('project_to_latent', False)):
                self.unimol_proj = nn.Linear(self.unimol_cls_dim, latent_dim)
        else:
            if self.unimol_cfg.get('use_outnorm', False):
                print("Warning: unimol.use_outnorm is set in config but unimol.enabled is False. Ignoring outnorm.")
        # validate outnorm method
        if self.unimol_cfg.get('outnorm_method', 'scalar') not in ['scalar', 'channel']:
            print(f"Warning: unimol.outnorm_method {self.unimol_cfg.get('outnorm_method')} unsupported. Falling back to 'scalar'.")
        # ---------------------------------------------------------------------------

    def safe_log_dict(self, dictionary, **kwargs):
        """Safe version of log_dict that checks if a logger is available."""
        try:
            if self.trainer is not None and self.trainer.logger is not None:
                self.log_dict(dictionary, **kwargs)
        except Exception:
            pass
        
    def print_model_params(self):
        """打印各模块参数量的方法"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        encoder_params = count_params(self.encoder)
        optimal_layer_params=count_params(self.Wh_log_var)+count_params(self.Wh_mu)+count_params(self.Wx_log_var)
        decoder_params = count_params(self.decoder)
        atom_num_predictor_params = count_params(self.atom_num_predictor)
        total_params = encoder_params + decoder_params+optimal_layer_params
        
        print("\n" + "="*50)
        print(f"Encoder可训练参数: {encoder_params/1e6:.2f}M")
        print(f"正则化层：: {optimal_layer_params/1e6:.2f}M")
        print(f"atom_num_predictor可训练参数: {atom_num_predictor_params/1e6:.2f}M")
        print(f"Decoder可训练参数: {decoder_params/1e6:.2f}M")
        print(f"总可训练参数: {total_params/1e6:.2f}M")
        print("="*50 + "\n")

    def forward(self, x):
        pass

    def encode(self,one_hot_h, x, batch_ligand,deterministic=False, return_nodes=False):
        #global_batch就是[0,0,……(8个),1,1,……(8个),…………]
        #global_h出来是hidden_dim的
        # call encoder — optionally can return node-level h/x if return_nodes True
        if return_nodes:
            global_h, global_x, global_batch, node_h_updated, node_x_updated = self.encoder(one_hot_h, x, batch_ligand, return_nodes=True)
        else:
            global_h, global_x, global_batch = self.encoder(one_hot_h, x, batch_ligand)
        
        self.safe_log_dict({'train/global_h_mean': global_h.mean(), 'train/global_x_mean': global_x.mean(),'train/global_x_var': global_x.var(),'train/global_h_var': global_h.var()})
        
        #mu和var;h从hidden_dim降到latent_dim
        Zh_mu = self.Wh_mu(global_h)
        self.safe_log_dict({'train/Zh_mu_mean': Zh_mu.mean(), 'train/Zh_mu_var': Zh_mu.var()})
        Zh_log_var = -torch.abs(self.Wh_log_var(global_h))
        Zx_mu = global_x.clone()
        # clamp to an upper bound (log(sigma2^2)) instead of subtracting it
        upper = torch.log(torch.tensor(self.cfg['train']['kl_loss']['sigma2']**2, device=Zx_mu.device, dtype=Zx_mu.dtype))
        raw = self.Wx_log_var(global_h).expand_as(Zx_mu)
        # Zx_log_var = upper-torch.abs(raw)
        Zx_log_var = -torch.abs(raw)
        # Zx_log_var = -torch.abs(self.Wx_log_var(global_h)).expand_as(Zx_mu)
        #分子数量
        data_size = torch.unique(global_batch).size(0)

        # # KL(q||p) for Zh (q ~ N(mu, var), p ~ N(0,1)) per-dim:
        Zh_var = torch.exp(Zh_log_var)
        # Zh_kl_per_dim = 0.5 * (Zh_mu * Zh_mu + Zh_var - 1.0 - Zh_log_var)
        Zx_var = torch.exp(Zx_log_var)
        self.safe_log_dict({'train/Zx_var_mean': Zx_var.mean(),'train/Zh_var_mean': Zh_var.mean()})
        # p is isotropic gaussian with sigma2 (allowed in config), KL per-dim for Zx is similar but scaled by sigma2
        sigma2 = float(self.cfg['train']['kl_loss'].get('sigma2'))
        # Zx_kl_per_dim = 0.5 * ((Zx_mu * Zx_mu + Zx_var) / (sigma2 ** 2) - 1.0 - Zx_log_var + 2.0 * torch.log(torch.tensor(sigma2, device=Zx_log_var.device, dtype=Zx_log_var.dtype)))
        # apply free bits (min KL per-dim)
        # free_bits = float(self.cfg['train']['kl_loss'].get('free_bits'))
        # if free_bits > 0.0:
        #     # free bits: allow a small KL budget per-dimension that is not penalized
        #     Zh_kl_per_dim = torch.clamp(Zh_kl_per_dim - free_bits, min=0.0)
        #     Zx_kl_per_dim = torch.clamp(Zx_kl_per_dim - free_bits, min=0.0)
        # # aggregate
        # Zh_kl_loss = torch.sum(Zh_kl_per_dim) / (data_size * Zh_mu.shape[-1])
        # Zx_kl_loss = torch.sum(Zx_kl_per_dim) / (data_size * Zx_mu.shape[-1])
        Zh_kl_loss = 0.5 * torch.sum(Zh_mu * Zh_mu) / (data_size * Zh_mu.shape[-1])
        
        Zx_kl_loss = -0.5 * torch.sum(1.0 + Zx_log_var #- 2*torch.log(torch.tensor(self.cfg['train']['kl_loss']['sigma2'],device=self.device,dtype=Zx_log_var.dtype))
                                      - (Zx_mu * Zx_mu + torch.exp(Zx_log_var))/(self.cfg['train']['kl_loss']['sigma2'])**2) / (data_size * Zx_mu.shape[-1])
        
        #rsample
        Zh_sampled = Zh_mu #if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx_mu)
        
        if return_nodes:
            return Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss, node_h_updated, node_x_updated
        return Zh_sampled, Zx_sampled,global_batch, Zh_kl_loss, Zx_kl_loss
        

        # ############################################### new svg ver ##############################################

        # #call encoder — optionally can return node-level h/x if return_nodes True
        # if return_nodes:
        #     global_h, global_x, global_batch, node_h_updated, node_x_updated = self.encoder(one_hot_h, x, batch_ligand, return_nodes=True)
        # else:
        #     global_h, global_x, global_batch = self.encoder(one_hot_h, x, batch_ligand)
        # self.safe_log_dict({'train/global_h_mean': global_h.mean(), 'train/global_x_mean': global_x.mean(),'train/global_x_var': global_x.var(),'train/global_h_var': global_h.var()})
        
        # #mu e var; h from hidden_dim to latent_dim
        # Zh_mu = self.Wh_mu(global_h)
        # self.safe_log_dict({'train/Zh_mu_mean': Zh_mu.mean(), 'train/Zh_mu_var': Zh_mu.var()})
        # # Zh_log_var = -torch.abs(self.Wh_log_var(global_h))
        # Zx_mu = global_x.clone()
        # # # clamp to an upper bound (log(sigma2^2)) instead of subtracting it
        # # upper = torch.log(torch.tensor(self.cfg['train']['kl_loss']['sigma2']**2, device=Zx_mu.device, dtype=Zx_mu.dtype))
        # # raw = self.Wx_log_var(global_h).expand_as(Zx_mu)
        # # Zx_log_var = upper-torch.abs(raw)
        # Zx_log_var = -torch.abs(self.Wx_log_var(global_h)).expand_as(Zx_mu)
        # #count molecular
        # data_size = torch.unique(global_batch).size(0)

        # # # compute per-dimension KL and optionally apply "free bits" to prevent collapse
        # # # KL(q||p) for Zh (q ~ N(mu, var), p ~ N(0,1)) per-dim:
        # # Zh_var = torch.exp(Zh_log_var)
        # # Zh_kl_per_dim = 0.5 * (Zh_mu * Zh_mu + Zh_var - 1.0 - Zh_log_var)
        # Zx_var = torch.exp(Zx_log_var)
        # self.safe_log_dict({'train/Zx_var_mean': Zx_var.mean()})   #,'train/Zh_var_mean': Zh_var.mean()})
        # # p is isotropic gaussian with sigma2 (allowed in config), KL per-dim for Zx is similar but scaled by sigma2
        # # sigma2 = float(self.cfg['train']['kl_loss'].get('sigma2'))
        # # Zx_kl_per_dim = 0.5 * ((Zx_mu * Zx_mu + Zx_var) / (sigma2 ** 2) - 1.0 - Zx_log_var + 2.0 * torch.log(torch.tensor(sigma2, device=Zx_log_var.device, dtype=Zx_log_var.dtype)))
        # # # apply free bits (min KL per-dim)
        # # free_bits = float(self.cfg['train']['kl_loss'].get('free_bits'))
        # # if free_bits > 0.0:
        # #     # free bits: allow a small KL budget per-dimension that is not penalized
        # #     Zh_kl_per_dim = torch.clamp(Zh_kl_per_dim - free_bits, min=0.0)
        # #     Zx_kl_per_dim = torch.clamp(Zx_kl_per_dim - free_bits, min=0.0)
        # # # aggregate
        # # Zh_kl_loss = torch.sum(Zh_kl_per_dim) / (data_size * Zh_mu.shape[-1])
        # # Zx_kl_loss = torch.sum(Zx_kl_per_dim) / (data_size * Zx_mu.shape[-1])
        # # Zh_kl_loss = -0.5 * torch.sum(1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var)) / (data_size * Zh_mu.shape[-1])
        
        # Zx_kl_loss = -0.5 * torch.sum(1.0 + Zx_log_var - 2*torch.log(torch.tensor(self.cfg['train']['kl_loss']['sigma2'],device=self.device,dtype=Zx_log_var.dtype))
        #                               - (Zx_mu * Zx_mu + torch.exp(Zx_log_var))/(self.cfg['train']['kl_loss']['sigma2'])**2) / (data_size * Zx_mu.shape[-1])
        # Zh_kl_loss = torch.zeros_like(Zx_kl_loss)
        
        # # #rsample
        # # Zh_sampled = Zh_mu #if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        # # Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx_mu)
        
        # # if return_nodes:
        # #     return Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss, node_h_updated, node_x_updated
        # # return Zh_sampled, Zx_sampled,global_batch, Zh_kl_loss, Zx_kl_loss

        # # final ver: pure noise
        # Zh_sampled = Zh_mu #if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)
        # Zx_sampled = Zx_mu if deterministic else Zx_mu + torch.randn_like(Zx_mu) # 直接用N(0,1) !
        
        # if return_nodes:
        #     return Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss, node_h_updated, node_x_updated
        # return Zh_sampled, Zx_sampled,global_batch, Zh_kl_loss, Zx_kl_loss

    def get_masked_token(self, h_indices: torch.Tensor, mask_token_prob: float):
        """Mask some tokens and return masked one-hot, target tokens, and mask boolean.
        h_indices: [N] long tensor
        returns: masked_one_hot (N x K), target_tokens (N long, -100 for ignore), mask_bool (N bool)
        """
        N = h_indices.shape[0]
        # determine num masked
        num_mask = int(mask_token_prob * N + np.random.rand())
        if num_mask <= 0:
            mask = torch.zeros(N, dtype=torch.bool, device=h_indices.device)
        else:
            mask_idc = np.random.choice(N, num_mask, replace=False)
            mask = torch.zeros(N, dtype=torch.bool, device=h_indices.device)
            mask[mask_idc] = True

        K = len(self.MAP_ATOM_TYPE_ONLY_TO_INDEX)
        masked_one_hot = F.one_hot(h_indices, K).float()
        masked_one_hot[mask] = 0.0
        target_tokens = torch.full_like(h_indices, -100)
        target_tokens[mask] = h_indices[mask]
        return masked_one_hot, target_tokens, mask

    def get_noised_coord(self, coord: torch.Tensor, mask_pos_prob: float, noise: float, noise_type: str):
        """Apply noise to some coordinates according to mask_pos_prob and return noised coords + boolean mask.
        coord: [N,3]
        """
        N = coord.shape[0]
        num_mask = int(mask_pos_prob * N + np.random.rand())
        if num_mask <= 0:
            mask = torch.zeros(N, dtype=torch.bool, device=coord.device)
            return coord.clone(), mask
        mask_idc = np.random.choice(N, num_mask, replace=False)
        mask = torch.zeros(N, dtype=torch.bool, device=coord.device)
        mask[mask_idc] = True
        noised = coord.clone()
        if noise_type == 'normal':
            n_tensor = torch.randn((num_mask, 3), device=coord.device) * noise
        elif noise_type == 'uniform':
            n_tensor = (torch.rand((num_mask, 3), device=coord.device) * 2 - 1) * noise
        else:  # fallback
            n_tensor = torch.randn((num_mask, 3), device=coord.device) * noise
        noised[mask] = noised[mask] + n_tensor
        return noised, mask
    
    def _attach_unimol_to_Zh(self, Zh, global_batch, h, x, batch_ligand):
        """Attach Uni-Mol CLS features to Zh by concatenation.

        - Zh: [N_nodes, latent_dim]
        - h: [N_nodes] atomic numbers (indices)
        - x: [N_nodes, 3] coordinates
        - batch_ligand: [N_nodes] batch index
        """
        if not getattr(self, 'unimol_enabled'):
            return Zh

        device = Zh.device
        num_graphs = int(batch_ligand.max().item()) + 1

        unimol_tensor = None
        
        # Try to run UniMol model if available
        unimol_batch_inputs = []
        for i in range(num_graphs):
            mask = (batch_ligand == i)
            # h is indices, map to atomic number then symbol
            atoms_indices = h[mask].cpu().numpy()
            atoms_nums = [self.MAP_INDEX_TO_ATOM_TYPE_ONLY[idx] for idx in atoms_indices]
            atoms_symbols = [Chem.GetPeriodicTable().GetElementSymbol(int(an)) for an in atoms_nums]
            
            coords = x[mask].cpu().numpy()
            
            unimol_input = coords2unimol(atoms_symbols, coords, self.unimol_model.dictionary, remove_hs=self.unimol_model.remove_hs)
            unimol_batch_inputs.append(unimol_input)
        
        # Collate
        max_len = max([len(u['src_tokens']) for u in unimol_batch_inputs])
        
        batch_src_tokens = []
        batch_src_distance = []
        batch_src_coord = []
        batch_src_edge_type = []
        
        for u in unimol_batch_inputs:
            diff = max_len - len(u['src_tokens'])
            
            # Pad tokens
            tokens = np.pad(u['src_tokens'], (0, diff), constant_values=self.unimol_model.dictionary.pad())
            batch_src_tokens.append(tokens)
            
            # Pad distance
            dist = np.pad(u['src_distance'], ((0, diff), (0, diff)), constant_values=0.0)
            batch_src_distance.append(dist)
            
            # Pad coord
            coord = np.pad(u['src_coord'], ((0, diff), (0, 0)), constant_values=0.0)
            batch_src_coord.append(coord)
            
            # Pad edge type
            edge_type = np.pad(u['src_edge_type'], ((0, diff), (0, diff)), constant_values=0)
            batch_src_edge_type.append(edge_type)
        
        src_tokens = torch.tensor(np.array(batch_src_tokens), dtype=torch.long, device=device)
        src_distance = torch.tensor(np.array(batch_src_distance), dtype=torch.float32, device=device)
        src_coord = torch.tensor(np.array(batch_src_coord), dtype=torch.float32, device=device)
        src_edge_type = torch.tensor(np.array(batch_src_edge_type), dtype=torch.long, device=device)
        # print(f"让我看看: src_tokens:{src_tokens}, src_distance:{src_distance}, src_coord:{src_coord}, src_edge_type:{src_edge_type}")
        
        # Forward
        out = self.unimol_model(
            src_tokens=src_tokens,
            src_distance=src_distance,
            src_coord=src_coord,
            src_edge_type=src_edge_type,
            return_repr=True
        )
        unimol_tensor = out['cls_repr']
        if unimol_tensor is None:
            print("Warning: Uni-Mol CLS features not available, skipping attachment.")
            return Zh

        # optionally project unimol tensor to latent dim
        if getattr(self, 'unimol_proj', None) is not None:
            unimol_tensor = self.unimol_proj(unimol_tensor)

        if self.unimol_cfg.get('use_outnorm'):
            Zh = self._match_Zh_to_unimol(Zh, unimol_tensor, global_batch, method=self.unimol_cfg.get('outnorm_method', 'scalar'))

        # split 512-dim cls_repr into 8 chunks of 64-dim and attach to 8 global nodes
        num_graphs = unimol_tensor.shape[0]
        unimol_parts = unimol_tensor.view(num_graphs, 8, 64) # [num_graphs, 8, 64]
        unimol_flat = unimol_parts.reshape(-1, 64) # [num_graphs * 8, 64]
        
        Zh_cat = torch.cat([Zh, unimol_flat], dim=-1)
        return Zh_cat

    def _match_Zh_to_unimol(self, Zh, unimol_tensor, global_batch, method='scalar', eps=1e-6):
        """Match Zh distribution to unimol's distribution (treat unimol as reference).

        Zh: [N_nodes, latent_dim]
        unimol_tensor: [num_graphs, cls_dim] OR [num_graphs, latent_dim] (if projected)
        global_batch: [N_nodes] - batch index for each node
        method: 'scalar' or 'channel'
        Returns: Zh_aligned (same shape and device as Zh) — do not change unimol_tensor.
        """
        device = Zh.device
        Zh_graph_mean = scatter_mean(Zh, global_batch, dim=0)  # [num_graphs, latent_dim]

        # Scalars for matching (global mean/std across graphs and channels)
        zh_mean_scalar = Zh_graph_mean.mean().detach()
        zh_std_scalar = Zh_graph_mean.std().detach()

        # unimol scalars (treat unimol as DINO scalar reference)
        u_mean_scalar = unimol_tensor.mean().detach()
        u_std_scalar = unimol_tensor.std().detach()

        if method == 'channel' and unimol_tensor.shape[-1] == Zh_graph_mean.shape[-1]:
            # per-channel (global per-channel) match: use dataset-level per-channel mean/std
            zh_mean_chan = Zh_graph_mean.mean(dim=0, keepdim=True).detach()  # [1, latent_dim]
            zh_std_chan = Zh_graph_mean.std(dim=0, keepdim=True).detach()
            u_mean_chan = unimol_tensor.mean(dim=0, keepdim=True).detach()
            u_std_chan = unimol_tensor.std(dim=0, keepdim=True).detach()
            # apply to node-level Zh
            Zh_normed = (Zh - zh_mean_chan.to(device)) / (zh_std_chan.to(device) + eps)
            Zh_aligned = Zh_normed * u_std_chan.to(device) + u_mean_chan.to(device)
            return Zh_aligned
        else:
            # scalar match: compute global scalar mean/std and map node-level Zh
            Zh_normed = (Zh - zh_mean_scalar.to(device)) / (zh_std_scalar.to(device) + eps)
            Zh_mapped_scalar = Zh_normed * u_std_scalar.to(device) + u_mean_scalar.to(device)
            return Zh_mapped_scalar
    
    def training_step(self, batch, batch_idx):

        t1 = time()
        B=len(batch)
        #==========================================for reconstruct and kl===========================================
        #随机取2个构象
        # samples = [random.sample(x, 2) for x in batch]          # [B, 2]
        samples = sum(batch, [])                              # flatten -> [2B]

        x_list = [s['x'] for s in samples]                      # len = 2B
        h_list = [s['h'] for s in samples]
        
        x = torch.cat(x_list, dim=0)
        h = torch.cat(h_list, dim=0)
        # print(f"device:{self.device},x.len{x.shape[0]}")
        batch_ligand = torch.cat([torch.full((len(h_list[i]),), i, dtype=torch.long,device=self.device)
                           for i in range(len(samples))])  # [2B].to(x.device)
        # print(f"len_samples:{len(samples)},len_x_list:{len(x_list)},len_h_list:{len(h_list)},len_x:{len(x)},len_h:{len(h)},len_batch_ligand:{len(batch_ligand)}")
        # h: [N_node,1]
        # x: [N_node,3]
        # batch_ligand: [N_node]

        t2 = time()
        #data aug,先不管它
        # with torch.no_grad():
        #     # add noise to protein_pos
        #     protein_noise = torch.randn_like(protein_pos) * self.cfg.train.pos_noise_std
        #     gt_protein_pos = batch.protein_pos + protein_noise
        #     # random rotation as data aug
        #     if self.cfg.train.random_rot:
        #         M = np.random.randn(3, 3)
        #         Q, __ = np.linalg.qr(M)
        #         Q = torch.from_numpy(Q.astype(np.float32)).to(ligand_pos.device)
        #         gt_protein_pos = gt_protein_pos @ Q
        #         ligand_pos = ligand_pos @ Q

        num_graphs = batch_ligand.max().item() + 1

        #ligand居中
        x, _ = center_pos(
            ligand_pos=x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        ) 

        t3 = time()
        # t：[num_graphs,1],取值是[0,1)
        t = torch.rand(
            [num_graphs, 1], dtype=x.dtype, device=x.device
        ).index_select(
            0, batch_ligand
        )  # different t for different molecules.

        #连续时间步下，防止t太小导致的计算问题
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)  # clamp t to [t_min,1]

        t4 = time()
        #把h转为0,1,2，再转为one-hot编码
        h = [self.MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h = torch.tensor(h, dtype=torch.long,device=self.device)
        K=len(self.MAP_ATOM_TYPE_ONLY_TO_INDEX)

        one_hot_h=F.one_hot(h, K).float()  # [N, K]
        # create masked token / masked coordinate inputs
        masked_one_hot_h, target_token, mask_token_bool = self.get_masked_token(h, self.cfg['train'].get('mask_token_prob'))
        masked_x, mask_coord_bool = self.get_noised_coord(x, self.cfg['train'].get('mask_pos_prob'), self.cfg['train'].get('noise'), self.cfg['train'].get('noise_type'))

        Zh_sampled, Zx_sampled, global_batch, Zh_kl_loss, Zx_kl_loss, node_h_updated, node_x_updated = self.encode(masked_one_hot_h, masked_x, batch_ligand,deterministic=False, return_nodes=True)
        
        # now attach unimol CLS features to Zh and pass to decoder for reconstruction losses
        Zh_for_decoder = self._attach_unimol_to_Zh(Zh_sampled,global_batch, h, x, batch_ligand)
        noise_cfg = self.cfg['train'].get('zh_noise')
        if noise_cfg.get('enabled', False):
            tau = noise_cfg.get('tau', 0.1)
            sigma = torch.abs(torch.randn(1, device=self.device) * tau)
            Zh_for_decoder = Zh_for_decoder + torch.randn_like(Zh_for_decoder) * sigma
        
        # print(f"Zh_sampled_shape:{Zh_sampled.shape}")
        # print(f"Zx_sampled_shape:{Zx_sampled.shape}")
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_for_decoder,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
        )
        
        # here the discretised_loss is close for current version.

        recon_loss = torch.mean(self.cfg['train']['recon_loss']['c_loss_weight'] *c_loss + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss + discretised_loss)
        kl_loss=torch.mean(self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] *Zh_kl_loss + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss)
        Zh_kl_loss=torch.mean(Zh_kl_loss)
        Zx_kl_loss=torch.mean(Zx_kl_loss)
        #======================================== Masked token + coord loss ========================================
        masked_token_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        masked_coord_loss = torch.tensor(0., dtype=torch.float32, device=self.device)
        # number of original nodes (exclude global nodes that were appended)
        n_original_nodes = one_hot_h.shape[0]
        if mask_token_bool.any():
            # node_h_updated now contains only original nodes (no global nodes),
            # so we can use it directly without slicing
            logits = self.token_classifier(node_h_updated)  # [N, K]
            masked_logits = logits[mask_token_bool]
            target = target_token[mask_token_bool].to(device=self.device)
            if masked_logits.shape[0] > 0:
                masked_token_loss = F.cross_entropy(masked_logits, target)
        if mask_coord_bool.any():
            pred_coords = node_x_updated[mask_coord_bool]
            target_coords = x[mask_coord_bool]
            if pred_coords.numel() > 0:
                masked_coord_loss = F.l1_loss(pred_coords, target_coords, reduction='mean')
        # add to loss with weights from cfg
        masked_token_loss_weight = self.cfg['train'].get('masked_token_loss_weight')
        masked_coord_loss_weight = self.cfg['train'].get('masked_coord_loss_weight')
        # ====================================== 计算两组构象的 constrastive loss ===================================
        conf1=(global_batch%2==0)
        conf2=(global_batch%2==1)
        if self.cfg['train']['contrastive_loss']['type']=='mse':
            Zh_conf1=Zh_sampled[conf1].detach()
            Zh_conf2=Zh_sampled[conf2]
            contrastive_loss = ((Zh_conf1 - Zh_conf2) ** 2).mean()
        elif self.cfg['train']['contrastive_loss']['type']=='ce_per_node':
            Zh_conf1=Zh_sampled[conf1].detach()
            Zh_conf2=Zh_sampled[conf2]
            p_teacher = F.softmax(Zh_conf1, dim=1)
            p_student = F.softmax(Zh_conf2, dim=1)
            contrastive_loss = -(p_teacher * p_student.log()).sum(dim=1).mean()
        elif self.cfg['train']['contrastive_loss']['type']=='ce_flatten':
            Zh_conf1=Zh_sampled[conf1].detach()
            Zh_conf2=Zh_sampled[conf2]
            Zh_conf1_reshaped = Zh_conf1.reshape(B, -1) 
            Zh_conf2_reshaped = Zh_conf2.reshape(B, -1)
            p_teacher = F.softmax(Zh_conf1_reshaped, dim=1)
            p_student = F.softmax(Zh_conf2_reshaped, dim=1)
            contrastive_loss = -(p_teacher * p_student.log()).sum(dim=1).mean()
        elif self.cfg['train']['contrastive_loss']['type']=='infonce':
            Zh_conf1=Zh_sampled[conf1]
            Zh_conf2=Zh_sampled[conf2]
            Zh_conf1_reshaped = Zh_conf1.reshape(B, -1)  # [B*num_global_nodes, latent_dim] → [B, num_global_nodes*latent_dim]
            Zh_conf2_reshaped = Zh_conf2.reshape(B, -1)
            logit_conf1_to_conf2 = torch.matmul(Zh_conf1_reshaped, Zh_conf2_reshaped.T) * self.logit_scale
            logit_conf2_to_conf1 = logit_conf1_to_conf2.T
            label = torch.arange(B, device=self.device)
            contrastive_loss = (F.cross_entropy(logit_conf1_to_conf2, label) + F.cross_entropy(logit_conf2_to_conf1, label))/2
       
        ########################################## 计算atom_num_predictor 的loss ########################################
        real_atom_nums = scatter_sum(torch.ones_like(batch_ligand,device=self.device), batch_ligand, dim=0)  # [num_graphs]
        global_representation = Zh_for_decoder.reshape(2*B,-1)   #2*B因为是两个构象
        # print(f"real_atom_nums.shape:{real_atom_nums.shape}")
        # print(f"global_representation.shape:{global_representation.shape}")
        pred_atom_nums = self.atom_num_predictor(global_representation).squeeze(-1) 
        atom_num_loss = F.mse_loss(pred_atom_nums, real_atom_nums.float()) 

        # Optional per-graph Zh stats and pred stats for debugging
        if self.cfg['train'].get('debug', False):
            try:
                num_graphs = Zh_sampled.shape[0] // self.encoder_config['global_node_num']
                Zh_r = Zh_sampled.view(num_graphs, self.encoder_config['global_node_num'], -1)
                zh_var_per_graph = Zh_r.var(dim=1).mean(dim=1)  # [num_graphs]
                zh_var_mean = float(zh_var_per_graph.mean().item())
                Zx_r = Zx_sampled.view(num_graphs, self.encoder_config['global_node_num'], -1)
                zx_var_per_graph = Zx_r.var(dim=1).mean(dim=1)  # [num_graphs]
                zx_var_mean = float(zx_var_per_graph.mean().item())
                pred_atom_mean = float(pred_atom_nums.mean().item())
                pred_atom_std = float(pred_atom_nums.std().item())
                # per-dim Zh/Zx stats
                Zh_mean = torch.mean(Zh_sampled, dim=0)
                Zh_var = torch.var(Zh_sampled, dim=0, unbiased=False)
                Zx_mean = torch.mean(Zx_sampled, dim=0)
                Zx_var = torch.var(Zx_sampled, dim=0, unbiased=False)

                # log compact debug scalars
                self.log_dict({
                    'debug/Zh_var_mean': zh_var_mean,
                    'debug/Zx_var_mean': zx_var_mean,
                    'debug/pred_atom_mean': pred_atom_mean,
                    'debug/pred_atom_std': pred_atom_std,
                }, on_step=True, prog_bar=False, sync_dist=True)

                # store for on_after_backward to use
                self._last_debug_stats = {
                    'Zh_var_mean': zh_var_mean,
                    'Zx_var_mean': zx_var_mean,
                    'pred_atom_mean': pred_atom_mean,
                    'pred_atom_std': pred_atom_std,
                }

                # log per-dimension values (may be many keys; only when debug)
                per_dim_log = {}
                for i, v in enumerate(Zh_mean.tolist()):
                    per_dim_log[f'train/Zh_mean_dim_{i}'] = v
                for i, v in enumerate(Zh_var.tolist()):
                    per_dim_log[f'train/Zh_var_dim_{i}'] = v
                for i, v in enumerate(Zx_mean.tolist()):
                    per_dim_log[f'train/Zx_mean_dim_{i}'] = v
                for i, v in enumerate(Zx_var.tolist()):
                    per_dim_log[f'train/Zx_var_dim_{i}'] = v
                per_dim_log['train/Zh_mean_overall'] = Zh_mean.mean().item()
                per_dim_log['train/Zh_var_overall'] = Zh_var.mean().item()
                per_dim_log['train/Zx_mean_overall'] = Zx_mean.mean().item()
                per_dim_log['train/Zx_var_overall'] = Zx_var.mean().item()
                self.log_dict(per_dim_log, on_step=True, prog_bar=False, sync_dist=True)
            except Exception as e:
                print(f"[DEBUG VAL ERROR] zh stats failure: {e}")

        #总loss
        contrastive_loss_weight=self.cfg['train']['contrastive_loss']['contrastive_loss_weight']
        recon_loss_weight=self.cfg['train']['recon_loss']['recon_loss_weight']
        kl_loss_weight=self.cfg['train']['kl_loss']['kl_loss_weight']
        atom_num_loss_weight=self.cfg['train']['atom_num_loss']['atom_num_loss_weight']
        # apply kl anneal factor to mitigate posterior collapse
        kl_cfg = self.cfg['train']['kl_loss']
        kl_anneal_cfg = kl_cfg.get('kl_anneal', {}) if kl_cfg is not None else {}
        kl_anneal_enabled = kl_anneal_cfg.get('enabled')
        if kl_anneal_enabled:
            start = float(kl_anneal_cfg.get('start', 0.0))
            end = float(kl_anneal_cfg.get('end', 1.0))
            total_steps = int(kl_anneal_cfg.get('total_steps', 20000))
            if total_steps <= 0:
                kl_scale = float(end)
            else:
                current = float(self.global_step if hasattr(self, 'global_step') else 0.0)
                kl_scale = start + (end - start) * min(current / float(total_steps), 1.0)
        else:
            kl_scale = 1.0

        
        loss= recon_loss_weight*recon_loss + kl_scale*(kl_loss_weight*kl_loss) + (contrastive_loss_weight*contrastive_loss+ atom_num_loss_weight*atom_num_loss)
        loss = loss + (masked_token_loss_weight * masked_token_loss + masked_coord_loss_weight * masked_coord_loss)

        # Compute gradient-norm for contrastive loss (pre-backward) for diagnostic purposes
        if self.cfg['train'].get('debug', False):
            try:
                params = [p for p in self.parameters() if p.requires_grad]
                grads = torch.autograd.grad(contrastive_loss, params, retain_graph=True, allow_unused=True)
                sum_sq = 0.0
                for g in grads:
                    if g is not None:
                        sum_sq += float((g.detach() ** 2).sum().item())
                contrastive_grad_norm = math.sqrt(sum_sq)
            except Exception as e:
                contrastive_grad_norm = 0.0
            try:
                # encoder-specific contribution
                enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
                enc_grads = torch.autograd.grad(contrastive_loss, enc_params, retain_graph=True, allow_unused=True)
                sum_sq_enc = 0.0
                for g in enc_grads:
                    if g is not None:
                        sum_sq_enc += float((g.detach() ** 2).sum().item())
                contrastive_grad_norm_encoder = math.sqrt(sum_sq_enc)
            except Exception:
                contrastive_grad_norm_encoder = 0.0
            # Log into lightning logger
            try:
                self.log_dict({'grad/contrastive_loss_norm': contrastive_grad_norm, 'grad/contrastive_loss_norm_encoder': contrastive_grad_norm_encoder}, on_step=True, prog_bar=False, sync_dist=True)
            except Exception:
                # fall back to print
                print(f"[DEBUG] contrastive_grad_norm: {contrastive_grad_norm}")

        # wandb.log({
        #     'lr': self.get_last_lr(),
        #     'train_loss': loss.item(),
        #     'train_recon_loss': recon_loss.item(),
        #     'train_kl_loss':kl_loss.item(),
        #     'train_contrastive_loss':contrastive_loss.item(),
        #     'atom_num_loss':atom_num_loss.item(),
        # })
        t5 = time()
        self.log_dict(
            {
                'lr': self.get_last_lr(),
                'train_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'kl_loss':kl_loss.item(),
                'kl_loss_Zh':Zh_kl_loss.item(),
                'kl_loss_Zx':Zx_kl_loss.item(),
                'kl_anneal_scale': kl_scale,
                'contrastive_loss':contrastive_loss.item(),
                'atom_num_loss':atom_num_loss.item(),
                'masked_token_loss': masked_token_loss.item(),
                'masked_coord_loss': masked_coord_loss.item(),
                'self.logit_scale':self.logit_scale.item(),
                # 'test':2
            },
            on_step=True,
            prog_bar=True,
            batch_size=self.cfg['train']['batch_size'],
            sync_dist=True
        )

        # check if loss is finite, skip update if not
        if not torch.isfinite(loss):
            return torch.tensor(0.0, device=self.device)
        self.train_losses.append(loss.clone().detach().cpu())

        t0 = time()

        if self.log_time:
            self.time_records = np.vstack((self.time_records, [t0, t1, t2, t3, t4, t5]))
            print(f'step total time: {self.time_records[-1, 0] - self.time_records[-1, 1]}, batch size: {num_graphs}')
            print(f'\tpl call & data access: {self.time_records[-1, 1] - self.time_records[-2, 0]}')
            print(f'\tunwrap data: {self.time_records[-1, 2] - self.time_records[-1, 1]}')
            print(f'\tadd noise & center pos: {self.time_records[-1, 3] - self.time_records[-1, 2]}')
            print(f'\tsample t: {self.time_records[-1, 4] - self.time_records[-1, 3]}')
            print(f'\tget loss: {self.time_records[-1, 5] - self.time_records[-1, 4]}')
            print(f'\tlogging: {self.time_records[-1, 0] - self.time_records[-1, 5]}')
        return loss

    def on_after_backward(self) -> None:
        """Lightning hook that runs after gradients are computed for the step.
        We use this to optionally log gradient norms for quick static checks.
        """
        if not self.cfg['train'].get('debug', False):
            return
        # log previously stored debug stats (from training_step)
        try:
            if hasattr(self, '_last_debug_stats'):
                self.log_dict({
                    'debug/Zh_var_mean': self._last_debug_stats.get('Zh_var_mean', 0.0),
                    'debug/pred_atom_mean': self._last_debug_stats.get('pred_atom_mean', 0.0),
                    'debug/pred_atom_std': self._last_debug_stats.get('pred_atom_std', 0.0),
                }, on_step=True, prog_bar=False, sync_dist=True)
        except Exception as e:
            print(f"[DEBUG ERROR] zh stats logging failure: {e}")

        # helper for grad norm computation
        def _grad_norm(param):
            if param is None or param.grad is None:
                return 0.0
            return float(param.grad.norm().item())

        grad_info = {}
        # Wh_mu/Wh_log_var important for Zh
        grad_info['grad_Wh_mu'] = _grad_norm(self.Wh_mu.weight) if hasattr(self, 'Wh_mu') else 0.0
        grad_info['grad_Wh_log_var'] = _grad_norm(self.Wh_log_var.weight) if hasattr(self, 'Wh_log_var') else 0.0
        grad_info['grad_Wx_log_var'] = _grad_norm(self.Wx_log_var.weight) if hasattr(self, 'Wx_log_var') else 0.0
        # atom_num_predictor params
        try:
            total = 0.0
            count = 0
            for p in self.atom_num_predictor.parameters():
                if p.grad is not None:
                    total += float(p.grad.norm().item())
                    count += 1
            grad_info['grad_atom_num_predictor_mean'] = total / count if count > 0 else 0.0
        except Exception:
            grad_info['grad_atom_num_predictor_mean'] = 0.0

        # total grad norm across all parameters
        try:
            total_sq = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_sq += float((p.grad.detach() ** 2).sum().item())
            total_grad_norm = math.sqrt(total_sq)
        except Exception:
            total_grad_norm = 0.0

        # also include mean across atom_num_predictor param grads which we computed
        try:
            total = 0.0
            count = 0
            for p in self.atom_num_predictor.parameters():
                if p.grad is not None:
                    total += float(p.grad.norm().item())
                    count += 1
            grad_info['grad_atom_num_predictor_mean'] = total / count if count > 0 else 0.0
        except Exception:
            grad_info['grad_atom_num_predictor_mean'] = 0.0

        # Log into logger
        try:
            self.log_dict({
                'grad/Wh_mu': grad_info['grad_Wh_mu'],
                'grad/Wh_log_var': grad_info['grad_Wh_log_var'],
                'grad/Wx_log_var': grad_info['grad_Wx_log_var'],
                'grad/atom_num_predictor_mean': grad_info['grad_atom_num_predictor_mean'],
                'grad/total': total_grad_norm,
            }, on_step=True, prog_bar=False, sync_dist=True)
        except Exception:
            print("[DEBUG GRAD]", grad_info, 'total_grad_norm=', total_grad_norm)

    def validation_step(self, batch, batch_idx):
        # samples = [random.choice(x) for x in batch]
        samples = sum(batch, [])  
        x_list = [s['x'] for s in samples]
        h_list = [s['h'] for s in samples]
        original_x = torch.cat(x_list, dim=0)
        original_h = torch.cat(h_list, dim=0)
        batch_ligand = torch.cat([torch.full((len(h_list[i]),), i, dtype=torch.long,device=self.device)
                           for i in range(len(samples))])
        # print(f"original_x.device",original_x.device)
        # print(f"original_h.device",original_h.device)
        # print(f"batch_ligand.device",batch_ligand.device)
        # original_h = batch['h']
        # original_x = batch['x']
        # batch_ligand = batch['batch']

        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=batch_ligand,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )
        
        num_graphs = batch_ligand.max().item() + 1
        t = torch.rand(
            [num_graphs, 1], dtype=x.dtype, device=x.device
        ).index_select(
            0, batch_ligand
        )
        
        if not self.cfg['decoder_config']['use_discrete_t'] and not self.cfg['decoder_config']['destination_prediction']:
            t = torch.clamp(t, min=self.decoder.t_min)
        
        # 把h转为one-hot
        # print(f"original_h",original_h)
        h = [self.MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long,device=self.device)
        K = len(self.MAP_ATOM_TYPE_ONLY_TO_INDEX)
        one_hot_h = F.one_hot(h, K).float()  # [N, K]
        # print(f"one_hot_h_shape:{one_hot_h.shape}")
        # print(f"x.shape:{x.shape}")
        # print(f"batch_ligand.shape:{batch_ligand.shape}")
    # create masked inputs for validation (deterministic)
        masked_one_hot_h, target_token_val, mask_token_bool_val = self.get_masked_token(h, self.cfg['train'].get('mask_token_prob'))
        masked_x_val, mask_coord_bool_val = self.get_noised_coord(x, self.cfg['train'].get('mask_pos_prob'), self.cfg['train'].get('noise'), self.cfg['train'].get('noise_type'))
        Zh_sampled, Zx_sampled, global_batch,Zh_kl_loss, Zx_kl_loss, node_h_updated_val, node_x_updated_val = self.encode(masked_one_hot_h, masked_x_val, batch_ligand,deterministic=True, return_nodes=True)

        # attach unimol CLS features (validation samples list of per-graph dicts)
        Zh_sampled_for_decoder = self._attach_unimol_to_Zh(Zh_sampled,global_batch, h, x, batch_ligand)
        # Collect Zh and Zx for validation-epoch statistics (store detached on CPU)
        # Initialize buffers if not present
        if not hasattr(self, '_val_Zh_list') or self._val_Zh_list is None:
            self._val_Zh_list = []
        if not hasattr(self, '_val_Zx_list') or self._val_Zx_list is None:
            self._val_Zx_list = []

        # Zh_sampled: [N_nodes, Zh_dim], Zx_sampled: [N_nodes, 3]
        try:
            self._val_Zh_list.append(Zh_sampled.detach().cpu())
            self._val_Zx_list.append(Zx_sampled.detach().cpu())
        except Exception:
            # if something unexpected, skip collection to avoid stopping validation
            pass

        # print(f"Zh_sampled_shape:{Zh_sampled.shape}")
        # print(f"Zx_sampled_shape:{Zx_sampled.shape}")
        c_loss, d_loss, discretised_loss = self.decoder.loss_one_step(
            t,
            protein_pos=Zx_sampled,
            protein_v=Zh_sampled_for_decoder,
            batch_protein=global_batch,
            ligand_pos=x,
            ligand_v=h,
            batch_ligand=batch_ligand,
        )
        
        # here the discretised_loss is close for current version.
        # record atom_num_loss
        B=len(batch)
        real_atom_nums = scatter_sum(torch.ones_like(batch_ligand,device=self.device), batch_ligand, dim=0)  # [num_graphs]
        global_representation = Zh_sampled_for_decoder.reshape(2*B,-1)   #2*B因为是两个构象
        # print(f"real_atom_nums.shape:{real_atom_nums.shape}")
        # print(f"global_representation.shape:{global_representation.shape}")
        pred_atom_nums = self.atom_num_predictor(global_representation).squeeze(-1) 
        atom_num_loss = F.mse_loss(pred_atom_nums, real_atom_nums.float()) 
        recon_loss = torch.mean(self.cfg['train']['recon_loss']['c_loss_weight'] *c_loss + self.cfg['train']['recon_loss']['d_loss_weight'] * d_loss + discretised_loss)
        kl_loss=torch.mean(self.cfg['train']['kl_loss']['Zh_kl_loss_weight'] *Zh_kl_loss + self.cfg['train']['kl_loss']['Zx_kl_loss_weight'] * Zx_kl_loss)
        Zh_kl_loss=torch.mean(Zh_kl_loss)
        Zx_kl_loss=torch.mean(Zx_kl_loss)
        #total loss
        loss=recon_loss*self.cfg['train']['recon_loss']['recon_loss_weight']+kl_loss*self.cfg['train']['kl_loss']['kl_loss_weight']+atom_num_loss*self.cfg['train']['atom_num_loss']['atom_num_loss_weight']
        # masked token & coord loss for validation
        masked_token_loss_val = torch.tensor(0., dtype=torch.float32, device=self.device)
        masked_coord_loss_val = torch.tensor(0., dtype=torch.float32, device=self.device)
        n_original_nodes = one_hot_h.shape[0]
        if mask_token_bool_val.any():
            logits_val = self.token_classifier(node_h_updated_val)
            masked_logits_val = logits_val[mask_token_bool_val]
            target_val = target_token_val[mask_token_bool_val].to(device=self.device)
            if masked_logits_val.shape[0] > 0:
                masked_token_loss_val = F.cross_entropy(masked_logits_val, target_val)
        if mask_coord_bool_val.any():
            pred_coords_val = node_x_updated_val[mask_coord_bool_val]
            target_coords_val = x[mask_coord_bool_val]
            if pred_coords_val.numel() > 0:
                masked_coord_loss_val = F.l1_loss(pred_coords_val, target_coords_val, reduction='mean')
        masked_token_loss_weight = self.cfg['train'].get('masked_token_loss_weight')
        masked_coord_loss_weight = self.cfg['train'].get('masked_coord_loss_weight')
        loss = loss + masked_token_loss_weight * masked_token_loss_val + masked_coord_loss_weight * masked_coord_loss_val

        generated_data = self.shared_sampling_step(batch, batch_idx, sample_num_atoms='ref', desc='Val',deterministic=True)

        molecule_builder = MoleculeBuilder()

        generated_h = torch.tensor(generated_data['h'], dtype=torch.long,device=self.device)
        generated_x = generated_data['x']
        generated_batch = generated_data['batch']

        unique_batches = torch.unique(batch_ligand)

        # 计算相似度
        similarities = []
        shape_sims = []
        for batch_idx_val in unique_batches:
            original_mask = batch_ligand == batch_idx_val

            original_atoms = original_h[original_mask].cpu().numpy().tolist()
            original_coords = original_x[original_mask].cpu().numpy()

            generated_mask = generated_batch == batch_idx_val

            generated_atoms = generated_h[generated_mask].cpu().numpy().tolist()
            generated_coords = generated_x[generated_mask].cpu().numpy()

            try:
                original_mol = molecule_builder.build_mol(original_coords, original_atoms)
            except Exception as e:
                print(f"Warning: Molecule building failed for original: {e}")
                original_mol = None

            try:
                generated_mol = molecule_builder.build_mol(generated_coords, generated_atoms)
            except Exception as e:
                print(f"Warning: Molecule building failed for generated: {e}")
                generated_mol = None

            if original_mol is None or generated_mol is None:
                print(f"Warning: Invalid molecule for batch_idx_val {batch_idx_val.item()}")
                continue

            try:
                similarity = molecule_builder.compute_iou(original_mol, generated_mol)
                similarities.append(similarity)
            except Exception as e:
                print(f"Warning: compute_iou failed: {e}")

            try:
                shape_dist = rdShapeHelpers.ShapeTanimotoDist(generated_mol, original_mol)
                shape_sim = 1.0 - shape_dist
                shape_sims.append(shape_sim)
            except Exception as e:
                print(f"Warning: shape similarity failed: {e}")

            # 保存分子 logic
            if batch_idx < 2:
                epoch = self.current_epoch
                save_path = os.path.join(self.save_dir, f'val_mols_epoch_{epoch}')
                os.makedirs(save_path, exist_ok=True)
                # Save original
                with Chem.SDWriter(os.path.join(save_path, f'batch_{batch_idx}_mol_{batch_idx_val.item()}_orig.sdf')) as w:
                    w.write(original_mol)
                # Save reconstructed
                with Chem.SDWriter(os.path.join(save_path, f'batch_{batch_idx}_mol_{batch_idx_val.item()}_recon.sdf')) as w:
                    w.write(generated_mol)

        if similarities:
            avg_similarity = torch.tensor(similarities).mean().item()
            avg_shape_similarity = torch.tensor(shape_sims).mean().item()
        else:
            avg_similarity = 0.0  
            avg_shape_similarity = 0.0
        # wandb.log({
        #     'val_loss': loss.item(),
        #     'val_recon_loss': recon_loss.item(),
        #     'val_kl_loss':kl_loss.item(),
        #     'val_atom_num_loss': atom_num_loss.item(),
        #     'val_similarity': avg_similarity,
        # })
        # ====================================== 计算两组构象的 constrastive loss ===================================
        conf1=(global_batch%2==0)
        conf2=(global_batch%2==1)
        if self.cfg['train']['contrastive_loss']['type']=='mse':
            Zh_conf1=Zh_sampled[conf1].detach()
            Zh_conf2=Zh_sampled[conf2]
            contrastive_loss = ((Zh_conf1 - Zh_conf2) ** 2).mean()
        elif self.cfg['train']['contrastive_loss']['type']=='ce_per_node':
            Zh_conf1=Zh_sampled[conf1].detach()
            Zh_conf2=Zh_sampled[conf2]
            p_teacher = F.softmax(Zh_conf1, dim=1)
            p_student = F.softmax(Zh_conf2, dim=1)
            contrastive_loss = -(p_teacher * p_student.log()).sum(dim=1).mean()
        elif self.cfg['train']['contrastive_loss']['type']=='ce_flatten':
            Zh_conf1=Zh_sampled[conf1].detach()
            Zh_conf2=Zh_sampled[conf2]
            Zh_conf1_reshaped = Zh_conf1.reshape(B, -1) 
            Zh_conf2_reshaped = Zh_conf2.reshape(B, -1)
            p_teacher = F.softmax(Zh_conf1_reshaped, dim=1)
            p_student = F.softmax(Zh_conf2_reshaped, dim=1)
            contrastive_loss = -(p_teacher * p_student.log()).sum(dim=1).mean()
        elif self.cfg['train']['contrastive_loss']['type']=='infonce':
            Zh_conf1=Zh_sampled[conf1]
            Zh_conf2=Zh_sampled[conf2]
            Zh_conf1_reshaped = Zh_conf1.reshape(B, -1)  # [B*num_global_nodes, latent_dim] → [B, num_global_nodes*latent_dim]
            Zh_conf2_reshaped = Zh_conf2.reshape(B, -1)
            logit_conf1_to_conf2 = torch.matmul(Zh_conf1_reshaped, Zh_conf2_reshaped.T) * self.logit_scale
            logit_conf2_to_conf1 = logit_conf1_to_conf2.T
            label = torch.arange(B, device=self.device)
            contrastive_loss = (F.cross_entropy(logit_conf1_to_conf2, label) + F.cross_entropy(logit_conf2_to_conf1, label))/2
       #====================================== 计算两组构象的 constrastive loss END ===================================
        self.log_dict({
            'val_loss': loss.item(),
        },
            prog_bar=True, 
            logger=True,
            on_step=True,
            # on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg['evaluation']['batch_size'],
        )
        self.log_dict({
            'val/val_loss': loss.item(),
            'val/val_recon_loss': recon_loss.item(),
            'val/val_kl_loss':kl_loss.item(),
            'val/val_kl_loss_Zh':Zh_kl_loss.item(),
            'val/val_kl_loss_Zx':Zx_kl_loss.item(),
            'val/val_atom_num_loss': atom_num_loss.item(),
            'val/val_similarity': avg_similarity,
            'val/val_shape_similarity': avg_shape_similarity,
            'val/val_contrastive_loss': contrastive_loss.item(),
            'val/val_masked_token_loss': masked_token_loss_val.item(),
            'val/val_masked_coord_loss': masked_coord_loss_val.item(),
        },
            prog_bar=True, 
            logger=True,
            # on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.cfg['evaluation']['batch_size'],
        )
        
        return loss

    def shared_sampling_step(self, batch, batch_idx, sample_num_atoms, desc='',deterministic=True):
        # here we need to sample the molecules in the validation step
        # samples = [random.choice(x) for x in batch]
        samples = sum(batch, [])  
        x_list = [s['x'] for s in samples]
        h_list = [s['h'] for s in samples]
        x = torch.cat(x_list, dim=0)
        h = torch.cat(h_list, dim=0)
        batch_ligand = torch.cat([torch.full((len(h_list[i]),), i, dtype=torch.long,device=self.device)
                           for i in range(len(samples))])
        # h = batch['h']
        # x = batch['x']
        # batch_ligand = batch['batch']

        num_graphs = batch_ligand.max().item() + 1  # B

        n_nodes = batch_ligand.size(0)  # N_lig

        #和train一样的居中
        x, offset = center_pos(
            ligand_pos=x,
            batch_ligand=batch_ligand,
            mode=True,
        ) 

        # 决定配体原子数目，其实和原来一样就可以，等到需要加入蛋白质信息的时候再采样蛋白质
        #prior根据蛋白质信息决定，后者根据传入的batch.ligand_element_batch，我们先用ref
        if sample_num_atoms == 'prior':
            # ligand_num_atoms = []
            # for data_id in range(len(batch)):
            #     data = batch[data_id]
            #     pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy() * self.cfg['data']['normalizer_dict']['pos'])
            #     ligand_num_atoms.append(atom_num.sample_atom_num(pocket_size).astype(int))
            # batch_ligand = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(ligand_num_atoms,device=self.device))
            # ligand_num_atoms = torch.tensor(ligand_num_atoms, dtype=torch.long, device=self.device)
            raise NotImplementedError("sample_num_atoms='prior' is not implemented yet.")
        elif sample_num_atoms == 'ref':
            batch_ligand = batch_ligand
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand,device=self.device), batch_ligand, dim=0)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")
        #累计和
        ligand_cum_atoms = torch.cat([
            torch.tensor([0], dtype=torch.long, device=x.device), 
            ligand_num_atoms.cumsum(dim=0)
        ])

        # TODO reuse for visualization and test
        #采样过程
        h = [self.MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in h]
        h_indices = torch.tensor(h, dtype=torch.long,device=self.device)
        K=len(self.MAP_ATOM_TYPE_ONLY_TO_INDEX)
        h=F.one_hot(h_indices, K).float()  # [N, K]

        # Zh_sampled, Zx_sampled, global_batch,Zh_kl_loss, Zx_kl_loss = self.encode(one_hot_h, x, batch_ligand)
        global_nodes,global_position,global_batch,Zh_kl_loss, Zx_kl_loss= self.encode(h, x, batch_ligand,deterministic=deterministic)
        
        # attach unimol CLS features for sampling
        global_nodes = self._attach_unimol_to_Zh(global_nodes,global_batch, h_indices, x, batch_ligand)

        # print("global_nodes",global_nodes.shape)
        # print("global_position",global_position.shape)
        # print("global_batch",global_batch.shape)
        theta_chain, sample_chain, y_chain = self.decoder.sample(
            protein_pos=global_position,
            protein_v=global_nodes,
            batch_protein=global_batch,
            batch_ligand=batch_ligand,
            # n_nodes=n_nodes,
            sample_steps=self.cfg['evaluation']['sample_steps'],
            n_nodes=num_graphs,
            # ligand_pos=ligand_pos,  # for debug only
            desc=desc,
        )

        #保存最终结果
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        # #反坐标归一化（之前也没看到哪里做了归一化啊）
        # pred_pos = pred_pos * torch.tensor(
        #     self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=x.device
        # )
        # out_batch = copy.deepcopy(batch)
        # out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
        #     self.cfg.data.normalizer_dict.pos, dtype=torch.float32, device=x.device
        # )

        #类型，转成原子序数
        pred_v = one_hot.argmax(dim=-1) #pred_v=[0,1,2……]
        pred_atom_type = [self.MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in pred_v.tolist()] #pred_atom_type=[6,7,8……]

        # for visualization
        atom_type = [self.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type]  # List[int]
        atom_type = torch.tensor(atom_type, dtype=torch.long, device=x.device)  # [N_lig]  [0,1,2……]

        # # 芳香性先不管
        # pred_aromatic = trans.is_aromatic_from_index(
        #     pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        # ) # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))

        out_data={'h':pred_atom_type,'x':pred_pos,'batch':batch_ligand}

        return out_data

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        print(f"epoch_loss: {epoch_loss}")
        self.log(
            "epoch_loss",
            epoch_loss.to(self.device),
            batch_size=self.cfg['train']['batch_size'],
            sync_dist=True
        )
        self.train_losses = []

    def on_validation_epoch_start(self) -> None:
        """Reset validation accumulators at start of validation epoch."""
        self._val_Zh_list = []
        self._val_Zx_list = []

    def on_validation_epoch_end(self) -> None:
        """Compute mean and var for collected Zh and Zx over the whole validation epoch and log to wandb."""
        # If no data collected, skip
        if not hasattr(self, '_val_Zh_list') or not self._val_Zh_list:
            print("No validation Zh/Zx collected this epoch.")
            return

        try:
            Zh_cat = torch.cat(self._val_Zh_list, dim=0)  # [N_total, Zh_dim]
            Zx_cat = torch.cat(self._val_Zx_list, dim=0)  # [N_total, 3]

            Zh_mean = torch.mean(Zh_cat, dim=0)
            Zh_var = torch.var(Zh_cat, dim=0, unbiased=False)

            Zx_mean = torch.mean(Zx_cat, dim=0)
            Zx_var = torch.var(Zx_cat, dim=0, unbiased=False)

            # Prepare dict for wandb logging (convert to floats/lists)
            log_dict = {}
            # Zh: log per-dimension mean/var with keys zh/mean/0 ... zh/var/0 ...
            for i, v in enumerate(Zh_mean.tolist()):
                log_dict[f'val/Zh_mean_dim_{i}'] = v
            for i, v in enumerate(Zh_var.tolist()):
                log_dict[f'val/Zh_var_dim_{i}'] = v

            # Zx (3 dims)
            for i, v in enumerate(Zx_mean.tolist()):
                log_dict[f'val/Zx_mean_dim_{i}'] = v
            for i, v in enumerate(Zx_var.tolist()):
                log_dict[f'val/Zx_var_dim_{i}'] = v

            # Also log aggregated scalars
            log_dict['val/Zh_mean_overall'] = Zh_mean.mean().item()
            log_dict['val/Zh_var_overall'] = Zh_var.mean().item()
            log_dict['val/Zx_mean_overall'] = Zx_mean.mean().item()
            log_dict['val/Zx_var_overall'] = Zx_var.mean().item()

            # Send to wandb (if available)
            try:
                self.log_dict(log_dict, on_epoch=True,sync_dist=True,batch_size=self.cfg['evaluation']['batch_size'],)
            except Exception as e:
                print(f"wandb logging failed: {e}")
        except Exception as e:
            print(f"Failed computing validation Zh/Zx stats: {e}")
    
    
    def configure_optimizers(self):
        # 将 self.cfg['train'] 转换为对象
        train_cfg = dict_to_namespace(self.cfg['train'])
        # 将 self.cfg['train']['optimizer'] 转换为对象
        optimizer_cfg = dict_to_namespace(self.cfg['train']['optimizer'])
        self.optim = get_optimizer(optimizer_cfg, self)
        self.scheduler, self.get_last_lr = get_scheduler(train_cfg, self.optim)
    
        return {
            'optimizer': self.optim, 
            'lr_scheduler': self.scheduler,
            # 'monitor': 'val_loss',  #同一个
        }

    def test_step(self, batch, batch_idx):
        original_x = batch['x']
        original_h = batch['h']
        original_batch = batch['batch']
        print(original_batch)

        # 居中配体
        x, _ = center_pos(
            ligand_pos=original_x,
            batch_ligand=original_batch,
            mode=self.cfg['decoder_config']['center_pos_mode']
        )

        # 使用 encoder 获取 global_position
        h = [self.MAP_ATOM_TYPE_ONLY_TO_INDEX[i.item()] for i in original_h]
        h = torch.tensor(h, dtype=torch.long,device=self.device)
        K = len(self.MAP_ATOM_TYPE_ONLY_TO_INDEX)
        one_hot_h = F.one_hot(h, K).float()  # [N, K]
        global_nodes, global_position, global_batch,_,_ = self.encode(one_hot_h, x, original_batch,deterministic=True)

        unique_batches = torch.unique(original_batch)
        original_unique_list = unique_batches.tolist()
        
        grouped_results = []
        for batch_idx_val in unique_batches:
            mask = original_batch == batch_idx_val
            grouped_results.append({
                'initial': {
                    'x': original_x[mask].cpu().numpy().tolist(),
                    'h': original_h[mask].cpu().numpy().tolist(),
                    'batch': original_batch[mask].cpu().numpy().tolist(),
                },
                'samples': []
            })
        
        # # 将 global_position 保存为 .xyz 文件
        # self._save_global_position_as_xyz(global_position, original_batch, batch_idx)

        n_samples = self.cfg['evaluation']['num_samples']
        samples = [self.shared_sampling_step(batch, batch_idx,
                                            sample_num_atoms=self.cfg['evaluation']['sample_num_atoms'],
                                            desc=f'Test-{i}/{n_samples}',deterministic=True)
                    for i in range(n_samples)]
        
        # 重组结果
        for sample_data in samples:
            sample_batch = sample_data['batch']
            for batch_idx_val in torch.unique(sample_batch):
                mask = sample_batch == batch_idx_val
                idx = original_unique_list.index(batch_idx_val.item())
                
                h_data = sample_data['h']
                if isinstance(h_data, torch.Tensor):
                    h_selected = h_data[mask].cpu().numpy().tolist()
                else:
                    mask_indices = torch.where(mask)[0].tolist()
                    h_selected = [h_data[i] for i in mask_indices]
                
                grouped_results[idx]['samples'].append({
                    'x': sample_data['x'][mask].cpu().numpy().tolist(),
                    'h': h_selected,
                    'batch': sample_batch[mask].cpu().numpy().tolist(),
                })
        
        if not hasattr(self, 'test_results') or self.test_results is None:
            self.test_results = []
        self.test_results.extend(grouped_results)
        return grouped_results
    
    def _save_global_position_as_xyz(self, global_position, batch, batch_idx):
        """将 global_position 保存为 .xyz 文件"""
        print(f"global_position shape: {global_position.shape}")
        print(f"global_position: {global_position}")
        
        all_pos = []
        
        # 确保 global_position 是二维数组
        if len(global_position.shape) == 1:
            global_position = global_position.reshape(-1, 3)
        
        all_pos = global_position.cpu().numpy()
        
        # 创建文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.save_dir, f'global_position_all_{timestamp}.xyz')
        
        # 写入 .xyz 文件
        with open(filename, 'w') as f:
            f.write(f"{len(all_pos)}\n")
            f.write(f"Global Position for All Batches\n")
            for (x, y, z) in all_pos:
                f.write(f"C {x} {y} {z}\n")
        
        print(f"Global position saved to: {filename}")

    def on_test_epoch_end(self):
        all_results = self.test_results
        
        processed_results = []
        for mol_group in all_results:
            processed_mol = {
                'initial': mol_group['initial'],
                'samples': mol_group['samples']
            }
            processed_results.append(processed_mol)
        
        self._save_results(processed_results)
        wandb.save(os.path.join(self.save_dir, f'test_results_*.json'))

    
    def _save_results(self, results):
        """保存结果到文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(self.save_dir, f'test_results_{timestamp}.json')
        
        output = {
            'config': self.cfg,
            'results': results,
            'save_time': timestamp,
            'num_molecules': len(results),
            'num_samples_per_mol': len(results[0]['samples']) if results else 0
        }
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n测试结果已保存至: {save_path}")