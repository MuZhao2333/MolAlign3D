import torch
import torch.nn as nn
from molalign3d.utils.common import MLP,compose_context
from molalign3d.models.uni_transformer import UniTransformerO2TwoUpdateGeneral


class Encoder(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, knn=32,
                 num_r_gaussian=20, edge_feat_dim=4, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='global', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False,
                 global_node_num=8,ligand_v_dim=9, name='unio2net',fix_real_x=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_node_num = global_node_num
        self.ligand_v_dim=ligand_v_dim
        self.input_adaptor=MLP(in_dim=ligand_v_dim, out_dim=hidden_dim, hidden_dim=hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False)
        # network name is passed by config
        self.net_name = name
        # If provided in kwargs (back compat not to break), pull it from arguments if passed via params dict
        # Create both adapters available
        # Instantiate UniTransformer by default
        self.unet = None
        self.pet = None
        if self.net_name == 'pet':
            # map hidden_dim to d_model for PET
            raise ValueError("This ZINC release supports the checkpoint-compatible unio2net encoder only.")
        else:
            self.unet = UniTransformerO2TwoUpdateGeneral(
                num_blocks=num_blocks,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                knn=knn,
                num_r_gaussian=num_r_gaussian,
                edge_feat_dim=edge_feat_dim,
                num_node_types=num_node_types,
                act_fn=act_fn,
                norm=norm,
                cutoff_mode=cutoff_mode,
                ew_net_type=ew_net_type,
                num_init_x2h=num_init_x2h,
                num_init_h2x=num_init_h2x,
                num_x2h=num_x2h,
                num_h2x=num_h2x,
                r_max=r_max,
                x2h_out_fc=x2h_out_fc,
                sync_twoup=sync_twoup
            )
        self.fix_real_x=fix_real_x
        # 可学习的global_h层
        self.global_h = nn.Parameter(torch.randn(self.global_node_num, self.hidden_dim))
        nn.init.normal_(self.global_h, mean=0, std=0.1) 
    def __repr__(self):
        return (
            f"Encoder:\n{self.unet}\n"
        )
    def forward(self, h, x, batch, return_nodes: bool = False, return_edge: bool = False):
        # h：[num_nodes, ligand_v_dim]
        # x：[num_nodes, 3]
        # mask_ligand：[num_nodes]
        # batch：[num_nodes]
        # num_global_nodes为每个分子添加的全局节点数量
        # Add global nodes
        batch_size = batch.max().item() + 1  # 分子数量
        # nn.par 原来的点过了全连接之后再cat
        # 定义到网络里面了
        # global_h = torch.zeros(
        #     (batch_size * self.global_node_num, self.ligand_v_dim), 
        #     device=h.device
        # )
        
        # 初始化成0
        global_x = torch.zeros(
            (batch_size * self.global_node_num, 3), 
            device=x.device
        )
        
        global_batch = torch.repeat_interleave(
            torch.arange(batch_size, device=batch.device), 
            repeats=self.global_node_num
        )

        # 组合真实节点与每个分子的global节点，确保每个分子的global节点紧随其真实原子
        # 使用compose_context保证按batch排序, 且在每个batch内保持protein(真实)/ligand(虚拟)顺序
        # h_new, x_new, batch_new, mask_ligand_new分别表示新的节点特征, 坐标, batch索引和是否为ligand(global)

        # Update h and x using the network
        # 把ligand_v_dim换到hidden_dim
        h = self.input_adaptor(h)
        global_h = self.global_h.repeat(batch_size, 1)
        # compose_context expects: (h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand)
        h_new, x_new, batch_new, mask_ligand_new = compose_context(
            h, global_h, x, global_x, batch, global_batch
        )
        # print("#####################进GNN前#################")
        # print(f'h_new:{h_new}')
        # print(f'x_new:{x_new}')
        # print(f'mask_ligand_new:{mask_ligand_new}')
        # print(f'batch_new:{batch_new}')
        # print("#####################进GNN前#################")
        # call network according to type
        if self.net_name == 'pet':
            # PET forward signature: (h, x, mask_ligand, batch, return_all=False,..)
            outputs = self.pet(h_new, x_new, mask_ligand_new, batch_new, return_all=return_nodes,fix_real_x=self.fix_real_x)
        else:
            outputs = self.unet(h_new, x_new, mask_ligand_new, batch_new, return_edge=return_edge, return_all=return_nodes,fix_real_x=self.fix_real_x)
        # print(f'output:{outputs}')
        h_updated = outputs['h']
        x_updated = outputs['x']

        # Extract global nodes using mask (works independent of node ordering)
        global_nodes_updated = h_updated[mask_ligand_new]
        global_positions_updated = x_updated[mask_ligand_new]

        if return_nodes:
            # also return node-level embeddings/coords for original (real) atoms only
            # compute node-level updates for original nodes (mask==False)
            node_mask = ~mask_ligand_new
            node_h_updated = h_updated[node_mask]
            node_x_updated = x_updated[node_mask]
            # compute updated global batch index from batch_new for global nodes
            global_batch_updated = batch_new[mask_ligand_new]
            return global_nodes_updated, global_positions_updated, global_batch_updated, node_h_updated, node_x_updated

        # compute updated global batch index from batch_new
        global_batch_updated = batch_new[mask_ligand_new]
        return global_nodes_updated, global_positions_updated, global_batch_updated

        