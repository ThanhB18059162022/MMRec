# coding: utf-8
# @email: y463213402@gmail.com
r"""
MGCN (đã chỉnh sửa: xử lý CPU/GPU an toàn, map_location khi torch.load, thay .cuda() bằng .to(self.device))
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class MGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MGCN, self).__init__(config, dataset)

        # --- device handling: an toàn khi không có GPU ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cpu':
            print("Warning: CUDA not available, running on CPU.")
        # --------------------------------------------------

        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # embeddings / modules
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # dataset files
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        # adjacency matrices (sparse scipy -> torch sparse)
        self.norm_adj = self.get_adj_mat()
        # convert and move to device safely
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # Features: nếu có, tạo embedding từ pretrained features
        # LƯU Ý: khi torch.load file chứa tensor được lưu trên GPU, dùng map_location để load về device hiện tại
        if getattr(self, 'v_feat', None) is not None and self.v_feat is not None:
            # pretrained visual features stored in self.v_feat (numpy or torch tensor)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file, map_location=self.device)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            # đảm bảo type/tensor ở device phù hợp
            if isinstance(image_adj, torch.Tensor):
                self.image_original_adj = image_adj.to(self.device)
            else:
                # nếu image_adj là scipy sparse -> convert
                self.image_original_adj = self.sparse_mx_to_torch_sparse_tensor(image_adj).float().to(self.device)

        if getattr(self, 't_feat', None) is not None and self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file, map_location=self.device)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            if isinstance(text_adj, torch.Tensor):
                self.text_original_adj = text_adj.to(self.device)
            else:
                self.text_original_adj = self.sparse_mx_to_torch_sparse_tensor(text_adj).float().to(self.device)

        # projection layers for features (move to device later)
        if getattr(self, 'v_feat', None) is not None and self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if getattr(self, 't_feat', None) is not None and self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # other modules
        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

        # Move model parameters to the chosen device (an toàn)
        self.to(self.device)

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        if isinstance(sparse_mx, torch.Tensor):
            return sparse_mx.coalesce() if sparse_mx.is_sparse else sparse_mx
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        tensor = torch.sparse.FloatTensor(indices, values, shape)
        return tensor.coalesce()

    def forward(self, adj, train=False):
        # ensure adj on same device
        if isinstance(adj, torch.Tensor):
            adj = adj.to(self.device)
        # feature projections
        if getattr(self, 'v_feat', None) is not None and self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight.to(self.device))
        if getattr(self, 't_feat', None) is not None and self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight.to(self.device))

        # Behavior-Guided Purifier
        image_item_embeds = torch.multiply(self.item_id_embedding.weight.to(self.device), self.gate_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight.to(self.device), self.gate_t(text_feats))

        # User-Item View
        item_embeds = self.item_id_embedding.weight.to(self.device)
        user_embeds = self.user_embedding.weight.to(self.device)
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        # image graph conv
        if getattr(self, 'image_original_adj', None) is not None:
            img_adj = self.image_original_adj.to(self.device) if isinstance(self.image_original_adj, torch.Tensor) else self.image_original_adj
            for i in range(self.n_layers):
                if self.sparse:
                    image_item_embeds = torch.sparse.mm(img_adj, image_item_embeds)
                else:
                    image_item_embeds = torch.mm(img_adj, image_item_embeds)
            image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
            image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)
        else:
            # fallback: use item embeddings if no image graph
            image_embeds = torch.cat([content_embeds[self.n_users:], content_embeds[self.n_users:]], dim=0)

        # text graph conv
        if getattr(self, 'text_original_adj', None) is not None:
            txt_adj = self.text_original_adj.to(self.device) if isinstance(self.text_original_adj, torch.Tensor) else self.text_original_adj
            for i in range(self.n_layers):
                if self.sparse:
                    text_item_embeds = torch.sparse.mm(txt_adj, text_item_embeds)
                else:
                    text_item_embeds = torch.mm(txt_adj, text_item_embeds)
            text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
            text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)
        else:
            text_embeds = torch.cat([content_embeds[self.n_users:], content_embeds[self.n_users:]], dim=0)

        # Behavior-Aware Fuser
        att_common = torch.cat([self.query_common(image_embeds), self.query_common(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        sep_image_embeds = image_embeds - common_embeds
        sep_text_embeds = text_embeds - common_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        sep_text_embeds = torch.multiply(text_prefer, sep_text_embeds)
        side_embeds = (sep_image_embeds + sep_text_embeds + common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
