import os
import sys
import inspect

import json
import torch
import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2
from sklearn.cluster import KMeans 
from sklearn.preprocessing import normalize

logging.getLogger().setLevel(logging.INFO)

class HalluEditCluster():
    def __init__(self, model, ebd='mean', centering=False, top_k_ranks=2, edit_layer_range=None, random_dps=True, alpha=1, n_clusters=2, cohesion_threshold=0.6):

        self.model = model
        self.model.model.eval()
        self.tokenizer = model.tokenizer

        self.alpha = alpha
        self.n_clusters = n_clusters
        self.cohesion_threshold = cohesion_threshold # ★ 응집도 임계값

        # --- [1] Text-Only 처리를 위한 이미지 토큰 길이 설정 ---
        # 모델별로 이미지 토큰 개수가 다릅니다.
        if 'MiniGPT4' in model.args.model_name:
            self.image_token_len = 32
        elif 'LLaVA' in model.args.model_name:
            self.image_token_len = 576
        elif 'mPLUG' in model.args.model_name:
            self.image_token_len = 65 # 예시값 (확인 필요)
        else:
            self.image_token_len = 576 # Default (LLaVA-1.5)
        
        print(f"[Config] Text-Only Mode Enabled: Ignoring first {self.image_token_len} image tokens.")
        print(f"[Config] Clustering Mode: k={self.n_clusters}, Threshold={self.cohesion_threshold}")

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)

        if model_config:
            self.D = model.model.config.hidden_size
            self.num_layers = model.model.config.num_hidden_layers
            self.E = model.model.lm_head
            self.lm_sep_idx = 2
        else:
            self.D = model.num_lm_hidden_size
            self.num_layers = model.num_lm_layers
            self.E = model.model.lm_head
            if model.args.model_name == ('MiniGPT4' or 'LLaVA-7B-HF'):
                self.lm_sep_idx = 3
            else:
                self.lm_sep_idx = 2
            
        print(f'args.model_name is {model.args.model_name}')

        self.ebd = ebd
        self.random_dps = random_dps
        self.centering = centering
        self.top_k_ranks = top_k_ranks
        if edit_layer_range is None:
            self.edit_layer_range = np.arange(self.num_layers)
        else:
            self.edit_layer_range = edit_layer_range

        self.f = open(f'logit_lens_final_{model.args.model_name}.txt', 'w')


    @staticmethod
    def project_into_vocabluary(vector, E, tokenizer, top_k=20, bottom_k=-1):
        vector = vector.to(torch.float32).to('cuda')
        E = E.to(torch.float32).to('cuda')
        vocab_ranking = E(vector) 
        sorted_token_ids = np.argsort(vocab_ranking.detach().cpu().numpy())[::-1]
        if bottom_k == -1:
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[:top_k]]
        else :
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[-bottom_k:][::-1]]
        return sorted_tokens


    def _get_hidden_sentence_embeddings(self, data):
        """ 
        ★ [핵심 수정] Text-Only Averaging 
        이미지 토큰을 제외하고 텍스트 토큰만 평균을 냅니다.
        """
        hidden_sent_embs = []
        for ins in tqdm(data, desc="Extracting Text Embeddings"):
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prompt = ins['question']
            answer = ins['answer']

            if self.ebd == 'mean':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                # hidden_states: [Layers, Batch, Seq_Len, Dim] -> [Layers, Seq_Len, Dim]
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0] 
                
                # --- Text Only Logic ---
                if hidden_states.shape[1] > self.image_token_len:
                    # 이미지 토큰 이후부터 끝까지 슬라이싱
                    text_states = hidden_states[:, self.image_token_len:, :]
                    hidden_sent_embs.append(text_states.mean(1).cpu())
                else:
                    # 예외 처리 (텍스트가 거의 없는 경우 등)
                    hidden_sent_embs.append(hidden_states.mean(1).cpu())

            elif self.ebd == 'last':
                # last 토큰은 이미 텍스트의 끝이므로 그대로 둠
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0] 
                hidden_sent_embs.append(hidden_states[:, -1].cpu())

            elif self.ebd == 'mlp_residual':
                _, mlp_residual, _, _, _, _ = self.model.get_activations(image, prompt, answer) 
                hidden_sent_embs.append(mlp_residual[:, -1].cpu()) 
            else:
                raise NotImplementedError
        
        hidden_sent_embs = torch.stack(hidden_sent_embs).permute(1, 0, 2)  # [Layer, N, Dim]
        return hidden_sent_embs


    def _get_difference_matrix_clusters(self, pos_data, neg_data):
        """
        [Step 2] Clustering & Filtering on Text-Only Data
        """
        # 1. Text-Only 데이터 추출 (위에서 수정된 함수 호출)
        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(pos_data) if isinstance(pos_data, list) else pos_data.permute(1, 0, 2)
        preferred_sent_embs = self._get_hidden_sentence_embeddings(neg_data) if isinstance(neg_data, list) else neg_data.permute(1, 0, 2)

        # 2. 차이 행렬 계산 (여기엔 이제 0점짜리 이미지 노이즈가 없습니다!)
        full_diff_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2 
        
        logging.info('Text-Only Difference matrix calculated. Starting Clustering & Filtering...')
        del non_preferred_sent_embs, preferred_sent_embs

        clustered_diff_matrices_by_layer = {} 

        # --- Layer 순회 ---
        for layer_num in self.edit_layer_range:
            
            E_layer_orig_numpy = full_diff_matrix[layer_num].cpu().numpy() 
            
            # 3. K-Means 수행 (정규화 안 된 원본 사용 -> 강도 정보 반영)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(E_layer_orig_numpy)
            
            layer_clusters = {}
            
            for k in range(self.n_clusters):
                indices = np.where(labels == k)[0]
                if len(indices) == 0: continue
                
                # 데이터 추출
                cluster_data_torch = full_diff_matrix[layer_num:layer_num+1, indices, :] # (1, N_c, D)
                if cluster_data_torch.dim() > 2:
                     cluster_data_torch = cluster_data_torch.squeeze() 
                     if cluster_data_torch.dim() == 1: cluster_data_torch = cluster_data_torch.unsqueeze(0)

                # 통계치 계산 (Norm & Cohesion)
                norms = torch.norm(cluster_data_torch, p=2, dim=1)
                avg_norm = torch.mean(norms).item()

                centroid = torch.mean(cluster_data_torch, dim=0, keepdim=True) 
                cluster_norm = cluster_data_torch / (cluster_data_torch.norm(dim=1, keepdim=True) + 1e-8)
                centroid_norm = centroid / (centroid.norm() + 1e-8)
                avg_cos_sim = torch.mean(torch.mm(cluster_norm, centroid_norm.t())).item()

                logging.info(f'Layer {layer_num} | Cluster {k}: N={len(indices)} | Norm={avg_norm:.2f} | Cohesion={avg_cos_sim:.3f}')

                # --- ★ [핵심] 필터링 로직 ★ ---
                if avg_cos_sim >= self.cohesion_threshold:
                    logging.info(f'   -> [SELECTED] Cluster {k} passed threshold.')
                    # 다시 3차원 형태로 저장 (SVD 호환용: 1, N_c, D)
                    layer_clusters[k] = cluster_data_torch.unsqueeze(0) 
                else:
                    logging.info(f'   -> [DROPPED] Cluster {k} removed (Noise).')

            if layer_clusters:
                clustered_diff_matrices_by_layer[layer_num] = layer_clusters
            else:
                logging.warning(f'Layer {layer_num}: All clusters dropped.')

        return clustered_diff_matrices_by_layer


    def get_edit_features_clusters(self, pos_data, neg_data):
        # 기존과 동일 (클러스터링된 데이터 매핑)
        clustered_diff_matrices_by_layer = self._get_difference_matrix_clusters(pos_data, neg_data)
        edit_features_by_layer_cluster = {} 

        for layer_num, cluster_dict in clustered_diff_matrices_by_layer.items():
            for cluster_id, diff_matrix_slice in cluster_dict.items():
                
                target_keys = []
                for key in self.model.model.state_dict():
                    if 'mlp' in key and 'weight' in key and not 'visual_encoder' in key and not 'vision_tower' in key:
                         parts = key.split('.')
                         try:
                             l_num = int(parts[self.lm_sep_idx])
                             if l_num == layer_num:
                                 target_keys.append(key)
                         except: continue
                
                if target_keys:
                    primary_key = target_keys[0] 
                    new_key = f"{primary_key}_cluster_{cluster_id}"
                    edit_features_by_layer_cluster[new_key] = diff_matrix_slice[0] # (N_c, D)

        return edit_features_by_layer_cluster

    
    def svd_on_edit_features(self, edit_features):
        # 기존과 동일
        svd = {}
        for key in edit_features:
            M = edit_features[key].to(torch.float32)
            if self.centering:
                M = M - M.mean(dim=0, keepdim=True)
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        return svd


    def find_p_hallu(self, svd):
        # 기존과 동일
        hallu_subspace = {}
        for key in svd.keys():
            parts = key.split('.')
            try: layer_num = int(parts[self.lm_sep_idx])
            except: 
                import re
                match = re.search(r'layers\.(\d+)\.', key)
                if match: layer_num = int(match.group(1))
                else: continue

            cluster_id = key.split('_')[-1]

            if layer_num not in self.edit_layer_range: continue
            
            singular_vectors = svd[key]['v']
            hallu_rank_list = np.arange(self.top_k_ranks)

            p_hallu = torch.zeros(self.D, self.D)
            for r in hallu_rank_list:
                if r >= singular_vectors.shape[1]: break 
                singular_vector = singular_vectors[:, r].unsqueeze(dim=1)
                p_hallu += singular_vector @ singular_vector.T

                sorted_tokens = self.project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=5)
                self.f.write(f'[L {layer_num}-C {cluster_id}] Rank {r}: {" | ".join([x for x in sorted_tokens])}\n')

            hallu_subspace[key] = p_hallu
        return hallu_subspace


    def edit_model_multi_cluster(self, hallu_subspace_by_key, edit_keys=True, edit_values=True, layer_range=None):
            """ Layer-wise / Cluster-wise HalluSpace 적용 """
            if layer_range is None: layer_range = np.arange(self.num_layers)
            
            total_shields_applied = 0
            edited_state_dict = self.model.model.state_dict()
            
            for key in edited_state_dict:
                if 'layers' not in key: continue
                try: layer_num = int(key.split('.')[self.lm_sep_idx])
                except: continue
                
                if layer_num in layer_range:
                    cluster_projections = []
                    for subspace_key, P_hallu in hallu_subspace_by_key.items():
                         if f"layers.{layer_num}." in subspace_key:
                             cluster_projections.append(P_hallu)

                    if not cluster_projections: continue
                    
                    weight = edited_state_dict[key].T 
                    modified_weight = weight.clone()
                    
                    if 'down_proj' in key:
                         logging.info(f'Layer {layer_num}: Applying {len(cluster_projections)} filtered shields to {key}.')

                    for P_hallu in cluster_projections:
                        Null_space = torch.eye(self.D) - self.alpha * P_hallu
                        
                        if self.model.args.model_name == 'MiniGPT4':
                            Null_space = Null_space.to(weight.device).to(self.model.model.llama_model.dtype)
                        else:
                            Null_space = Null_space.to(weight.device).to(self.model.model.dtype)
                        
                        if edit_keys and 'up_proj' in key:
                            modified_weight = Null_space @ modified_weight
                        elif edit_values and 'down_proj' in key:
                            modified_weight = modified_weight @ Null_space
                        elif 'c_proj' in key: 
                            modified_weight = modified_weight @ Null_space
                        
                        total_shields_applied += 1

                    modified_weight = modified_weight.T 
                    edited_state_dict[key] = modified_weight.to('cuda').contiguous()

            self.model.model.load_state_dict(edited_state_dict, assign=True)
            logging.info(f'Layer-wise Multi-cluster Edited model created. (Total operations: {total_shields_applied})')
            return self.model.model


    def setup_for_edits_clusters(self, pos_data, neg_data):
        clustered_diff_matrices_by_layer = self._get_difference_matrix_clusters(pos_data, neg_data)
        
        all_hallu_subspaces = {}
        for layer_num, cluster_dict in clustered_diff_matrices_by_layer.items():
            for cluster_id, features in cluster_dict.items():
                logging.info(f"Processing SVD for Layer {layer_num} - Cluster {cluster_id}...")
                
                single_cluster_edit_features = {}
                target_key = None
                for key in self.model.model.state_dict():
                     if 'mlp' in key and 'weight' in key:
                         try:
                             if int(key.split('.')[self.lm_sep_idx]) == layer_num:
                                 target_key = key
                                 break
                         except: continue
                
                if target_key:
                    new_key = f"{target_key}_cluster_{cluster_id}"
                    single_cluster_edit_features[new_key] = features[0]

                    svd = self.svd_on_edit_features(single_cluster_edit_features)
                    hallu_subspace = self.find_p_hallu(svd)
                    all_hallu_subspaces.update(hallu_subspace)
                    
                    del svd, single_cluster_edit_features
                    torch.cuda.empty_cache()
            
        self.all_hallu_subspaces = all_hallu_subspaces


    def apply_edit_end_to_end(self, pos_data, neg_data, edit_keys=True, edit_values=True, layer_range=None):
        self.setup_for_edits_clusters(pos_data, neg_data)
        edited_model = self.edit_model_multi_cluster(self.all_hallu_subspaces, edit_keys, edit_values, layer_range)
        torch.cuda.empty_cache()
        return edited_model