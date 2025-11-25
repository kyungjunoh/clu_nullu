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
        self.cohesion_threshold = cohesion_threshold # ★ 응집도 임계값 설정

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
        print(f'Cluster-wise Nullu enabled with k={self.n_clusters}')
        print(f'Selective Editing Enabled: Cohesion Threshold >= {self.cohesion_threshold}')

        self.ebd = ebd
        self.random_dps = random_dps
        self.centering = centering
        self.top_k_ranks = top_k_ranks
        if edit_layer_range is None:
            self.edit_layer_range = np.arange(self.num_layers)
        else:
            self.edit_layer_range = edit_layer_range

        self.f = open(f'logit_lens_test_{model.args.model_name}_cluster.txt', 'w')


    @staticmethod
    def project_into_vocabluary(vector, E, tokenizer, top_k=20, bottom_k=-1):
        """ HalluSpace 방향을 단어로 해석하는 함수 """
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
        """ 은닉 상태 추출 """
        hidden_sent_embs = []
        for ins in tqdm(data):
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prompt = ins['question']
            answer = ins['answer']

            if self.ebd == 'mean':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0] 
                hidden_sent_embs.append(hidden_states.mean(1).cpu())
            elif self.ebd == 'last':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0] 
                hidden_sent_embs.append(hidden_states[:, -1].cpu())
            elif self.ebd == 'mlp_residual':
                _, mlp_residual, _, _, _, _ = self.model.get_activations(image, prompt, answer) 
                hidden_sent_embs.append(mlp_residual[:, -1].cpu()) 
            else:
                raise NotImplementedError
        
        hidden_sent_embs = torch.stack(hidden_sent_embs).permute(1, 0, 2)  # [L, N, D]
        return hidden_sent_embs


    def _get_difference_matrix_clusters(self, pos_data, neg_data):
        """
        Layer-Wise Clustering 및 응집도 기반 필터링 수행
        """
        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(pos_data) if isinstance(pos_data, list) else pos_data.permute(1, 0, 2)
        preferred_sent_embs = self._get_hidden_sentence_embeddings(neg_data) if isinstance(neg_data, list) else neg_data.permute(1, 0, 2)

        # (L, N, D) -> 전체 차이 행렬 계산
        full_diff_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2 
        
        logging.info('Full Difference matrix calculated. Starting Layer-wise Clustering & Selective Filtering...')
        del non_preferred_sent_embs, preferred_sent_embs

        clustered_diff_matrices_by_layer = {} 

        # --- [Layer 순회 및 개별 클러스터링] ---
        for layer_num in self.edit_layer_range:
            
            # 1. 해당 레이어의 차이 벡터 추출 (N, D)
            E_layer_orig_numpy = full_diff_matrix[layer_num].cpu().numpy() 
            
            # 2. K-Means 수행 (정규화 안 된 데이터라도 방향성 군집화 시도)
            # (Tip: 성능 향상을 위해 필요하다면 여기서 E_layer_orig_numpy를 정규화해서 kmeans에 넣을 수 있음.
            #  하지만 사용자가 "Norm 안 한 게 성능이 좋다"고 했으므로 원본 사용 유지)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(E_layer_orig_numpy)
            
            layer_clusters = {}
            
            for k in range(self.n_clusters):
                indices = np.where(labels == k)[0]
                if len(indices) == 0: 
                    logging.warning(f'Layer {layer_num}: Cluster {k} is empty.')
                    continue
                
                # 3. 데이터 추출 및 차원 정리 (3D -> 2D 에러 방지)
                cluster_data_torch = full_diff_matrix[layer_num:layer_num+1, indices, :] # (1, N_c, D)
                if cluster_data_torch.dim() > 2:
                     cluster_data_torch = cluster_data_torch.squeeze() # (N_c, D)
                     if cluster_data_torch.dim() == 1: # 샘플이 1개일 경우 1D가 될 수 있음
                         cluster_data_torch = cluster_data_torch.unsqueeze(0)

                # 4. 평균 Norm (강도) 계산
                norms = torch.norm(cluster_data_torch, p=2, dim=1)
                avg_norm = torch.mean(norms).item()

                # 5. 응집도 (Cohesion) 계산
                centroid = torch.mean(cluster_data_torch, dim=0, keepdim=True) # (1, D)
                
                # 정규화 (Cohesion 계산용)
                cluster_norm = cluster_data_torch / (cluster_data_torch.norm(dim=1, keepdim=True) + 1e-8)
                centroid_norm = centroid / (centroid.norm() + 1e-8)

                # 코사인 유사도 계산
                avg_cos_sim = torch.mean(torch.mm(cluster_norm, centroid_norm.t())).item()

                logging.info(f'Layer {layer_num} | Cluster {k}: N={len(indices)} | Norm={avg_norm:.2f} | Cohesion={avg_cos_sim:.3f}')

                # --- ★ [핵심] 응집도 기반 필터링 (Selective Filtering) ★ ---
                if avg_cos_sim >= self.cohesion_threshold:
                    logging.info(f'   -> [SELECTED] Cluster {k} passed threshold ({self.cohesion_threshold}). Adding to Edit list.')
                    # 다시 3차원 형태로 저장 (SVD 함수와의 호환성 유지: 1, N_c, D)
                    layer_clusters[k] = cluster_data_torch.unsqueeze(0) 
                else:
                    logging.info(f'   -> [DROPPED] Cluster {k} failed threshold ({self.cohesion_threshold}). Treating as noise.')

            # 선택된 클러스터가 하나라도 있을 때만 저장
            if layer_clusters:
                clustered_diff_matrices_by_layer[layer_num] = layer_clusters
            else:
                logging.warning(f'Layer {layer_num}: No clusters passed the cohesion threshold.')

        return clustered_diff_matrices_by_layer


    def get_edit_features_clusters(self, pos_data, neg_data):
        """ Layer-wise Clustering 결과를 받아 SVD를 위한 특징 딕셔너리로 변환합니다. """
        clustered_diff_matrices_by_layer = self._get_difference_matrix_clusters(pos_data, neg_data)
        
        edit_features_by_layer_cluster = {} 

        for layer_num, cluster_dict in clustered_diff_matrices_by_layer.items():
            for cluster_id, diff_matrix_slice in cluster_dict.items():
                
                # 해당 레이어에 맞는 모델 파라미터 키 찾기
                target_keys = []
                for key in self.model.model.state_dict():
                    if 'mlp' in key and 'weight' in key and not 'visual_encoder' in key and not 'vision_tower' in key:
                         # 키 파싱 (모델마다 구조가 다를 수 있으니 주의)
                         parts = key.split('.')
                         try:
                             l_num = int(parts[self.lm_sep_idx])
                             if l_num == layer_num:
                                 target_keys.append(key)
                         except:
                             continue
                
                # 해당 레이어의 키가 발견되면 (보통 mlp layer 키 하나임)
                if target_keys:
                    # 대표 키 하나만 사용 (SVD는 레이어 단위로 하므로)
                    # 여기서는 down_proj 등을 예시로 쓰지만, 어차피 SVD용 데이터는 diff_matrix_slice임
                    # 키 이름 유니크하게 만들기
                    primary_key = target_keys[0] 
                    new_key = f"{primary_key}_cluster_{cluster_id}"
                    edit_features_by_layer_cluster[new_key] = diff_matrix_slice[0] # (N_c, D)

        return edit_features_by_layer_cluster

    
    def svd_on_edit_features(self, edit_features):
        """ SVD 수행 """
        svd = {}
        for key in edit_features:
            M = edit_features[key].to(torch.float32)
            # Centering (선택사항)
            if self.centering:
                M = M - M.mean(dim=0, keepdim=True)
                
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        return svd


    def find_p_hallu(self, svd):
        """ HalluSpace 투영 행렬 P 계산 """
        hallu_subspace = {}
        for key in svd.keys():
            # 키 파싱 (ex: model.layers.16.mlp..._cluster_0)
            parts = key.split('.')
            try:
                layer_num = int(parts[self.lm_sep_idx])
            except:
                # 키 형식이 다를 경우 안전장치
                import re
                match = re.search(r'layers\.(\d+)\.', key)
                if match:
                    layer_num = int(match.group(1))
                else:
                    continue

            cluster_id = key.split('_')[-1]

            if layer_num not in self.edit_layer_range:
                continue
            
            singular_vectors = svd[key]['v']
            hallu_rank_list = np.arange(self.top_k_ranks)

            p_hallu = torch.zeros(self.D, self.D)
            for r in hallu_rank_list:
                if r >= singular_vectors.shape[1]: break # 랭크가 데이터 수보다 클 경우 방지

                singular_vector = singular_vectors[:, r].unsqueeze(dim=1)
                p_hallu += singular_vector @ singular_vector.T

                # (로그 기록)
                sorted_tokens = self.project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=5)
                self.f.write(f'[L {layer_num}-C {cluster_id}] Rank {r}: {" | ".join([x for x in sorted_tokens])}\n')

            hallu_subspace[key] = p_hallu
        return hallu_subspace


    def edit_model_multi_cluster(self, hallu_subspace_by_key, edit_keys=True, edit_values=True, layer_range=None):
            """
            Layer-wise / Cluster-wise HalluSpace 적용
            """
            if layer_range is None: layer_range = np.arange(self.num_layers)
            
            total_shields_applied = 0
            
            edited_state_dict = self.model.model.state_dict()
            
            for key in edited_state_dict:
                if 'layers' not in key:
                    continue
                    
                try:
                    layer_num = int(key.split('.')[self.lm_sep_idx])
                except (ValueError, IndexError):
                    continue
                
                if layer_num in layer_range:
                    # 현재 레이어에 해당하는 모든 클러스터 방패 수집
                    cluster_projections = []
                    
                    # key 예시: model.layers.16.mlp.down_proj.weight
                    # subspace_key 예시: model.layers.16.mlp.down_proj.weight_cluster_0
                    # 주의: subspace_key는 대표 키로 생성되었으므로, 현재 key가 subspace_key에 포함되는지보다는
                    # 레이어 번호와 클러스터 ID를 매칭하는 것이 더 정확할 수 있음.
                    
                    # 간단한 매칭 로직: subspace_key가 현재 key의 'prefix'를 포함하는지 확인
                    # (여기서는 단순화를 위해 레이어 번호가 같은 모든 subspace를 가져오되,
                    #  MLP 모듈 전체에 동일한 SVD 결과를 적용한다고 가정)
                    
                    for subspace_key, P_hallu in hallu_subspace_by_key.items():
                         # subspace_key에서 레이어 번호 추출
                         if f"layers.{layer_num}." in subspace_key:
                             cluster_projections.append(P_hallu)

                    # 중복 제거 (키 이름 때문에 중복될 수 있음) - P_hallu 텐서 주소 비교 또는 딕셔너리 구조 개선 필요
                    # 현재 구조에서는 get_edit_features_clusters에서 고유 키를 만들었으므로 
                    # 해당 레이어의 모든 클러스터(0, 1 등)가 들어옴.
                    
                    if not cluster_projections: continue
                    
                    # 동일한 P_hallu가 여러 번 적용되지 않도록 set으로 관리하거나(불가능), 리스트 유지
                    # (여기서는 레이어당 클러스터 개수만큼 적용됨)

                    weight = edited_state_dict[key].T 
                    modified_weight = weight.clone()
                    
                    # 적용 전 로깅 (최초 1회만)
                    if 'down_proj' in key: # 로그가 너무 많으니 down_proj에서만 찍기
                         logging.info(f'Layer {layer_num}: Applying {len(cluster_projections)} cluster shields to {key}.')

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
        """ 파이프라인 관리 """
        # 1. 클러스터링 및 필터링 (Cohesion < 0.6 제거됨)
        clustered_diff_matrices_by_layer = self._get_difference_matrix_clusters(pos_data, neg_data)
        
        all_hallu_subspaces = {}
        
        for layer_num, cluster_dict in clustered_diff_matrices_by_layer.items():
            for cluster_id, features in cluster_dict.items():
                logging.info(f"Processing SVD for Layer {layer_num} - Cluster {cluster_id}...")
                
                single_cluster_edit_features = {}
                # 키 생성 (대표 키 하나 찾기)
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
                    # features는 (1, N_c, D) -> [0] -> (N_c, D)
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