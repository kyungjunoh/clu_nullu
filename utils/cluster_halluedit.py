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
from sklearn.preprocessing import normalize # ★ K-Means 정규화를 위해 추가

logging.getLogger().setLevel(logging.INFO)

class HalluEditCluster():
    def __init__(self, model, ebd='mean', centering=False, top_k_ranks=2, edit_layer_range=None, random_dps=True, alpha=1, n_clusters=2):

        self.model = model
        self.model.model.eval()
        self.tokenizer = model.tokenizer

        self.alpha = alpha
        self.n_clusters = n_clusters

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)

        if model_config:
            self.D = model.model.config.hidden_size
            self.num_layers = model.model.config.num_hidden_layers
            self.E = model.model.lm_head
            self.lm_sep_idx = 2
        else:
            self.D = model.num_lm_hidden_size
            self.num_layers = model.num_lm_layers
            self.E = model.lm_head
            if model.args.model_name == ('MiniGPT4' or 'LLaVA-7B-HF'):
                self.lm_sep_idx = 3
            else:
                self.lm_sep_idx = 2
            
        print(f'args.model_name is {model.args.model_name}')
        print(f'Cluster-wise Nullu enabled with k={self.n_clusters}')

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
        """ HalluSpace 방향을 단어로 해석하는 함수 (동일) """
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
        """ 은닉 상태 추출 (동일) """
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
        ★★ 최종 수정: Layer-Wise Clustering 및 SVD용 데이터 분리 ★★
        """
        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(pos_data) if isinstance(pos_data, list) else pos_data.permute(1, 0, 2)
        preferred_sent_embs = self._get_hidden_sentence_embeddings(neg_data) if isinstance(neg_data, list) else neg_data.permute(1, 0, 2)

        # (L, N, D) -> 전체 차이 행렬 계산 (이것이 SVD에 사용될 '원본 데이터'입니다)
        full_diff_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2 
        
        logging.info('Full Difference matrix calculated. Starting Layer-wise Clustering...')
        del non_preferred_sent_embs, preferred_sent_embs

        # 최종 결과 구조: {Layer_Num: {Cluster_ID: Diff_Matrix_Slice (1, N_c, D)}}
        clustered_diff_matrices_by_layer = {} 

        # --- ▼ [Layer 순회 및 개별 클러스터링] ▼ ---
        for layer_num in self.edit_layer_range:
            
            # 1. 해당 레이어의 차이 벡터 추출 (N, D) - NumPy 변환
            E_layer_orig_numpy = full_diff_matrix[layer_num].cpu().numpy() 
            
            # 2. 정규화 (Normalization) - K-Means의 입력 데이터
            # 이것이 t-SNE와 K-Means를 성공시키는 핵심 단계입니다.

            # 3. K-Means (레이어별 군집화)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(E_layer_orig_numpy)
            
            layer_clusters = {}
            
            for k in range(self.n_clusters):
                indices = np.where(labels == k)[0]
                if len(indices) == 0: 
                    logging.warning(f'Layer {layer_num}: Cluster {k} is empty.')
                    continue
                
                # 4. SVD용 데이터 슬라이싱: 정규화되지 않은 원본 Torch 텐서 사용
                cluster_data_torch = full_diff_matrix[layer_num:layer_num+1, indices, :] # (1, N_c, D)

                layer_clusters[k] = cluster_data_torch
                num_samples = len(indices)

                # 2. 평균 Norm (강도)
                norms = torch.norm(cluster_data_torch, p=2, dim=1)
                avg_norm = torch.mean(norms).item()

                # 3. 응집도 (방향성)
                if cluster_data_torch.dim() > 2:
                    cluster_data_torch = cluster_data_torch.squeeze() 
                    # 혹은 안전하게: cluster_data_torch = cluster_data_torch.reshape(cluster_data_torch.shape[0], -1)

                # 2. 중심 벡터 계산 (Mean vector)
                centroid = torch.mean(cluster_data_torch, dim=0, keepdim=True) # (1, D)

                # 3. 정규화 (Normalization)
                # cluster_data_torch가 (N, D)인지 확인되었으므로 dim=1은 Feature 차원입니다.
                cluster_norm = cluster_data_torch / (cluster_data_torch.norm(dim=1, keepdim=True) + 1e-8)
                centroid_norm = centroid / (centroid.norm() + 1e-8)

                # 4. 코사인 유사도 계산 (Matrix Multiplication)
                # 이제 centroid_norm은 (1, D)인 2차원이므로 .t() (Transpose)가 정상 작동합니다.
                avg_cos_sim = torch.mean(torch.mm(cluster_norm, centroid_norm.t())).item()

                logging.info(f'Layer {layer_num} | Cluster {k}: N={num_samples} | Norm={avg_norm:.2f} | Cohesion={avg_cos_sim:.3f}')
                logging.info(f'Layer {layer_num}: Cluster {k} samples: {len(indices)}')
                
            clustered_diff_matrices_by_layer[layer_num] = layer_clusters

        return clustered_diff_matrices_by_layer


    def get_edit_features_clusters(self, pos_data, neg_data):
        """
        Layer-wise Clustering 결과를 받아 SVD를 위한 특징 딕셔너리로 변환합니다.
        """
        clustered_diff_matrices_by_layer = self._get_difference_matrix_clusters(pos_data, neg_data)
        
        # Layer_Num과 Cluster_ID를 모두 키로 사용하여 SVD에 전달할 준비
        edit_features_by_layer_cluster = {} # {key: diff_matrix (1, N_c, D)}

        for layer_num, cluster_dict in clustered_diff_matrices_by_layer.items():
            for cluster_id, diff_matrix_slice in cluster_dict.items():
                
                for key in self.model.model.state_dict():
                    # (이전 코드와 동일한 모델별 키 필터링 로직)
                    if 'mlp' in key and 'weight' in key and not 'visual_encoder' in key and not 'vision_tower' in key:
                         if int(key.split('.')[self.lm_sep_idx]) == layer_num:
                             # 키 이름에 cluster ID를 붙여서 저장
                             new_key = f"{key}_cluster_{cluster_id}"
                             edit_features_by_layer_cluster[new_key] = diff_matrix_slice[0] # (N_c, D)

        return edit_features_by_layer_cluster

    
    def svd_on_edit_features(self, edit_features):
        """ 
        SVD 수행 및 클러스터 평균 벡터 추출 
        """
        svd = {}
        cluster_means = {} # ★ 추가: 클러스터의 '대표 성향'(평균) 저장

        for key in edit_features:
            M = edit_features[key].to(torch.float32)
            
            # ★ [추가] 클러스터의 평균 환각 방향 저장
            # M shape: (N_samples, D) -> Mean shape: (D,)
            cluster_means[key] = M.mean(dim=0) 

            if self.centering:
                M = M - M.mean(dim=0, keepdim=True)
                
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False)
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
            
        return svd, cluster_means


    def find_p_hallu(self, svd, cluster_means): # ★ 인자 추가: cluster_means
        """ 
        HalluSpace 투영 행렬 P 계산 및 Before/After 시뮬레이션 로깅 
        """
        hallu_subspace = {}
        
        # 로그 헤더 (한 번만 출력되도록 조건문 추가하면 좋지만, 여기선 간단히 매번 출력)
        # self.f.write("\n=== HalluSpace Analysis (Cluster-wise) ===\n")

        for key in svd.keys():
            # 키 파싱
            parts = key.split('.')
            try: layer_num = int(parts[self.lm_sep_idx])
            except: 
                import re
                match = re.search(r'layers\.(\d+)\.', key)
                if match: layer_num = int(match.group(1))
                else: continue

            cluster_id = key.split('_')[-1] # 클러스터 ID

            if layer_num not in self.edit_layer_range: continue
            
            singular_vectors = svd[key]['v']
            hallu_rank_list = np.arange(self.top_k_ranks)

            # --- [1] Before: 편집 전 클러스터의 대표 단어 ---
            mean_vec = cluster_means[key].to('cuda')
            before_tokens = self.project_into_vocabluary(mean_vec, self.E.cpu(), self.tokenizer, top_k=8)
            
            self.f.write(f"\n--- Layer {layer_num} | Cluster {cluster_id} ---\n")
            self.f.write(f"[Before Edit] Cluster Context : {' | '.join(before_tokens)}\n")
            self.f.write("-" * 40 + "\n")

            p_hallu = torch.zeros(self.D, self.D)
            for r in hallu_rank_list:
                if r >= singular_vectors.shape[1]: break
                
                singular_vector = singular_vectors[:, r].unsqueeze(dim=1) # (D, 1)
                p_hallu += singular_vector @ singular_vector.T

                # 제거 타겟 단어 로깅
                rank_tokens = self.project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=5)
                self.f.write(f"   > [Target Rank {r}]: {' | '.join(rank_tokens)}\n")

            # --- [2] After: 편집(영공간 투영) 후 남는 단어 시뮬레이션 ---
            p_hallu_cuda = p_hallu.to('cuda')
            I = torch.eye(self.D).to('cuda')
            
            # 영공간 투영
            projected_mean_vec = (I - p_hallu_cuda) @ mean_vec
            
            after_tokens = self.project_into_vocabluary(projected_mean_vec, self.E.cpu(), self.tokenizer, top_k=8)
            
            self.f.write("-" * 40 + "\n")
            self.f.write(f"[After Edit]  Remaining Context: {' | '.join(after_tokens)}\n")
            self.f.flush()

            hallu_subspace[key] = p_hallu
        return hallu_subspace


    def edit_model_multi_cluster(self, hallu_subspace_by_key, edit_keys=True, edit_values=True, layer_range=None):
            """
            ★★ Layer-wise / Cluster-wise HalluSpace를 순차적으로 적용하여 모델을 편집합니다. ★★
            """
            if layer_range is None: layer_range = np.arange(self.num_layers)
            
            # 로깅용: 전체 적용된 방패 개수 카운트
            total_shields_applied = 0
            
            edited_state_dict = self.model.model.state_dict()
            
            for key in edited_state_dict:
                # ★★★ [수정됨] 에러 방지: 'layers'가 포함되지 않은 키(예: embedding)는 건너뜁니다. ★★★
                if 'layers' not in key:
                    continue
                    
                try:
                    layer_num = int(key.split('.')[self.lm_sep_idx])
                except (ValueError, IndexError):
                    # 혹시라도 파싱이 안 되는 키가 있으면 안전하게 건너뜁니다.
                    continue
                
                if layer_num in layer_range:
                    
                    # 해당 레이어의 키(key)에 해당하는 모든 클러스터 방패(P_hallu)를 수집
                    # 예: key='model.layers.16.mlp.down_proj.weight'
                    # 찾을 대상: 'model.layers.16.mlp.down_proj.weight_cluster_0', '..._cluster_1'
                    cluster_projections = []
                    
                    # hallu_subspace_by_key는 { 'key_name_cluster_0': P_matrix, ... } 형태
                    for subspace_key, P_hallu in hallu_subspace_by_key.items():
                        # subspace_key가 현재 모델 key로 시작하는지 확인
                        if subspace_key.startswith(key): 
                            cluster_projections.append(P_hallu)

                    if not cluster_projections: continue
                    
                    logging.info(f'Editing {key}: Applying {len(cluster_projections)} cluster shields.')

                    weight = edited_state_dict[key].T 
                    modified_weight = weight.clone()
                    
                    # --- ★ 핵심: 모든 클러스터 방패를 순차적으로 적용 ★ ---
                    for P_hallu in cluster_projections:
                        # Null Space Projection Matrix (I - alpha * P)
                        Null_space = torch.eye(self.D) - self.alpha * P_hallu
                        
                        if self.model.args.model_name == 'MiniGPT4':
                            Null_space = Null_space.to(weight.device).to(self.model.model.llama_model.dtype)
                        else:
                            Null_space = Null_space.to(weight.device).to(self.model.model.dtype)
                        
                        # 순차 적용
                        if edit_keys and 'up_proj' in key:
                            modified_weight = Null_space @ modified_weight
                        elif edit_values and 'down_proj' in key:
                            modified_weight = modified_weight @ Null_space
                        elif 'c_proj' in key: # Qwen 등
                            modified_weight = modified_weight @ Null_space
                        # (기타 모델 키 로직 생략)
                        
                        total_shields_applied += 1

                    if torch.allclose(weight, modified_weight) and ('gate_proj' not in key):
                            logging.warning(f'Module {key} not edited after projection.')

                    modified_weight = modified_weight.T 
                    edited_state_dict[key] = modified_weight.to('cuda').contiguous()

            self.model.model.load_state_dict(edited_state_dict, assign=True)
            logging.info(f'Layer-wise Multi-cluster Edited model created. (Total shields applied: {total_shields_applied})')
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

                    # ★ [수정됨] svd 함수가 cluster_means도 반환
                    svd, cluster_means = self.svd_on_edit_features(single_cluster_edit_features)
                    
                    # ★ [수정됨] find_p_hallu에 cluster_means 전달
                    hallu_subspace = self.find_p_hallu(svd, cluster_means)
                    
                    all_hallu_subspaces.update(hallu_subspace)
                    
                    del svd, single_cluster_edit_features, cluster_means
                    torch.cuda.empty_cache()
            
        self.all_hallu_subspaces = all_hallu_subspaces


    def apply_edit_end_to_end(self, pos_data, neg_data, edit_keys=True, edit_values=True, layer_range=None):
        # 1. Layer-wise HalluSpace 계산
        self.setup_for_edits_clusters(pos_data, neg_data)

        # 2. 순차적 편집 적용
        edited_model = self.edit_model_multi_cluster(self.all_hallu_subspaces, edit_keys, edit_values, layer_range)
        torch.cuda.empty_cache()

        return edited_model

# (참고: main 함수에서 HalluEditCluster 클래스로 변경하여 사용해야 합니다.)