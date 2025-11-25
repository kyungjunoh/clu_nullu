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

logging.getLogger().setLevel(logging.INFO)

class HalluEdit():
    def __init__(self, model, ebd='mean', centering=False, top_k_ranks=2, edit_layer_range=None, random_dps=True, alpha=1):

        self.model = model
        self.model.model.eval()
        self.tokenizer = model.tokenizer

        self.alpha = alpha

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)

        if model_config: # model.model.config.model_type
            model_type = getattr(model_config, 'model_type', None)
            self.D = model.model.config.hidden_size
            self.num_layers = model.model.config.num_hidden_layers
            self.E = model.model.lm_head
            self.lm_sep_idx = 2
        else: # model.args.model_name
            self.D = model.num_lm_hidden_size
            self.num_layers = model.num_lm_layers
            self.E = model.lm_head
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

        # 로그 파일 생성
        self.f = open(f'logit_lens_test_{model.args.model_name}.txt', 'w')


    @staticmethod
    def project_into_vocabluary(vector, E, tokenizer, top_k=20, bottom_k=-1):
        """
        Project a vector into the vocabulary space and return the top_k tokens.
        """
        vector = vector.to(torch.float32).to('cuda')
        E = E.to(torch.float32).to('cuda')
        
        vocab_ranking = E(vector)     # (V,)
        sorted_token_ids = np.argsort(vocab_ranking.detach().cpu().numpy())[::-1]  # Descending order
        
        if bottom_k == -1:
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[:top_k]]
            # logging.debug([(sorted_token_ids[i], sorted_tokens[i], vocab_ranking[sorted_token_ids[i]].item()) for i in range(top_k)])
        else :
            sorted_tokens = [tokenizer.decode(x).strip() for x in sorted_token_ids[-bottom_k:][::-1]]  # Least score to most score
        return sorted_tokens


    def _get_hidden_sentence_embeddings(self, data):
        hidden_sent_embs = []
        for ins in tqdm(data):
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            prompt = ins['question']
            answer = ins['answer']

            if self.ebd == 'mean':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
                hidden_sent_embs.append(hidden_states.mean(1).cpu())   # sentence mean, [32, 4096]
            elif self.ebd == 'last':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0]   # [32, seq_len, 4096]
                hidden_sent_embs.append(hidden_states[:, -1].cpu())   # last token, [32, 4096]
            elif self.ebd == 'mlp_residual':
                _, mlp_residual, _, _, _, _ = self.model.get_activations(image, prompt, answer)   # [32, seq_len, 4096]
                hidden_sent_embs.append(mlp_residual[:, -1].cpu())   # last token, [32, 4096]
            else:
                raise NotImplementedError
        
        hidden_sent_embs = torch.stack(hidden_sent_embs).permute(1, 0, 2)   # [32, N, 4096]
        return hidden_sent_embs


    def _get_difference_matrix(self, pos_data, neg_data):
        
        # ============================================================
        # ★ [수정됨] 현재 분석 중인 문장 쌍(정답 vs 환각)을 로그에 기록
        # ============================================================
        self.f.write("\n" + "="*60 + "\n")
        self.f.write("   Single Data Analysis Check\n")
        self.f.write("="*60 + "\n")
        
        if isinstance(pos_data, list) and len(pos_data) >= 1:
            # 리스트의 첫 번째 요소(현재 분석 대상)만 기록
            truth_sent = neg_data[0].get('answer', 'No Answer Found')
            hallu_sent = pos_data[0].get('answer', 'No Answer Found')
            
            self.f.write(f"[Truthful Sentence]   : {truth_sent}\n")
            self.f.write(f"[Hallucinated Sentence]: {hallu_sent}\n")
            self.f.write("-" * 40 + "\n")
        else:
            self.f.write("Data is not in list format or empty.\n")
        
        self.f.flush() # 파일에 즉시 쓰기
        # ============================================================

        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(pos_data) if isinstance(pos_data, list) else pos_data.permute(1, 0, 2)  # (L, N, D)
        preferred_sent_embs = self._get_hidden_sentence_embeddings(neg_data) if isinstance(neg_data, list) else neg_data.permute(1, 0, 2)  # (L, N, D)

        difference_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2  # (L, N, D)

        logging.info('Difference matrix calculated.')
        del non_preferred_sent_embs

        if self.centering:
            logging.info('Centering: Removing first singular vector from preference matrix.')

            for layer_num in range(difference_matrix.shape[0]):
                d = difference_matrix[layer_num].to(torch.float32)
                pref = deepcopy(preferred_sent_embs[layer_num].to(torch.float32))

                u, s, vt = torch.linalg.svd(pref, full_matrices=False)  # (N, D) -> (N, N), (N,), (N, D)
                projection_vector = vt[0].unsqueeze(dim=-1)  # (D, 1)

                sorted_tokens = self.project_into_vocabluary(projection_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=10)
                self.f.write(f'Layer {layer_num} - mu: {" | ".join([x for x in sorted_tokens])}\n')

                P = projection_vector @ projection_vector.T  # (D, D)
                I = torch.eye(projection_vector.shape[0]).to(pref.device)  # (D, D)

                d = d @ (I - self.alpha * P)  # (N, D) @ (D, D) -> (N, D) alpha
                difference_matrix[layer_num] = d.to(difference_matrix[layer_num].dtype) # d

        return difference_matrix


    def get_edit_features(self, pos_data, neg_data):

        difference_matrix = self._get_difference_matrix(pos_data, neg_data)  # (L, N, D)
        edit_features = {}

        for key in self.model.model.state_dict():
            # 모델별 키 필터링 로직 (MiniGPT4, Qwen, mPLUG 등)
            if self.model.args.model_name == 'MiniGPT4':
                if ('weight' in key and 'mlp' in key and '_format' not in key 
                    and not 'visual_encoder' in key and not 'gate_proj' in key and not 'up_proj' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    edit_features[key] = difference_matrix[layer_num]
            
            elif self.model.args.model_name == 'Qwen_VL_Chat':
                if ('mlp.c_proj.weight' in key and not 'visual' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    edit_features[key] = difference_matrix[layer_num]
            
            elif self.model.args.model_name == 'mPLUG_Owl2':
                if ('weight' in key and 'mlp' in key and not 'vision' in key 
                    and not 'gate_proj' in key and not 'up_proj' in key and not 'visual' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    edit_features[key] = difference_matrix[layer_num]
            
            else: # LLaVA 등
                if ('weight' in key and 'mlp' in key and not 'vision_tower' in key 
                    and not 'gate_proj' in key and not 'up_proj' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])
                    edit_features[key] = difference_matrix[layer_num]
        return edit_features
    
    
    def svd_on_edit_features(self, edit_features):
        '''
        SVD 수행 및 평균 벡터 추출
        '''
        svd = {}
        mean_vectors = {} # ★ 각 레이어의 평균 차이 벡터 저장용

        for key in edit_features:
            M = edit_features[key].to(torch.float32) # (N, D)
            
            # ★ 이 레이어의 '평균 환각 방향'을 저장 (데이터가 1개면 그 데이터 자체가 됨)
            mean_vectors[key] = M.mean(dim=0) 

            if self.centering:
                M = M - M.mean(dim=0, keepdim=True)
                
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False) # Skinny SVD
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
            
        logging.info('SVD of Edit_Features calculated.')
        return svd, mean_vectors


    def find_p_hallu(self, svd, mean_vectors):
        hallu_subspace = {}
        
        self.f.write("\n" + "="*60 + "\n")
        self.f.write("   HalluSpace Analysis: Before vs After (Original HalluEdit)\n")
        self.f.write("="*60 + "\n")

        for key in svd.keys():
            layer_num = int(key.split('.')[self.lm_sep_idx])
            if layer_num not in self.edit_layer_range:
                continue
            
            self.f.write(f'\n--- Layer {layer_num} ({key}) ---\n')
            logging.info(f'Calculating hallu subspace for: {key}')

            singular_vectors = svd[key]['v'] # (D, N)
            hallu_rank_list = np.arange(self.top_k_ranks)

            # --- [1] Before: 편집 전 환각 내용 ---
            mean_vec = mean_vectors[key].to('cuda')
            before_tokens = self.project_into_vocabluary(mean_vec, self.E.cpu(), self.tokenizer, top_k=8)
            
            self.f.write(f"[Before Edit] Mean Hallucination : {' | '.join(before_tokens)}\n")
            self.f.write("-" * 40 + "\n")

            p_hallu = torch.zeros(self.D, self.D)
            for r in hallu_rank_list:
                if r >= singular_vectors.shape[1]: break
                
                singular_vector = singular_vectors[:, r].unsqueeze(dim=1) # (D, 1)
                p_hallu += singular_vector @ singular_vector.T # (D, D)

                # 각 Rank가 타겟팅하는 단어들
                sorted_tokens = self.project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=5)
                self.f.write(f"   > [Target Rank {r}]: {' | '.join(sorted_tokens)}\n")

            # --- [2] After: 편집(영공간 투영) 후 남는 내용 시뮬레이션 ---
            # 공식: Projected = (I - P) @ Mean_Vector
            p_hallu_cuda = p_hallu.to('cuda')
            I = torch.eye(self.D).to('cuda')
            
            projected_mean_vec = (I - p_hallu_cuda) @ mean_vec
            
            after_tokens = self.project_into_vocabluary(projected_mean_vec, self.E.cpu(), self.tokenizer, top_k=8)
            
            self.f.write("-" * 40 + "\n")
            self.f.write(f"[After Edit]  Remaining Context : {' | '.join(after_tokens)}\n")
            self.f.flush()

            hallu_subspace[key] = p_hallu
            
        logging.info('Hallu subspace calculated.')
        return hallu_subspace


    def edit_model(self, hallu_subspace, edit_keys=True, edit_values=True, layer_range=None):
        assert edit_keys or edit_values, 'At least one of edit_keys or edit_values should be True'
        logging.info(f'Editing keys: {edit_keys}, Editing values: {edit_values}.')

        if layer_range is None:
            layer_range = np.arange(self.num_layers)
        logging.info(f'Editing layers: {layer_range}')

        edited_state_dict = self.model.model.state_dict()
        for key in edited_state_dict:
            if key in hallu_subspace:
                layer_num = int(key.split('.')[self.lm_sep_idx])
                if layer_num in layer_range:
                    logging.info(f'Editing: {key}')
                    
                    Null_space = torch.eye(self.D) - hallu_subspace[key]
                    if self.model.args.model_name == 'MiniGPT4':
                        Null_space = Null_space.to(edited_state_dict[key].device).to(self.model.model.llama_model.dtype)
                    else:
                        Null_space = Null_space.to(edited_state_dict[key].device).to(self.model.model.dtype)

                    weight = edited_state_dict[key]
                    weight = weight.T

                    if edit_keys and 'up_proj' in key:
                        modified_weight = Null_space @ weight 
                    elif edit_values and 'down_proj' in key:
                        modified_weight = weight @ Null_space
                    elif 'c_proj' in key: 
                        modified_weight = weight @ Null_space
                    else:
                        continue
                    
                    modified_weight = modified_weight.T
                    edited_state_dict[key] = modified_weight.to('cuda').contiguous()

        self.model.model.load_state_dict(edited_state_dict, assign=True)
        logging.info('Edited model created.')
        return self.model.model


    def setup_for_edits(self, pos_data, neg_data):
        edit_features = self.get_edit_features(pos_data, neg_data)
        svd, mean_vectors = self.svd_on_edit_features(edit_features)
        
        del edit_features
        self.hallu_subspace = self.find_p_hallu(svd, mean_vectors)
        
        del svd, mean_vectors
        torch.cuda.empty_cache()


    def apply_edit_end_to_end(self, pos_data, neg_data, edit_keys=True, edit_values=True, layer_range=None):
        self.setup_for_edits(pos_data, neg_data)
        edited_model = self.edit_model(self.hallu_subspace, edit_keys, edit_values, layer_range)
        torch.cuda.empty_cache()
        return edited_model