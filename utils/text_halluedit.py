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

        # --- ★ [수정 1] 모델별 이미지 토큰 길이 설정 ★ ---
        # 모델에 따라 이미지 토큰 개수가 다릅니다. (LLaVA: 576, MiniGPT4: 32 등)
        # 정확한 값은 사용하는 모델 설정에 맞춰야 합니다.
        if 'MiniGPT4' in model.args.model_name:
            self.image_token_len = 32
        elif 'LLaVA' in model.args.model_name:
            self.image_token_len = 576
        elif 'mPLUG' in model.args.model_name:
            self.image_token_len = 65 # mPLUG-Owl2 예시 (상황에 따라 다를 수 있음)
        else:
            self.image_token_len = 576 # Default (LLaVA-1.5 standard)
        
        print(f"Text-Only Mode Enabled. Filtering out first {self.image_token_len} tokens.")

        model_config = getattr(model, 'model', None) and getattr(model.model, 'config', None)

        if model_config: 
            model_type = getattr(model_config, 'model_type', None)
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

        self.f = open(f'logit_lens_test_{model.args.model_name}_text_only.txt', 'w')


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
        ★ [핵심 수정] 이미지 토큰을 제외하고 텍스트 토큰만 평균을 냅니다.
        """
        hidden_sent_embs = []
        for ins in tqdm(data):
            image = cv2.imread(ins['img_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            prompt = ins['question']
            answer = ins['answer']

            # 모델 Forward
            if self.ebd == 'mean':
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                # hidden_states: [Layers, Batch, Seq_Len, Dim] -> [Layers, Seq_Len, Dim] (Batch=1 가정)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0] 
                
                # --- ★ Text-Only Averaging Logic ★ ---
                # 시퀀스 길이가 이미지 토큰보다 긴 경우에만 자릅니다.
                if hidden_states.shape[1] > self.image_token_len:
                    # [:, 576:, :] -> 이미지 토큰(0~575)을 버림
                    text_states = hidden_states[:, self.image_token_len:, :]
                    # 남은 텍스트 토큰들에 대해서만 평균 (Dim=1)
                    hidden_sent_embs.append(text_states.mean(1).cpu())
                else:
                    # 예외 처리: 텍스트 토큰이 없거나 길이가 짧은 경우 (기존 방식대로 전체 평균)
                    hidden_sent_embs.append(hidden_states.mean(1).cpu())

            elif self.ebd == 'last':
                # 'last'는 어차피 마지막 토큰(텍스트 끝)이므로 수정 불필요하지만 확인 차원
                outputs, _ = self.model._basic_forward(image, prompt, answer, return_dict=True)
                hidden_states = torch.stack(outputs.hidden_states)[1:, 0] 
                hidden_sent_embs.append(hidden_states[:, -1].cpu()) 
                
            elif self.ebd == 'mlp_residual':
                _, mlp_residual, _, _, _, _ = self.model.get_activations(image, prompt, answer) 
                hidden_sent_embs.append(mlp_residual[:, -1].cpu()) 
            else:
                raise NotImplementedError
        
        hidden_sent_embs = torch.stack(hidden_sent_embs).permute(1, 0, 2)   # [Layer, N, Dim]
        return hidden_sent_embs


    def _get_difference_matrix(self, pos_data, neg_data):
        # 위에서 수정된 _get_hidden_sentence_embeddings를 호출하므로
        # 여기서 반환되는 값은 이미 "텍스트 토큰만의 평균"입니다.
        non_preferred_sent_embs = self._get_hidden_sentence_embeddings(pos_data) if isinstance(pos_data, list) else pos_data.permute(1, 0, 2) 
        preferred_sent_embs = self._get_hidden_sentence_embeddings(neg_data) if isinstance(neg_data, list) else neg_data.permute(1, 0, 2) 

        difference_matrix = (preferred_sent_embs - non_preferred_sent_embs) / 2 

        logging.info('Text-Only Difference matrix calculated.')
        del non_preferred_sent_embs

        if self.centering:
            # Centering 로직은 유지
            for layer_num in range(difference_matrix.shape[0]):
                d = difference_matrix[layer_num].to(torch.float32)
                pref = deepcopy(preferred_sent_embs[layer_num].to(torch.float32))

                u, s, vt = torch.linalg.svd(pref, full_matrices=False) 
                projection_vector = vt[0].unsqueeze(dim=-1) 

                P = projection_vector @ projection_vector.T 
                I = torch.eye(projection_vector.shape[0]).to(pref.device) 

                d = d @ (I - self.alpha * P) 
                difference_matrix[layer_num] = d.to(difference_matrix[layer_num].dtype) 

        return difference_matrix


    def get_edit_features(self, pos_data, neg_data):
        # 기존 로직 동일 (difference_matrix만 Text-Only로 변경됨)
        difference_matrix = self._get_difference_matrix(pos_data, neg_data) 
        edit_features = {}

        for key in self.model.model.state_dict():
            # 모델별 키 필터링 로직 (기존 코드 유지)
            if self.model.args.model_name == 'MiniGPT4':
                if ('weight' in key and 'mlp' in key and '_format' not in key and not 'visual_encoder' in key and not 'gate_proj' in key and not 'up_proj' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    edit_features[key] = difference_matrix[layer_num]
            elif self.model.args.model_name == 'Qwen_VL_Chat':
                 if ('mlp.c_proj.weight' in key and not 'visual' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    edit_features[key] = difference_matrix[layer_num]
            elif self.model.args.model_name == 'mPLUG_Owl2':
                if ('weight' in key and 'mlp' in key and not 'vision' in key and not 'gate_proj' in key and not 'up_proj' in key and not 'visual' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx])  
                    edit_features[key] = difference_matrix[layer_num]
            else: # LLaVA, etc.
                if ('weight' in key and 'mlp' in key and not 'vision_tower' in key and not 'gate_proj' in key and not 'up_proj' in key):
                    layer_num = int(key.split('.')[self.lm_sep_idx]) 
                    edit_features[key] = difference_matrix[layer_num]
        return edit_features
    
    
    def svd_on_edit_features(self, edit_features):
        # 기존 로직 동일
        svd = {}
        for key in edit_features:
            logging.debug(f'Calculating SVD for: {key}')
            M = edit_features[key].to(torch.float32) 
            u, s, vt = torch.linalg.svd(M.cuda(), full_matrices=False) 
            svd[key] = {'u': u.cpu(), 's': s.cpu(), 'v': vt.T.cpu()}
        logging.info('SVD of Edit_Features calculated.')
        return svd


    def find_p_hallu(self, svd, rank_range=20):
        # 기존 로직 동일
        hallu_subspace = {}
        for key in svd.keys():
            layer_num = int(key.split('.')[self.lm_sep_idx]) 
            if layer_num not in self.edit_layer_range:
                continue

            singular_vectors = svd[key]['v'] 
            hallu_rank_list = np.arange(self.top_k_ranks) 

            p_hallu = torch.zeros(self.D, self.D)
            for r in hallu_rank_list:
                singular_vector = singular_vectors[:, r].unsqueeze(dim=1) 
                p_hallu += singular_vector @ singular_vector.T 
                
                # 로그용
                sorted_tokens = self.project_into_vocabluary(singular_vector.squeeze(), self.E.cpu(), self.tokenizer, top_k=10)
                self.f.write(f'Layer {layer_num} - rank{r}: {" | ".join([x for x in sorted_tokens])}\n')

            hallu_subspace[key] = p_hallu
        return hallu_subspace


    def edit_model(self, hallu_subspace, edit_keys=True, edit_values=True, layer_range=None):
        # 기존 로직 동일
        assert edit_keys or edit_values, 'At least one of edit_keys or edit_values should be True'
        if layer_range is None:
            layer_range = np.arange(self.num_layers)
            
        edited_state_dict = self.model.model.state_dict()
        for key in edited_state_dict:
            if key in hallu_subspace:
                layer_num = int(key.split('.')[self.lm_sep_idx])
                if layer_num in layer_range:
                    Null_space = torch.eye(self.D) - hallu_subspace[key]
                    
                    # dtype 맞춰주기
                    if self.model.args.model_name == 'MiniGPT4':
                         dtype = self.model.model.llama_model.dtype
                    else:
                         dtype = self.model.model.dtype
                    Null_space = Null_space.to(edited_state_dict[key].device).to(dtype)

                    weight = edited_state_dict[key].T 
                    
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
        logging.info('Edited model created (Text-Only SVD applied).')
        return self.model.model


    def setup_for_edits(self, pos_data, neg_data):
        edit_features = self.get_edit_features(pos_data, neg_data)
        svd = self.svd_on_edit_features(edit_features)
        del edit_features
        self.hallu_subspace = self.find_p_hallu(svd)
        del svd
        torch.cuda.empty_cache()


    def apply_edit_end_to_end(self, pos_data, neg_data, edit_keys=True, edit_values=True, layer_range=None):
        self.setup_for_edits(pos_data, neg_data)
        edited_model = self.edit_model(self.hallu_subspace, edit_keys, edit_values, layer_range)
        torch.cuda.empty_cache()
        return edited_model