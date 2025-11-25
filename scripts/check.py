import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import logging
import argparse
from tqdm import tqdm
import numpy as np

# 앞서 작성한 클래스가 있는 파일 임포트 (파일명에 맞게 수정 필요)
from utils.cluster_halluedit import HalluEditCluster
# 편의를 위해 클래스 정의가 포함된 상태라고 가정하거나, 
# 실제 사용 시에는 위 import 문을 활성화하세요.

from model import build_model
from dataset import build_dataset

# 로그 설정 (콘솔 출력용)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_logit_lens_verification(args):
    print(f"▶ 검증 시작: 모델 {args.model_name} 로딩 중...")
    model = build_model(args)
    
    print("▶ 데이터셋 구축 중 (Text-Only & Clustering을 위한 데이터)...")
    # 검증용이므로 샘플 수를 적게(예: 100~200개) 설정해도 됩니다.
    pos_data, neg_data = build_dataset(args.dataset, args.split, args.sampling, args.num_samples)
    
    # HalluEditCluster 인스턴스 생성
    # cohesion_threshold=0.6 설정 확인
    editor = HalluEditCluster(
        model, 
        ebd='mean', 
        n_clusters=2, 
        cohesion_threshold=0.6, 
        top_k_ranks=3 # 상위 3개 방향까지 확인
    )

    print("\n▶ [Step 1] Text-Only 데이터 추출 및 클러스터링 수행...")
    # 내부적으로 Clustering -> Filtering -> SVD가 수행됩니다.
    editor.setup_for_edits_clusters(pos_data, neg_data)
    
    print("\n▶ [Step 2] Logit Lens 결과 분석 (HalluSpace Decoding)")
    print("="*80)
    print(f"{'Layer':<6} | {'Cluster':<8} | {'Rank':<5} | {'Top Interpreted Tokens (The Meaning of Hallucination)'}")
    print("="*80)

    # all_hallu_subspaces에는 SVD 결과가 저장되어 있지 않으므로(P 행렬만 저장됨),
    # setup_for_edits_clusters 과정에서 생성된 로그 파일이나
    # SVD 수행 직후에 벡터를 가져오는 로직이 필요합니다.
    # 여기서는 editor 객체의 내부 정보를 활용해 다시 투영하거나, 
    # setup 과정에서 저장된 로그 파일을 읽어와서 깔끔하게 보여주는 방식을 씁니다.
    
    log_file_path = f'logit_lens_final_{args.model_name}.txt'
    
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # 로그 파일 형식: [L 16-C 1] Rank 0: chair | table | ...
            if "Rank" in line:
                # 파싱해서 예쁘게 출력
                try:
                    parts = line.strip().split('] ')
                    meta = parts[0].replace('[', '') # L 16-C 1
                    content = parts[1] # Rank 0: chair | ...
                    
                    layer_info = meta.split('-')[0].replace('L ', '')
                    cluster_info = meta.split('-')[1].replace('C ', '')
                    
                    rank_info = content.split(':')[0].replace('Rank ', '')
                    tokens = content.split(':')[1].strip()
                    
                    print(f"{layer_info:<6} | {cluster_info:<8} | {rank_info:<5} | {tokens}")
                except:
                    print(line.strip())
    else:
        print("⚠️ 로그 파일을 찾을 수 없습니다. 코드가 실행되면서 생성된 파일을 확인하세요.")

    print("="*80)
    print("\n▶ 결과 해석 가이드:")
    print("1. [구체적 사물] (chair, dog, car...) -> '물체(Object) 환각' 유형")
    print("2. [색상/속성] (red, small, old...) -> '속성(Attribute) 환각' 유형")
    print("3. [위치/관계] (on, under, next...) -> '관계(Relation) 환각' 유형")
    print("4. [기능어/문법] (the, is, a, .) -> '단순 문법적 편향' (환각 아닐 수 있음)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="LLaVA-7B")
    parser.add_argument("--model_path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--dataset", default="lure")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=200) # 빠른 검증을 위해 200개만
    parser.add_argument("--sampling", default='first')
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 앞서 정의한 HalluEditCluster 클래스가 필요합니다.
    # 이 스크립트 내에 클래스를 포함하거나 import 해서 사용하세요.
    run_logit_lens_verification(args)