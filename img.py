import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from collections import defaultdict

def analyze_hallucination_source(pkl_path):
    """
    저장된 pkl 파일을 로드하여 Image vs Text 토큰의 환각 강도(Norm)를 비교 분석합니다.
    """
    if not os.path.exists(pkl_path):
        print(f"오류: 파일이 존재하지 않습니다 - {pkl_path}")
        return

    print(f"▶ 데이터 로딩 중: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 1. 데이터 쌍(Pair) 매칭
    # 같은 이미지에 대해 환각(Label 0)과 진실(Label 1)이 쌍으로 존재해야 차이를 구할 수 있습니다.
    # 구조: paired_data[image_id] = {0: hallu_entry, 1: truth_entry}
    paired_data = defaultdict(dict)
    
    for entry in data:
        img_id = entry['image']
        label = entry['label']
        paired_data[img_id][label] = entry

    # 분석 결과를 저장할 리스트
    # (Batch_Size, Num_Layers) 형태가 됨
    img_diff_norms = []
    text_diff_norms = []
    
    valid_pairs = 0
    
    print("▶ 차이 벡터(Diff) 및 Norm 계산 중...")
    
    for img_id, pair in paired_data.items():
        # 환각(0)과 진실(1) 데이터가 모두 있어야 함
        if 0 not in pair or 1 not in pair:
            continue
            
        hallu = pair[0]
        truth = pair[1]
        
        # Tensor 가져오기 (CPU로 이동)
        # Shape: (Num_Layers, Hidden_Dim) 예: (32, 4096)
        h_img = hallu['hidden_states_img'].float()
        t_img = truth['hidden_states_img'].float()
        
        h_text = hallu['hidden_states_text'].float()
        t_text = truth['hidden_states_text'].float()
        
        # --- ★ 핵심 계산: 차이 벡터의 크기(Norm) 계산 ★ ---
        # 1. 이미지 영역 차이
        diff_img = h_img - t_img
        # dim=1 (Hidden Dimension)에 대해 Norm을 구함 -> 결과: (Num_Layers,)
        norm_img = torch.norm(diff_img, p=2, dim=1).numpy()
        
        # 2. 텍스트 영역 차이
        diff_text = h_text - t_text
        norm_text = torch.norm(diff_text, p=2, dim=1).numpy()
        
        img_diff_norms.append(norm_img)
        text_diff_norms.append(norm_text)
        
        valid_pairs += 1

    if valid_pairs == 0:
        print("오류: 매칭되는 데이터 쌍(Hallucination vs Truth)을 찾을 수 없습니다.")
        return

    # Numpy 배열 변환: (Samples, Layers)
    img_diff_norms = np.array(img_diff_norms)
    text_diff_norms = np.array(text_diff_norms)
    
    print(f"▶ 분석 완료: 총 {valid_pairs}개의 쌍을 분석했습니다.")
    
    # 시각화 결과 저장 폴더
    save_dir = "visual/source_analysis"
    os.makedirs(save_dir, exist_ok=True)

    # --- 분석 1: 레이어별 평균 Norm 비교 (Line Plot) ---
    plot_layer_comparison(img_diff_norms, text_diff_norms, save_dir)
    
    # --- 분석 2: 특정 레이어(Early, Middle, Late)에서의 분포 (Scatter Plot) ---
    # LLaVA-7B 기준 (총 32레이어) -> 10(초반), 16(중반-환각형성), 31(후반)
    target_layers = [10, 16, 31] 
    plot_scatter_distribution(img_diff_norms, text_diff_norms, target_layers, save_dir)


def plot_layer_comparison(img_norms, text_norms, save_dir):
    """
    모든 레이어에 대해 Image Norm 평균과 Text Norm 평균을 꺾은선 그래프로 비교
    """
    # (Samples, Layers) -> (Layers,) 평균 계산
    avg_img = np.mean(img_norms, axis=0)
    avg_text = np.mean(text_norms, axis=0)
    layers = np.arange(len(avg_img))
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, avg_img, label='Image Token Diff Norm (Visual Uncertainty)', color='blue', marker='o')
    plt.plot(layers, avg_text, label='Text Token Diff Norm (LLM Prior)', color='red', marker='s')
    
    plt.title("Layer-wise Hallucination Source Analysis: Image vs Text")
    plt.xlabel("Layer Index")
    plt.ylabel("Average L2 Norm of Difference Vector")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "layer_wise_norm_comparison.png")
    plt.savefig(save_path)
    print(f"   [저장 완료] 레이어별 비교 그래프: {save_path}")
    plt.close()


def plot_scatter_distribution(img_norms, text_norms, target_layers, save_dir):
    """
    특정 레이어에서 샘플별 분포를 산점도로 시각화
    """
    for layer in target_layers:
        # 해당 레이어의 데이터 추출
        # img_norms shape: (Samples, Layers)
        if layer >= img_norms.shape[1]: continue

        vals_img = img_norms[:, layer]
        vals_text = text_norms[:, layer]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(vals_img, vals_text, alpha=0.5, color='purple', s=10)
        
        # y=x 기준선 (대각선)
        max_val = max(vals_img.max(), vals_text.max())
        plt.plot([0, max_val], [0, max_val], 'k--', label='Equal Impact (y=x)')
        
        plt.title(f"Source Distribution at Layer {layer}")
        plt.xlabel("Image Token Diff Norm (Visual)")
        plt.ylabel("Text Token Diff Norm (Language)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 영역 해석 텍스트 추가
        plt.text(max_val*0.1, max_val*0.9, "Language Dominant\n(LLM Prior)", 
                 color='red', ha='left', va='top', fontweight='bold')
        plt.text(max_val*0.9, max_val*0.1, "Visual Dominant\n(Visual Uncertainty)", 
                 color='blue', ha='right', va='bottom', fontweight='bold')

        save_path = os.path.join(save_dir, f"scatter_layer_{layer}.png")
        plt.savefig(save_path)
        print(f"   [저장 완료] Layer {layer} 산점도: {save_path}")
        plt.close()

def main():
    # ★ 중요: 이전 단계에서 생성한 pkl 파일 경로를 정확히 입력하세요 ★
    # 예: output/LLaVA-7B/lure_train_first100_0_activations.pkl
    PKL_PATH = "/data/Nullu/output/LLaVA-7B/lure_train_0_activations.pkl" 
    
    analyze_hallucination_source(PKL_PATH)

if __name__ == "__main__":
    main()