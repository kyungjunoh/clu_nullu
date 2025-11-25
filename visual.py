import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize

def load_and_compute_diff(pkl_path, layer_to_visualize, feature_key='hidden_states', max_samples=2000):
    """
    데이터를 로드하고, Pos(환각) - Neg(진실) = Diff(차이) 벡터를 계산합니다.
    """
    if not os.path.exists(pkl_path):
        print(f"오류: {pkl_path} 파일을 찾을 수 없습니다.")
        return None

    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    pos_features = []
    neg_features = []

    for entry in data:
        try:
            # 텐서를 넘파이로 변환
            feature_vector = entry[feature_key][layer_to_visualize].cpu().numpy()
            if entry['label'] == 0: # 0 = Hallucination
                pos_features.append(feature_vector)
            else: # 1 = Truthful
                neg_features.append(feature_vector)
        except Exception as e:
            continue

    # 짝이 맞는지 확인 및 길이 조정
    min_len = min(len(pos_features), len(neg_features))
    if min_len == 0:
        print("오류: 데이터가 비어있습니다.")
        return None

    if max_samples:
        min_len = min(min_len, max_samples)
    
    pos_features = pos_features[:min_len]
    neg_features = neg_features[:min_len]

    X_pos = np.array(pos_features)
    X_neg = np.array(neg_features)

    # ★ 핵심: 차이 벡터 계산 (Diff = Hallu - Truth)
    X_diff = X_pos - X_neg

    # ★ 코사인 유사도 계산을 위해 미리 L2 정규화 수행 (필수)
    # 정규화된 벡터끼리의 내적(Dot Product)은 코사인 유사도와 같습니다.
    X_diff = normalize(X_diff, norm='l2')
    
    print(f"Layer {layer_to_visualize}: 차이 벡터 계산 완료 {X_diff.shape}")
    return X_diff

def plot_cosine_similarity_histogram(X_diff, LAYER):
    """
    모든 차이 벡터 쌍 간의 코사인 유사도를 계산하고 히스토그램을 그립니다.
    """
    print("1. 모든 벡터 쌍 간의 코사인 유사도 계산 중...")
    
    # X_diff가 이미 정규화되어 있으므로, 행렬 곱(Dot Product)만으로 코사인 유사도 행렬을 얻습니다.
    # (N, D) @ (D, N) -> (N, N)
    similarity_matrix = np.dot(X_diff, X_diff.T)
    
    # 대각선 성분(자기 자신과의 유사도=1)과 중복된 하삼각 행렬을 제외하고
    # 상삼각 행렬(Upper Triangle)의 값만 추출합니다.
    upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_tri_indices]

    # 통계치 계산
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    print(f"2. 히스토그램 그리는 중... (평균 유사도: {mean_sim:.4f}, 표준편차: {std_sim:.4f})")

    # 시각화
    plt.figure(figsize=(10, 6))
    
    # 히스토그램
    plt.hist(similarities, bins=100, range=(-1.0, 1.0), color='skyblue', edgecolor='black', alpha=0.7)
    
    # 평균선 표시
    plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sim:.2f}')
    
    plt.title(f"Histogram of Pairwise Cosine Similarities (Layer {LAYER})\n(Diff Vectors: Hallucination - Truth)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency (Count)")
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # 저장 경로 설정
    save_dir = "visual/cosine_hist"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/layer_{LAYER}_cosine_hist.png"
    
    plt.savefig(save_path)
    plt.close() # 메모리 해제
    
    print(f"결과 저장 완료: {save_path}")
    print("-" * 50)

def main():
    # 파일 경로 수정
    PKL_FILE_PATH = "/data/Nullu/output/LLaVA-7B/lure_train_0_activations.pkl" 
    
    # 시각화 결과 저장 폴더 생성
    os.makedirs("visual", exist_ok=True)
    
    # Layer 16부터 31까지 순회
    for i in range(16, 32):
        LAYER = i
        # 'hidden_states'는 [Layer, Batch, Seq, Dim] 등의 구조일 수 있으므로 로드 함수에서 주의
        KEY = 'hidden_states' 

        X_diff = load_and_compute_diff(PKL_FILE_PATH, LAYER, KEY, max_samples=2000) # 샘플 너무 많으면 계산 오래 걸림 (N^2)
        
        if X_diff is not None:
            plot_cosine_similarity_histogram(X_diff, LAYER=LAYER)

if __name__ == "__main__":
    main()