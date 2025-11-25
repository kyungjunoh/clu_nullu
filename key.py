import pickle
import numpy as np
import torch
from sklearn.cluster import KMeans
from numpy.linalg import norm
import os
import sys
from sklearn.preprocessing import normalize

# -------------------------------------------------------------------------
# 1. 데이터 로드 및 차이 벡터 계산 함수 (디버깅 기능 추가됨)
# -------------------------------------------------------------------------
def load_and_compute_diff(pkl_path, layer_to_visualize, feature_key='hidden_states', max_samples=2000):
    if not os.path.exists(pkl_path):
        print(f"🔴 [오류] 파일을 찾을 수 없습니다: {pkl_path}")
        return None

    print(f"📂 데이터 로드 중... {pkl_path}")
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    pos_features = []
    neg_features = []
    
    error_count = 0

    for i, entry in enumerate(data):
        try:
            feature_vector = entry[feature_key][layer_to_visualize].numpy()
            
            if entry['label'] == 0: # 0 = Hallucination
                pos_features.append(feature_vector)
            else: # 1 = Truthful
                neg_features.append(feature_vector)
        except Exception as e:
            if error_count < 5:
                print(f"⚠️ [데이터 처리 에러] 샘플 {i}: {e}")
            error_count += 1
            continue

    # 짝 맞추기 및 샘플 수 조절
    min_len = min(len(pos_features), len(neg_features))
    if min_len == 0:
        print("🔴 [오류] 데이터가 부족합니다.")
        return None
        
    if max_samples:
        min_len = min(min_len, max_samples)
    
    pos_features = pos_features[:min_len]
    neg_features = neg_features[:min_len]

    # ★★★ [수정됨] dtype=np.float32 추가 ★★★
    X_pos = np.array(pos_features, dtype=np.float32)
    X_neg = np.array(neg_features, dtype=np.float32)

    X_diff = X_pos - X_neg
    X_diff = normalize(X_diff, norm='l2')
    
    print(f"✅ 차이 벡터 계산 완료: {X_diff.shape} (Type: {X_diff.dtype})")
    return X_diff

# -------------------------------------------------------------------------
# 2. 분석 함수들
# -------------------------------------------------------------------------
def compute_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def compute_projection_residual(vectors, subspace_vectors):
    # ★★★ [수정됨] 안전하게 float32로 한번 더 변환 ★★★
    vectors = vectors.astype(np.float32)
    subspace_vectors = subspace_vectors.astype(np.float32)

    # 1. 부분공간의 기저(Basis) 찾기 (SVD)
    # 이제 float32이므로 에러가 나지 않습니다.
    U, S, Vt = np.linalg.svd(subspace_vectors.T, full_matrices=False)
    
    k = 4  # 상위 k개 방향
    Vk = U[:, :k] 
    
    # 2. 투영 행렬 P = V V^T
    P = Vk @ Vk.T
    
    # 3. 영공간 투영 (I - P)
    I = np.eye(P.shape[0], dtype=np.float32)
    Null_P = I - P
    
    # 4. 투영 후 크기 비율 계산
    original_norms = np.linalg.norm(vectors, axis=1).mean()
    projected_vectors = vectors @ Null_P
    projected_norms = np.linalg.norm(projected_vectors, axis=1).mean()
    
    ratio = projected_norms / original_norms
    return ratio

# -------------------------------------------------------------------------
# 3. 메인 실행 함수
# -------------------------------------------------------------------------
def main():
    # 파일 경로 확인 필수!
    PKL_FILE_PATH = "/data/Nullu/output/LLaVA-7B/lure_train_0_activations.pkl"
    
    # ★ 중요: t-SNE에서 분리가 잘 되었던 'hidden_states'(마지막 토큰)를 사용하는 것을 추천합니다.
    # 논문처럼 'mean'을 쓰려면 'hidden_states_mean'으로 하시면 됩니다.
    KEY = 'hidden_states_mean' 
    LAYER = 31
    
    print("=== 분석 시작 ===")
    X_diff = load_and_compute_diff(PKL_FILE_PATH, LAYER, KEY, max_samples=5000)
    
    if X_diff is None:
        print("❌ 분석을 중단합니다.")
        return

    # K-Means로 그룹 분리 (k=2)
    print(f"🌀 K-Means 클러스터링 (k=2) 수행 중...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_diff)
    labels = kmeans.labels_
    
    group_A = X_diff[labels == 0]
    group_B = X_diff[labels == 1]
    
    print(f"   - Group A 샘플 수: {len(group_A)}")
    print(f"   - Group B 샘플 수: {len(group_B)}")
    
    # --- 실험 1: 코사인 유사도 ---
    mean_A = np.mean(group_A, axis=0)
    mean_B = np.mean(group_B, axis=0)
    
    similarity = compute_cosine_similarity(mean_A, mean_B)
    print("\n" + "=" * 40)
    print(f"🧪 [실험 1] 두 그룹 평균 벡터의 코사인 유사도")
    print("=" * 40)
    print(f"▶ 결과: {similarity:.4f}")
    print(f"   (해석: 0에 가까우면 '직교(다른 방향)', 1에 가까우면 '같은 방향')")
    
    # --- 실험 2: 교차 투영 (Cross-Projection) ---
    remaining_ratio_B_by_A = compute_projection_residual(group_B, group_A)
    remaining_ratio_A_by_B = compute_projection_residual(group_A, group_B)
    
    print("\n" + "=" * 40)
    print("🧪 [실험 2] 교차 투영 테스트 (Cross-Projection)")
    print("=" * 40)
    print(f"▶ A의 방패로 B를 막았을 때, B가 살아남은 비율: {remaining_ratio_B_by_A * 100:.2f}%")
    print(f"▶ B의 방패로 A를 막았을 때, A가 살아남은 비율: {remaining_ratio_A_by_B * 100:.2f}%")
    print("   (해석: 100%에 가까울수록 서로 전혀 막지 못하는 '독립적인 방향'임)")

# ★★★ 이 부분이 빠져 있어서 실행이 안 되었던 것입니다! ★★★
if __name__ == "__main__":
    main()