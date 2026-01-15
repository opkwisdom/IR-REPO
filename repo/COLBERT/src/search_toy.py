import torch

# 보기 편하게 설정
torch.set_printoptions(precision=2, sci_mode=False)

print("=== [1. 데이터 준비] ===")
# 상황: (Batch, Token, Doc) -> Score
# 0. B0, T0, D10 -> 0.5 (Max 경쟁 패배 예정)
# 1. B0, T0, D10 -> 0.9 (Max 경쟁 승리 예정)
# 2. B0, T1, D10 -> 0.3 (승리한 0.9와 Sum 예정)
# 3. B0, T1, D20 -> 0.4 (단독)
# 4. B1, T0, D10 -> 0.7 (배치가 다름, 별도 보존)

scores    = torch.tensor([0.1, 0.9, 0.5, 0.2,  # B0 관련
                          0.8,                 # B1 관련
                          0.1, 0.2, 0.3,       # B2 관련
                          0.5, 0.5])           # B3 관련

# Batch ID (0~3)
batch_ids = torch.tensor([0,   0,   0,   0,
                          1,
                          2,   2,   2,
                          3,   3])

# Token ID (0~4) - 누가 찾았는가?
token_ids = torch.tensor([0,   0,   1,   4,    # B0: T0가 2번, T1, T4
                          0,                   # B1: T0 (B0의 T0와 섞이는지 체크)
                          2,   2,   2,         # B2: T2가 3번 (Max 경쟁)
                          0,   1])             # B3: T0, T1 (서로 다름)

# Doc ID (10, 30, 40)
doc_ids   = torch.tensor([10,  10,  10,  10,
                          10,
                          30,  30,  30,
                          40,  40])

print(f"Scores: {scores}")
print(f"Batch : {batch_ids}")
print(f"Token : {token_ids}")
print(f"Doc   : {doc_ids}")
print("-" * 60)

# -------------------------------------------------------------------------
# Step 1. 64-bit 좌표 압축 (Composite Key)
# (Batch, Token, Doc)을 하나로 묶어서 "유니크한 이벤트" 식별
# -------------------------------------------------------------------------
# 식: (Batch << 48) | (Token << 32) | (Doc)
unique_keys = (batch_ids << 48) | (token_ids << 32) | (doc_ids)

# -------------------------------------------------------------------------
# Step 2. 중복 제거 및 매핑 지도 생성
# -------------------------------------------------------------------------
unique_keys_sorted, inverse_indices = torch.unique(
    unique_keys, sorted=True, return_inverse=True
)

print("=== [2. Unique Key & Inverse Map] ===")
print(f"원본 데이터 개수: {len(scores)}개")
print(f"유니크 그룹 개수: {len(unique_keys_sorted)}개 (5개 -> 4개로 압축됨)")
print(f"매핑 지도(Inverse): {inverse_indices}")
# 예상 지도: [0, 0, 1, 2, 3]
# 해설: 0번째(0.5)와 1번째(0.9) 데이터는 둘 다 '0'번 그룹(B0-T0-D10)으로 가라!

# -------------------------------------------------------------------------
# Step 3. Max Reduction (같은 토큰끼리의 경쟁)
# -------------------------------------------------------------------------
num_unique = unique_keys_sorted.size(0)
# 초기값은 아주 작은 수 (-infinity)
max_scores = torch.full((num_unique,), -1e9)

# ★ 핵심: inverse_indices를 따라가서 Max값만 남김
max_scores.scatter_reduce_(
    0, inverse_indices, scores, reduce="amax", include_self=False
)

print("-" * 60)
print("=== [3. Max Reduction 결과] ===")
print(f"Max Scores: {max_scores}")
# 예상: [0.90, 0.30, 0.40, 0.70]
# - 0번(B0-T0-D10): Max(0.5, 0.9) -> 0.9
# - 1번(B0-T1-D10): Max(0.3)      -> 0.3
# - 2번(B0-T1-D20): Max(0.4)