# CDNow LTV 예측 프로젝트
## BG/NBD vs PyTorch DeepSurv 비교 실험

---

## 프로젝트 개요

CDNow 데이터셋(1997–1998)을 활용한 Customer Lifetime Value(LTV) 예측 모델 개발 및 비교 실험.
통계 모델(BG/NBD)과 딥러닝 생존 분석 모델(PyTorch DeepSurv)을 구현하고 RMSE · C-index로 성능을 비교합니다.

---

## 데이터셋

| 항목 | 값 |
|------|-----|
| 출처 | CDNow (온라인 CD 쇼핑몰) |
| 고객 수 | 23,570명 |
| 거래 건수 | 69,659건 |
| 기간 | 1997-01 ~ 1998-06 (18개월) |
| 컬럼 | customer_id, date, quantity, price |

---

## 모델 아키텍처

### 1. BG/NBD (Beta-Geometric / Negative Binomial Distribution)
- **목적**: 미래 t일 내 구매 횟수 예측
- **라이브러리**: `lifetimes`
- **입력**: Recency, Frequency, T (고객 연령)
- **출력**: 기대 구매 횟수 → CLV 환산

### 2. PyTorch DeepSurv
- **목적**: 첫 구매 후 재구매까지 시간 예측 (생존 분석)
- **Loss**: Cox Partial Likelihood (Breslow 근사)
- **구조**: FC(64) → BN → ReLU → Dropout(0.3) → FC(64) → FC(32) → FC(1)
- **입력**: 7개 RFM 파생 피처 (StandardScaler 정규화)
- **출력**: Log-hazard → 예상 재구매 일수

---

## 피처 엔지니어링

| 피처 | 설명 |
|------|------|
| recency_days | 관측 종료일 기준 마지막 구매 경과일 |
| frequency | 총 구매 횟수 |
| monetary | 총 구매 금액 ($) |
| T_days | 첫 구매 ~ 관측 종료 (고객 연령) |
| avg_order_value | 평균 주문 금액 |
| avg_quantity | 평균 구매 수량 |
| purchase_span | 첫~마지막 구매 간격 |

---

## 프로젝트 구조

```
cdnow_ltv/
├── main.py                 # 파이프라인 실행 스크립트
├── requirements.txt
├── README.md
├── data/
│   └── cdnow.csv           # ← 여기에 원본 파일 배치
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # 데이터 로드 & RFM 피처 추출
│   ├── bgnbd.py            # BG/NBD + Gamma-Gamma 모델
│   ├── deepsurv.py         # PyTorch DeepSurv 모델
│   └── evaluation.py       # RMSE, C-index, 시각화
└── outputs/                # 실행 후 생성
    ├── model_comparison.csv
    ├── model_comparison.png
    ├── training_curve.png
    ├── rfm_distribution.png
    ├── predicted_vs_actual.png
    └── deepsurv_weights.pt
```

---

## 실행 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 배치
mkdir data
cp cdnow.csv data/cdnow.csv

# 3. 실행 (기본 설정)
python main.py

# 4. 커스텀 설정
python main.py \
  --epochs 200 \
  --hidden_dims 128 64 32 \
  --dropout 0.2 \
  --lr 5e-4 \
  --batch_size 256
```

---

## 평가 지표

| 지표 | 설명 | 범위 |
|------|------|------|
| **RMSE** | 재구매 예상 일수 오차 (낮을수록 좋음) | 0 ~ ∞ |
| **C-index** | 생존 순위 예측 정확도 (높을수록 좋음) | 0.5(랜덤) ~ 1.0(완벽) |

---

## 핵심 설계 결정

1. **생존 레이블**: `duration` = 첫 구매 ~ 두 번째 구매까지 일수, `event` = 재구매 여부
2. **BG/NBD → Duration 변환**: 예상 구매 횟수를 역산하여 duration 척도로 통일
3. **Early Stopping**: validation loss 기준 patience=15 에포크
4. **C-index 가속**: 벡터화 구현으로 23K 고객 평가 시간 단축
