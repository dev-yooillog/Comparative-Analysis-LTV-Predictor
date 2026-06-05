import argparse
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.processor import load_cdnow, build_rfm, make_survival_labels, prepare_ml_data
from src.stats_model import BGNBDModel
from src.surv_nn import DeepSurvModel  
from src.metrics import (              
    evaluate_model, 
    plot_comparison, 
    plot_training_curve, 
    plot_rfm_distribution, 
    plot_predicted_vs_actual
)

def parse_args():
    p = argparse.ArgumentParser(description="CDNow LTV 예측 파이프라인")
    p.add_argument("--data_path", default="data/cdnow.csv", help="CDNow CSV 경로")
    p.add_argument("--epochs", type=int, default=100, help="DeepSurv 최대 에포크")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 64, 32])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="outputs", help="결과물 저장 디렉토리")
    p.add_argument("--no_plot", action="store_true", help="그래프 저장만 (화면 표시 안 함)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    print("   CDNow LTV 예측 | BG/NBD vs DeepSurv")

    print("\n 데이터 로드 & RFM 피처 엔지니어링")
    df = load_cdnow(args.data_path)
    obs_end = df["date"].max()

    print(f"  거래 건수  : {len(df):,}")
    print(f"  고객 수    : {df['customer_id'].nunique():,}")
    print(f"  기간       : {df['date'].min().date()} ~ {obs_end.date()}")

    rfm = build_rfm(df, obs_end)
    rfm_full = make_survival_labels(rfm, df, obs_end)

    print(f"  1회 구매자 : {(rfm_full['frequency']==1).sum():,}")
    print(f"  재구매자   : {(rfm_full['frequency']>=2).sum():,}")
    print(f"  재구매 이벤트(event=1): {rfm_full['event'].sum():,} ({rfm_full['event'].mean()*100:.1f}%)")

    print("\n Train/Test Split & 스케일링")
    (
        X_tr, X_te,
        dur_tr, dur_te,
        evt_tr, evt_te,
        scaler, idx_tr, idx_te,
    ) = prepare_ml_data(rfm_full, test_size=args.test_size, random_state=args.seed)

    rfm_tr = rfm_full.iloc[idx_tr].reset_index(drop=True)
    rfm_te = rfm_full.iloc[idx_te].reset_index(drop=True)

    print(f"  Train: {len(X_tr):,}명  |  Test: {len(X_te):,}명")
    print(f"  피처: {X_tr.shape[1]}개")

    print("\n BG/NBD 모델")
    bgnbd = BGNBDModel(penalizer_coef=1e-4)
    bgnbd.fit(rfm_tr)

    pred_bgnbd_te = bgnbd.predict_expected_duration(rfm_te, t=int(obs_end.timestamp() - df["date"].min().timestamp()) // 86400)

    result_bgnbd = evaluate_model(
        "BG/NBD",
        dur_te.astype(np.float32),
        pred_bgnbd_te,
        evt_te,
    )

    print("\n DeepSurv (PyTorch) 모델")
    deepsurv = DeepSurvModel(
        in_features=X_tr.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        lr=args.lr,
    )
    deepsurv.fit(
        X_tr, dur_tr, evt_tr,
        X_te, dur_te, evt_te,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    pred_deepsurv_te = deepsurv.predict_expected_duration(X_te, max_duration=float(rfm_full["duration"].max()))

    result_deepsurv = evaluate_model(
        "DeepSurv",
        dur_te.astype(np.float32),
        pred_deepsurv_te,
        evt_te,
    )

    print("\n결과 비교 & 시각화")
    results = [result_bgnbd, result_deepsurv]

    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.output_dir, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    print("  최종 결과 요약")
    print(results_df.to_string(index=False))

    winner_rmse = results_df.loc[results_df["rmse"].idxmin(), "model"]
    winner_cindex = results_df.loc[results_df["c_index"].idxmax(), "model"]
    print(f"\n  RMSE   우승: {winner_rmse}")
    print(f"  C-index 우승: {winner_cindex}")

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

    plot_comparison(results, save_path=os.path.join(args.output_dir, "model_comparison.png"))
    plot_training_curve(
        deepsurv.train_losses, deepsurv.val_losses,
        save_path=os.path.join(args.output_dir, "training_curve.png")
    )
    plot_rfm_distribution(rfm_full, save_path=os.path.join(args.output_dir, "rfm_distribution.png"))
    plot_predicted_vs_actual(
        dur_te, pred_bgnbd_te, pred_deepsurv_te, evt_te,
        save_path=os.path.join(args.output_dir, "predicted_vs_actual.png")
    )

    model_path = os.path.join(args.output_dir, "deepsurv_weights.pt")
    deepsurv.save(model_path)
    print(f"\n  [저장] DeepSurv 가중치: {model_path}")
    print(f"  [저장] 결과 CSV: {results_path}")
    print("\n파이프라인 완료")

    return results


if __name__ == "__main__":
    main()
