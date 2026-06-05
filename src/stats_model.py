import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter

class BGNBDModel:
    def __init__(self, penalizer_coef: float = 1e-4):
        self.bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
        self.ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
        self._fitted = False

    def fit(self, rfm: pd.DataFrame) -> "BGNBDModel":
        self.bgf.fit(
            rfm["repeat_frequency"],
            rfm["recency_T"],
            rfm["T_days"],
        )

        ggf_data = rfm[rfm["repeat_frequency"] > 0]
        self.ggf.fit(ggf_data["repeat_frequency"], ggf_data["avg_order_value"])

        self._fitted = True
        return self

    def predict_purchases(self, rfm: pd.DataFrame, t: int = 90) -> np.ndarray:
        self._check_fitted()
        return self.bgf.conditional_expected_number_of_purchases_up_to_time(
            t,
            rfm["repeat_frequency"],
            rfm["recency_T"],
            rfm["T_days"],
        ).values

    def predict_alive_probability(self, rfm: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.bgf.conditional_probability_alive(
            rfm["repeat_frequency"],
            rfm["recency_T"],
            rfm["T_days"],
        ).values

    def predict_clv(self, rfm: pd.DataFrame, months: int = 12, discount_rate: float = 0.01) -> np.ndarray:
        self._check_fitted()
        clv = self.ggf.customer_lifetime_value(
            self.bgf,
            rfm["repeat_frequency"],
            rfm["recency_T"],
            rfm["T_days"],
            rfm["avg_order_value"],
            time=months,
            discount_rate=discount_rate,
        )
        return clv.values

    def predict_expected_duration(self, rfm: pd.DataFrame, t: int = 545) -> np.ndarray:
        self._check_fitted()
        pred_purchases = self.predict_purchases(rfm, t=t)
        
        avg_T = rfm["T_days"].values
        expected_days = np.where(
            pred_purchases > 0,
            avg_T / (pred_purchases + 1),
            avg_T,
        )
        return expected_days.astype(np.float32)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("fit() must be called before prediction.")

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"parameter": list(self.bgf.params_.keys()),
             "value": list(self.bgf.params_.values())}
        )