"""
Explainable AI (XAI) Engine using TreeSHAP for Delivery Bottlenecks
"""
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class DeliveryDelayExplainer:
    def __init__(self, model: Any, preprocessor: Any, feature_names: List[str]):
        """Initializes TreeExplainer for the trained XGBoost delivery model."""
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)
        
    def compute_shap_values(self, X_transformed: np.ndarray) -> np.ndarray:
        """Computes SHAP delay attributions."""
        return self.explainer.shap_values(X_transformed)
        
    def get_global_delay_drivers(self, shap_values: np.ndarray) -> pd.DataFrame:
        """Calculates global mean absolute delay impact (in minutes)."""
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        df_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_Delay_Impact_Minutes': np.round(mean_abs, 2)
        }).sort_values('Mean_Delay_Impact_Minutes', ascending=False).reset_index(drop=True)
        return df_imp
        
    def explain_single_delivery(self, 
                                X_transformed_row: np.ndarray, 
                                top_k: int = 5) -> Dict[str, Any]:
        """Decomposes a single delivery's delay into minute-by-minute root causes."""
        shap_vals = self.explainer.shap_values(X_transformed_row)[0]
        base_val = float(self.explainer.expected_value)
        
        attr_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Delay_Contribution_Mins': np.round(shap_vals, 2)
        })
        
        delay_pushers = attr_df[attr_df['Delay_Contribution_Mins'] > 0].sort_values(
            'Delay_Contribution_Mins', ascending=False
        ).head(top_k).to_dict(orient='records')
        
        speed_helpers = attr_df[attr_df['Delay_Contribution_Mins'] < 0].sort_values(
            'Delay_Contribution_Mins', ascending=True
        ).head(top_k).to_dict(orient='records')
        
        return {
            'base_eta_minutes': round(base_val, 1),
            'top_delay_causes': delay_pushers,
            'top_speed_factors': speed_helpers
        }
