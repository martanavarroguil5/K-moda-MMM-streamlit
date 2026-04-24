from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelPackage:
    scaler: StandardScaler
    model: ElasticNetCV | ElasticNet
    feature_columns: List[str]
    media_feature_columns: List[str]
    control_columns: List[str]
    media_params: Dict[str, Dict[str, float | int | str]]
    city_dummy_columns: List[str]
    spec_name: str
