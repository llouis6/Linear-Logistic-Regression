"""
UCI ML Repository Dataset Loaders
Fetches datasets directly from UCI repository using ucimlrepo package.
"""
from typing import Tuple, Dict
import pandas as pd
from ucimlrepo import fetch_ucirepo

DataBundle = Tuple[pd.DataFrame, pd.Series, Dict, pd.DataFrame]
# returns: (X, y, metadata, variables)


def load_breast_cancer_uci() -> DataBundle:
    """
    Wisconsin Diagnostic Breast Cancer (UCI id=17).
    
    Returns:
        X: features (DataFrame)
        y: targets as a 1D Series (Malignant/Benign)
        metadata: dict of dataset-level info
        variables: DataFrame describing columns
    """
    ds = fetch_ucirepo(id=17)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze()  # Series
    return X, y, ds.metadata, ds.variables


def load_wine_uci() -> DataBundle:
    """
    Wine (UCI id=109).
    
    Returns:
        X: features (DataFrame)
        y: targets as a 1D Series (class labels)
        metadata: dict of dataset-level info
        variables: DataFrame describing columns
    """
    ds = fetch_ucirepo(id=109)
    X = ds.data.features.copy()
    y = ds.data.targets.squeeze()
    return X, y, ds.metadata, ds.variables

