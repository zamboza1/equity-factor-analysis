"""
Factor data sources.

This module provides functions to load factor data (Market, Value, Size, Momentum).
Initially uses yfinance to download real factor data.
"""

from typing import Optional
import pandas as pd
from pathlib import Path
from backend.errors import DataError





def get_factor_data(
    start_date: str,
    end_date: str,
    cache = None
) -> pd.DataFrame:
    """
    Fetch real academic Fama-French factors from Ken French Data Library.
    
    Factors:
        - Mkt-RF: Excess Market Return
        - SMB: Small Minus Big (Size)
        - HML: High Minus Low (Value)
        - MOM: Momentum
    """
    import pandas_datareader.data as web
    from datetime import datetime
    
    try:
        # Fama-French 3 Factors (Daily)
        ff3 = web.DataReader(
            'F-F_Research_Data_Factors_Daily', 
            'famafrench', 
            start_date, 
            end_date
        )
        ff3_df = ff3[0] # The first table contains daily returns
        
        # Momentum Factor (Daily)
        mom = web.DataReader(
            'F-F_Momentum_Factor_Daily', 
            'famafrench', 
            start_date, 
            end_date
        )
        mom_df = mom[0]
        
        # Merge factors
        df = pd.merge(ff3_df, mom_df, left_index=True, right_index=True)
        
        # Ken French data is in percent (e.g. 1.0 means 1%), convert to decimal
        df = df / 100.0
        
        # Rename for consistency with the app
        df = df.rename(columns={
            "Mkt-RF": "Market",
            "HML": "Value",
            "SMB": "Size",
            "Mom  ": "Momentum" # Note the double space in Ken French MOM column
        })
        
        # Strip any accidental whitespace in column names
        df.columns = [c.strip() for c in df.columns]
        
        # Fix Momentum column name if strip didn't catch peculiar formatting
        if "Mom" in df.columns:
            df = df.rename(columns={"Mom": "Momentum"})
            
        return df[["Market", "Value", "Size", "Momentum", "RF"]]
        
    except Exception as e:
        raise DataError(f"Failed to fetch Fama-French factors from Ken French Library: {e}")


