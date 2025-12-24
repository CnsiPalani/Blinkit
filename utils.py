

import mysql.connector
import pandas as pd
import numpy as np
import os

def get_db_connection():
    """
    Establishes and returns a MySQL database connection using environment variables.
    """
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=os.getenv("DB_NAME", "blinkit"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "password")
    )
    return conn

def fetch_query(query, params=None):
    """
    Executes a SQL query and returns the result as a pandas DataFrame.
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()
    return df

def safe_parse_datetime(series):
    return pd.to_datetime(series, errors='coerce')

def compute_daily_orders(orders_df):
    orders_df['order_day'] = safe_parse_datetime(orders_df['order_date']).dt.date
    return (orders_df.groupby('order_day')
            .agg(orders_count=('order_total', 'size'),
                 total_revenue=('order_total', 'sum'))
            .reset_index())

def compute_daily_marketing(mkt_df):
    mkt_df['spend_day'] = safe_parse_datetime(mkt_df['date']).dt.date
    return (mkt_df.groupby('spend_day')
            .agg(total_spend=('spend', 'sum'),
                 total_impressions=('impressions', 'sum'),
                 channels=('channel', lambda x: sorted(set(x))))
            .reset_index())

def join_daily(d_orders, d_marketing):
    d_orders = d_orders.rename(columns={'order_day': 'day'})
    d_marketing = d_marketing.rename(columns={'spend_day': 'day'})
    joined = pd.merge(d_orders, d_marketing, on='day', how='outer')
    joined['orders_count'] = joined['orders_count'].fillna(0).astype(int)
    joined['total_revenue'] = joined['total_revenue'].fillna(0.0)
    joined['total_spend'] = joined['total_spend'].fillna(0.0)
    joined['total_impressions'] = joined['total_impressions'].fillna(0)
    joined['channels'] = joined['channels'].apply(lambda x: x if isinstance(x, list) else [])
    joined['roas'] = np.where(joined['total_spend'] > 0,
                              joined['total_revenue'] / joined['total_spend'],
                              np.nan)
    return joined.sort_values('day')

def derive_is_late(orders_df, threshold_min=10):
    # requires promised_time & actual_time; else returns NaN target
    if not {'promised_time', 'actual_time'}.issubset(orders_df.columns):
        orders_df['is_late'] = np.nan
        return orders_df
    promised = safe_parse_datetime(orders_df['promised_time'])
    actual   = safe_parse_datetime(orders_df['actual_time'])
    delta_min = (actual - promised).dt.total_seconds() / 60.0
    orders_df['is_late'] = (delta_min > threshold_min).astype(int)
    return orders_df

def feature_engineer(orders_df):
    dt = safe_parse_datetime(orders_df['order_date'])
    orders_df['hour_of_day'] = dt.dt.hour
    orders_df['day_of_week'] = dt.dt.dayofweek  # Monday=0
    if 'region' not in orders_df.columns:
        orders_df['region'] = 'Unknown'
    return orders_df
