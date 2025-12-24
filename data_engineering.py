

from src.utils import fetch_query, compute_daily_orders, compute_daily_marketing, join_daily


def build_master_view():
    # Fetch marketing data
    mkt_df = fetch_query("SELECT date, channel, spend, impressions FROM blinkit_marketing_performance")
    # Fetch orders data
    orders_df = fetch_query("SELECT order_id, order_date, order_total FROM blinkit_orders")
    d_orders = compute_daily_orders(orders_df)
    d_mkt    = compute_daily_marketing(mkt_df)
    master   = join_daily(d_orders, d_mkt)
    return master


if __name__ == '__main__':
    master = build_master_view()
    print(master.head())
    print(f"Master view generated with {len(master)} rows")
