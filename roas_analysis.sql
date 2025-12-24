
-- sql/roas_analysis.sql
-- Purpose: Build a daily master analytical view joining Marketing (daily) and Orders (transactional)
-- Database: PostgreSQL

WITH orders_daily AS (
    SELECT
        CAST(order_date AS DATE) AS order_day,
        COUNT(*) AS orders_count,
        SUM(order_total) AS total_revenue
    FROM blinkit_orders
    GROUP BY CAST(order_date AS DATE)
),
marketing_daily AS (
    SELECT
        CAST(date AS DATE) AS spend_day,
        SUM(spend) AS total_spend,
        SUM(impressions) AS total_impressions,
        ARRAY_AGG(DISTINCT channel) AS channels
    FROM blinkit_marketing_performance
    GROUP BY CAST(date AS DATE)
)
SELECT
    COALESCE(od.order_day, md.spend_day) AS day,
    COALESCE(od.total_revenue, 0) AS total_revenue,
    COALESCE(md.total_spend, 0) AS total_spend,
    COALESCE(md.total_impressions, 0) AS total_impressions,
    COALESCE(od.orders_count, 0) AS orders_count,
    CASE
        WHEN COALESCE(md.total_spend, 0) = 0 THEN NULL           -- avoid infinite ROAS
        ELSE ROUND(COALESCE(od.total_revenue, 0) / NULLIF(md.total_spend, 0), 4)
    END AS roas,
    md.channels
FROM orders_daily od
FULL OUTER JOIN marketing_daily md
    ON od.order_day = md.spend_day
ORDER BY day;
