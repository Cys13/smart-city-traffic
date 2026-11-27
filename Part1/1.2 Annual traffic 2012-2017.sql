WITH yearly AS (
    SELECT
        strftime('%Y', date_time) AS year,
        COUNT(*) AS hourly_records,
        SUM(CAST(traffic_volume AS INTEGER)) AS total_traffic,
        ROUND(AVG(CAST(traffic_volume AS REAL)), 2) AS avg_hourly_traffic
    FROM "Metro_Interstate_Traffic_Volume"
    WHERE strftime('%Y', date_time) IN ('2012', '2013', '2014', '2015', '2016', '2017')
    GROUP BY year
)
SELECT
    year,
    hourly_records,
    total_traffic,
    total_traffic - LAG(total_traffic) OVER (ORDER BY year) AS "YoY Change (Vehicles)",
    ROUND(
        (total_traffic - LAG(total_traffic) OVER (ORDER BY year)) * 100.0 /
        LAG(total_traffic) OVER (ORDER BY year),
        2
    ) AS "YoY % Change",
    avg_hourly_traffic
FROM yearly
ORDER BY year;