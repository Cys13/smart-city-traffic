WITH holiday_dates AS (
    SELECT DISTINCT
        strftime('%Y', date_time) AS year,
        holiday,
        date(date_time) AS holiday_date
    FROM "Metro_Interstate_Traffic_Volume"
    WHERE strftime('%Y', date_time) IN ('2015', '2016', '2017')
      AND holiday IN ('New Years Day', 'Labor Day')
),
tagged_stats AS (
    SELECT
        strftime('%Y', date_time) AS year,
        holiday,
        COUNT(*) AS tagged_holiday_records,
        ROUND(AVG(CAST(temp AS REAL)), 2) AS tagged_temp_k
    FROM "Metro_Interstate_Traffic_Volume"
    WHERE strftime('%Y', date_time) IN ('2015', '2016', '2017')
      AND holiday IN ('New Years Day', 'Labor Day')
    GROUP BY year, holiday
),
fullday_stats AS (
    SELECT
        h.year,
        h.holiday,
        ROUND(AVG(CAST(t.temp AS REAL)), 2) AS full_day_avg_temp_k,
        ROUND(AVG(CAST(t.temp AS REAL)) - 273.15, 2) AS full_day_avg_temp_c,
        ROUND(AVG(CAST(t.traffic_volume AS REAL)), 2) AS full_day_avg_traffic
    FROM holiday_dates h
    JOIN "Metro_Interstate_Traffic_Volume" t
      ON date(t.date_time) = h.holiday_date
    GROUP BY h.year, h.holiday
)
SELECT
    f.holiday,
    f.year,
    COALESCE(s.tagged_holiday_records, 0) AS "Tagged Holiday Records",
    s.tagged_temp_k AS "Tagged Temp (K)",
    f.full_day_avg_temp_k || ' K / ' || f.full_day_avg_temp_c || ' °C' AS "Full-Day Avg Temp (K / °C)",
    f.full_day_avg_traffic AS "Full-Day Avg Traffic"
FROM fullday_stats f
LEFT JOIN tagged_stats s
  ON f.year = s.year AND f.holiday = s.holiday
ORDER BY f.holiday, f.year;