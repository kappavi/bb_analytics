WITH price_periods AS (
        WITH cte1 AS (
            SELECT
                cps.id,
                cps.store_id,
                role_id,
                duration_interval,
                (cps.anchor_datetime AT TIME ZONE ss.timezone) + (
                    floor(EXTRACT(EPOCH FROM (now() - cps.anchor_datetime)) / EXTRACT(EPOCH FROM cps.duration_interval)) * cps.duration_interval
                ) AS latest_refresh_utc,
                FLOOR(EXTRACT(EPOCH FROM '1 year'::interval) / EXTRACT(EPOCH FROM cps.duration_interval)) AS num_intervals_back
            FROM customer.price_schedules cps
            JOIN
                scraping.stores ss
                ON ss.store_id = cps.store_id
            WHERE
                duration_interval IS NOT NULL -- only for fixed price periods
            and
                role_id = 15
        )
             ,
        cte2 AS (
            SELECT
                c1.id AS price_schedule_id,
                c1.store_id,
                c1.role_id,
                c1.duration_interval,
                generate_series(
                    c1.latest_refresh_utc - (c1.num_intervals_back * c1.duration_interval),
                    c1.latest_refresh_utc,
                    c1.duration_interval
                ) AS period_start
            FROM cte1 c1
        ),

        cte3 AS (
            SELECT
                *,
                period_start + duration_interval AS period_end
            FROM cte2
        ),
        cte4 AS (
            SELECT
                DISTINCT period_start, period_end,
                DENSE_RANK() OVER (ORDER BY period_start) - 1 AS period_id,
                'P' || TO_CHAR(period_start, 'MMDDYY') AS period_name

            FROM cte3
        )
        SELECT
            c3.price_schedule_id,
            c3.store_id,
            c3.role_id,
            c3.period_start,
            c3.period_end,
            c4.period_id,
            c4.period_name
        FROM cte3 c3
        JOIN cte4 c4 USING (period_start, period_end)
        ORDER BY c3.price_schedule_id, c3.period_start
),


customer_store AS (
    SELECT
        da.area_name,
        da.area_id,
        das.store_id

    FROM dashboard.areas da
    JOIN dashboard.roles dr
        ON dr.role_id = da.role_id
    JOIN dashboard.areas_stores das
        ON da.area_id = das.area_id
    JOIN
        scraping.stores ss
        ON (
            ss.store_id = das.store_id
            AND ss.merchant_id = dr.merchant_id
        )
    WHERE
        dr.role_name = 'h_mart'
),

all_dates AS (
    SELECT
        generate_series(
            DATE('2024-11-01'),
            current_date,
            interval '1 day'
        )::date AS day,
        store_id,
        area_name
    FROM
        customer_store
),

all_prices AS (
    SELECT
        si.name,
        si.merchant_item_id::int,
        si.bb_id,
        sp.store_id,
        sp.current_price AS scraping_price,
        DATE_TRUNC('day', sp.timestamp_utc) AS day,
        ROW_NUMBER() OVER (
            PARTITION BY sp.store_id, si.merchant_item_id, DATE_TRUNC('day', sp.timestamp_utc)
            ORDER BY sp.timestamp_utc DESC
        ) AS row_num

    FROM
        scraping.items si
    JOIN
        scraping.prices sp
        ON sp.item_id = si.item_id
    JOIN
        customer_store cs ON sp.store_id = cs.store_id
    WHERE
        sp.timestamp_utc > DATE('2024-11-01')
    and merchant_item_id = '32'
),

latest_prices_of_day AS (
    SELECT
        bb_id,
        store_id,
        merchant_item_id,
        name,
        scraping_price,
        day
  FROM
      all_prices
  WHERE
      row_num = 1
  ),

all_sales AS (
    SELECT
        ci.bb_id,
        cp.order_id,
        cp.store_id,
        ci.customer_item_id::int,
        DATE_TRUNC('day', cp.date) AS day,
        MAX(cp.selling_price) AS selling_price,
        SUM(cp.units_sold) AS units_sold
    FROM
        customer.items ci
        JOIN customer.prices cp
            ON cp.item_id = ci.item_id
        JOIN customer_store cs
            ON cp.store_id = cs.store_id
    WHERE
        cp.date > DATE('2024-11-01')
    and customer_item_id = '32'
    GROUP BY
        1,
        2,
        3,
        4,
        5
),

all_dates_with_prices AS (
    SELECT
        ap.merchant_item_id,
        ap.name,
        ap.store_id,
        ap.bb_id,
        ald.day,
        ald.area_name,
        ap.scraping_price::numeric
    FROM
        all_dates ald
        LEFT JOIN latest_prices_of_day ap
        ON ald.day = ap.day
        AND ald.store_id = ap.store_id
),

all_dates_with_sales AS (
    SELECT
        asl.customer_item_id,
        asl.bb_id,
        asl.store_id,
        asl.order_id,
        ald.day,
        asl.selling_price::numeric,
        asl.units_sold::numeric

    FROM
        all_dates ald
        LEFT JOIN all_sales asl
        ON ald.day = asl.day
        AND ald.store_id = asl.store_id
)

SELECT
    pp.period_id,
    pp.period_name,
    als.order_id,
    adwp.area_name,
    adwp.merchant_item_id,
    adwp.name,
    si.url,
    adwp.bb_id,
    adwp.day,
    adwp.scraping_price::numeric,
    als.selling_price::numeric,
    als.units_sold::numeric
FROM
    all_dates_with_sales als
    LEFT JOIN all_dates_with_prices adwp
        ON (
            als.day = adwp.day
            AND adwp.merchant_item_id = als.customer_item_id
            AND als.store_id = adwp.store_id
        )
    JOIN
        scraping.items si ON
        (
            si.merchant_item_id::int = als.customer_item_id
            AND si.merchant_id = 17
        )
    JOIN
        price_periods pp ON
        (
            adwp.day >= pp.period_start
            AND adwp.day < pp.period_end
            AND adwp.store_id = pp.store_id
        )
ORDER BY
    pp.period_id,
    adwp.merchant_item_id,
    adwp.day