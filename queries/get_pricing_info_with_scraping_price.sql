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
first_period AS (
  SELECT *
    FROM price_periods
   WHERE period_id = $(price_period)     -- e.g. 0
),

-- the customer-side prices within a pay period 
prices_in_period AS (
  SELECT
    fp.store_id,
    p.item_id,
    DATE_TRUNC('day', p.date) AS day,
    p.regular_price
  FROM first_period fp
  JOIN customer.prices p
    ON p.store_id = fp.store_id
   AND p.date    >= fp.period_start
   AND p.date    <  fp.period_end
  JOIN customer.items ci
    ON ci.item_id    = p.item_id
   AND ci.merchant_id = $(merchant_id)
),
-- sales data within a pay period 
sales_in_period AS (
  SELECT
    fp.store_id,
    s.item_id,
    DATE_TRUNC('day', s.timestamp_utc) AS day, -- because there are multiple sales in a day, we truncate to the day to normalize. but all sales are sitll considered
    s.selling_price,
    s.units_sold
  FROM first_period fp
  JOIN customer.sales s
    ON s.store_id = fp.store_id
   AND s.timestamp_utc >= fp.period_start
   AND s.timestamp_utc <  fp.period_end
  JOIN customer.items ci
    ON ci.item_id    = s.item_id
   AND ci.merchant_id = $(merchant_id)
),
-- cost data within a pay period

-- grabbing scraping data for all items within a ceratin pay period 
scraping_all AS (
  SELECT
    si.merchant_item_id,
    si.item_id,
    sp.store_id,
    DATE_TRUNC('day', sp.timestamp_utc) AS day,
    sp.current_price AS sale_price
  FROM scraping.items si
  JOIN scraping.prices sp
    ON sp.item_id = si.item_id
   AND si.merchant_id = $(merchant_id)  
--    and si.merchant_item_id = '429'
   and sp.old_price is not null
  JOIN first_period fp
    ON sp.store_id = fp.store_id
   AND sp.timestamp_utc >= fp.period_start
   AND sp.timestamp_utc <  fp.period_end
)

SELECT
  ci.bb_id,
  p.store_id,
  p.item_id,
  ci.name,
  p.day,
  p.regular_price,
  sa.sale_price,
  s.selling_price,
  c.unit_cost,
  SUM(s.units_sold) as units_sold
FROM prices_in_period p

-- sales data
LEFT JOIN sales_in_period s
  ON p.store_id = s.store_id
 AND p.item_id  = s.item_id
 AND p.day      = s.day

-- scraping data (sale price)
LEFT JOIN scraping_all sa
  ON  p.store_id                = sa.store_id
  AND p.day                     = sa.day

-- item data 
JOIN customer.items ci
  ON ci.item_id = p.item_id
-- cost data 
AND ci.customer_item_id       = sa.merchant_item_id
LEFT JOIN customer.costs c on c.item_id = p.item_id and c.store_id = p.store_id

GROUP BY p.item_id, p.store_id, p.day, c.unit_cost, ci.bb_id, ci.name, p.regular_price, sa.sale_price, s.selling_price

ORDER BY p.item_id, p.day;