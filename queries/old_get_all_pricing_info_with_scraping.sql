DROP TABLE IF EXISTS temporary.analysis;

CREATE TABLE temporary.analysis AS 
WITH price_periods AS (
            WITH cte1 AS (
                SELECT
                    cps.id,
                    cps.store_id,
                    role_id,
                    duration_interval,
                    ss.timezone,
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
                    c1.timezone,
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
                c3.timezone,
                c4.period_id,
                c4.period_name
            FROM cte3 c3
            JOIN cte4 c4 USING (period_start, period_end)
            ORDER BY c3.price_schedule_id, c3.period_start
),

-- the customer-side prices within a pay period 
prices_in_period AS (
  SELECT
    pp.store_id,
    pp.timezone,
    pp.period_id,
    p.item_id,
    date_trunc('day', p.date AT TIME ZONE pp.timezone) AS day, -- this is already the strucure of prices
    p.regular_price
  FROM price_periods pp
  JOIN customer.prices p
    ON p.store_id = pp.store_id
   AND p.date >= pp.period_start
   AND p.date <  pp.period_end
),

-- sales data within a pay period 
sales_in_period AS (
  SELECT
    pp.store_id,
    pp.period_id,
    s.item_id,
    date_trunc('day', s.timestamp_utc AT TIME ZONE pp.timezone) AS day, -- because there are multiple sales in a day, we truncate to the day to normalize. but all sales are sitll considered
    s.selling_price,
    s.units_sold,
    s.basket_size
  FROM price_periods pp
  JOIN customer.sales s
    ON s.store_id = pp.store_id
   AND s.timestamp_utc >= pp.period_start
   AND s.timestamp_utc <  pp.period_end
),

-- costs in period as well 
costs_in_period AS (
  SELECT
    pp.store_id,
    pp.period_id,
    c.item_id,
    c.unit_cost,
    date_trunc('day', c.date AT TIME ZONE pp.timezone) AS day -- there should be only one cost per day
  FROM price_periods pp
  JOIN customer.costs c
    ON c.store_id = pp.store_id
   AND c.date >= pp.period_start
   AND c.date <  pp.period_end
),

-- scraping data for all items within a pay period 
scraping_all AS (
  SELECT
    si.bb_id,
    si.merchant_item_id,
    si.item_id,
    pp.store_id,
    pp.period_id,
    date_trunc('day', sp.timestamp_utc AT TIME ZONE pp.timezone) AS day,
    sp.current_price AS sale_price
  FROM scraping.items si
  JOIN scraping.prices sp
    ON sp.item_id = si.item_id
  AND sp.old_price IS NOT NULL
  JOIN price_periods pp
    ON sp.store_id = pp.store_id
   AND sp.timestamp_utc >= pp.period_start
   AND sp.timestamp_utc <  pp.period_end
),

SELECT
  p.store_id,
  p.item_id,
  p.day,

  -- trivial aggregators for non group columns
  MAX(p.timezone)           AS timezone, -- trivial because store_id
  MAX(ci.bb_id)              AS bb_id, -- trivial beecause item_id
  MAX(p.period_id)           AS period_id, -- trivial because day 
  MAX(ci.merchant_id)       AS merchant_id, -- trivial because store_id
  MAX(ci.name)              AS name,
  MAX(sa.sale_price)        AS promo_price,
  MAX(c.unit_cost)          AS unit_cost,

  -- actual aggregations
  SUM(s.units_sold)         AS units_sold,
  SUM(s.basket_size)        AS total_basket_size,
  COUNT(s.basket_size)      AS num_baskets,
  AVG(s.basket_size)        AS avg_basket_size,

  -- weighted average of selling_price by units_sold. there are sometimes diff, selling prices in a day
  CASE
    WHEN SUM(s.units_sold) = 0 THEN NULL
    ELSE
      SUM(s.selling_price * s.units_sold)::numeric
      / SUM(s.units_sold)
  END                        AS avg_selling_price,
  -- do weighted average of regular price by units_sold (JUST IN CASE) there are multiple regular prices in a day
  CASE
    WHEN SUM(p.regular_price * s.units_sold) = 0 THEN NULL
    ELSE
      SUM(p.regular_price * s.units_sold)::numeric
      / SUM(s.units_sold)
  END                        AS regular_price

FROM prices_in_period p

-- all sales, aggregated by day
LEFT JOIN sales_in_period s
  ON p.store_id          = s.store_id
  AND p.period_id         = s.period_id
  AND p.item_id           = s.item_id
  AND p.day               = s.day 

-- all scraping data (sale_price)
LEFT JOIN scraping_all sa
  ON p.store_id          = sa.store_id
  AND p.period_id         = sa.period_id
  AND p.day               = sa.day

-- item lookup for bb_id, merchant_id, name
JOIN customer.items ci
  ON ci.item_id = p.item_id
  AND ci.bb_id = sa.bb_id
  AND ci.customer_item_id       = sa.merchant_item_id

-- cost lookup (may vary by item)
LEFT JOIN costs_in_period c
  ON c.item_id = p.item_id
  and c.store_id = p.store_id
  and c.day = p.day

GROUP BY
  p.store_id,
  p.item_id,
  p.day

ORDER BY
  period_id,
  p.store_id,
  p.day,
  p.item_id;

-- select all to load into df
SELECT * FROM temporary.analysis;