
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
            ),
                -- generates all periods for each store
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
                -- generates the end date of each period 
            cte3 AS (
                SELECT
                    *,
                    period_start + duration_interval AS period_end
                FROM cte2
            ),
            -- generates period id and period name for each period 
            cte4 AS (
                SELECT
                    DISTINCT period_start, period_end,
                    DENSE_RANK() OVER (ORDER BY period_start) - 1 AS period_id,
                    'P' || TO_CHAR(period_start, 'MMDDYY') AS period_name

                FROM cte3
            )
            -- final table with all periods for each store 
            SELECT
                c3.price_schedule_id,
                c3.store_id,
                c3.role_id,
                (c3.period_start AT TIME ZONE c3.timezone)::timestamp AS period_start,
                (c3.period_end AT TIME ZONE c3.timezone)::timestamp AS period_end,
                c3.timezone,
                c4.period_id,
                c4.period_name
            FROM cte3 c3
            JOIN cte4 c4 USING (period_start, period_end)
            ORDER BY c3.price_schedule_id, period_start
),

-- the customer-side prices within a pay period 
prices_in_period AS (
  SELECT
    pp.store_id,
    pp.timezone,
    pp.period_id,
    p.item_id,
    date_trunc('day', p.timestamp_local) AS day, -- this is already the strucure of prices, but convert to date
    p.regular_price
  FROM price_periods pp
  JOIN customer.prices p
    ON p.store_id = pp.store_id
   AND p.timestamp_local >= pp.period_start 
   AND p.timestamp_local <  pp.period_end
   where pp.store_id in (1235, 1173)
   and item_id = 8225846
),

-- sales data within a pay period 
sales_in_period AS (
  SELECT
    pp.store_id,
    pp.period_id,
    s.item_id,
    date_trunc('day', s.timestamp_local) AS day, -- because there are multiple sales in a day, we truncate to the day to normalize. but all sales are sitll considered
    s.selling_price,
    s.units_sold,
    s.basket_size
  FROM price_periods pp
  JOIN customer.sales s
    ON s.store_id = pp.store_id
    and s.item_id = 8225846
   AND s.timestamp_local >= pp.period_start
   AND s.timestamp_local <  pp.period_end
   where pp.store_id in (1235, 1173)
),

-- costs in period as well 
costs_in_period AS (
  SELECT
    pp.store_id,
    pp.period_id,
    c.item_id,
    c.unit_cost,
    date_trunc('day', c.timestamp_local) AS day -- there should be only one cost per day
  FROM price_periods pp
  JOIN customer.costs c
    ON c.store_id = pp.store_id
    and c.store_id in (1235, 1173)
    and c.item_id = 8225846
   AND c.timestamp_local >= pp.period_start
   AND c.timestamp_local <  pp.period_end
),

-- scraping data for all items within a pay period 
scraping_all AS (
  SELECT
    ci.bb_id,
    ci.customer_item_id,
    ci.item_id,
    pp.store_id,
    pp.period_id,
    date_trunc('day', sp.timestamp_utc AT TIME ZONE pp.timezone) AS day,
    -- regular price from scraipng if oldprice is null
    CASE
      WHEN sp.old_price IS NULL THEN sp.current_price
      ELSE NULL
    END AS scraping_regular_price,

    -- promo pricing from scraping if old price is not null 
    CASE
      WHEN sp.old_price IS NOT NULL THEN sp.current_price
      ELSE NULL
    END AS promo_price
  FROM scraping.items si
  JOIN scraping.prices sp
    ON sp.item_id = si.item_id
    and sp.bb_id = 3437
  JOIN price_periods pp
    ON sp.store_id = pp.store_id
    and sp.store_id in (1235, 1173)
   AND sp.timestamp_utc at time zone pp.timezone >= pp.period_start
   AND sp.timestamp_utc at time zone pp.timezone <  pp.period_end
   JOIN customer.items ci
      ON ci.bb_id             = 3437
     AND ci.customer_item_id  = si.merchant_item_id
),

-- all days where a store id / item-id / day combination exist in a table
all_event_days AS (
    SELECT store_id, item_id, period_id, day FROM prices_in_period where item_id = 8225846
    UNION
    SELECT store_id, item_id, period_id, day FROM sales_in_period where item_id = 8225846
    UNION
    SELECT store_id, item_id, period_id, day FROM costs_in_period where item_id = 8225846
    UNION
    SELECT store_id, item_id, period_id, day FROM scraping_all where bb_id = 3437
  )

SELECT
      ed.store_id,
      ed.item_id,
      ed.day,

      -- non‐group columns (trivial per group keys)
      MAX(pp.timezone)      AS timezone,
      MAX(ed.period_id)     AS period_id,
      MAX(ci.bb_id)         AS bb_id,
      MAX(ci.merchant_id)   AS merchant_id,
      MAX(ci.name)          AS name,

      -- spot values
      MAX(pp.period_start) AS period_start_local,
      MAX(pp.period_end) AS period_end_local,
      MAX(sa.promo_price)   AS promo_price,
      MAX(sa.scraping_regular_price) AS scraping_regular_price,
      MAX(co.unit_cost)     AS unit_cost,

      -- your existing aggregations
      SUM(s.units_sold)     AS units_sold,
      SUM(s.basket_size)    AS total_basket_size,
      COUNT(s.basket_size)  AS num_baskets,
      AVG(s.basket_size)    AS avg_basket_size,

      -- weighted avg selling_price
      CASE
        WHEN SUM(s.units_sold) FILTER (WHERE s.selling_price IS NOT NULL) = 0
            THEN NULL
        ELSE
            SUM(s.selling_price * s.units_sold)
            FILTER (WHERE s.selling_price IS NOT NULL)
            / SUM(s.units_sold)
            FILTER (WHERE s.selling_price IS NOT NULL)
        END AS avg_selling_price,

      -- weighted avg regular_price by units_sold. this is just to safeguard against variations in regular price, which should not be the case
      CASE
        WHEN SUM(s.units_sold) FILTER (WHERE pi.regular_price IS NOT NULL) = 0
            THEN NULL
        ELSE
            SUM(pi.regular_price * s.units_sold)
            FILTER (WHERE pi.regular_price IS NOT NULL)
            / SUM(s.units_sold)
            FILTER (WHERE pi.regular_price IS NOT NULL)
        END AS regular_price

    FROM all_event_days ed
    JOIN price_periods pp
      ON pp.store_id   = ed.store_id
     AND pp.period_id  = ed.period_id

    LEFT JOIN prices_in_period pi
      ON pi.store_id   = ed.store_id
     AND pi.item_id    = ed.item_id
     AND pi.period_id  = ed.period_id
     AND pi.day        = ed.day

    LEFT JOIN sales_in_period s
      ON s.store_id    = ed.store_id
     AND s.item_id     = ed.item_id
     AND s.period_id   = ed.period_id
     AND s.day         = ed.day

    LEFT JOIN costs_in_period co
      ON co.store_id   = ed.store_id
     AND co.item_id    = ed.item_id
     AND co.period_id  = ed.period_id
     AND co.day        = ed.day

    LEFT JOIN scraping_all sa
      ON sa.store_id   = ed.store_id
     AND sa.item_id    = ed.item_id
     AND sa.period_id  = ed.period_id
     AND sa.day        = ed.day

    JOIN customer.items ci
      ON ci.item_id    = ed.item_id

    GROUP BY ed.store_id, ed.item_id, ed.day

ORDER BY
  period_id,
  store_id,
  day,
  item_id;
