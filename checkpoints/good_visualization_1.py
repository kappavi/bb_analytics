import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import os
import re
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

load_dotenv('../../.env')

# Create the engine
def get_engine():
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD') 
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    db_database = os.getenv('DB_DATABASE')
    
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
    
    engine = create_engine(
        connection_string,
        connect_args={
            "sslmode": "require"  # Azure PostgreSQL requires SSL
        }
    )
    return engine

def format_query(query_text, params):
    """
    Format query by replacing $(param_name) with actual values
    Similar to the Node.js formatQuery function
    """
    formatted_query = query_text
    for key, value in params.items():
        # Replace $(key) with the actual value
        pattern = f"\\$\\({key}\\)"
        if isinstance(value, str):
            formatted_query = re.sub(pattern, f"'{value}'", formatted_query)
        else:
            formatted_query = re.sub(pattern, str(value), formatted_query)
    
    return formatted_query

def run_query(query_filename, rawQuery=None, params=None, engine=None):
    """
    Run a SQL query from a file with parameters
    Similar to the Node.js runQuery function
    """
    if engine is None:
        engine = get_engine()
    
    if params is None:
        params = {}
    
    try:
        if rawQuery is None:
            # Read the SQL file
            query_path = os.path.join('queries', query_filename)
            with open(query_path, 'r') as file:
                query_text = file.read()
        else:
            query_text = rawQuery
        
        # Format the query with parameters
        formatted_query = format_query(query_text, params)
        
        # Execute the query and return DataFrame
        df = pd.read_sql(formatted_query, engine)
        return df
        
    except Exception as error:
        print(f"Error while running query: {query_filename} -> {error}")
        print(f"Formatted query:\n{formatted_query}")
        raise error
    

def write_to_table(df, schema_name, table_name, engine=None):
    """
    Write a DataFrame to a table
    """
    if engine is None:
        engine = get_engine()
    df.to_sql(table_name, engine, schema=schema_name, if_exists='replace', index=False)

def aggregate_data_by_period(df):
    df = (
        df
        .groupby(
            ['period_id', 'store_id', 'item_id'],
            as_index=False
        )
        .agg(
            period_start        = ('day',           'min'),
            period_end          = ('day',           'max'),
            avg_active_price    = ('active_price', 'mean'),
            avg_regular_price   = ('regular_price', 'mean'), 
            avg_promo_price      = ('promo_price',    'mean'),
            avg_selling_price   = ('avg_selling_price', 'mean'),
            avg_unit_cost       = ('unit_cost',     'mean'),
            total_units_sold    = ('units_sold',    'sum'),
            avg_basket_size     = ('avg_basket_size','sum'),
            total_num_baskets   = ('num_baskets', 'sum'),
            total_promo_spending = ('promo_spending','sum'),
            total_revenue       = ('revenue', 'sum'),
            total_profit        = ('profit', 'sum'),
            total_promo_spending_selling_price_based = ('promo_spending (selling_price_based)', 'sum'),
            total_revenue_selling_price_based = ('revenue (selling_price_based)', 'sum'),
            total_profit_selling_price_based = ('profit (selling_price_based)', 'sum'),
            # we are not going to aggregate filled from data here because that was row level 
            # non aggregations here
            name = ('name', 'first'),
            bb_id = ('bb_id', 'first'),
            merchant_id = ('merchant_id', 'first'),
            timezone = ('timezone', 'first')

            
        )
    )
    df['weighted_average_promo_spending_ratio'] = df['total_promo_spending'] / df['total_revenue']
    df['weighted_average_promo_spending_ratio_selling_price_based'] = df['total_promo_spending_selling_price_based'] / df['total_revenue_selling_price_based']
    return df



def detect_promotions(df):
    """
    Detect when items are on promotion based on price comparison
    """
    df = df.copy()
    
    # An item is on promotion if it has a promo price that differs from regular price
    df['has_promo_price'] = pd.notna(df['avg_promo_price'])
    df['is_on_promotion'] = (
        df['has_promo_price'] & 
        (abs(df['avg_promo_price'] - df['avg_regular_price']) > 0.01)  # Allow for small rounding differences
    )
    
    # Calculate promotion discount
    df['promo_discount_pct'] = np.where(
        df['is_on_promotion'],
        ((df['avg_regular_price'] - df['avg_promo_price']) / df['avg_regular_price'] * 100),
        0
    )
    
    # Validate active price logic: active_price should equal promo_price when on promotion, regular_price otherwise
    df['expected_active_price'] = np.where(
        df['is_on_promotion'],
        df['avg_promo_price'],
        df['avg_regular_price']
    )
    
    # Flag any discrepancies
    df['price_logic_check'] = abs(df['avg_active_price'] - df['expected_active_price']) < 0.01
    
    return df

# 1. — Load your data — 
# Replace with wherever you get your DataFrame
@st.cache_data
def load_data():
    query = "SELECT * FROM temporary.analysis_cleaned"
    df = run_query(None, rawQuery=query)
    return df  

# Main
st.set_page_config(page_title="Item-Centric Promotion Analysis Dashboard", layout="wide")
df = load_data()

df_analyzed = aggregate_data_by_period(df)
# Add promotion detection
df_analyzed = detect_promotions(df_analyzed)
# order by total_profit descending
df_analyzed = df_analyzed.sort_values('total_profit', ascending=False)

st.title("🎯 Item-Centric Promotion Analysis Dashboard")

# 3. — Sidebar controls —
st.sidebar.header("Filters")

# Store selector (first level filter)
stores = df_analyzed[['store_id', 'merchant_id']].drop_duplicates().sort_values('store_id')
stores['display_name'] = stores['store_id'].astype(str) + " (Merchant: " + stores['merchant_id'].astype(str) + ")"
selected_store = st.sidebar.selectbox(
    "Select Store:",
    options=stores['store_id'].tolist(),
    format_func=lambda sid: stores[stores['store_id'] == sid]['display_name'].iloc[0]
)

# Filter data by selected store
df_store_filtered = df_analyzed[df_analyzed['store_id'] == selected_store]

# Item selector (filtered by store)
items_in_store = df_store_filtered[['item_id','name']].drop_duplicates().set_index('item_id')
selected_item = st.sidebar.selectbox(
    "Select Item:",
    options=items_in_store.index,
    format_func=lambda iid: f"{iid} — {items_in_store.loc[iid,'name']}"
)

# Period selection for comparison
periods = sorted(df_store_filtered['period_id'].unique().tolist())
prev_period = st.sidebar.selectbox("Previous Period:", options=periods, index=0)
curr_period = st.sidebar.selectbox("Current Period:", options=periods, index=len(periods)-1)

# Enhanced Metrics selectors - active_price as primary
st.sidebar.subheader("📊 Price Metrics")
price_metrics = ['avg_active_price', 'avg_regular_price', 'avg_promo_price', 'avg_selling_price', 'avg_unit_cost']
selected_price = st.sidebar.multiselect(
    "Price metrics to plot:", 
    price_metrics, 
    default=['avg_active_price', 'avg_regular_price', 'avg_promo_price'],
    help="Active price is the primary metric - equals promo price when on promotion, regular price otherwise"
)

st.sidebar.subheader("💰 Financial Metrics")
fin_metrics = ['total_revenue','total_profit','total_units_sold','weighted_average_promo_spending_ratio']
selected_fin = st.sidebar.multiselect("Financial metrics to plot:", fin_metrics, default=['total_revenue','total_profit'])

st.sidebar.subheader("🎁 Promotion Filters")
promotion_filter = st.sidebar.radio(
    "Show periods:",
    ["All periods", "Promotion periods only", "Non-promotion periods only"]
)

# Show promotion periods only filter
if promotion_filter == "Promotion periods only":
    promo_periods = df_store_filtered[df_store_filtered['is_on_promotion']]['period_id'].unique()
    periods = [p for p in periods if p in promo_periods]
elif promotion_filter == "Non-promotion periods only":
    non_promo_periods = df_store_filtered[~df_store_filtered['is_on_promotion']]['period_id'].unique()
    periods = [p for p in periods if p in non_promo_periods]

# Show store info with promotion summary
promo_count = len(df_store_filtered[df_store_filtered['is_on_promotion']])
total_count = len(df_store_filtered)
st.info(f"📍 Store {selected_store} | Items: {df_store_filtered['item_id'].nunique()} | Periods: {len(periods)} | Promotions: {promo_count}/{total_count} ({promo_count/total_count*100:.1f}%)")

# 4. — Filter data for the selected item and store —
item_df = df_store_filtered[df_store_filtered['item_id'] == selected_item].sort_values('period_id')

# Check if we have data for the selected item
if item_df.empty:
    st.error(f"No data found for item {selected_item} in store {selected_store}")
    st.stop()

# Apply promotion filter to item data
if promotion_filter == "Promotion periods only":
    item_df_filtered = item_df[item_df['is_on_promotion']]
elif promotion_filter == "Non-promotion periods only":
    item_df_filtered = item_df[~item_df['is_on_promotion']]
else:
    item_df_filtered = item_df

# 5. — Enhanced Time-series plots with promotion indicators —
st.header(f"📈 Analysis for {items_in_store.loc[selected_item,'name']} (Store {selected_store})")

# Show item promotion summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    promo_periods_count = item_df['is_on_promotion'].sum()
    st.metric("Promotion Periods", f"{promo_periods_count}/{len(item_df)}")
with col2:
    avg_discount = item_df[item_df['is_on_promotion']]['promo_discount_pct'].mean()
    st.metric("Avg Discount %", f"{avg_discount:.1f}%" if not pd.isna(avg_discount) else "N/A")
with col3:
    price_range = f"${item_df['avg_active_price'].min():.2f} - ${item_df['avg_active_price'].max():.2f}"
    st.metric("Price Range", price_range)
with col4:
    price_logic_issues = (~item_df['price_logic_check']).sum()
    st.metric("Price Logic Issues", price_logic_issues, delta_color="inverse")

def create_enhanced_chart_with_promotions(data, metrics, title, item_data):
    """Create plotly chart with promotion period highlighting"""
    fig = go.Figure()
    
    # Add background highlighting for promotion periods
    for _, row in item_data.iterrows():
        if row['is_on_promotion']:
            period_start = pd.to_datetime(row['period_start'])
            period_end = pd.to_datetime(row['period_end'])
            
            # Add background rectangle for promotion period
            fig.add_shape(
                type="rect",
                x0=period_start, x1=period_end,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below",
                line_width=0,
            )
    
    # Add lines for each metric
    for i, metric in enumerate(metrics):
        # Different line styles for different price types
        line_style = dict(width=3) if metric == 'avg_active_price' else dict(width=2)
        if metric == 'avg_active_price':
            line_style['color'] = 'blue'
        elif metric == 'avg_promo_price':
            line_style['dash'] = 'dash'
            line_style['color'] = 'red'
        elif metric == 'avg_regular_price':
            line_style['dash'] = 'dot'
            line_style['color'] = 'green'
            
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[metric],
            mode='lines+markers',
            name=metric.replace('avg_', '').replace('_', ' ').title(),
            line=line_style,
            marker=dict(size=8 if metric == 'avg_active_price' else 6)
        ))
    
    # Add period boundary lines
    for _, row in item_data.iterrows():
        period_start = pd.to_datetime(row['period_start'])
        
        # Period start line with promotion indicator
        line_color = "red" if row['is_on_promotion'] else "gray"
        line_width = 3 if row['is_on_promotion'] else 1
        
        fig.add_shape(
            type="line",
            x0=period_start, x1=period_start,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=line_color, dash="dot", width=line_width)
        )
        
        # Annotation with promotion status
        annotation_text = f"P{row['period_id']}"
        if row['is_on_promotion']:
            annotation_text += f" 🎁 (-{row['promo_discount_pct']:.1f}%)"
        
        fig.add_annotation(
            x=period_start,
            y=1.02,
            xref="x", yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=10, color=line_color, weight="bold" if row['is_on_promotion'] else "normal"),
            bgcolor="white" if row['is_on_promotion'] else None,
            bordercolor=line_color if row['is_on_promotion'] else None,
            borderwidth=1 if row['is_on_promotion'] else 0
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    return fig

# Price analysis with promotion highlighting
if selected_price and not item_df_filtered.empty:
    price_ts = item_df_filtered.set_index('period_start')[selected_price]
    fig_price = create_enhanced_chart_with_promotions(
        price_ts, selected_price, 
        "💰 Price Fluctuation Analysis (Red background = Promotion periods)", 
        item_df_filtered
    )
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Price insights
    st.subheader("💡 Price Insights")
    col1, col2 = st.columns(2)
    with col1:
        if 'avg_active_price' in selected_price:
            active_price_volatility = item_df['avg_active_price'].std()
            st.write(f"**Active Price Volatility:** ${active_price_volatility:.2f}")
            
            # Price trend
            if len(item_df) >= 2:
                price_trend = ((item_df['avg_active_price'].iloc[-1] / item_df['avg_active_price'].iloc[0]) - 1) * 100
                trend_direction = "📈" if price_trend > 0 else "📉" if price_trend < 0 else "➡️"
                st.write(f"**Overall Price Trend:** {trend_direction} {price_trend:+.1f}%")
    
    with col2:
        promo_periods = item_df[item_df['is_on_promotion']]
        if not promo_periods.empty:
            avg_promo_discount = promo_periods['promo_discount_pct'].mean()
            max_promo_discount = promo_periods['promo_discount_pct'].max()
            st.write(f"**Average Promotion Discount:** {avg_promo_discount:.1f}%")
            st.write(f"**Maximum Promotion Discount:** {max_promo_discount:.1f}%")

# Financial metrics with promotion context
if selected_fin and not item_df_filtered.empty:
    fin_ts = item_df_filtered.set_index('period_start')[selected_fin]
    fig_fin = create_enhanced_chart_with_promotions(
        fin_ts, selected_fin, 
        "📊 Financial Performance (Red background = Promotion periods)", 
        item_df_filtered
    )
    st.plotly_chart(fig_fin, use_container_width=True)

# 6. — Enhanced Period Comparison with Promotion Context —
st.header("📊 Period-over-Period Analysis")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Metric Comparison & Lift Analysis")
    if prev_period in item_df['period_id'].values and curr_period in item_df['period_id'].values:
        prev_data = item_df[item_df['period_id'] == prev_period].iloc[0]
        curr_data = item_df[item_df['period_id'] == curr_period].iloc[0]
        
        # Enhanced comparison with promotion context
        comparison_metrics = ['avg_active_price', 'avg_regular_price', 'avg_promo_price', 'total_revenue', 
                            'total_profit', 'total_units_sold', 'weighted_average_promo_spending_ratio']
        
        comp_data = []
        for metric in comparison_metrics:
            if metric in prev_data.index and metric in curr_data.index:
                prev_val = prev_data[metric] if not pd.isna(prev_data[metric]) else 0
                curr_val = curr_data[metric] if not pd.isna(curr_data[metric]) else 0
                
                lift = ((curr_val / prev_val) - 1) * 100 if prev_val != 0 else 0
                
                comp_data.append({
                    'Metric': metric.replace('avg_', '').replace('_', ' ').title(),
                    f'Period {prev_period}': prev_val,
                    f'Period {curr_period}': curr_val,
                    'Lift %': lift
                })
        
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True)
        
        # Promotion status comparison
        st.subheader("🎁 Promotion Status")
        prev_promo = prev_data['is_on_promotion']
        curr_promo = curr_data['is_on_promotion']
        
        col_a, col_b = st.columns(2)
        with col_a:
            status_prev = "🎁 On Promotion" if prev_promo else "📦 Regular Price"
            discount_prev = f"(-{prev_data['promo_discount_pct']:.1f}%)" if prev_promo else ""
            st.write(f"**Period {prev_period}:** {status_prev} {discount_prev}")
        
        with col_b:
            status_curr = "🎁 On Promotion" if curr_promo else "📦 Regular Price"
            discount_curr = f"(-{curr_data['promo_discount_pct']:.1f}%)" if curr_promo else ""
            st.write(f"**Period {curr_period}:** {status_curr} {discount_curr}")
        
        # Promotion impact analysis
        if prev_promo != curr_promo:
            if curr_promo and not prev_promo:
                st.success("🎯 **Promotion Started** - Analyze promotion impact on sales metrics!")
            elif not curr_promo and prev_promo:
                st.info("📈 **Promotion Ended** - Check for any residual effects on sales!")
    else:
        st.warning("Selected periods not available for this item")

with col2:
    st.subheader("📊 Visual Lift Analysis")
    if prev_period in item_df['period_id'].values and curr_period in item_df['period_id'].values:
        # Create comparison chart
        key_metrics = ['total_revenue', 'total_profit', 'total_units_sold']
        available_metrics = [m for m in key_metrics if m in comp_df['Metric'].str.lower().str.replace(' ', '_').values]
        
        if available_metrics:
            # Prepare data for bar chart
            chart_data = []
            for _, row in comp_df.iterrows():
                metric_name = row['Metric']
                chart_data.append({
                    'Metric': metric_name,
                    'Period': f'Period {prev_period}',
                    'Value': row[f'Period {prev_period}']
                })
                chart_data.append({
                    'Metric': metric_name,
                    'Period': f'Period {curr_period}',
                    'Value': row[f'Period {curr_period}']
                })
            
            chart_df = pd.DataFrame(chart_data)
            
            fig_comparison = px.bar(
                chart_df, 
                x='Metric', 
                y='Value', 
                color='Period',
                title="Period Comparison",
                barmode='group'
            )
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)

# 7. — Promotion Impact Analysis —
st.header("🎁 Promotion Impact Analysis")

promo_periods = item_df[item_df['is_on_promotion']]
non_promo_periods = item_df[~item_df['is_on_promotion']]

if not promo_periods.empty and not non_promo_periods.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Sales Performance During Promotions")
        promo_avg_revenue = promo_periods['total_revenue'].mean()
        non_promo_avg_revenue = non_promo_periods['total_revenue'].mean()
        revenue_lift = ((promo_avg_revenue / non_promo_avg_revenue) - 1) * 100
        
        st.metric("Average Revenue Lift", f"{revenue_lift:+.1f}%")
        
        promo_avg_units = promo_periods['total_units_sold'].mean()
        non_promo_avg_units = non_promo_periods['total_units_sold'].mean()
        units_lift = ((promo_avg_units / non_promo_avg_units) - 1) * 100
        
        st.metric("Average Units Sold Lift", f"{units_lift:+.1f}%")
    
    with col2:
        st.subheader("💰 Profitability Analysis")
        promo_avg_profit = promo_periods['total_profit'].mean()
        non_promo_avg_profit = non_promo_periods['total_profit'].mean()
        profit_lift = ((promo_avg_profit / non_promo_avg_profit) - 1) * 100
        
        st.metric("Average Profit Lift", f"{profit_lift:+.1f}%")
        
        # ROI of promotions
        avg_promo_spending = promo_periods['total_promo_spending'].mean()
        profit_per_promo_dollar = promo_avg_profit / avg_promo_spending if avg_promo_spending > 0 else 0
        
        st.metric("Profit per Promo $", f"${profit_per_promo_dollar:.2f}")
    
    with col3:
        st.subheader("📊 Promotion Effectiveness")
        # Calculate elasticity approximation
        avg_price_change = promo_periods['promo_discount_pct'].mean()
        elasticity = units_lift / (-avg_price_change) if avg_price_change != 0 else 0
        
        st.metric("Price Elasticity Estimate", f"{elasticity:.2f}")
        
        # Best performing promotion
        if len(promo_periods) > 1:
            best_promo = promo_periods.loc[promo_periods['total_revenue'].idxmax()]
            st.metric("Best Promotion Period", f"Period {best_promo['period_id']}")

    # Detailed promotion performance table
    st.subheader("📋 Detailed Promotion Performance")
    if not promo_periods.empty:
        promo_summary = promo_periods[['period_id', 'period_start', 'promo_discount_pct', 
                                     'total_revenue', 'total_profit', 'total_units_sold']].copy()
        promo_summary['period_start'] = pd.to_datetime(promo_summary['period_start']).dt.strftime('%Y-%m-%d')
        st.dataframe(promo_summary, use_container_width=True)

else:
    if promo_periods.empty:
        st.info("🔍 No promotion periods found for this item in the selected time range.")
    if non_promo_periods.empty:
        st.info("🔍 No non-promotion periods found for this item in the selected time range.")

# 8. — Top Items Analysis (Store-level with promotion context) —
st.header(f"🏪 Store {selected_store} - Promotion Performance Overview")

tab1, tab2, tab3 = st.tabs(["📈 Top Revenue Items", "🎁 Most Promoted Items", "💡 Promotion Effectiveness"])

with tab1:
    st.subheader("Top Revenue Items by Promotion Status")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**During Promotions**")
        top_promo_revenue = df_store_filtered[df_store_filtered['is_on_promotion']].groupby('item_id').agg({
            'name': 'first',
            'total_revenue': 'sum',
            'period_id': 'count'
        }).sort_values('total_revenue', ascending=False).head(10)
        top_promo_revenue.columns = ['Name', 'Total Revenue', 'Promo Periods']
        st.dataframe(top_promo_revenue, use_container_width=True)
    
    with col2:
        st.write("**Regular Pricing**")
        top_regular_revenue = df_store_filtered[~df_store_filtered['is_on_promotion']].groupby('item_id').agg({
            'name': 'first',
            'total_revenue': 'sum',
            'period_id': 'count'
        }).sort_values('total_revenue', ascending=False).head(10)
        top_regular_revenue.columns = ['Name', 'Total Revenue', 'Regular Periods']
        st.dataframe(top_regular_revenue, use_container_width=True)

with tab2:
    st.subheader("Most Frequently Promoted Items")
    promotion_frequency = df_store_filtered.groupby('item_id').agg({
        'name': 'first',
        'is_on_promotion': ['sum', 'count'],
        'promo_discount_pct': 'mean',
        'total_revenue': 'sum'
    }).round(2)
    
    promotion_frequency.columns = ['Name', 'Promo Periods', 'Total Periods', 'Avg Discount %', 'Total Revenue']
    promotion_frequency['Promo Frequency %'] = (promotion_frequency['Promo Periods'] / promotion_frequency['Total Periods'] * 100).round(1)
    promotion_frequency = promotion_frequency.sort_values('Promo Periods', ascending=False).head(15)
    
    st.dataframe(promotion_frequency, use_container_width=True)

with tab3:
    st.subheader("Promotion Effectiveness Analysis")
    
    # Calculate promotion impact for each item
    effectiveness_data = []
    for item_id in df_store_filtered['item_id'].unique():
        item_data = df_store_filtered[df_store_filtered['item_id'] == item_id]
        promo_data = item_data[item_data['is_on_promotion']]
        regular_data = item_data[~item_data['is_on_promotion']]
        
        if not promo_data.empty and not regular_data.empty:
            revenue_lift = ((promo_data['total_revenue'].mean() / regular_data['total_revenue'].mean()) - 1) * 100
            units_lift = ((promo_data['total_units_sold'].mean() / regular_data['total_units_sold'].mean()) - 1) * 100
            avg_discount = promo_data['promo_discount_pct'].mean()
            
            effectiveness_data.append({
                'Item ID': item_id,
                'Name': item_data['name'].iloc[0],
                'Avg Discount %': avg_discount,
                'Revenue Lift %': revenue_lift,
                'Units Lift %': units_lift,
                'Promo Periods': len(promo_data)
            })
    
    if effectiveness_data:
        effectiveness_df = pd.DataFrame(effectiveness_data)
        effectiveness_df = effectiveness_df.sort_values('Revenue Lift %', ascending=False)
        st.dataframe(effectiveness_df, use_container_width=True)
        
        # Visualization of effectiveness
        fig_effectiveness = px.scatter(
            effectiveness_df,
            x='Avg Discount %',
            y='Revenue Lift %',
            size='Promo Periods',
            hover_data=['Name', 'Units Lift %'],
            title="Promotion Effectiveness: Discount vs Revenue Lift",
            labels={'Avg Discount %': 'Average Discount (%)', 'Revenue Lift %': 'Revenue Lift (%)'}
        )
        fig_effectiveness.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_effectiveness, use_container_width=True)

# 9. — Download enhanced data —
enhanced_item_df = item_df.copy()
enhanced_item_df['promotion_status'] = enhanced_item_df['is_on_promotion'].map({True: 'On Promotion', False: 'Regular Price'})

csv = enhanced_item_df.to_csv(index=False)
st.download_button(
    "📥 Download Enhanced Analysis Data", 
    csv, 
    file_name=f"store_{selected_store}_item_{selected_item}_promotion_analysis.csv"
)

# Footer with insights
st.markdown("---")
st.markdown("""
**💡 Key Insights to Look For:**
- **Price Elasticity**: How sensitive are sales to price changes during promotions?
- **Promotion ROI**: Are the additional sales during promotions worth the discount cost?
- **Optimal Discount**: What discount level maximizes profit while driving sales?
- **Promotion Timing**: Are there patterns in when promotions are most effective?
""")