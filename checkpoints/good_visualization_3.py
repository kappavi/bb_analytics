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

load_dotenv('.env')

# setting types: 
LOAD_SETTINGS = ['csv', 'db']
# if data is detected in data/analysis_cleaned.csv and data/analysis_staging.csv, use csv, otherwise use db
if os.path.exists('data/analysis_cleaned.csv') and os.path.exists('data/analysis_staging.csv'):
    CURRENT_LOAD_SETTING = 'csv'
else:
    CURRENT_LOAD_SETTING = 'db'

# Create the engine
def get_engine():
    # firstly tryt o use st.secrets, otherwise use.env
    if st.secrets:
        db_user = st.secrets['DB_USER']
        db_password = st.secrets['DB_PASSWORD']
        db_host = st.secrets['DB_HOST']
        db_port = st.secrets['DB_PORT']
        db_database = st.secrets['DB_DATABASE']
    else:   
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
            active_price    = ('active_price', 'first'), # use the first active price but mark changes that are made
            active_price_changed = ('active_price', lambda s: s.nunique()>1), # mark if the active price changed so we can plot it later
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
            timezone = ('timezone', 'first'),
            period_start_local = ('period_start_local', 'first'),
            period_end_local = ('period_end_local', 'first'),
            on_promotion = ('on_promotion', 'first')
            
            
        )
    )
    df['weighted_average_promo_spending_ratio'] = df['total_promo_spending'] / df['total_revenue']
    df['weighted_average_promo_spending_ratio_selling_price_based'] = df['total_promo_spending_selling_price_based'] / df['total_revenue_selling_price_based']
    return df



def detect_promotions(df):
    """
    Use on_promotion flag to determine promotion status and validate price logic
    """
    df = df.copy()
    
    # Use the on_promotion flag as the definitive source of promotion status
    df['is_on_promotion'] = df['on_promotion'].fillna(False)
    df['has_promo_price'] = pd.notna(df['avg_promo_price'])
    
    # Calculate promotion discount when on promotion and promo price exists
    df['promo_discount_pct'] = np.where(
        df['is_on_promotion'] & df['has_promo_price'],
        ((df['avg_regular_price'] - df['avg_promo_price']) / df['avg_regular_price'] * 100),
        0
    )
    
    # Validate active price logic: 
    # - During promotion period: active_price should equal promo_price (if available) or regular_price
    # - During non-promotion period: active_price should equal regular_price
    df['expected_active_price'] = np.where(
        df['is_on_promotion'] & df['has_promo_price'],
        df['avg_promo_price'],  # Promotion period with promo price
        df['avg_regular_price']  # Non-promotion period OR promotion period without promo price
    )
    
    # Flag any discrepancies (allowing for small rounding differences)
    df['price_logic_check'] = abs(df['active_price'] - df['expected_active_price']) < 0.01
    
    return df

# 1. — Load your data — 
# Replace with wherever you get your DataFrame
@st.cache_data
def load_data(method=CURRENT_LOAD_SETTING):
    # This will only show progress when not cached
    progress_bar = st.progress(0)
    status_text = st.empty()
    if method == 'csv':
        status_text.text('Loading data from CSV...')
        progress_bar.progress(25)
        df = pd.read_csv('data/analysis_cleaned.csv')
        progress_bar.progress(50)
        df_raw = pd.read_csv('data/analysis_staging.csv')
        progress_bar.progress(100)

    else:
        status_text.text('Loading data from DB...')
        progress_bar.progress(25)
        query = "SELECT * FROM temporary.analysis_cleaned"
        # df = run_query(None, rawQuery=query)
        df = run_query(None, rawQuery=query)
        
        status_text.text('Loading raw staging data...')
        progress_bar.progress(75)
        query = "SELECT * FROM temporary.analysis_staging"
        # df_raw = run_query(None, rawQuery=query)
        df_raw = run_query(None, rawQuery=query)
        
        status_text.text('Data processing complete!')
        progress_bar.progress(100)
    
    # Clear the progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return df, df_raw  


# Main
st.set_page_config(page_title="Item-Centric Promotion Analysis Dashboard", layout="wide")

# Show loading message
with st.spinner('Loading data...'):
    df, df_raw = load_data()

df_analyzed = aggregate_data_by_period(df)
# Add promotion detection
df_analyzed = detect_promotions(df_analyzed)
# order by total_profit descending
df_analyzed = df_analyzed.sort_values('total_profit', ascending=False)

st.title("🎯 Item-Centric Promotion Analysis Dashboard")

def create_enhanced_chart_with_promotions(data, metrics, title, item_data):
    """Create plotly chart with promotion period highlighting and enhanced hover"""
    # Check if we need dual y-axis for active_price
    has_price_metric = 'active_price' in metrics
    other_metrics = [m for m in metrics if m != 'active_price']
    
    if has_price_metric and other_metrics:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    # Add background highlighting for promotion periods
    for _, row in item_data.iterrows():
        if row['is_on_promotion']:
            period_start = pd.to_datetime(row['period_start_local'])
            period_end = pd.to_datetime(row['period_end_local'])
            
            fig.add_shape(
                type="rect",
                x0=period_start, x1=period_end,
                y0=0, y1=1,
                xref="x", yref="paper",
                fillcolor="rgba(255, 0, 0, 0.1)",  # Red background back to original
                layer="below",
                line_width=0,
            )
    
    # Create period midpoints for plotting data points
    period_midpoints = []
    for _, row in item_data.iterrows():
        period_start = pd.to_datetime(row['period_start_local'])
        period_end = pd.to_datetime(row['period_end_local'])
        midpoint = period_start + (period_end - period_start) / 2
        period_midpoints.append(midpoint)
    
    # Prepare lift calculations for hover
    def calculate_lifts(current_idx, item_data):
        if current_idx == 0:
            return "First Period", "First Period"
        
        current_row = item_data.iloc[current_idx]
        prev_row = item_data.iloc[current_idx - 1]
        
        # Find last non-promotional period
        last_non_promo_idx = None
        for i in range(current_idx - 1, -1, -1):
            if not item_data.iloc[i]['is_on_promotion']:
                last_non_promo_idx = i
                break
        
        # Calculate vs previous period
        prev_lifts = []
        for metric in ['active_price', 'avg_regular_price', 'avg_promo_price', 'total_revenue', 'total_profit', 'total_units_sold', 'weighted_average_promo_spending_ratio']:
            if metric in current_row.index and metric in prev_row.index:
                curr_val = current_row[metric] if not pd.isna(current_row[metric]) else 0
                prev_val = prev_row[metric] if not pd.isna(prev_row[metric]) else 0
                lift = ((curr_val / prev_val - 1) * 100) if prev_val != 0 else 0
                prev_lifts.append(f"{metric.replace('avg_', '').replace('_', ' ').title()}: {lift:+.1f}%")
        
        # Calculate vs last non-promotional period
        if last_non_promo_idx is not None:
            non_promo_row = item_data.iloc[last_non_promo_idx]
            non_promo_lifts = []
            for metric in ['active_price', 'avg_regular_price', 'avg_promo_price', 'total_revenue', 'total_profit', 'total_units_sold', 'weighted_average_promo_spending_ratio']:
                if metric in current_row.index and metric in non_promo_row.index:
                    curr_val = current_row[metric] if not pd.isna(current_row[metric]) else 0
                    non_promo_val = non_promo_row[metric] if not pd.isna(non_promo_row[metric]) else 0
                    lift = ((curr_val / non_promo_val - 1) * 100) if non_promo_val != 0 else 0
                    non_promo_lifts.append(f"{metric.replace('avg_', '').replace('_', ' ').title()}: {lift:+.1f}%")
            non_promo_text = "<br>".join(non_promo_lifts)
        else:
            non_promo_text = "No prior non-promo period"
        
        prev_text = "<br>".join(prev_lifts)
        return prev_text, non_promo_text
    
    # Add lines for each metric - using period midpoints for x-axis
    for i, metric in enumerate(metrics):
        # Get the metric values aligned with periods
        metric_values = []
        for _, row in item_data.iterrows():
            metric_values.append(row[metric])
        
        # Create simplified hover text focused on key metrics
        hover_texts = []
        for idx, value in enumerate(metric_values):
            period_info = item_data.iloc[idx]
            
            # Calculate lifts vs previous period for key metrics
            if idx > 0:
                prev_row = item_data.iloc[idx - 1]
                
                # Revenue lift
                curr_revenue = period_info.get('total_revenue', 0)
                prev_revenue = prev_row.get('total_revenue', 0)
                revenue_lift = ((curr_revenue / prev_revenue - 1) * 100) if prev_revenue != 0 else 0
                
                # Profit lift
                curr_profit = period_info.get('total_profit', 0)
                prev_profit = prev_row.get('total_profit', 0)
                profit_lift = ((curr_profit / prev_profit - 1) * 100) if prev_profit != 0 else 0
                
                # Units lift
                curr_units = period_info.get('total_units_sold', 0)
                prev_units = prev_row.get('total_units_sold', 0)
                units_lift = ((curr_units / prev_units - 1) * 100) if prev_units != 0 else 0
                
                lift_text = (
                    f"<b>Lift vs Previous Period:</b><br>"
                    f"Revenue: {revenue_lift:+.1f}%<br>"
                    f"Profit: {profit_lift:+.1f}%<br>"
                    f"Units Sold: {units_lift:+.1f}%"
                )
            else:
                lift_text = "<b>Lift vs Previous Period:</b><br>First Period - No Comparison"
            
            # Build unified hover text
            price_change_warning = " ⚠️" if period_info.get('active_price_changed', False) else ""
            hover_text = (
                f"<b>Period {period_info['period_id']}{price_change_warning}</b><br>"
                f"Active Price: ${period_info.get('active_price', 0):.2f}<br>"
                f"Total Units Sold: {period_info.get('total_units_sold', 0):,.0f}<br><br>"
                f"{lift_text}"
            )
            hover_texts.append(hover_text)
        
        # Line styling
        line_style = dict(width=3) if metric == 'active_price' else dict(width=2)
        if metric == 'active_price':
            line_style['color'] = 'blue'
        elif metric == 'avg_promo_price':
            line_style['dash'] = 'dash'
            line_style['color'] = 'red'
        elif metric == 'avg_regular_price':
            line_style['dash'] = 'dot'
            line_style['color'] = 'green'
        
        # Determine which y-axis to use
        secondary_y = (metric == 'active_price' and has_price_metric and other_metrics)
        
        # Create trace with step-like appearance
        # Add points at period boundaries to create flat lines within periods
        extended_x = []
        extended_y = []
        extended_hover = []
        
        for idx, (midpoint, value, hover_text) in enumerate(zip(period_midpoints, metric_values, hover_texts)):
            period_start = pd.to_datetime(item_data.iloc[idx]['period_start_local'])
            period_end = pd.to_datetime(item_data.iloc[idx]['period_end_local'])
            
            # Add start point, midpoint, and end point for each period
            if idx == 0:
                # For first period, start from period start
                extended_x.extend([period_start, midpoint, period_end])
                extended_y.extend([value, value, value])
                extended_hover.extend([hover_text, hover_text, hover_text])
            else:
                # For subsequent periods, connect from previous period end
                extended_x.extend([period_start, midpoint, period_end])
                extended_y.extend([value, value, value])
                extended_hover.extend([hover_text, hover_text, hover_text])
        
        # Create trace - only show detailed hover with lift information to the first metric to avoid duplication
        trace = go.Scatter(
            x=extended_x,
            y=extended_y,
            mode='lines+markers',
            name=metric.replace('avg_', '').replace('_', ' ').title(),
            line=line_style,
            marker=dict(
                size=[6 if i % 3 != 1 else (8 if metric == 'active_price' else 6) for i in range(len(extended_x))],
                opacity=[0.3 if i % 3 != 1 else 1.0 for i in range(len(extended_x))]  # Highlight midpoints
            ),
            connectgaps=True
        )
        
        # Only add detailed hover with lift information to the first metric
        if i == 0:
            trace.update(
                hovertemplate="%{customdata}<extra></extra>",
                customdata=extended_hover
            )
        else:
            # For other metrics, show simple hover with just the metric value
            simple_hover = [f"<b>{metric.replace('avg_', '').replace('_', ' ').title()}</b><br>Value: {val:.2f}" for val in extended_y]
            trace.update(
                hovertemplate="%{customdata}<extra></extra>",
                customdata=simple_hover
            )
        
        if has_price_metric and other_metrics:
            fig.add_trace(trace, secondary_y=secondary_y)
        else:
            fig.add_trace(trace)
    
    # Add period separation lines and annotations (one line between periods)
    for idx, row in enumerate(item_data.iterrows()):
        _, row = row
        period_start = pd.to_datetime(row['period_start_local'])
        line_color = "red" if row['is_on_promotion'] else "gray"
        line_width = 2 if row['is_on_promotion'] else 1
        
        # Only add separation line at period start (except for the first period)
        if idx > 0:
            fig.add_shape(
                type="line",
                x0=period_start, x1=period_start,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=line_color, dash="dot", width=line_width)
            )
        
        # Annotation at bottom of period line for better visibility
        annotation_text = f"P{row['period_id']}"
        if row['is_on_promotion']:
            annotation_text += f" 🎁"
        if row.get('active_price_changed', False):
            annotation_text += " ⚠️"
        
        fig.add_annotation(
            x=period_start,
            y=1.1,  # Place at top of chart
            xref="x", yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=10, color="rgba(255, 255, 255, 1)", weight="bold" if row['is_on_promotion'] else "normal"),
            bgcolor="rgba(0, 0, 0, 0.5)" if row['is_on_promotion'] else None,
            bordercolor="rgba(255, 255, 255, 1)" if row['is_on_promotion'] else None,
            borderwidth=1 if row['is_on_promotion'] else 0,
            xanchor="left",
            yanchor="top"
        )
    
    # Update layout with soft gridlines
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        hovermode='x unified',  # Unified hover for all traces at x position
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="right", x=1),
        height=500,
        plot_bgcolor="white",  # Clean white background
        paper_bgcolor="black",
        font=dict(color="rgba(32, 32, 32, 1)"),  # Darker gray text
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",  # Soft gray vertical lines
            gridwidth=1,
            color="rgba(32, 32, 32, 1)"  # Darker gray axis text
        ),
        yaxis=dict(
            color="rgba(32, 32, 32, 1)"  # Darker gray axis text
        )
    )
    
    if has_price_metric and other_metrics:
        # Get the data ranges for both axes to set appropriate tick spacing
        price_data = [row['active_price'] for _, row in item_data.iterrows() if not pd.isna(row['active_price'])]
        
        # Get other metrics data for left axis
        other_data = []
        for metric in other_metrics:
            other_data.extend([row[metric] for _, row in item_data.iterrows() if not pd.isna(row[metric])])
        
        if price_data and other_data:
            price_min, price_max = min(price_data), max(price_data)
            other_min, other_max = min(other_data), max(other_data)
            
            # Calculate appropriate tick spacing
            price_range = price_max - price_min
            other_range = other_max - other_min
            
            # Set tick spacing based on data range
            if price_range > 0:
                # For price: steps of $1 if range < $10, else $2, $5, $10, etc.
                if price_range <= 10:
                    price_tick = 1
                elif price_range <= 20:
                    price_tick = 2
                elif price_range <= 50:
                    price_tick = 5
                else:
                    price_tick = 10
            else:
                price_tick = 1
            
            if other_range > 0:
                # For financial metrics: steps of $1000 if range < $10k, else $2k, $5k, $10k, etc.
                if other_range <= 10000:
                    other_tick = 1000
                elif other_range <= 20000:
                    other_tick = 2000
                elif other_range <= 50000:
                    other_tick = 5000
                elif other_range <= 100000:
                    other_tick = 10000
                else:
                    other_tick = 20000
            else:
                other_tick = 1000
            
            # Calculate target number of ticks - use financial metrics as the base
            target_ticks = 7  # Standard number of grid lines
            
            # Calculate tick size for financial metrics to get exactly target_ticks
            if other_range > 0:
                other_tick_calculated = other_max / target_ticks
                # Round to nice numbers
                if other_tick_calculated <= 1000:
                    other_tick = max(100, round(other_tick_calculated / 100) * 100)
                elif other_tick_calculated <= 5000:
                    other_tick = max(500, round(other_tick_calculated / 500) * 500)
                elif other_tick_calculated <= 10000:
                    other_tick = max(1000, round(other_tick_calculated / 1000) * 1000)
                else:
                    other_tick = max(2000, round(other_tick_calculated / 2000) * 2000)
            else:
                other_tick = 1000
            
            # Calculate tick size for price to get exactly target_ticks  
            if price_range > 0:
                price_tick_calculated = price_max / target_ticks
                # Round to nice numbers
                if price_tick_calculated <= 1:
                    price_tick = max(0.5, round(price_tick_calculated * 2) / 2)
                elif price_tick_calculated <= 5:
                    price_tick = max(1, round(price_tick_calculated))
                elif price_tick_calculated <= 10:
                    price_tick = max(2, round(price_tick_calculated / 2) * 2)
                else:
                    price_tick = max(5, round(price_tick_calculated / 5) * 5)
            else:
                price_tick = 1
            
            # Calculate actual max values based on tick sizes to ensure same number of lines
            other_axis_max = other_tick * target_ticks
            price_axis_max = price_tick * target_ticks
            
            # Set the y-axis properties with shared soft gridlines
            fig.update_yaxes(
                title_text="Financial Metrics ($)", 
                secondary_y=False,
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.2)",  # Soft gray with low opacity
                gridwidth=1,
                dtick=other_tick,
                range=[0, other_axis_max],  # Start at 0, use calculated max
                color="rgba(32, 32, 32, 1)"  # Darker gray axis text
            )
            fig.update_yaxes(
                title_text="Price ($)", 
                secondary_y=True,
                showgrid=False,  # Don't show grid for secondary axis to avoid duplication
                overlaying="y",
                side="right",
                dtick=price_tick,
                range=[0, price_axis_max],  # Start at 0, use calculated max
                color="rgba(32, 32, 32, 1)"  # Darker gray axis text
            )
        else:
            # Fallback if no data
            fig.update_yaxes(title_text="Financial Metrics ($)", secondary_y=False, color="rgba(32, 32, 32, 1)")
            fig.update_yaxes(title_text="Price ($)", secondary_y=True, color="rgba(32, 32, 32, 1)")
    else:
        fig.update_yaxes(title_text="Value", color="rgba(32, 32, 32, 1)")
    
    return fig

# Create main tabs
analytics_tab, customer_tab = st.tabs(["🔬 Raw Data Analytics", "👥 For Customer"])

with analytics_tab:
    st.header("🔬 Raw Data Analytics - Daily Level Analysis")
    
    # Analytics page sidebar controls
    st.sidebar.header("🔬 Analytics Filters")
    
    # Store selector for raw data
    stores_raw = df_raw[['store_id', 'merchant_id']].drop_duplicates().sort_values('store_id')
    stores_raw['display_name'] = stores_raw['store_id'].astype(str) + " (Merchant: " + stores_raw['merchant_id'].astype(str) + ")"
    selected_store_raw = st.sidebar.selectbox(
        "Select Store (Analytics):",
        options=stores_raw['store_id'].tolist(),
        format_func=lambda sid: stores_raw[stores_raw['store_id'] == sid]['display_name'].iloc[0],
        key="analytics_store"
    )
    
    # Filter raw data by selected store
    df_raw_store_filtered = df_raw[df_raw['store_id'] == selected_store_raw]
    
    # Item selector for raw data (filtered by store)
    items_in_store_raw = df_raw_store_filtered[['item_id','name']].drop_duplicates().set_index('item_id')
    selected_item_raw = st.sidebar.selectbox(
        "Select Item (Analytics):",
        options=items_in_store_raw.index,
        format_func=lambda iid: f"{iid} — {items_in_store_raw.loc[iid,'name']}",
        key="analytics_item"
    )
    
    # Metrics selector for raw data
    st.sidebar.subheader("📊 Daily Metrics")
    # Use only columns that exist in df_raw
    available_columns = df_raw.columns.tolist()
    price_metrics = [col for col in ['regular_price', 'scraping_regular_price', 'promo_price', 'avg_selling_price', 'unit_cost'] if col in available_columns]
    volume_metrics = [col for col in ['units_sold', 'total_basket_size', 'num_baskets', 'avg_basket_size'] if col in available_columns]
    
    st.sidebar.write("**Price Metrics:**")
    selected_price_metrics = st.sidebar.multiselect(
        "Select price metrics:", 
        price_metrics, 
        default=price_metrics[:3] if len(price_metrics) >= 3 else price_metrics,
        key="analytics_price_metrics"
    )
    
    st.sidebar.write("**Volume Metrics:**")
    selected_volume_metrics = st.sidebar.multiselect(
        "Select volume metrics:", 
        volume_metrics, 
        default=[],
        key="analytics_volume_metrics"
    )
    
    # Combine selected metrics
    selected_metrics_raw = selected_price_metrics + selected_volume_metrics
    
    # Filter data for selected item and store
    item_raw_df = df_raw_store_filtered[df_raw_store_filtered['item_id'] == selected_item_raw].copy()
    
    if item_raw_df.empty:
        st.error(f"No raw data found for item {selected_item_raw} in store {selected_store_raw}")
    else:
        # Sort by date
        if 'day' in item_raw_df.columns:
            item_raw_df['day'] = pd.to_datetime(item_raw_df['day'])
            item_raw_df = item_raw_df.sort_values('day')
        
        # Show data info
        total_days = len(item_raw_df)
        date_range = f"{item_raw_df['day'].min().strftime('%Y-%m-%d')} to {item_raw_df['day'].max().strftime('%Y-%m-%d')}" if 'day' in item_raw_df.columns else "Unknown"
        st.info(f"📊 **Item {selected_item_raw}** in Store {selected_store_raw} | {total_days} days of data | Date range: {date_range}")
        
        # Analyze data quality patterns
        st.subheader("🔍 Data Quality Analysis")
        
        # Create analysis of the 3 specific scenarios
        analysis_df = item_raw_df.copy()
        analysis_df['regular_price_missing'] = pd.isna(analysis_df['regular_price'])
        analysis_df['scraping_regular_missing'] = pd.isna(analysis_df['scraping_regular_price'])
        analysis_df['promo_price_missing'] = pd.isna(analysis_df['promo_price'])
        
        # Define the 3 scenarios (updated logic)
        analysis_df['scenario_1_no_prices'] = (analysis_df['regular_price_missing'] & analysis_df['scraping_regular_missing'] & analysis_df['promo_price_missing'])
        analysis_df['scenario_2_no_scraped_or_promo'] = (analysis_df['scraping_regular_missing'] & analysis_df['promo_price_missing'])
        analysis_df['scenario_3_regular_vs_scraped_diff'] = (
            (~analysis_df['regular_price_missing']) & 
            (~analysis_df['scraping_regular_missing']) & 
            (abs(analysis_df['regular_price'] - analysis_df['scraping_regular_price']) > 0.01)
        )
        
        # Count scenarios
        col1, col2, col3 = st.columns(3)
        with col1:
            scenario_1_count = analysis_df['scenario_1_no_prices'].sum()
            st.metric("🔴 No Prices Available", f"{scenario_1_count} days", 
                     delta=f"{scenario_1_count/len(analysis_df)*100:.1f}%" if len(analysis_df) > 0 else "0%")
        
        with col2:
            scenario_2_count = analysis_df['scenario_2_no_scraped_or_promo'].sum()
            st.metric("🟠 No Scraped/Promo Price", f"{scenario_2_count} days", 
                     delta=f"{scenario_2_count/len(analysis_df)*100:.1f}%" if len(analysis_df) > 0 else "0%")
        
        with col3:
            scenario_3_count = analysis_df['scenario_3_regular_vs_scraped_diff'].sum()
            st.metric("🔵 Regular ≠ Scraped", f"{scenario_3_count} days", 
                     delta=f"{scenario_3_count/len(analysis_df)*100:.1f}%" if len(analysis_df) > 0 else "0%")

        # Create daily metrics chart with missing data handling
        if selected_metrics_raw and not item_raw_df.empty:
            def create_daily_metrics_chart(data, metrics, title):
                """Create horizontally scrollable chart with missing data indicators and scenario highlighting"""
                # Check if we need dual y-axis (price vs volume metrics)
                price_metrics_in_selection = [m for m in metrics if m in ['regular_price', 'scraping_regular_price', 'promo_price', 'avg_selling_price', 'unit_cost']]
                volume_metrics_in_selection = [m for m in metrics if m in ['units_sold', 'total_basket_size', 'num_baskets', 'avg_basket_size']]
                
                if price_metrics_in_selection and volume_metrics_in_selection:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                else:
                    fig = make_subplots(specs=[[{"secondary_y": False}]])
                
                # Add background highlighting for data quality scenarios with hover information
                x_dates = data['day'] if 'day' in data.columns else data.index
                
                # Collect data for invisible hover traces for each scenario
                scenario_1_dates, scenario_1_hover = [], []
                scenario_2_dates, scenario_2_hover = [], []
                scenario_3_dates, scenario_3_hover = [], []
                
                for idx, (date, row) in enumerate(zip(x_dates, data.itertuples())):
                    # Format date for display
                    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                    
                    # Scenario highlighting - 3 scenarios with priority order
                    if hasattr(row, 'scenario_1_no_prices') and row.scenario_1_no_prices:
                        # Red background for no prices available (highest priority)
                        fig.add_shape(
                            type="rect",
                            x0=date, x1=date,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            fillcolor="rgba(255, 0, 0, 0.25)",
                            layer="below",
                            line=dict(color="red", width=2)
                        )
                        
                        # Collect hover info for scenario 1
                        scenario_1_dates.append(date)
                        regular_price = getattr(row, 'regular_price', 'N/A')
                        scraping_price = getattr(row, 'scraping_regular_price', 'N/A')
                        promo_price = getattr(row, 'promo_price', 'N/A')
                        
                        hover_text = (
                            f"<b>🔴 DATA QUALITY ISSUE</b><br>"
                            f"Date: {date_str}<br>"
                            f"<b>Problem:</b> All prices missing<br><br>"
                            f"Regular Price: {regular_price}<br>"
                            f"Scraping Price: {scraping_price}<br>"
                            f"Promo Price: {promo_price}<br><br>"
                            f"<b>Impact:</b> Cannot determine actual pricing"
                        )
                        scenario_1_hover.append(hover_text)
                        
                    elif hasattr(row, 'scenario_2_no_scraped_or_promo') and row.scenario_2_no_scraped_or_promo:
                        # Orange background for no scraped/promo price
                        fig.add_shape(
                            type="rect",
                            x0=date, x1=date,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            fillcolor="rgba(255, 165, 0, 0.2)",
                            layer="below",
                            line=dict(color="orange", width=2)
                        )
                        
                        # Collect hover info for scenario 2
                        scenario_2_dates.append(date)
                        regular_price = getattr(row, 'regular_price', 'N/A')
                        scraping_price = getattr(row, 'scraping_regular_price', 'N/A')
                        promo_price = getattr(row, 'promo_price', 'N/A')
                        
                        missing_items = []
                        if pd.isna(getattr(row, 'scraping_regular_price', None)):
                            missing_items.append("Scraping Price")
                        if pd.isna(getattr(row, 'promo_price', None)):
                            missing_items.append("Promo Price")
                        
                        # Format prices properly
                        reg_price_str = f"${regular_price:.2f}" if not pd.isna(regular_price) else "N/A"
                        scrap_price_str = f"${scraping_price:.2f}" if not pd.isna(scraping_price) else "N/A"
                        promo_price_str = f"${promo_price:.2f}" if not pd.isna(promo_price) else "N/A"
                        
                        hover_text = (
                            f"<b>🟠 DATA QUALITY ISSUE</b><br>"
                            f"Date: {date_str}<br>"
                            f"<b>Problem:</b> Missing {' & '.join(missing_items)}<br><br>"
                            f"Regular Price: {reg_price_str}<br>"
                            f"Scraping Price: {scrap_price_str}<br>"
                            f"Promo Price: {promo_price_str}<br><br>"
                            f"<b>Impact:</b> Limited price validation capability"
                        )
                        scenario_2_hover.append(hover_text)
                        
                    elif hasattr(row, 'scenario_3_regular_vs_scraped_diff') and row.scenario_3_regular_vs_scraped_diff:
                        # Teal background for price differences
                        fig.add_shape(
                            type="rect",
                            x0=date, x1=date,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            fillcolor="rgba(0, 128, 128, 0.2)",
                            layer="below",
                            line=dict(color="teal", width=2)
                        )
                        
                        # Collect hover info for scenario 3
                        scenario_3_dates.append(date)
                        regular_price = getattr(row, 'regular_price', 0)
                        scraping_price = getattr(row, 'scraping_regular_price', 0)
                        price_diff = abs(regular_price - scraping_price)
                        
                        hover_text = (
                            f"<b>🔵 PRICE DISCREPANCY</b><br>"
                            f"Date: {date_str}<br>"
                            f"<b>Problem:</b> Price mismatch detected<br><br>"
                            f"Regular Price: ${regular_price:.2f}<br>"
                            f"Scraping Price: ${scraping_price:.2f}<br>"
                            f"<b>Difference: ${price_diff:.2f}</b><br><br>"
                            f"<b>Impact:</b> Data consistency issue"
                        )
                        scenario_3_hover.append(hover_text)
                
                # Store scenario data for later use after we know the y-axis ranges
                scenario_data = {
                    'scenario_1': (scenario_1_dates, scenario_1_hover),
                    'scenario_2': (scenario_2_dates, scenario_2_hover), 
                    'scenario_3': (scenario_3_dates, scenario_3_hover)
                }
                
                # Color palette for different metrics
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
                
                for i, metric in enumerate(metrics):
                    if metric not in data.columns:
                        st.warning(f"Metric '{metric}' not found in data")
                        continue
                    
                    # Prepare data with missing value handling
                    x_dates = data['day'] if 'day' in data.columns else data.index
                    y_values = data[metric].copy()
                    
                    # Special handling for when regular_price and scraping_regular_price are the same
                    line_style = dict(color=colors[i % len(colors)], width=2)
                    marker_colors = [colors[i % len(colors)]] * len(y_values)
                    marker_sizes = [6] * len(y_values)
                    
                    if metric == 'scraping_regular_price' and 'regular_price' in data.columns:
                        # Mark points purple when regular and scraping prices are the same
                        for idx, (reg_price, scrap_price) in enumerate(zip(data['regular_price'], data['scraping_regular_price'])):
                            if pd.notna(reg_price) and pd.notna(scrap_price) and abs(reg_price - scrap_price) < 0.01:
                                marker_colors[idx] = 'purple'
                                marker_sizes[idx] = 8  # Slightly larger for visibility
                    
                    # Create custom hover text showing data availability
                    hover_texts = []
                    is_price_metric = metric in ['regular_price', 'scraping_regular_price', 'promo_price', 'avg_selling_price', 'unit_cost']
                    
                    for idx, (date, value) in enumerate(zip(x_dates, y_values)):
                        # Format date properly
                        if hasattr(date, 'strftime'):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                            
                        if pd.isna(value):
                            hover_text = f"<b>{metric.replace('_', ' ').title()}</b><br>Date: {date_str}<br>Value: N/A ❌"
                        else:
                            try:
                                if is_price_metric:
                                    hover_text = f"<b>{metric.replace('_', ' ').title()}</b><br>Date: {date_str}<br>Value: ${float(value):.2f}"
                                else:
                                    hover_text = f"<b>{metric.replace('_', ' ').title()}</b><br>Date: {date_str}<br>Value: {float(value):.0f}"
                            except (ValueError, TypeError):
                                # If value cannot be converted to float, display as string
                                hover_text = f"<b>{metric.replace('_', ' ').title()}</b><br>Date: {date_str}<br>Value: {value}"
                        hover_texts.append(hover_text)
                    
                    # Determine which y-axis to use
                    is_volume_metric = metric in volume_metrics_in_selection
                    secondary_y = is_volume_metric and price_metrics_in_selection and volume_metrics_in_selection
                    
                    # Create trace with enhanced markers for price matching
                    trace = go.Scatter(
                        x=x_dates,
                        y=y_values,
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=line_style,
                        marker=dict(
                            size=marker_sizes,
                            symbol='circle',
                            color=marker_colors,
                            line=dict(width=1, color='white')
                        ),
                        connectgaps=False,  # Don't connect across missing data
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=hover_texts
                    )
                    
                    if price_metrics_in_selection and volume_metrics_in_selection:
                        fig.add_trace(trace, secondary_y=secondary_y)
                    else:
                        fig.add_trace(trace)
                
                # Update layout for horizontal scrolling
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    hovermode='closest',
                    height=600,
                    # Enable horizontal scrolling for many data points
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date"
                    ),
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=1.02, 
                        xanchor="right", 
                        x=1
                    )
                )
                
                # Set y-axis labels based on what metrics are selected
                if price_metrics_in_selection and volume_metrics_in_selection:
                    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
                    fig.update_yaxes(title_text="Volume/Count", secondary_y=True)
                elif price_metrics_in_selection:
                    fig.update_yaxes(title_text="Price ($)")
                elif volume_metrics_in_selection:
                    fig.update_yaxes(title_text="Volume/Count")
                else:
                    fig.update_yaxes(title_text="Value")
                
                # Add invisible hover traces for data quality scenarios
                # Get the y-axis range to position hover points appropriately
                y_min, y_max = 0, 1
                if metrics and len(data) > 0:
                    all_values = []
                    for metric in metrics:
                        if metric in data.columns:
                            metric_values = data[metric].dropna()
                            if not metric_values.empty:
                                all_values.extend(metric_values.tolist())
                    
                    if all_values:
                        y_min, y_max = min(all_values), max(all_values)
                        y_mid = (y_min + y_max) / 2
                    else:
                        y_mid = 0.5
                else:
                    y_mid = 0.5
                
                # Add scenario hover traces
                scenario_colors = {'scenario_1': 'red', 'scenario_2': 'orange', 'scenario_3': 'teal'}
                scenario_names = {'scenario_1': 'Data Quality Issues', 'scenario_2': 'Missing Data Issues', 'scenario_3': 'Price Discrepancies'}
                
                for scenario_key, (dates, hover_texts) in scenario_data.items():
                    if dates and hover_texts:
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=[y_mid] * len(dates),  # Position at middle of y-range
                            mode='markers',
                            marker=dict(
                                size=20, 
                                color=scenario_colors[scenario_key], 
                                opacity=0.01,  # Nearly invisible
                                symbol='square'
                            ),
                            name=scenario_names[scenario_key],
                            showlegend=False,
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=hover_texts
                        ))
                
                return fig
            
            # Create and display the chart with enhanced analysis
            fig_daily = create_daily_metrics_chart(
                analysis_df,  # Use the enhanced analysis dataframe
                selected_metrics_raw, 
                f"Daily Metrics for Item {selected_item_raw} - Store {selected_store_raw}"
            )
            
            # Add legend for data quality indicators
            st.info("""
            **📊 Chart Legend:**
            - 🔴 **Red Background**: No regular price OR scraping price OR promo price (big issue)
            - 🟠 **Orange Background**: No scraped_regular_price OR promo_price available  
            - 🔵 **Teal Background**: regular_price ≠ scraped_regular_price (difference > $0.01)
            - 🟣 **Purple Dots**: Points where regular_price = scraping_regular_price
            
            **💡 Tip:** Hover over highlighted background areas to see detailed price information and data quality issues!
            """)
            
            st.plotly_chart(fig_daily, use_container_width=True)
            
            # Data quality summary
            st.subheader("📊 Data Quality Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Missing Data Count:**")
                for metric in selected_metrics_raw:
                    if metric in item_raw_df.columns:
                        missing_count = item_raw_df[metric].isna().sum()
                        total_count = len(item_raw_df)
                        missing_pct = (missing_count / total_count) * 100
                        st.write(f"{metric}: {missing_count}/{total_count} ({missing_pct:.1f}%)")
            
            with col2:
                st.write("**Value Ranges:**")
                for metric in selected_metrics_raw:
                    if metric in item_raw_df.columns and not item_raw_df[metric].isna().all():
                        min_val = item_raw_df[metric].min()
                        max_val = item_raw_df[metric].max()
                        st.write(f"{metric}: ${min_val:.2f} - ${max_val:.2f}")
            
            with col3:
                st.write("**Latest Values:**")
                latest_data = item_raw_df.tail(1)
                for metric in selected_metrics_raw:
                    if metric in latest_data.columns:
                        latest_val = latest_data[metric].iloc[0]
                        if pd.isna(latest_val):
                            st.write(f"{metric}: N/A ❌")
                        else:
                            st.write(f"{metric}: ${latest_val:.2f}")
            
            # Raw data table (last 10 days)
            st.subheader("📋 Recent Raw Data (Last 10 Days)")
            recent_data = item_raw_df.tail(10)
            display_columns = ['day'] + selected_metrics_raw if 'day' in recent_data.columns else selected_metrics_raw
            display_data = recent_data[display_columns].copy()
            
            if 'day' in display_data.columns:
                display_data['day'] = display_data['day'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_data, use_container_width=True)
            
            # Download raw data
            csv_raw = item_raw_df.to_csv(index=False)
            st.download_button(
                "📥 Download Raw Data", 
                csv_raw, 
                file_name=f"store_{selected_store_raw}_item_{selected_item_raw}_raw_data.csv",
                key="download_raw"
            )

with customer_tab:
    st.header("👥 Customer Dashboard")
    st.info("This page shows simplified insights focusing on customer-relevant metrics.")
    
    # Customer page controls on main page
    st.subheader("🔧 Select Store and Item")
    col1, col2 = st.columns(2)
    
    with col1:
        # Store selector for customer
        stores_customer = df_analyzed[['store_id', 'merchant_id']].drop_duplicates().sort_values('store_id')
        stores_customer['display_name'] = stores_customer['store_id'].astype(str) + " (Merchant: " + stores_customer['merchant_id'].astype(str) + ")"
        selected_store_customer = st.selectbox(
            "Select Store:",
            options=stores_customer['store_id'].tolist(),
            format_func=lambda sid: stores_customer[stores_customer['store_id'] == sid]['display_name'].iloc[0],
            key="customer_store"
        )
    
    # Filter data by selected store for customer
    df_store_filtered_customer = df_analyzed[df_analyzed['store_id'] == selected_store_customer]
    
    with col2:
        # Item selector for customer (filtered by store)
        items_in_store_customer = df_store_filtered_customer[['item_id','name']].drop_duplicates().set_index('item_id')
        selected_item_customer = st.selectbox(
            "Select Item:",
            options=items_in_store_customer.index,
            format_func=lambda iid: f"{iid} — {items_in_store_customer.loc[iid,'name']}",
            key="customer_item"
        )
    
    # Filter data for the selected item and store
    item_df_customer = df_store_filtered_customer[df_store_filtered_customer['item_id'] == selected_item_customer].sort_values('period_id')
    
    if not item_df_customer.empty:
        st.header(f"📈 Analysis for {items_in_store_customer.loc[selected_item_customer,'name']} (Store {selected_store_customer})")
        
        # 1. Financial insights graph with total revenue, total profit, active price
        st.subheader("💰 Financial Performance By Period")
        customer_metrics = ['total_revenue', 'total_profit', 'active_price']
        fin_ts_customer = item_df_customer.set_index('period_start_local')[customer_metrics]
        fig_fin_customer = create_enhanced_chart_with_promotions(
            fin_ts_customer, customer_metrics, 
            f"Financial Metrics - {items_in_store_customer.loc[selected_item_customer,'name']}", 
            item_df_customer
        )
        st.plotly_chart(fig_fin_customer, use_container_width=True)
        
        # 1.5. Financial Performance by Day graph (daily data points with period-based features)
        st.subheader("📅 Financial Performance by Day")
        
        # Get daily data for this specific item and store
        daily_data_customer = df[(df['store_id'] == selected_store_customer) & 
                                    (df['item_id'] == selected_item_customer)].copy()
        
        if not daily_data_customer.empty:
            # Sort by date
            daily_data_customer['day'] = pd.to_datetime(daily_data_customer['day'])
            daily_data_customer = daily_data_customer.sort_values('day')
            
            def create_daily_financial_chart_with_period_features(daily_data, period_data, title):
                """Create daily financial chart with period-based hover and promotion backgrounds"""
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add background highlighting for promotion periods
                for _, row in period_data.iterrows():
                    if row['is_on_promotion']:
                        period_start = pd.to_datetime(row['period_start_local'])
                        period_end = pd.to_datetime(row['period_end_local'])
                        
                        fig.add_shape(
                            type="rect",
                            x0=period_start, x1=period_end,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            fillcolor="rgba(255, 0, 0, 0.1)",  # Red background for promotions
                            layer="below",
                            line_width=0,
                        )
                
                # Add period boundary vertical lines to match the period-based chart
                for idx, (_, row) in enumerate(period_data.iterrows()):
                    period_start = pd.to_datetime(row['period_start_local'])
                    period_end = pd.to_datetime(row['period_end_local'])
                    line_color = "red" if row['is_on_promotion'] else "gray"
                    line_width = 2 if row['is_on_promotion'] else 1
                    
                    # Add line at period start (except for the first period)
                    if idx > 0:
                        fig.add_shape(
                            type="line",
                            x0=period_start, x1=period_start,
                            y0=0, y1=1,
                            xref="x", yref="paper",
                            line=dict(color=line_color, dash="dot", width=line_width),
                            layer="above"
                        )
                    
                    # Add line at period end for all periods
                    fig.add_shape(
                        type="line",
                        x0=period_end, x1=period_end,
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color=line_color, dash="dot", width=line_width),
                        layer="above"
                    )
                
                # Create period mapping for hover text
                period_hover_map = {}
                for _, row in period_data.iterrows():
                    period_start = pd.to_datetime(row['period_start_local'])
                    period_end = pd.to_datetime(row['period_end_local'])
                    
                    # Calculate lifts vs previous period for hover
                    period_idx = list(period_data.index).index(row.name)
                    if period_idx > 0:
                        prev_row = period_data.iloc[period_idx - 1]
                        
                        # Revenue lift
                        curr_revenue = row.get('total_revenue', 0)
                        prev_revenue = prev_row.get('total_revenue', 0)
                        revenue_lift = ((curr_revenue / prev_revenue - 1) * 100) if prev_revenue != 0 else 0
                        
                        # Profit lift
                        curr_profit = row.get('total_profit', 0)
                        prev_profit = prev_row.get('total_profit', 0)
                        profit_lift = ((curr_profit / prev_profit - 1) * 100) if prev_profit != 0 else 0
                        
                        # Units lift
                        curr_units = row.get('total_units_sold', 0)
                        prev_units = prev_row.get('total_units_sold', 0)
                        units_lift = ((curr_units / prev_units - 1) * 100) if prev_units != 0 else 0
                        
                        lift_text = (
                            f"<b>Lift vs Previous Period:</b><br>"
                            f"Revenue: {revenue_lift:+.1f}%<br>"
                            f"Profit: {profit_lift:+.1f}%<br>"
                            f"Units Sold: {units_lift:+.1f}%"
                        )
                    else:
                        lift_text = "<b>Lift vs Previous Period:</b><br>First Period - No Comparison"
                    
                    # Build hover text for this period
                    price_change_warning = " ⚠️" if row.get('active_price_changed', False) else ""
                    period_hover_text = (
                        f"<b>Period {row['period_id']}{price_change_warning}</b><br>"
                        f"Active Price: ${row.get('active_price', 0):.2f}<br>"
                        f"Total Units Sold: {row.get('total_units_sold', 0):,.0f}<br><br>"
                        f"{lift_text}"
                    )
                    
                    # Map all days in this period to the period hover text
                    current_date = period_start
                    while current_date <= period_end:
                        period_hover_map[current_date.strftime('%Y-%m-%d')] = period_hover_text
                        current_date += pd.Timedelta(days=1)
                
                                # Add daily data traces with period-based hover
                daily_metrics = ['revenue', 'profit', 'active_price']
                colors = ['#87ceeb', '#1f77b4', 'blue']  # ocean blue, light blue, blue, purple
                
                for i, metric in enumerate(daily_metrics):
                    if metric in daily_data.columns:
                        # Determine which y-axis to use
                        secondary_y = (metric == 'active_price')
                        
                        # Only show detailed hover with daily financial information on the first metric
                        if i == 0:
                            # Create detailed hover text for each day with daily revenue and profit
                            hover_texts = []
                            for _, day_row in daily_data.iterrows():
                                
                                # Get daily financial values
                                daily_revenue = day_row.get('revenue', 0)
                                
                                hover_text = (
                                    f"<b>Revenue:</b> ${daily_revenue:,.2f}<br><b>Units Sold:</b> {day_row.get('units_sold', 0):,.0f}"
                                )
                                hover_texts.append(hover_text)
                        else:
                            # For other metrics, show simple hover with just the metric value
                            hover_texts = [f"<b>{metric.replace('_', ' ').title()}</b> Value: {row[metric]:.2f}" 
                                         for _, row in daily_data.iterrows()]
                        
                        trace = go.Scatter(
                            x=daily_data['day'],
                            y=daily_data[metric],
                            mode='lines+markers',
                            name=metric.replace('_', ' ').title(),
                            line=dict(color=colors[i], width=2),
                            marker=dict(size=4, opacity=0.7),
                            hovertemplate="%{customdata}<extra></extra>",
                            customdata=hover_texts,
                            connectgaps=False
                        )
                        
                        fig.add_trace(trace, secondary_y=secondary_y)

                # Add period annotations to match the period-based chart
                for idx, (_, row) in enumerate(period_data.iterrows()):
                    period_start = pd.to_datetime(row['period_start_local'])
                    
                    # Annotation at top of chart to match period-based chart
                    annotation_text = f"P{row['period_id']}"
                    if row['is_on_promotion']:
                        annotation_text += f" 🎁"
                    if row.get('active_price_changed', False):
                        annotation_text += " ⚠️"
                    
                    fig.add_annotation(
                        x=period_start,
                        y=1.1,  # Place at top of chart
                        xref="x", yref="paper",
                        text=annotation_text,
                        showarrow=False,
                        font=dict(size=10, color="rgba(255, 255, 255, 1)", weight="bold" if row['is_on_promotion'] else "normal"),
                        bgcolor="rgba(0, 0, 0, 0.5)" if row['is_on_promotion'] else None,
                        bordercolor="rgba(255, 255, 255, 1)" if row['is_on_promotion'] else None,
                        borderwidth=1 if row['is_on_promotion'] else 0,
                        xanchor="left",
                        yanchor="top"
                    )
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="right", x=1),
                    height=500,
                    plot_bgcolor="white",
                    paper_bgcolor="black",
                    font=dict(color="rgba(32, 32, 32, 1)"),
                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(128, 128, 128, 0.2)",
                        gridwidth=1,
                        color="rgba(32, 32, 32, 1)"
                    ),
                    yaxis=dict(
                        color="rgba(32, 32, 32, 1)"
                    )
                )
                
                # Calculate synchronized y-axis ranges for both sides
                target_ticks = 7  # Standard number of grid lines
                
                # Get data ranges for both axes
                financial_data = []
                for metric in ['revenue', 'profit']:
                    if metric in daily_data.columns:
                        metric_values = daily_data[metric].dropna()
                        if not metric_values.empty:
                            financial_data.extend(metric_values.tolist())
                
                if financial_data:
                    financial_max = max(financial_data)
                    # Calculate tick size for financial metrics
                    financial_tick_calculated = financial_max / target_ticks
                    if financial_tick_calculated <= 1000:
                        financial_tick = max(100, round(financial_tick_calculated / 100) * 100)
                    elif financial_tick_calculated <= 5000:
                        financial_tick = max(500, round(financial_tick_calculated / 500) * 500)
                    elif financial_tick_calculated <= 10000:
                        financial_tick = max(1000, round(financial_tick_calculated / 1000) * 1000)
                    else:
                        financial_tick = max(2000, round(financial_tick_calculated / 2000) * 2000)
                    
                    financial_axis_max = financial_tick * target_ticks
                else:
                    financial_tick = 1000
                    financial_axis_max = 7000
                
                # Set y-axis properties for financial metrics
                fig.update_yaxes(
                    title_text="Financial Metrics ($)", 
                    secondary_y=False,
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    gridwidth=1,
                    dtick=financial_tick,
                    range=[0, financial_axis_max],  # Start at 0, use calculated max
                    color="rgba(32, 32, 32, 1)"
                )
                
                # Get price range for synchronized y-axis scaling
                if 'active_price' in daily_data.columns:
                    price_data_daily = daily_data['active_price'].dropna()
                    if not price_data_daily.empty:
                        daily_price_max = price_data_daily.max()
                        # Calculate tick size for price to get exactly target_ticks
                        price_tick_calculated = daily_price_max / target_ticks
                        if price_tick_calculated <= 1:
                            price_tick = max(0.5, round(price_tick_calculated * 2) / 2)
                        elif price_tick_calculated <= 5:
                            price_tick = max(1, round(price_tick_calculated))
                        elif price_tick_calculated <= 10:
                            price_tick = max(2, round(price_tick_calculated / 2) * 2)
                        else:
                            price_tick = max(5, round(price_tick_calculated / 5) * 5)
                        
                        price_axis_max = price_tick * target_ticks
                        
                        fig.update_yaxes(
                            title_text="Active Price ($)", 
                            secondary_y=True,
                            showgrid=False,
                            overlaying="y",
                            side="right",
                            dtick=price_tick,
                            range=[0, price_axis_max],  # Start at 0, use calculated max
                            color="rgba(32, 32, 32, 1)"
                        )
                    else:
                        fig.update_yaxes(
                            title_text="Active Price ($)", 
                            secondary_y=True,
                            showgrid=False,
                            overlaying="y",
                            side="right",
                            color="rgba(32, 32, 32, 1)"
                        )
                else:
                    fig.update_yaxes(
                        title_text="Active Price ($)", 
                        secondary_y=True,
                        showgrid=False,
                        overlaying="y",
                        side="right",
                        color="rgba(32, 32, 32, 1)"
                    )
                
                return fig
            
            # Create and display the daily financial chart
            fig_daily_fin_customer = create_daily_financial_chart_with_period_features(
                daily_data_customer, 
                item_df_customer,
                f"Daily Financial Metrics - {items_in_store_customer.loc[selected_item_customer,'name']}"
            )
            st.plotly_chart(fig_daily_fin_customer, use_container_width=True)
            
        else:
            st.info("No daily data available for this item and store combination.")
        
        # 2. Store Promotion Performance Overview (all 4 tabs)
        st.header(f"🏪 Store {selected_store_customer} - Promotion Performance Overview")
        
        # Calculate promotion frequency and total revenue lift for all items in the store (moved up here)
        promotion_impact_data = []
        
        for item_id in df_store_filtered_customer['item_id'].unique():
            item_data = df_store_filtered_customer[df_store_filtered_customer['item_id'] == item_id]
            promo_periods = item_data[item_data['is_on_promotion']]
            non_promo_periods = item_data[~item_data['is_on_promotion']]
            
            # Count promotion frequency
            promotion_count = len(promo_periods)
            
            # Calculate total revenue lift only if we have both promo and non-promo periods
            if not promo_periods.empty and not non_promo_periods.empty:
                # Sum of revenue during all promotion periods
                total_promo_revenue = promo_periods['total_revenue'].sum()
                # Sum of revenue during all non-promotion periods  
                total_non_promo_revenue = non_promo_periods['total_revenue'].sum()
                
                # Calculate total lift: (total_promo / total_non_promo - 1) * 100
                total_revenue_lift = ((total_promo_revenue / total_non_promo_revenue) - 1) * 100 if total_non_promo_revenue > 0 else 0
                
                # Calculate average lift per promotion to normalize for promotion count
                avg_lift_per_promo = total_revenue_lift / promotion_count if promotion_count > 0 else 0
                
                promotion_impact_data.append({
                    'Item ID': item_id,
                    'Item Name': item_data['name'].iloc[0],
                    'Promotion Count': promotion_count,
                    'Avg Lift per Promo %': avg_lift_per_promo,
                    'Avg Revenue per Promo': promo_periods['total_revenue'].mean(),
                    'Total Revenue': item_data['total_revenue'].sum()
                })
            elif promotion_count > 0:
                # Item has promotions but no non-promo periods for comparison
                promotion_impact_data.append({
                    'Item ID': item_id,
                    'Item Name': item_data['name'].iloc[0],
                    'Promotion Count': promotion_count,
                    'Avg Lift per Promo %': 0,  # Can't calculate lift without baseline
                    'Avg Revenue per Promo': promo_periods['total_revenue'].mean(),
                    'Total Revenue': item_data['total_revenue'].sum()
                })
        
        # Store Impact Overview - Scatter Plot
        if promotion_impact_data:
            promotion_impact_df = pd.DataFrame(promotion_impact_data)
            
            # Create scatter plot with better visibility and more vertical space
            fig_promotion_impact = px.scatter(
                promotion_impact_df,
                x='Promotion Count',
                y='Avg Lift per Promo %',
                size='Total Revenue',
                hover_data=['Item Name', 'Avg Revenue per Promo'],
                title="Store Items: Promotion Frequency vs Average Lift per Promotion",
                labels={
                    'Promotion Count': 'Number of Promotion Periods',
                    'Avg Lift per Promo %': 'Average Lift per Promotion (%)',
                    'Total Revenue': 'Total Revenue ($)'
                },
                color_discrete_sequence=['#00ff88']  # Bright green color
            )
            
            # Set y-axis range to focus on the actual data range (with some padding)
            y_min = promotion_impact_df['Avg Lift per Promo %'].min()
            y_max = promotion_impact_df['Avg Lift per Promo %'].max()
            y_range = y_max - y_min
            y_padding = max(y_range * 0.1, 10)  # At least 10% padding or 10 units
            
            # Add horizontal line at 0% lift
            fig_promotion_impact.add_hline(y=0, line_dash="dash", line_color="white", line_width=2, opacity=0.8)
            
            # Update layout for high contrast and readability with increased height
            fig_promotion_impact.update_layout(
                height=600,  # Increased from 450 for more vertical space
                showlegend=False,
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(size=14, color="white"),
                title=dict(font=dict(size=18, color="white"), x=0.5),
                margin=dict(l=80, r=80, t=80, b=80)
            )
            
            # Update axes for high contrast and focused range
            fig_promotion_impact.update_xaxes(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.3)",
                gridwidth=1,
                title_font=dict(size=16, color="white"),
                tickfont=dict(size=13, color="white"),
                color="white"
            )
            fig_promotion_impact.update_yaxes(
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.3)",
                gridwidth=1,
                title_font=dict(size=16, color="white"),
                tickfont=dict(size=13, color="white"),
                color="white",
                range=[y_min - y_padding, y_max + y_padding]  # Focus on actual data range
            )
            
            # Update markers for better visibility
            fig_promotion_impact.update_traces(
                marker=dict(
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    sizemin=8,  # Minimum marker size
                    sizemode='diameter'
                )
            )
            
            st.plotly_chart(fig_promotion_impact, use_container_width=True)
            
            # Simple insights for customers
            col1, col2, col3 = st.columns(3)
            
            # Filter out items with 0 lift (no comparison data)
            valid_impact_df = promotion_impact_df[promotion_impact_df['Avg Lift per Promo %'] != 0]
            
            with col1:
                if not valid_impact_df.empty:
                    best_performer = valid_impact_df.loc[valid_impact_df['Avg Lift per Promo %'].idxmax()]
                    st.metric(
                        "🏆 Best Promotion Performer", 
                        f"{best_performer['Item Name'][:20]}...",
                        f"+{best_performer['Avg Lift per Promo %']:.0f}% per promo"
                    )
                else:
                    st.metric("🏆 Best Promotion Performer", "No data available")
            
            with col2:
                if not valid_impact_df.empty:
                    most_promoted = promotion_impact_df.loc[promotion_impact_df['Promotion Count'].idxmax()]
                    st.metric(
                        "🎁 Most Promoted Item", 
                        f"{most_promoted['Item Name'][:20]}...",
                        f"{most_promoted['Promotion Count']} times"
                    )
                else:
                    st.metric("🎁 Most Promoted Item", "No data available")
            
            with col3:
                avg_lift_per_promotion = valid_impact_df['Avg Lift per Promo %'].mean() if not valid_impact_df.empty else 0
                st.metric(
                    "📊 Store Avg Lift per Promotion", 
                    f"{avg_lift_per_promotion:.1f}%",
                    "across all items"
                )
        else:
            st.info("No promotion data available for analysis.")
        
        # Detailed Analysis Tabs - Promotion Effectiveness moved to first position
        tab1_cust, tab2_cust, tab3_cust, tab4_cust = st.tabs(["💡 Promotion Effectiveness", "📈 Top Revenue Items", "🎁 Most Promoted Items", "🏆 Top Promotion Periods"])
            
        with tab1_cust:
            st.subheader("📊 Promotion Effectiveness")
            
            # Explanation of BB Promo Score
            st.info("""
            **🎯 BB Promo Score Methodology:**
            
            The BB Promo Score is a composite metric that combines four key promotion performance indicators:
            
            **Formula:** Score = 0.4×(Normalized Lift per Promo) + 0.3×(Normalized Total Revenue) + 0.2×(Normalized Delta Revenue) + 0.1×(Normalized Promo Count)
            
            **Normalization:** Each metric is normalized using min-max scaling: norm(value) = (value - min) / (max - min)
            
            **Weights:**
            - 40% Avg Lift per Promo (effectiveness per promotion)
            - 30% Total Revenue (absolute revenue impact) 
            - 20% Total Delta Revenue (incremental revenue gain)
            - 10% Promotion Count (frequency bonus)
            
            **Range:** 0.0 (worst performer) to 1.0 (best performer)
            """)
            
            effectiveness_data = []
            
            for item_id in df_store_filtered_customer['item_id'].unique():
                item_data = df_store_filtered_customer[df_store_filtered_customer['item_id'] == item_id].sort_values('period_id')
                promo_data = item_data[item_data['is_on_promotion']]
                non_promo_data = item_data[~item_data['is_on_promotion']]
                
                if not promo_data.empty and not non_promo_data.empty:
                    # Calculate total revenue lift using the same method as scatter plot
                    total_promo_revenue = promo_data['total_revenue'].sum()
                    total_non_promo_revenue = non_promo_data['total_revenue'].sum()
                    total_revenue_lift = ((total_promo_revenue / total_non_promo_revenue) - 1) * 100 if total_non_promo_revenue > 0 else 0
                    
                    # Calculate average lift per promotion to normalize for promotion count
                    promotion_count = len(promo_data)
                    avg_lift_per_promo = total_revenue_lift / promotion_count if promotion_count > 0 else 0
                    
                    # Calculate total delta revenue: total promo revenue - total non-promo revenue
                    total_delta_revenue = total_promo_revenue - total_non_promo_revenue
                    
                    effectiveness_data.append({
                        'Item ID': item_id,
                        'Name': item_data['name'].iloc[0],
                        'Avg Lift per Promo %': avg_lift_per_promo,
                        'Total Revenue ($)': total_promo_revenue,
                        'Total Delta Revenue ($)': total_delta_revenue,
                        'Promotion Periods': len(promo_data)
                    })
            
            if effectiveness_data:
                effectiveness_df = pd.DataFrame(effectiveness_data)
                
                # Calculate BB Promo Score using min-max normalization and weighted sum
                # Weights: Lift per Promo (0.4), Total Revenue (0.3), Delta Revenue (0.2), Promo Count (0.1)
                w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
                
                # Get clean data for normalization (no null values)
                clean_df = effectiveness_df.dropna(subset=['Avg Lift per Promo %'])
                
                if len(clean_df) > 1:  # Need at least 2 data points for normalization
                    # Extract metrics for normalization
                    lift_values = clean_df['Avg Lift per Promo %'].values
                    revenue_values = clean_df['Total Revenue ($)'].values
                    delta_values = clean_df['Total Delta Revenue ($)'].values
                    count_values = clean_df['Promotion Periods'].values
                    
                    # Min-max normalization function
                    def normalize(values):
                        min_val, max_val = values.min(), values.max()
                        if max_val == min_val:  # Avoid division by zero
                            return np.ones_like(values) * 0.5  # Return 0.5 for all if no variation
                        return (values - min_val) / (max_val - min_val)
                    
                    # Normalize each metric
                    norm_lift = normalize(lift_values)
                    norm_revenue = normalize(revenue_values)
                    norm_delta = normalize(delta_values)
                    norm_count = normalize(count_values)
                    
                    # Calculate BB Promo Score
                    bb_scores = w1 * norm_lift + w2 * norm_revenue + w3 * norm_delta + w4 * norm_count
                    
                    # Add scores back to the clean dataframe
                    clean_df = clean_df.copy()
                    clean_df['BB Promo Score'] = bb_scores
                    
                    # Reorder columns to put BB Promo Score after Name
                    cols = ['Item ID', 'Name', 'BB Promo Score', 'Avg Lift per Promo %', 'Total Revenue ($)', 'Total Delta Revenue ($)', 'Promotion Periods']
                    clean_df = clean_df[cols]
                    
                    # Sort by BB Promo Score (highest first)
                    effectiveness_df_sorted = clean_df.sort_values('BB Promo Score', ascending=False)
                    st.dataframe(effectiveness_df_sorted.round(3), use_container_width=True)
                    
                else:
                    # Fallback if insufficient data for scoring
                    effectiveness_df_sorted = clean_df.sort_values('Avg Lift per Promo %', ascending=False) if not clean_df.empty else clean_df
                    st.dataframe(effectiveness_df_sorted.round(2), use_container_width=True)
            else:
                st.info("Insufficient data to calculate promotion effectiveness")
            
        with tab2_cust:
            st.subheader("Top Revenue Items by Promotion Status")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎁 Items with Promotions**")
                promo_items = df_store_filtered_customer[df_store_filtered_customer['is_on_promotion']].groupby('item_id').agg({
                    'name': 'first',
                    'total_revenue': 'sum',
                    'total_profit': 'sum',
                    'total_units_sold': 'sum',
                    'promo_discount_pct': 'mean'
                }).sort_values('total_revenue', ascending=False).head(10)
                
                if not promo_items.empty:
                    st.dataframe(promo_items.round(2), use_container_width=True)
                else:
                    st.info("No promoted items found in this store")
                
            with col2:
                st.write("**📦 Items without Promotions**")
                non_promo_items = df_store_filtered_customer[~df_store_filtered_customer['is_on_promotion']].groupby('item_id').agg({
                    'name': 'first',
                    'total_revenue': 'sum',
                    'total_profit': 'sum',
                    'total_units_sold': 'sum'
                }).sort_values('total_revenue', ascending=False).head(10)
                
                if not non_promo_items.empty:
                    st.dataframe(non_promo_items.round(2), use_container_width=True)
                else:
                    st.info("No non-promoted items found in this store")
            
        with tab3_cust:
            st.subheader("Most Frequently Promoted Items")
            promotion_frequency = df_store_filtered_customer.groupby('item_id').agg({
                'name': 'first',
                'is_on_promotion': 'sum',
                'period_id': 'count',
                'total_revenue': 'sum',
                'promo_discount_pct': 'mean'
            }).rename(columns={'is_on_promotion': 'promotion_periods', 'period_id': 'total_periods'})
            
            promotion_frequency['promotion_rate'] = (promotion_frequency['promotion_periods'] / promotion_frequency['total_periods'] * 100)
            promotion_frequency = promotion_frequency.sort_values('promotion_periods', ascending=False).head(15)
            
            if not promotion_frequency.empty:
                st.dataframe(promotion_frequency.round(2), use_container_width=True)
            else:
                st.info("No promotion data available for this store")
                
        with tab4_cust:
            st.subheader("🏆 Top Performing Promotion Periods")
            
            # Calculate period-level performance across all items in the store
            promo_periods_agg_customer = df_store_filtered_customer[df_store_filtered_customer['is_on_promotion']].groupby('period_id').agg({
                'period_start_local': 'first',
                'total_revenue': 'sum',
                'total_profit': 'sum',
                'total_units_sold': 'sum',
                'total_promo_spending': 'sum',
                'promo_discount_pct': 'mean',
                'item_id': 'count'
            }).reset_index()
            
            promo_periods_agg_customer.columns = ['Period ID', 'Period Start', 'Total Revenue', 'Total Profit', 
                                        'Total Units Sold', 'Total Promo Spending', 'Avg Discount %', 'Items on Promo']
            promo_periods_agg_customer['Period Start'] = pd.to_datetime(promo_periods_agg_customer['Period Start']).dt.strftime('%Y-%m-%d')
            
            if not promo_periods_agg_customer.empty:
                col1, col2 = st.columns(2)
        
                with col1:
                    st.write("**🥇 Highest Total Revenue Periods**")
                    top_revenue_periods = promo_periods_agg_customer.sort_values('Total Revenue', ascending=False).head(10)
                    display_cols = ['Period ID', 'Period Start', 'Items on Promo', 'Avg Discount %', 'Total Revenue', 'Total Profit']
                    st.dataframe(top_revenue_periods[display_cols].round(2), use_container_width=True)
                
                with col2:
                    st.write("**💰 Highest Total Profit Periods**")
                    top_profit_periods = promo_periods_agg_customer.sort_values('Total Profit', ascending=False).head(10)
                    st.dataframe(top_profit_periods[display_cols].round(2), use_container_width=True)
            else:
                    st.info("No promotion periods found for this store.")
            
        # 3. Store promotion summary (without discount analysis)
        if not df_store_filtered_customer[df_store_filtered_customer['is_on_promotion']].empty:
            promo_periods_agg_summary = df_store_filtered_customer[df_store_filtered_customer['is_on_promotion']].groupby('period_id').agg({
                'period_start_local': 'first',
                'total_revenue': 'sum',
                'total_profit': 'sum',
                'total_units_sold': 'sum',
                'total_promo_spending': 'sum',
                'promo_discount_pct': 'mean',
                'item_id': 'count'
            }).reset_index()
            
            promo_periods_agg_summary.columns = ['Period ID', 'Period Start', 'Total Revenue', 'Total Profit', 
                                        'Total Units Sold', 'Total Promo Spending', 'Avg Discount %', 'Items on Promo']
            
            # Overall summary metrics
            st.subheader("📊 Store Promotion Summary")
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                total_promo_revenue = promo_periods_agg_summary['Total Revenue'].sum()
                st.metric("Total Promo Revenue", f"${total_promo_revenue:,.2f}")
            
            with col2:
                total_promo_profit = promo_periods_agg_summary['Total Profit'].sum()
                st.metric("Total Promo Profit", f"${total_promo_profit:,.2f}")
                    
            with col3:
                avg_revenue_per_period = promo_periods_agg_summary['Total Revenue'].mean()
                st.metric("Avg Revenue per Period", f"${avg_revenue_per_period:,.2f}")
            
            with col4:
                total_promo_periods_count = len(promo_periods_agg_summary)
                st.metric("Total Promo Periods", total_promo_periods_count)
        
            # Performance trends visualization
            st.subheader("📈 Promotion Period Performance Trends")
            
            # Create time series chart showing revenue and profit over periods
            fig_trends_customer = make_subplots(
                rows=1, cols=1,
                subplot_titles=('Revenue and Profit by Promotion Period',)
            )
            
            # Revenue trend
            fig_trends_customer.add_trace(
                go.Scatter(
                    x=promo_periods_agg_summary['Period ID'],
                    y=promo_periods_agg_summary['Total Revenue'],
                    mode='lines+markers',
                    name='Total Revenue',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                )
            )
            
            # Profit trend
            fig_trends_customer.add_trace(
                go.Scatter(
                    x=promo_periods_agg_summary['Period ID'],
                    y=promo_periods_agg_summary['Total Profit'],
                    mode='lines+markers',
                    name='Total Profit',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                )
            )
            
            fig_trends_customer.update_layout(
                height=400,
                title_text="Promotion Period Performance",
                showlegend=True,
                xaxis_title="Period ID",
                yaxis_title="Revenue/Profit ($)"
            )
            
            st.plotly_chart(fig_trends_customer, use_container_width=True)
            
            # 4. Complete Promotion Period Analysis
            st.subheader("📋 Complete Promotion Period Analysis")
            st.dataframe(promo_periods_agg_summary.sort_values('Total Revenue', ascending=False).round(2), use_container_width=True)
            
        else:
            st.info("No promotion data available for this store.")
    
    else:
        st.error(f"No data found for item {selected_item_customer} in store {selected_store_customer}")
