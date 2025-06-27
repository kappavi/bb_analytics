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

def load_data_in_chunks(query, chunksize=200_000):
    engine = get_engine()
    it = pd.read_sql(text(query), engine, chunksize=chunksize)
    return pd.concat(it, ignore_index=True)

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
def load_data():
    # This will only show progress when not cached
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text('Loading cleaned analysis data...')
    progress_bar.progress(25)
    query = "SELECT * FROM temporary.analysis_cleaned"
    # df = run_query(None, rawQuery=query)
    df = load_data_in_chunks(query)
    
    status_text.text('Loading raw staging data...')
    progress_bar.progress(75)
    query = "SELECT * FROM temporary.analysis_staging"
    # df_raw = run_query(None, rawQuery=query)
    df_raw = load_data_in_chunks(query)
    
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

# Create main tabs
main_tab, analytics_tab, customer_tab = st.tabs(["📊 Promotion Analysis", "🔬 Raw Data Analytics", "👥 For Customer"])

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

with main_tab:
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
    # Enhanced Metrics selectors - active_price as primary
    st.sidebar.subheader("📊 Price Metrics")
    price_metrics = ['active_price', 'avg_regular_price', 'avg_promo_price', 'avg_selling_price', 'avg_unit_cost']
    selected_price = st.sidebar.multiselect(
        "Price metrics to plot:", 
        price_metrics, 
        default=['active_price', 'avg_regular_price', 'avg_promo_price'],
        help="Active price is the primary metric - equals promo price when on promotion, regular price otherwise"
    )

    st.sidebar.subheader("💰 Financial Metrics")
    fin_metrics = ['total_revenue','total_profit','total_units_sold','weighted_average_promo_spending_ratio', 'active_price']
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

    # 4.5 — Period Length Analysis —
    if not item_df.empty:
        # Calculate period lengths
        item_df_copy = item_df.copy()
        item_df_copy['period_start_local'] = pd.to_datetime(item_df_copy['period_start_local'])
        item_df_copy['period_end_local'] = pd.to_datetime(item_df_copy['period_end_local'])
        item_df_copy['period_length_days'] = (item_df_copy['period_end_local'] - item_df_copy['period_start_local']).dt.days + 1  # +1 to include both start and end days
        
        # Check for variations in period length
        unique_lengths = item_df_copy['period_length_days'].unique()
        min_length = item_df_copy['period_length_days'].min()
        max_length = item_df_copy['period_length_days'].max()
        avg_length = item_df_copy['period_length_days'].mean()
        
        if len(unique_lengths) > 1: # TODO: remove this once we have a better way to handle period length variation
            st.warning(f"⚠️ **Period Length Variation Detected!**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min Period Length", f"{min_length} days")
            with col2:
                st.metric("Max Period Length", f"{max_length} days")
            with col3:
                st.metric("Avg Period Length", f"{avg_length:.1f} days")
            with col4:
                st.metric("Unique Lengths", len(unique_lengths))
            
            # Show detailed period length breakdown
            with st.expander("📊 Detailed Period Length Analysis"):
                period_lengths = item_df_copy[['period_id', 'period_start_local', 'period_end_local', 'period_length_days', 'is_on_promotion']].copy()
                period_lengths['period_start_local'] = period_lengths['period_start_local'].dt.strftime('%Y-%m-%d')
                period_lengths['period_end_local'] = period_lengths['period_end_local'].dt.strftime('%Y-%m-%d')
                period_lengths['promotion_status'] = period_lengths['is_on_promotion'].map({True: '🎁 Promo', False: '📦 Regular'})
                
                st.dataframe(
                    period_lengths[['period_id', 'period_start_local', 'period_end_local', 'period_length_days', 'promotion_status']].sort_values('period_id'),
                    use_container_width=True
                )
                
                # Length distribution chart
                import plotly.express as px
                fig_lengths = px.histogram(
                    period_lengths, 
                    x='period_length_days',
                    title="Distribution of Period Lengths",
                    labels={'period_length_days': 'Period Length (Days)', 'count': 'Number of Periods'},
                    nbins=20
                )
                st.plotly_chart(fig_lengths, use_container_width=True)
                
                # Length by period chart
                fig_period_lengths = px.bar(
                    period_lengths,
                    x='period_id',
                    y='period_length_days',
                    color='promotion_status',
                    title="Period Length by Period ID",
                    labels={'period_length_days': 'Length (Days)', 'period_id': 'Period ID'}
                )
                st.plotly_chart(fig_period_lengths, use_container_width=True)
        else:
            st.success(f"✅ **Consistent Period Lengths**: All periods are {unique_lengths[0]} days long")

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
        price_range = f"${item_df['active_price'].min():.2f} - ${item_df['active_price'].max():.2f}"
        st.metric("Price Range", price_range)
    with col4:
        price_logic_issues = (~item_df['price_logic_check']).sum()
        st.metric("Price Logic Issues", price_logic_issues, delta_color="inverse")

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
            
            # Only show hover info on the first metric to avoid duplication
            if i == 0:
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
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=extended_hover,
                    connectgaps=True
                )
            else:
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
                    hoverinfo='skip',  # Skip hover for other metrics
                    connectgaps=True
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
            line_width = 3 if row['is_on_promotion'] else 1
            
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
                annotation_text += f" 🎁 (-{row['promo_discount_pct']:.1f}%)"
            if row.get('active_price_changed', False):
                annotation_text += " ⚠️"
            
            fig.add_annotation(
                x=period_start,
                y=-0.1,  # Place at bottom of chart
                xref="x", yref="paper",
                text=annotation_text,
                showarrow=False,
                font=dict(size=10, color="rgba(32, 32, 32, 1)", weight="bold" if row['is_on_promotion'] else "normal"),
                bgcolor="white" if row['is_on_promotion'] else None,
                bordercolor="rgba(32, 32, 32, 1)" if row['is_on_promotion'] else None,
                borderwidth=1 if row['is_on_promotion'] else 0,
                xanchor="left",
                yanchor="top"
            )
        
        # Update layout with soft gridlines
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            hovermode='x unified',  # Unified hover for all traces at x position
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
                
                # Calculate number of ticks to make them similar
                price_ticks = max(5, int(price_range / price_tick) + 1)
                other_ticks = max(5, int(other_range / other_tick) + 1)
                
                # Adjust to have similar number of grid lines (5-7 lines)
                target_ticks = 6
                if price_ticks > target_ticks * 1.5:
                    price_tick = price_tick * 2
                if other_ticks > target_ticks * 1.5:
                    other_tick = other_tick * 2
                
                # Set the y-axis properties with shared soft gridlines
                fig.update_yaxes(
                    title_text="Financial Metrics ($)", 
                    secondary_y=False,
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",  # Soft gray with low opacity
                    gridwidth=1,
                    dtick=other_tick,
                    color="rgba(32, 32, 32, 1)"  # Darker gray axis text
                )
                fig.update_yaxes(
                    title_text="Price ($)", 
                    secondary_y=True,
                    showgrid=False,  # Don't show grid for secondary axis to avoid duplication
                    overlaying="y",
                    side="right",
                    dtick=price_tick,
                    color="rgba(32, 32, 32, 1)"  # Darker gray axis text
                )
            else:
                # Fallback if no data
                fig.update_yaxes(title_text="Financial Metrics ($)", secondary_y=False, color="rgba(32, 32, 32, 1)")
                fig.update_yaxes(title_text="Price ($)", secondary_y=True, color="rgba(32, 32, 32, 1)")
        else:
            fig.update_yaxes(title_text="Value", color="rgba(32, 32, 32, 1)")
        
        return fig

    # Price analysis with promotion highlighting
    if selected_price and not item_df_filtered.empty:
        price_ts = item_df_filtered.set_index('period_start_local')[selected_price]
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
            if 'active_price' in selected_price:
                active_price_volatility = item_df['active_price'].std()
                st.write(f"**Active Price Volatility:** ${active_price_volatility:.2f}")
                
                # Price trend
                if len(item_df) >= 2:
                    price_trend = ((item_df['active_price'].iloc[-1] / item_df['active_price'].iloc[0]) - 1) * 100
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
        fin_ts = item_df_filtered.set_index('period_start_local')[selected_fin]
        fig_fin = create_enhanced_chart_with_promotions(
            fin_ts, selected_fin, 
            "📊 Financial Performance (Red background = Promotion periods)", 
            item_df_filtered
        )
        st.plotly_chart(fig_fin, use_container_width=True)

    # 8. — Top Items Analysis (Store-level with promotion context) —
    st.header(f"🏪 Store {selected_store} - Promotion Performance Overview")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Top Revenue Items", "🎁 Most Promoted Items", "💡 Promotion Effectiveness", "🏆 Top Promotion Periods"])

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
                profit_lift = ((promo_data['total_profit'].mean() / regular_data['total_profit'].mean()) - 1) * 100
                avg_discount = promo_data['promo_discount_pct'].mean()
                
                # Calculate ROI
                avg_promo_spending = promo_data['total_promo_spending'].mean()
                promo_roi = (profit_lift / avg_discount) if avg_discount > 0 else 0
                
                effectiveness_data.append({
                    'Item ID': item_id,
                    'Name': item_data['name'].iloc[0],
                    'Avg Discount %': avg_discount,
                    'Revenue Lift %': revenue_lift,
                    'Units Lift %': units_lift,
                    'Profit Lift %': profit_lift,
                    'Promo ROI': promo_roi,
                    'Promo Periods': len(promo_data)
                })
            else:
                # Items with no comparison data go to bottom
                effectiveness_data.append({
                    'Item ID': item_id,
                    'Name': item_data['name'].iloc[0],
                    'Avg Discount %': 0,
                    'Revenue Lift %': None,
                    'Units Lift %': None,
                    'Profit Lift %': None,
                    'Promo ROI': None,
                    'Promo Periods': len(promo_data) if not promo_data.empty else 0
                })
        
        if effectiveness_data:
            effectiveness_df = pd.DataFrame(effectiveness_data)
            # Sort by Revenue Lift %, putting null values at bottom
            effectiveness_df = effectiveness_df.sort_values('Revenue Lift %', ascending=False, na_position='last')
            st.dataframe(effectiveness_df, use_container_width=True)
            
            # Enhanced visualization
            valid_data = effectiveness_df.dropna(subset=['Revenue Lift %'])
            if not valid_data.empty:
                fig_effectiveness = px.scatter(
                    valid_data,
                    x='Avg Discount %',
                    y='Revenue Lift %',
                    size='Promo Periods',
                    color='Profit Lift %',
                    hover_data=['Name', 'Units Lift %', 'Promo ROI'],
                    title="Enhanced Promotion Effectiveness: Discount vs Revenue Lift (Color = Profit Lift)",
                    labels={'Avg Discount %': 'Average Discount (%)', 'Revenue Lift %': 'Revenue Lift (%)'},
                    color_continuous_scale="RdYlGn"
                )
                fig_effectiveness.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_effectiveness.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_effectiveness, use_container_width=True)

    with tab4:
        st.subheader("🏆 Top Performing Promotion Periods")
        
        # Calculate period-level performance across all items in the store
        
        # 1. Get promotion periods aggregated by period_id
        promo_periods_agg = df_store_filtered[df_store_filtered['is_on_promotion']].groupby('period_id').agg({
            'period_start_local': 'first',
            'total_revenue': 'sum',
            'total_profit': 'sum',
            'total_units_sold': 'sum',
            'total_promo_spending': 'sum',
            'promo_discount_pct': 'mean',  # Average discount across all items in that period
            'item_id': 'count'  # Number of items on promotion
        }).reset_index()
        
        promo_periods_agg.columns = ['Period ID', 'Period Start', 'Total Revenue', 'Total Profit', 
                                    'Total Units Sold', 'Total Promo Spending', 'Avg Discount %', 'Items on Promo']
        promo_periods_agg['Period Start'] = pd.to_datetime(promo_periods_agg['Period Start']).dt.strftime('%Y-%m-%d')
        
        # 2. Calculate period-over-period lifts
        promo_periods_agg = promo_periods_agg.sort_values('Period ID')
        
        # Calculate lifts vs previous period
        promo_periods_agg['Revenue Lift vs Prev %'] = 0.0
        promo_periods_agg['Profit Lift vs Prev %'] = 0.0
        promo_periods_agg['Units Lift vs Prev %'] = 0.0
        
        for i in range(1, len(promo_periods_agg)):
            current_row = promo_periods_agg.iloc[i]
            prev_row = promo_periods_agg.iloc[i-1]
            
            # Revenue lift
            if prev_row['Total Revenue'] > 0:
                revenue_lift = ((current_row['Total Revenue'] / prev_row['Total Revenue']) - 1) * 100
                promo_periods_agg.iloc[i, promo_periods_agg.columns.get_loc('Revenue Lift vs Prev %')] = revenue_lift
            
            # Profit lift
            if prev_row['Total Profit'] > 0:
                profit_lift = ((current_row['Total Profit'] / prev_row['Total Profit']) - 1) * 100
                promo_periods_agg.iloc[i, promo_periods_agg.columns.get_loc('Profit Lift vs Prev %')] = profit_lift
            
            # Units lift
            if prev_row['Total Units Sold'] > 0:
                units_lift = ((current_row['Total Units Sold'] / prev_row['Total Units Sold']) - 1) * 100
                promo_periods_agg.iloc[i, promo_periods_agg.columns.get_loc('Units Lift vs Prev %')] = units_lift
        
        if not promo_periods_agg.empty:
            # Create two ranking tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🥇 Highest Total Revenue Periods**")
                top_revenue_periods = promo_periods_agg.sort_values('Total Revenue', ascending=False).head(10)
                display_cols = ['Period ID', 'Period Start', 'Items on Promo', 'Avg Discount %', 'Total Revenue', 'Total Profit']
                st.dataframe(top_revenue_periods[display_cols].round(2), use_container_width=True)
            
            with col2:
                st.write("**📈 Highest Period-over-Period Lift**")
                # Filter out first period (no previous period to compare)
                lift_data = promo_periods_agg[promo_periods_agg['Revenue Lift vs Prev %'] != 0]
                if not lift_data.empty:
                    top_lift_periods = lift_data.sort_values('Revenue Lift vs Prev %', ascending=False).head(10)
                    display_cols = ['Period ID', 'Period Start', 'Revenue Lift vs Prev %', 'Profit Lift vs Prev %', 'Units Lift vs Prev %', 'Total Revenue']
                    st.dataframe(top_lift_periods[display_cols].round(2), use_container_width=True)
                else:
                    st.write("No period-over-period data available")
            
            # Overall summary metrics
            st.subheader("📊 Store Promotion Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_promo_revenue = promo_periods_agg['Total Revenue'].sum()
                st.metric("Total Promo Revenue", f"${total_promo_revenue:,.2f}")
            
            with col2:
                total_promo_profit = promo_periods_agg['Total Profit'].sum()
                st.metric("Total Promo Profit", f"${total_promo_profit:,.2f}")
            
            with col3:
                avg_revenue_per_period = promo_periods_agg['Total Revenue'].mean()
                st.metric("Avg Revenue per Period", f"${avg_revenue_per_period:,.2f}")
            
            with col4:
                total_promo_periods = len(promo_periods_agg)
                st.metric("Total Promo Periods", total_promo_periods)
            
            # Performance trends visualization
            st.subheader("📈 Promotion Period Performance Trends")
            
            # Create time series chart showing revenue and profit over periods
            fig_trends = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Revenue by Promotion Period', 'Period-over-Period Lift %'),
                vertical_spacing=0.1
            )
            
            # Revenue trend
            fig_trends.add_trace(
                go.Scatter(
                    x=promo_periods_agg['Period ID'],
                    y=promo_periods_agg['Total Revenue'],
                    mode='lines+markers',
                    name='Total Revenue',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Profit trend
            fig_trends.add_trace(
                go.Scatter(
                    x=promo_periods_agg['Period ID'],
                    y=promo_periods_agg['Total Profit'],
                    mode='lines+markers',
                    name='Total Profit',
                    line=dict(color='green', width=3),
                    marker=dict(size=8),
                    yaxis='y2'
                ),
                row=1, col=1
            )
            
            # Lift trends (skip first period)
            lift_data = promo_periods_agg[promo_periods_agg['Revenue Lift vs Prev %'] != 0]
            if not lift_data.empty:
                fig_trends.add_trace(
                    go.Bar(
                        x=lift_data['Period ID'],
                        y=lift_data['Revenue Lift vs Prev %'],
                        name='Revenue Lift %',
                        marker=dict(color='lightblue'),
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                fig_trends.add_trace(
                    go.Bar(
                        x=lift_data['Period ID'],
                        y=lift_data['Profit Lift vs Prev %'],
                        name='Profit Lift %',
                        marker=dict(color='lightgreen'),
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            fig_trends.update_layout(
                height=600,
                title_text="Promotion Period Analysis",
                showlegend=True
            )
            
            fig_trends.update_xaxes(title_text="Period ID", row=2, col=1)
            fig_trends.update_yaxes(title_text="Revenue/Profit ($)", row=1, col=1)
            fig_trends.update_yaxes(title_text="Lift (%)", row=2, col=1)
            
            # Add zero line for lift chart
            if not lift_data.empty:
                fig_trends.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Detailed performance analysis
            st.subheader("📊 Promotion Performance vs Discount Analysis")
            
            fig_discount_analysis = px.scatter(
                promo_periods_agg,
                x='Avg Discount %',
                y='Total Revenue',
                size='Items on Promo',
                color='Total Profit',
                hover_data=['Period ID', 'Period Start', 'Total Units Sold'],
                title="Store-level: Discount % vs Total Revenue (Size = Items on Promo, Color = Total Profit)",
                labels={'Avg Discount %': 'Average Discount (%)', 'Total Revenue': 'Total Revenue ($)'},
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_discount_analysis, use_container_width=True)
            
            # Complete promotion periods table
            st.subheader("📋 Complete Promotion Period Analysis")
            st.dataframe(promo_periods_agg.sort_values('Total Revenue', ascending=False).round(2), use_container_width=True)
            
        else:
            st.info("No promotion periods found for this store.")

    # 9. — Download enhanced data —
    enhanced_item_df = item_df.copy()
    enhanced_item_df['promotion_status'] = enhanced_item_df['is_on_promotion'].map({True: 'On Promotion', False: 'Regular Price'})

    csv = enhanced_item_df.to_csv(index=False)
    st.download_button(
        "📥 Download Enhanced Analysis Data", 
        csv, 
        file_name=f"store_{selected_store}_item_{selected_item}_promotion_analysis.csv"
    )

    # 6. — Enhanced Period Comparison with Promotion Context —
    col1, col2 = st.columns(2)
    with col1:
        prev_period = st.selectbox("Previous Period:", options=periods, index=0, key="prev_main")
    with col2:
        curr_period = st.selectbox("Current Period:", options=periods, index=len(periods)-1, key="curr_main")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Metric Comparison & Lift Analysis")
        if prev_period in item_df['period_id'].values and curr_period in item_df['period_id'].values:
            prev_data = item_df[item_df['period_id'] == prev_period].iloc[0]
            curr_data = item_df[item_df['period_id'] == curr_period].iloc[0]
            
            # Enhanced comparison with promotion context
            comparison_metrics = ['active_price', 'avg_regular_price', 'avg_promo_price', 'total_revenue', 
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
                promo_summary = promo_periods[['period_id', 'period_start_local', 'promo_discount_pct', 
                                            'total_revenue', 'total_profit', 'total_units_sold']].copy()
                promo_summary['period_start_local'] = pd.to_datetime(promo_summary['period_start_local']).dt.strftime('%Y-%m-%d')
                st.dataframe(promo_summary, use_container_width=True)

            else:
                if promo_periods.empty:
                    st.info("🔍 No promotion periods found for this item in the selected time range.")
                if non_promo_periods.empty:
                    st.info("🔍 No non-promotion periods found for this item in the selected time range.")

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
        st.subheader("💰 Financial Performance")
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
                colors = ['blue', 'green', 'red']
                
                for i, metric in enumerate(daily_metrics):
                    if metric in daily_data.columns:
                        # Create hover text for each day
                        hover_texts = []
                        for _, day_row in daily_data.iterrows():
                            day_str = day_row['day'].strftime('%Y-%m-%d')
                            if day_str in period_hover_map:
                                hover_texts.append(period_hover_map[day_str])
                            else:
                                hover_texts.append(f"<b>Date: {day_str}</b><br>No period data available")
                        
                        # Determine which y-axis to use
                        secondary_y = (metric == 'active_price')
                        
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
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
                
                # Set y-axis properties
                fig.update_yaxes(
                    title_text="Financial Metrics ($)", 
                    secondary_y=False,
                    showgrid=True,
                    gridcolor="rgba(128, 128, 128, 0.2)",
                    gridwidth=1,
                    color="rgba(32, 32, 32, 1)"
                )
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
        
        # 2. Store-wide Promotion Impact Analysis
        st.subheader("🏪 Store Promotion Impact Overview")
        
        # Calculate promotion frequency and total revenue lift for all items in the store
        promotion_impact_data = []
        
        for item_id in df_store_filtered_customer['item_id'].unique():
            item_data = df_store_filtered_customer[df_store_filtered_customer['item_id'] == item_id]
            promo_periods = item_data[item_data['is_on_promotion']]
            non_promo_periods = item_data[~item_data['is_on_promotion']]
            
            # Count promotion frequency
            promotion_count = len(promo_periods)
            
            # Calculate total revenue lift only if we have both promo and non-promo periods
            if not promo_periods.empty and not non_promo_periods.empty:
                # Average revenue during non-promotion periods
                avg_non_promo_revenue = non_promo_periods['total_revenue'].mean()
                
                # Calculate lift for each promotion period and sum them
                total_revenue_lift = 0
                for _, promo_period in promo_periods.iterrows():
                    promo_revenue = promo_period['total_revenue']
                    if avg_non_promo_revenue > 0:
                        period_lift = ((promo_revenue / avg_non_promo_revenue) - 1) * 100
                        total_revenue_lift += period_lift
                
                promotion_impact_data.append({
                    'Item ID': item_id,
                    'Item Name': item_data['name'].iloc[0],
                    'Promotion Count': promotion_count,
                    'Total Revenue Lift %': total_revenue_lift,
                    'Avg Revenue per Promo': promo_periods['total_revenue'].mean(),
                    'Total Revenue': item_data['total_revenue'].sum()
                })
            elif promotion_count > 0:
                # Item has promotions but no non-promo periods for comparison
                promotion_impact_data.append({
                    'Item ID': item_id,
                    'Item Name': item_data['name'].iloc[0],
                    'Promotion Count': promotion_count,
                    'Total Revenue Lift %': 0,  # Can't calculate lift without baseline
                    'Avg Revenue per Promo': promo_periods['total_revenue'].mean(),
                    'Total Revenue': item_data['total_revenue'].sum()
                })
        
        if promotion_impact_data:
            promotion_impact_df = pd.DataFrame(promotion_impact_data)
            
            # Create scatter plot with better visibility
            fig_promotion_impact = px.scatter(
                promotion_impact_df,
                x='Promotion Count',
                y='Total Revenue Lift %',
                size='Total Revenue',
                hover_data=['Item Name', 'Avg Revenue per Promo'],
                title="Store Items: Promotion Frequency vs Total Revenue Lift",
                labels={
                    'Promotion Count': 'Number of Promotion Periods',
                    'Total Revenue Lift %': 'Total Revenue Lift (%)',
                    'Total Revenue': 'Total Revenue ($)'
                },
                color_discrete_sequence=['#00ff88']  # Bright green color
            )
            
            # Set y-axis range to focus on the actual data range (with some padding)
            y_min = promotion_impact_df['Total Revenue Lift %'].min()
            y_max = promotion_impact_df['Total Revenue Lift %'].max()
            y_range = y_max - y_min
            y_padding = max(y_range * 0.1, 10)  # At least 10% padding or 10 units
            
            # Add horizontal line at 0% lift
            fig_promotion_impact.add_hline(y=0, line_dash="dash", line_color="white", line_width=2, opacity=0.8)
            
            # Update layout for high contrast and readability
            fig_promotion_impact.update_layout(
                height=450,
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
            valid_impact_df = promotion_impact_df[promotion_impact_df['Total Revenue Lift %'] != 0]
            
            with col1:
                if not valid_impact_df.empty:
                    best_performer = valid_impact_df.loc[valid_impact_df['Total Revenue Lift %'].idxmax()]
                    st.metric(
                        "🏆 Best Promotion Performer", 
                        f"{best_performer['Item Name'][:20]}...",
                        f"+{best_performer['Total Revenue Lift %']:.0f}% total lift"
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
                avg_lift_per_promotion = valid_impact_df['Total Revenue Lift %'].sum() / valid_impact_df['Promotion Count'].sum() if not valid_impact_df.empty and valid_impact_df['Promotion Count'].sum() > 0 else 0
                st.metric(
                    "📊 Avg Lift per Promotion", 
                    f"{avg_lift_per_promotion:.1f}%",
                    "across all items"
                )
        else:
            st.info("No promotion data available for analysis.")
            
        # 3. Store Promotion Performance Overview (all 4 tabs)
        st.header(f"🏪 Store {selected_store_customer} - Promotion Performance Overview")
        
        tab1_cust, tab2_cust, tab3_cust, tab4_cust = st.tabs(["📈 Top Revenue Items", "🎁 Most Promoted Items", "💡 Promotion Effectiveness", "🏆 Top Promotion Periods"])
            
        with tab1_cust:
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
            
        with tab2_cust:
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
                
        with tab3_cust:
            st.subheader("📊 Promotion Effectiveness")
            effectiveness_data = []
            
            for item_id in df_store_filtered_customer['item_id'].unique():
                item_data = df_store_filtered_customer[df_store_filtered_customer['item_id'] == item_id]
                promo_data = item_data[item_data['is_on_promotion']]
                non_promo_data = item_data[~item_data['is_on_promotion']]
                
                if not promo_data.empty and not non_promo_data.empty:
                    avg_promo_revenue = promo_data['total_revenue'].mean()
                    avg_non_promo_revenue = non_promo_data['total_revenue'].mean()
                    revenue_lift = ((avg_promo_revenue / avg_non_promo_revenue) - 1) * 100 if avg_non_promo_revenue > 0 else 0
                    
                    avg_promo_profit = promo_data['total_profit'].mean()
                    avg_non_promo_profit = non_promo_data['total_profit'].mean()
                    profit_lift = ((avg_promo_profit / avg_non_promo_profit) - 1) * 100 if avg_non_promo_profit > 0 else 0
                    
                    effectiveness_data.append({
                        'Item ID': item_id,
                        'Name': item_data['name'].iloc[0],
                        'Revenue Lift %': revenue_lift,
                        'Profit Lift %': profit_lift,
                        'Avg Discount %': promo_data['promo_discount_pct'].mean(),
                        'Promotion Periods': len(promo_data)
                    })
            
            if effectiveness_data:
                effectiveness_df = pd.DataFrame(effectiveness_data)
                # Sort by revenue lift but put null lifts at bottom
                effectiveness_df_sorted = effectiveness_df.dropna(subset=['Revenue Lift %']).sort_values('Revenue Lift %', ascending=False)
                st.dataframe(effectiveness_df_sorted.round(2), use_container_width=True)
            else:
                st.info("Insufficient data to calculate promotion effectiveness")
                
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
