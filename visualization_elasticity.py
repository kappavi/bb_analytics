import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import defaultdict

@st.cache_data
def load_data():
    return pd.read_csv('data/price_grouped_hmart.csv')

@st.cache_data
def load_category_data():
    """Load category aggregated data"""
    return pd.read_csv('data/category_grouped_hmart.csv')

@st.cache_data
def compute_item_metrics(df):
    """Compute elasticity and price point metrics for all items"""
    metrics = []
    categories_map = defaultdict(list)
    
    for item_id in df['item_id'].unique():
        grp = df[df['item_id'] == item_id].copy()
        
        # Skip items with insufficient data
        if len(grp) < 3:
            continue
            
        try:
            # Calculate elasticity (beta)
            grp['log_P'] = np.log(grp['active_price'])
            grp['log_Q'] = np.log(grp['avg_units_sold'])
            
            # Check for invalid log values (inf, -inf, or NaN)
            if not (np.isfinite(grp['log_P']).all() and np.isfinite(grp['log_Q']).all()):
                continue
                
            X = pd.DataFrame({'log_P': grp['log_P']})
            X = sm.add_constant(X)
            y = grp['log_Q']
            model = sm.OLS(y, X).fit()
            alpha = model.params['const']
            beta = model.params['log_P']
            r_squared = model.rsquared
            
            # Validate that our regression results are finite numbers
            if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(r_squared)):
                continue
            
            # Count unique price points
            unique_prices = grp['active_price'].nunique()
            
            # Get item name
            item_name = grp['name'].iloc[0]
            
            # Handle categories - allow multiple categories per item
            item_categories = []
            if 'categories' in grp.columns:
                categories_raw = grp['categories'].iloc[0]
                if isinstance(categories_raw, list):
                    item_categories = categories_raw
                elif pd.notna(categories_raw):
                    # If it's a string, try to parse it or treat as single category
                    if isinstance(categories_raw, str) and ',' in categories_raw:
                        item_categories = [cat.strip() for cat in categories_raw.split(',')]
                    else:
                        item_categories = [str(categories_raw)]
                else:
                    item_categories = ['Other']
            else:
                item_categories = ['Other']
            
            # If no categories found, default to 'Other'
            if not item_categories:
                item_categories = ['Other']
            
            # Add item to categories_map for all its categories (for category analysis)
            for cat in item_categories:
                categories_map[cat].append(item_id)
            
            # Create only ONE row per item (not per category)
            metrics.append({
                'item_id': item_id,
                'name': item_name,
                'categories': ', '.join(item_categories),  # All categories in one column
                'elasticity': beta,  # This is beta (elasticity)
                'alpha': alpha,      # This is alpha (constant)
                'r_squared': r_squared,
                'unique_prices': unique_prices
            })
        except:
            # Skip items where log-log regression fails
            continue
    
    return pd.DataFrame(metrics), dict(categories_map)

@st.cache_data
def compute_category_elasticity(df, metrics_df, categories_map):
    """Compute average elasticity by category"""
    category_stats = []
    
    for category, item_ids in categories_map.items():
        # Get metrics for items in this category
        category_items = metrics_df[metrics_df['item_id'].isin(item_ids)]
        
        if len(category_items) == 0:
            continue
        
        # Calculate average elasticity for the category
        avg_elasticity = category_items['elasticity'].mean()
        median_elasticity = category_items['elasticity'].median()
        std_elasticity = category_items['elasticity'].std()
        item_count = len(category_items)  # Number of unique items in this category
        
        # Calculate weighted average by number of price points
        if 'unique_prices' in category_items.columns:
            weights = category_items['unique_prices']
            weighted_avg_elasticity = (category_items['elasticity'] * weights).sum() / weights.sum()
        else:
            weighted_avg_elasticity = avg_elasticity
        
        category_stats.append({
            'category': category,
            'item_count': item_count,
            'avg_elasticity': avg_elasticity,
            'median_elasticity': median_elasticity,
            'std_elasticity': std_elasticity,
            'weighted_avg_elasticity': weighted_avg_elasticity
        })
    
    return pd.DataFrame(category_stats).sort_values('avg_elasticity')

df = load_data()
metrics_df, categories_map = compute_item_metrics(df)

# Load category aggregated data
df_category_agg = load_category_data()

# Filter out rows with NaN values in key columns to prevent undefined values in sorting
metrics_df = metrics_df.dropna(subset=['elasticity', 'alpha', 'r_squared'])
st.write(f"📈 Loaded {len(metrics_df)} valid items with complete elasticity data")

category_elasticity_df = compute_category_elasticity(df, metrics_df, categories_map)

st.set_page_config(
    page_title="Demand Curve and Optimal Price Explorer",
    layout="wide",              # <-- switch to wide mode
    initial_sidebar_state="expanded"
)

st.title("Demand Curve and Optimal Price Explorer")

# Add tabs for different views
tab1, tab2, tab3 = st.tabs(["Item Analysis", "Category Analysis", "Categories Map"])

with tab1:
    # Sidebar for sorting and filtering
    st.sidebar.header("Item Ranking & Filtering")

    # Sorting options
    sort_by = st.sidebar.selectbox(
        "Sort items by:",
        ["Item ID", "Elasticity (most elastic)", "Elasticity (least elastic)", 
         "Most price points", "Fewest price points", "Best R²", "Highest Alpha", 
         "Lowest Alpha", "Category"]
    )

    # Apply sorting
    if sort_by == "Elasticity (most elastic)":
        metrics_df_sorted = metrics_df.sort_values('elasticity', ascending=True, na_position='last')  # Most negative = most elastic
    elif sort_by == "Elasticity (least elastic)":
        metrics_df_sorted = metrics_df.sort_values('elasticity', ascending=False, na_position='last')  # Least negative = least elastic
    elif sort_by == "Most price points":
        metrics_df_sorted = metrics_df.sort_values('unique_prices', ascending=False, na_position='last')
    elif sort_by == "Fewest price points":
        metrics_df_sorted = metrics_df.sort_values('unique_prices', ascending=True, na_position='last')
    elif sort_by == "Best R²":
        metrics_df_sorted = metrics_df.sort_values('r_squared', ascending=False, na_position='last')
    elif sort_by == "Highest Alpha":
        metrics_df_sorted = metrics_df.sort_values('alpha', ascending=False, na_position='last')
    elif sort_by == "Lowest Alpha":
        metrics_df_sorted = metrics_df.sort_values('alpha', ascending=True, na_position='last')
    elif sort_by == "Category":
        metrics_df_sorted = metrics_df.sort_values(['categories', 'item_id'], na_position='last')
    else:  # Item ID
        metrics_df_sorted = metrics_df.sort_values('item_id', na_position='last')

    # Filter by minimum price points
    min_price_points = st.sidebar.slider(
        "Minimum unique price points:",
        min_value=int(metrics_df['unique_prices'].min()),
        max_value=int(metrics_df['unique_prices'].max()),
        value=3
    )

    # Filter by category - extract unique categories from the comma-separated values
    all_categories = set()
    for categories_str in metrics_df['categories'].dropna():
        for cat in categories_str.split(', '):
            all_categories.add(cat.strip())
    
    categories = ['All'] + sorted(all_categories)
    selected_category = st.sidebar.selectbox("Filter by category:", categories)

    filtered_df = metrics_df_sorted[metrics_df_sorted['unique_prices'] >= min_price_points]
    if selected_category != 'All':
        # Filter items that contain the selected category
        filtered_df = filtered_df[filtered_df['categories'].str.contains(selected_category, na=False)]

    # Display item ranking metrics
    st.subheader("Item Rankings")
    
    display_cols = ['item_id', 'name', 'categories', 'elasticity', 'alpha', 'r_squared', 'unique_prices']
    
    # Add download button for the table
    col1, col2 = st.columns([1, 4])
    with col1:
        csv_data = filtered_df[display_cols].round(3).to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"item_rankings_{selected_category.lower().replace(' ', '_')}_min{min_price_points}pts.csv",
            mime="text/csv",
            help="Download the current filtered table as a CSV file"
        )
    with col2:
        st.write("💡 **Tip:** Click on any row to automatically view that item's graph below!")
    
    # Interactive dataframe with row selection
    selection = st.dataframe(
        filtered_df[display_cols].round(3),
        use_container_width=True,
        height=400,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "item_id": st.column_config.NumberColumn("Item ID", width="small"),
            "name": st.column_config.TextColumn("Product Name", width="large"),
            "categories": st.column_config.TextColumn("Categories", width="large"),
            "elasticity": st.column_config.NumberColumn("Elasticity (β)", width="small", format="%.3f"),
            "alpha": st.column_config.NumberColumn("Alpha (α)", width="small", format="%.3f"),
            "r_squared": st.column_config.NumberColumn("R²", width="small", format="%.3f"),
            "unique_prices": st.column_config.NumberColumn("Price Points", width="small")
        }
    )

    # Determine selected item - either from table selection or dropdown fallback
    if selection.selection.rows:
        # Get the selected row index and corresponding item_id
        selected_row_idx = selection.selection.rows[0]
        selected_item_id = filtered_df.iloc[selected_row_idx]['item_id']
        st.success(f"📊 Showing analysis for Item {selected_item_id}: {filtered_df.iloc[selected_row_idx]['name']}")
    else:
        # Fallback to dropdown selection if no row is selected
        st.info("👆 Click on a row above to select an item, or use the dropdown below:")
        unique_items = filtered_df.drop_duplicates('item_id')
        item_options = [f"{row['item_id']}: {row['name']}" for _, row in unique_items.iterrows()]
        item_display = st.selectbox("Select Item:", item_options)
        
        # Extract item_id from selection
        if item_display:
            selected_item_id = int(item_display.split(':')[0])
        else:
            selected_item_id = unique_items.iloc[0]['item_id']

    # Get selected item data (get first row for this item_id)
    selected_metrics = metrics_df[metrics_df['item_id'] == selected_item_id].iloc[0]

    # 2) isolate its blocks
    grp = df[df['item_id'] == selected_item_id].copy()
    P_obs = grp['active_price'].values
    Q_obs = grp['avg_units_sold'].values
    R_obs = P_obs * Q_obs

    # 3) fit the log–log power law
    grp['log_P'] = np.log(grp['active_price'])
    grp['log_Q'] = np.log(grp['avg_units_sold'])
    X = pd.DataFrame({'log_P': grp['log_P']})
    X = sm.add_constant(X)
    y = grp['log_Q']
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    beta  = model.params['log_P']

    # 4) predict on observed range for fitted curve
    P_grid = np.linspace(P_obs.min(), P_obs.max(), 300)
    Q_fit = np.exp(alpha) * P_grid**beta
    R_fit = P_grid * Q_fit

    # 5) LOWESS smoothing of actual points
    smoothed = lowess(Q_obs, P_obs, frac=0.6, return_sorted=True)
    P_lo, Q_lo = smoothed[:,0], smoothed[:,1]
    R_lo = P_lo * Q_lo

    # 6) find optima for each method
    #   a) observed
    idx_obs = np.argmax(R_obs)
    P_obs_best, Q_obs_best, R_obs_best = P_obs[idx_obs], Q_obs[idx_obs], R_obs[idx_obs]
    #   b) fitted
    idx_fit = np.argmax(R_fit)
    P_fit_best, Q_fit_best, R_fit_best = P_grid[idx_fit], Q_fit[idx_fit], R_fit[idx_fit]
    #   c) LOWESS
    idx_lo = np.argmax(R_lo)
    P_lo_best, Q_lo_best, R_lo_best = P_lo[idx_lo], Q_lo[idx_lo], R_lo[idx_lo]

    # 7) plot everything - made smaller
    with st.container():
        fig, ax = plt.subplots(figsize=(16, 8))  # Reduced from (20, 10)
        # observed scatter
        ax.scatter(P_obs, Q_obs, label='Observed data', color='C0', alpha=0.7, s=80)
        # fitted curve
        ax.plot(P_grid, Q_fit,   label='Log–Log fit',      color='C1', linewidth=3)
        # LOWESS curve
        ax.plot(P_lo,   Q_lo,    label='LOWESS smooth',    color='C2', linestyle='--', linewidth=3)
        # annotate each optimum with arrows
        # Calculate offset for arrow positioning (10% of axis range)
        x_range = P_obs.max() - P_obs.min()
        y_range = Q_obs.max() - Q_obs.min()
        arrow_offset_x = x_range * 0.1
        arrow_offset_y = y_range * 0.1
        
        # Observed optimum arrow
        ax.annotate('Obs-opt', xy=(P_obs_best, Q_obs_best), 
                   xytext=(P_obs_best + arrow_offset_x, Q_obs_best + arrow_offset_y),
                   arrowprops=dict(arrowstyle='->', color='C0', lw=2),
                   fontsize=12, color='C0', fontweight='bold')
        
        # Fitted optimum arrow
        ax.annotate('Fit-opt', xy=(P_fit_best, Q_fit_best), 
                   xytext=(P_fit_best - arrow_offset_x, Q_fit_best + arrow_offset_y),
                   arrowprops=dict(arrowstyle='->', color='C1', lw=2),
                   fontsize=12, color='C1', fontweight='bold')
        
        # LOWESS optimum arrow
        ax.annotate('LOWESS-opt', xy=(P_lo_best, Q_lo_best), 
                   xytext=(P_lo_best + arrow_offset_x, Q_lo_best - arrow_offset_y),
                   arrowprops=dict(arrowstyle='->', color='C2', lw=2),
                   fontsize=12, color='C2', fontweight='bold')

        ax.set_xlabel('Price', fontsize=14)
        ax.set_ylabel('Avg Units Sold', fontsize=14)
        ax.set_title(f'Item {selected_item_id}: {selected_metrics["name"]} ({selected_metrics["categories"]})\n'
                     f'Elasticity: {beta:.3f} | R²: {model.rsquared:.3f} | Price Points: {selected_metrics["unique_prices"]}', 
                     fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # 8) display all three optima and item details
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Optimal Pricing Analysis")
        st.markdown(f"""
        **Observed-optimum:**  
        • Price = ${P_obs_best:.2f}  
        • Units = {Q_obs_best:.1f}  
        • Revenue = ${R_obs_best:.2f}  

        **Fitted-optimum (log–log):**  
        • Price = ${P_fit_best:.2f}  
        • Units = {Q_fit_best:.1f}  
        • Revenue = ${R_fit_best:.2f}  

        **LOWESS-optimum:**  
        • Price = ${P_lo_best:.2f}  
        • Units = {Q_lo_best:.1f}  
        • Revenue = ${R_lo_best:.2f}
        """)

    with col2:
        st.markdown("### Item Characteristics")
        elasticity_description = "Highly elastic" if beta < -1.5 else "Moderately elastic" if beta < -1 else "Inelastic"
        st.markdown(f"""
        **Product:** {selected_metrics["name"]}  
        **Categories:** {selected_metrics["categories"]}  
        **Item ID:** {selected_item_id}  
        **Price Elasticity:** {beta:.3f} ({elasticity_description})  
        **Model Fit (R²):** {model.rsquared:.3f}  
        **Unique Price Points:** {selected_metrics["unique_prices"]}  
        **Price Range:** ${P_obs.min():.2f} - ${P_obs.max():.2f}
        """)

with tab2:
    st.subheader("Category Time-Series Elasticity Analysis")
    
    # Filter out NaN values from category data
    df_category_clean = df_category_agg.dropna(subset=['avg_price', 'avg_qty'])
    
    # Category selection
    available_categories = sorted(df_category_clean['specific_category'].unique())
    selected_category = st.selectbox(
        "Select a category to analyze:",
        available_categories,
        help="Choose a category to see its price-quantity relationship over time"
    )
    
    # Filter data for selected category
    cat_data = df_category_clean[df_category_clean['specific_category'] == selected_category].copy()
    cat_data = cat_data.sort_values('day')
    
    if len(cat_data) < 3:
        st.warning(f"Not enough data points for {selected_category}. Need at least 3 data points for analysis.")
    else:
        # Convert day to datetime for better plotting
        cat_data['day'] = pd.to_datetime(cat_data['day'])
        
        # Calculate elasticity using log-log regression
        try:
            # Filter out zero or negative values for log transformation
            valid_data = cat_data[(cat_data['avg_price'] > 0) & (cat_data['avg_qty'] > 0)].copy()
            
            if len(valid_data) < 3:
                st.warning(f"Not enough valid data points for {selected_category} after filtering zero/negative values.")
            else:
                # Log transformation
                valid_data['log_P'] = np.log(valid_data['avg_price'])
                valid_data['log_Q'] = np.log(valid_data['avg_qty'])
                
                # Regression analysis
                X = pd.DataFrame({'log_P': valid_data['log_P']})
                X = sm.add_constant(X)
                y = valid_data['log_Q']
                model = sm.OLS(y, X).fit()
                alpha = model.params['const']
                beta = model.params['log_P']
                r_squared = model.rsquared
                
                # Create price grid for fitted curve
                P_grid = np.linspace(valid_data['avg_price'].min(), valid_data['avg_price'].max(), 300)
                Q_fit = np.exp(alpha) * P_grid**beta
                R_fit = P_grid * Q_fit
                
                # LOWESS smoothing
                if len(valid_data) >= 5:  # Need enough points for LOWESS
                    smoothed = lowess(valid_data['avg_qty'], valid_data['avg_price'], frac=0.6, return_sorted=True)
                    P_lo, Q_lo = smoothed[:,0], smoothed[:,1]
                    R_lo = P_lo * Q_lo
                else:
                    P_lo, Q_lo, R_lo = None, None, None
                
                # Find optima
                # Observed optimum
                valid_data['revenue'] = valid_data['avg_price'] * valid_data['avg_qty']
                idx_obs = valid_data['revenue'].idxmax()
                P_obs_best = valid_data.loc[idx_obs, 'avg_price']
                Q_obs_best = valid_data.loc[idx_obs, 'avg_qty']
                R_obs_best = valid_data.loc[idx_obs, 'revenue']
                
                # Fitted optimum
                idx_fit = np.argmax(R_fit)
                P_fit_best, Q_fit_best, R_fit_best = P_grid[idx_fit], Q_fit[idx_fit], R_fit[idx_fit]
                
                # LOWESS optimum
                if P_lo is not None:
                    idx_lo = np.argmax(R_lo)
                    P_lo_best, Q_lo_best, R_lo_best = P_lo[idx_lo], Q_lo[idx_lo], R_lo[idx_lo]
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time series plot
                    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
                    
                    # Price over time
                    ax1.plot(cat_data['day'], cat_data['avg_price'], marker='o', linewidth=2, markersize=6)
                    ax1.set_ylabel('Average Price ($)')
                    ax1.set_title(f'{selected_category} - Price Over Time')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)
                    
                    # Quantity over time
                    ax2.plot(cat_data['day'], cat_data['avg_qty'], marker='s', color='orange', linewidth=2, markersize=6)
                    ax2.set_ylabel('Average Quantity')
                    ax2.set_title(f'{selected_category} - Quantity Over Time')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
                    
                    # Revenue over time
                    cat_data['revenue'] = cat_data['avg_price'] * cat_data['avg_qty']
                    ax3.plot(cat_data['day'], cat_data['revenue'], marker='D', color='green', linewidth=2, markersize=6)
                    ax3.set_ylabel('Total Revenue ($)')
                    ax3.set_xlabel('Date')
                    ax3.set_title(f'{selected_category} - Revenue Over Time')
                    ax3.tick_params(axis='x', rotation=45)
                    ax3.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                with col2:
                    # Demand curve analysis
                    fig2, ax = plt.subplots(figsize=(12, 8))
                    
                    # Scatter plot of observed points
                    ax.scatter(valid_data['avg_price'], valid_data['avg_qty'], 
                              label='Observed data', color='C0', alpha=0.7, s=80)
                    
                    # Fitted curve
                    ax.plot(P_grid, Q_fit, label='Log-Log fit', color='C1', linewidth=3)
                    
                    # LOWESS curve if available
                    if P_lo is not None:
                        ax.plot(P_lo, Q_lo, label='LOWESS smooth', color='C2', linestyle='--', linewidth=3)
                    
                    # Mark optima with arrows
                    # Calculate offset for arrow positioning (10% of axis range)
                    x_range = valid_data['avg_price'].max() - valid_data['avg_price'].min()
                    y_range = valid_data['avg_qty'].max() - valid_data['avg_qty'].min()
                    arrow_offset_x = x_range * 0.1
                    arrow_offset_y = y_range * 0.1
                    
                    # Observed optimum arrow
                    ax.annotate('Obs-opt', xy=(P_obs_best, Q_obs_best), 
                               xytext=(P_obs_best + arrow_offset_x, Q_obs_best + arrow_offset_y),
                               arrowprops=dict(arrowstyle='->', color='C0', lw=2),
                               fontsize=12, color='C0', fontweight='bold')
                    
                    # Fitted optimum arrow
                    ax.annotate('Fit-opt', xy=(P_fit_best, Q_fit_best), 
                               xytext=(P_fit_best - arrow_offset_x, Q_fit_best + arrow_offset_y),
                               arrowprops=dict(arrowstyle='->', color='C1', lw=2),
                               fontsize=12, color='C1', fontweight='bold')
                    
                    # LOWESS optimum arrow (if available)
                    if P_lo is not None:
                        ax.annotate('LOWESS-opt', xy=(P_lo_best, Q_lo_best), 
                                   xytext=(P_lo_best + arrow_offset_x, Q_lo_best - arrow_offset_y),
                                   arrowprops=dict(arrowstyle='->', color='C2', lw=2),
                                   fontsize=12, color='C2', fontweight='bold')
                    
                    ax.set_xlabel('Average Price ($)', fontsize=14)
                    ax.set_ylabel('Average Quantity', fontsize=14)
                    ax.set_title(f'{selected_category} - Demand Curve\n'
                               f'Elasticity: {beta:.3f} | R²: {r_squared:.3f} | Data Points: {len(valid_data)}', 
                               fontsize=16)
                    ax.legend(fontsize=12)
                    ax.tick_params(axis='both', labelsize=12)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                # Display analysis results
                st.markdown("### Category Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Optimal Pricing Analysis")
                    st.markdown(f"""
                    **Observed-optimum:**  
                    • Price = ${P_obs_best:.2f}  
                    • Quantity = {Q_obs_best:.2f}  
                    • Revenue = ${R_obs_best:.2f}  

                    **Fitted-optimum (log–log):**  
                    • Price = ${P_fit_best:.2f}  
                    • Quantity = {Q_fit_best:.2f}  
                    • Revenue = ${R_fit_best:.2f}  
                    """)
                    
                    if P_lo is not None:
                        st.markdown(f"""
                        **LOWESS-optimum:**  
                        • Price = ${P_lo_best:.2f}  
                        • Quantity = {Q_lo_best:.2f}  
                        • Revenue = ${R_lo_best:.2f}
                        """)
                
                with col2:
                    st.markdown("#### Category Characteristics")
                    elasticity_description = "Highly elastic" if beta < -1.5 else "Moderately elastic" if beta < -1 else "Inelastic"
                    st.markdown(f"""
                    **Category:** {selected_category}  
                    **Price Elasticity:** {beta:.3f} ({elasticity_description})  
                    **Model Fit (R²):** {r_squared:.3f}  
                    **Data Points:** {len(valid_data)}  
                    **Date Range:** {cat_data['day'].min().strftime('%Y-%m-%d')} to {cat_data['day'].max().strftime('%Y-%m-%d')}  
                    **Price Range:** ${valid_data['avg_price'].min():.2f} - ${valid_data['avg_price'].max():.2f}  
                    **Quantity Range:** {valid_data['avg_qty'].min():.2f} - {valid_data['avg_qty'].max():.2f}
                    """)
                
                # Data table
                st.markdown("### Raw Data")
                display_data = cat_data[['day', 'avg_price', 'avg_qty', 'total_revenue']].copy()
                display_data['day'] = display_data['day'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    display_data.round(2),
                    use_container_width=True,
                    column_config={
                        "day": st.column_config.TextColumn("Date"),
                        "avg_price": st.column_config.NumberColumn("Avg Price ($)", format="%.2f"),
                        "avg_qty": st.column_config.NumberColumn("Avg Quantity", format="%.2f"),
                        "total_revenue": st.column_config.NumberColumn("Total Revenue ($)", format="%.2f")
                    }
                )
                
        except Exception as e:
            st.error(f"Error in elasticity analysis: {str(e)}")
            st.write("Displaying raw data instead:")
            st.dataframe(cat_data)

with tab3:
    st.subheader("Categories Map")
    st.write("This shows which items belong to each category:")
    
    # Display categories map in an expandable format
    for category, item_list in categories_map.items():
        with st.expander(f"**{category}** ({len(item_list)} items)"):
            # Get item names for this category
            category_items = metrics_df[metrics_df['item_id'].isin(item_list)].drop_duplicates('item_id')
            if not category_items.empty:
                display_items = category_items[['item_id', 'name', 'categories', 'elasticity', 'alpha', 'r_squared', 'unique_prices']].copy()
                st.dataframe(
                    display_items.round(3),
                    use_container_width=True,
                    height=min(300, len(display_items) * 35 + 50),
                    column_config={
                        "item_id": st.column_config.NumberColumn("Item ID", width="small"),
                        "name": st.column_config.TextColumn("Product Name", width="large"),
                        "categories": st.column_config.TextColumn("All Categories", width="large"), 
                        "elasticity": st.column_config.NumberColumn("Elasticity", width="medium", format="%.3f"),
                        "alpha": st.column_config.NumberColumn("Alpha (α)", width="small", format="%.3f"),
                        "r_squared": st.column_config.NumberColumn("R²", width="small", format="%.3f"),
                        "unique_prices": st.column_config.NumberColumn("Price Points", width="small")
                    }
                )
            else:
                st.write("No items found for this category.")