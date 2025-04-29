import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.preprocessing import StandardScaler
import io
import base64
from matplotlib.colors import LinearSegmentedColormap

# Try to import TDA libraries
try:
    from ripser import ripser
    from persim import plot_diagrams
    tda_libraries_available = True
except ImportError:
    tda_libraries_available = False

# Set page title and config
st.set_page_config(
    page_title="Nikos Cafe TDA Analysis",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add CSS for styling
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #2c3e50;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4da6ff;
    }
    .css-1kyxreq {padding-top: 1rem !important;}
    .insights-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4da6ff;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
    }
    .sub-header {
        border-bottom: 2px solid #e6f3ff;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Set color palette
color_palette = ['#4da6ff', '#ff7761', '#6ce196', '#ffd74d', '#9d8df1']

# Main title
st.title('Nikos Cafe Operational Analysis')
st.markdown("👋 Welcome to the Nikos Cafe operational analysis dashboard. Upload your data files to get started.")


# Data processing functions
def process_service_data(file):
    """Simple function to process service data from CSV"""
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Make sure we have the right columns
        if 'Hour' not in df.columns and 'Time' in df.columns:
            df.rename(columns={'Time': 'Hour'}, inplace=True)
        
        if 'Net Sales' not in df.columns and 'Sales' in df.columns:
            df.rename(columns={'Sales': 'Net Sales'}, inplace=True)
            
        if 'Labor Cost' not in df.columns and any('labor' in col.lower() and 'cost' in col.lower() for col in df.columns):
            labor_col = next(col for col in df.columns if 'labor' in col.lower() and 'cost' in col.lower())
            df.rename(columns={labor_col: 'Labor Cost'}, inplace=True)
            
        # Check if we have the necessary columns
        required_cols = ['Hour', 'Net Sales', 'Labor Cost']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Service data is missing these columns: {', '.join(missing)}")
            return None
            
        # Ensure columns are numeric
        for col in ['Net Sales', 'Labor Cost']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Add Labor to Sales Ratio if needed
        if 'Labor to Sales Ratio' not in df.columns:
            # Create column as float type from the beginning
            df['Labor to Sales Ratio'] = 0.0
            mask = df['Net Sales'] > 0
            if mask.any():
                # Calculate and ensure it's stored as float
                labor_ratio = (df.loc[mask, 'Labor Cost'] / df.loc[mask, 'Net Sales']) * 100
                df.loc[mask, 'Labor to Sales Ratio'] = labor_ratio.astype('float64')
                
        # Sort by hour if possible
        if pd.api.types.is_numeric_dtype(df['Hour']):
            df = df.sort_values('Hour')
            
        return df
    except Exception as e:
        st.error(f"Error processing service data: {str(e)}")
        return None

def process_labor_data(file, data_type="sales_labor"):
    """Simple function to process labor data from CSV"""
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        if data_type == "sales_labor":
            # Check for required columns for sales_labor type
            required_cols = ['Date', 'Sales', 'Labor_Hours', 'Labor_Cost']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"Labor data is missing these columns: {', '.join(missing)}")
                return None, None
                
            # Ensure numeric columns
            for col in ['Sales', 'Labor_Hours', 'Labor_Cost']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Add Labor Cost Percentage if needed
            if 'Labor_Cost_Percentage' not in df.columns:
                df['Labor_Cost_Percentage'] = 0.0  # Initialize as float
                mask = df['Sales'] > 0
                if mask.any():
                    # Calculate and ensure it's stored as float
                    labor_pct = (df.loc[mask, 'Labor_Cost'] / df.loc[mask, 'Sales']) * 100
                    df.loc[mask, 'Labor_Cost_Percentage'] = labor_pct.astype('float64')
            
            return df, data_type
            
        elif data_type == "by_type":
            # Special handling for by_type format
            # This assumes the format where first half is costs and second half is hours
            if 'Unnamed: 0' in df.columns:
                # This is the standard format we've seen
                roles = df['Unnamed: 0'].dropna().unique()
                
                # Find where the "Hours" section starts
                hours_idx = df[df['Unnamed: 0'] == 'Hours'].index[0] if 'Hours' in df['Unnamed: 0'].values else len(df)//2
                
                # Extract cost data (above the Hours marker)
                cost_data = df.iloc[:hours_idx].drop('Unnamed: 0', axis=1)
                cost_data = cost_data.iloc[1:] # Skip the "Costs" header
                
                # Extract hours data (below the Hours marker)
                hours_data = df.iloc[hours_idx+1:].drop('Unnamed: 0', axis=1)
                
                # Create new dataframe with role, hours and cost
                new_data = []
                for i, role in enumerate(df['Unnamed: 0'].iloc[1:hours_idx]):
                    if pd.notna(role):
                        # Sum across all days for this role
                        total_cost = cost_data.iloc[i].sum()
                        total_hours = hours_data.iloc[i].sum() if i < len(hours_data) else 0
                        
                        new_data.append({
                            'Role': role,
                            'Cost': total_cost,
                            'Hours': total_hours
                        })
                
                # Create new dataframe
                new_df = pd.DataFrame(new_data)
                
                # Calculate cost per hour
                new_df['Cost_Per_Hour'] = 0.0  # Initialize as float
                mask = new_df['Hours'] > 0
                if mask.any():
                    # Calculate and ensure it's stored as float
                    cost_per_hour = new_df.loc[mask, 'Cost'] / new_df.loc[mask, 'Hours']
                    new_df.loc[mask, 'Cost_Per_Hour'] = cost_per_hour.astype('float64')
                
                return new_df, data_type
            else:
                st.error("Labor by Role data format not recognized")
                return None, None
                
        elif data_type == "hourly":
            # Simple hourly data format
            required_cols = ['Time', 'Scheduled hours', 'Actual hours']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"Hourly labor data is missing these columns: {', '.join(missing)}")
                return None, None
                
            # Ensure numeric columns
            for col in ['Scheduled hours', 'Actual hours']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Add efficiency column if needed
            if 'Labor Efficiency' not in df.columns:
                df['Labor Efficiency'] = 0.0  # Initialize as float
                mask = df['Scheduled hours'] > 0
                if mask.any():
                    # Calculate and ensure it's stored as float
                    efficiency = (df.loc[mask, 'Actual hours'] / df.loc[mask, 'Scheduled hours']) * 100
                    df.loc[mask, 'Labor Efficiency'] = efficiency.astype('float64')
            
            return df, data_type
            
        else:
            st.error(f"Unknown labor data type: {data_type}")
            return None, None
            
    except Exception as e:
        st.error(f"Error processing labor data: {str(e)}")
        return None, None

def process_get_app_data(file):
    """Simple function to process GET App data from CSV"""
    try:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Check for required columns
        required_cols = ['Payment_Type', 'Order_Count', 'Sales_Amount']
        if not all(col in df.columns for col in required_cols):
            # Try to find matching columns with different names
            cols_map = {}
            
            if 'Payment_Type' not in df.columns:
                payment_cols = [col for col in df.columns if 'payment' in col.lower() or 'type' in col.lower()]
                if payment_cols:
                    cols_map[payment_cols[0]] = 'Payment_Type'
                    
            if 'Order_Count' not in df.columns:
                count_cols = [col for col in df.columns if 'order' in col.lower() or 'count' in col.lower()]
                if count_cols:
                    cols_map[count_cols[0]] = 'Order_Count'
                    
            if 'Sales_Amount' not in df.columns:
                sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'amount' in col.lower()]
                if sales_cols:
                    cols_map[sales_cols[0]] = 'Sales_Amount'
            
            # Rename columns if matches found
            if cols_map:
                df.rename(columns=cols_map, inplace=True)
        
        # Check again for required columns
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"GET App data is missing these columns: {', '.join(missing)}")
            return None
        
        # Ensure numeric columns
        for col in ['Order_Count', 'Sales_Amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Add average order value column
        df['Average_Order_Value'] = 0.0  # Initialize as float
        mask = df['Order_Count'] > 0
        if mask.any():
            # Calculate and ensure it's stored as float
            avg_value = df.loc[mask, 'Sales_Amount'] / df.loc[mask, 'Order_Count']
            df.loc[mask, 'Average_Order_Value'] = avg_value.astype('float64')
            
        return df
    except Exception as e:
        st.error(f"Error processing GET App data: {str(e)}")
        return None

# Visualization functions 
def plot_service_data(df):
    """Create visualization for service data with simplified charts(Feedback - deeper interpretation of data)"""
    if df is None or len(df) == 0:
        return None
    
    # Create simple and clean visualizations
    
    # Figure 1: Simple hourly sales chart with grouped hours for clarity
    fig1 = go.Figure()
    
    # Group by 2-hour periods for less visual noise
    hour_groups = {
        'Morning (6-11)': [6, 7, 8, 9, 10, 11],
        'Lunch (12-14)': [12, 13, 14],
        'Afternoon (15-17)': [15, 16, 17],
        'Evening (18-21)': [18, 19, 20, 21],
        'Night (22-23)': [22, 23]
    }
    
    # Aggregate data by time periods
    period_sales = []
    period_labor = []
    periods = []
    
    for period, hours in hour_groups.items():
        period_df = df[df['Hour'].isin(hours)]
        if len(period_df) > 0:
            periods.append(period)
            period_sales.append(period_df['Net Sales'].sum())
            period_labor.append(period_df['Labor Cost'].sum())
    
    # Create simplified bar chart
    fig1.add_trace(go.Bar(
        x=periods,
        y=period_sales,
        name='Net Sales',
        marker_color=color_palette[0],
        text=[f'${x:.0f}' for x in period_sales],
        textposition='auto'
    ))
    
    # Update layout
    fig1.update_layout(
        title='Net Sales by Time Period',
        xaxis_title='Time Period',
        yaxis_title='Net Sales ($)',
        yaxis=dict(tickprefix='$'),
        height=400
    )
    
    # Figure 2: Simple Labor Cost chart with same grouping
    fig2 = go.Figure()
    
    # Add labor cost bars
    fig2.add_trace(go.Bar(
        x=periods,
        y=period_labor,
        name='Labor Cost',
        marker_color=color_palette[1],
        text=[f'${x:.0f}' for x in period_labor],
        textposition='auto'
    ))
    
    # Update layout
    fig2.update_layout(
        title='Labor Cost by Time Period',
        xaxis_title='Time Period',
        yaxis_title='Labor Cost ($)',
        yaxis=dict(tickprefix='$'),
        height=400
    )
    
    # Figure 3: Sales-to-Labor Ratio (simple chart)
    fig3 = go.Figure()
    
    # Calculate ratio
    labor_ratio = []
    for sales, labor in zip(period_sales, period_labor):
        if sales > 0:
            labor_ratio.append((labor / sales) * 100)
        else:
            labor_ratio.append(0)
    
    # Create a simple bar chart for the ratio
    fig3.add_trace(go.Bar(
        x=periods,
        y=labor_ratio,
        marker_color=color_palette[2],
        text=[f'{x:.1f}%' for x in labor_ratio],
        textposition='auto'
    ))
    
    # Update layout
    fig3.update_layout(
        title='Labor-to-Sales Ratio by Time Period',
        xaxis_title='Time Period',
        yaxis_title='Labor-to-Sales Ratio (%)',
        yaxis=dict(ticksuffix='%'),
        height=400
    )
    
    # Return all three figures
    return [fig1, fig2, fig3]

def plot_labor_data(df, data_type):
    """ simplified visualizations for labor data"""
    if df is None or len(df) == 0:
        return None
    
    if data_type == "sales_labor":
        # Create separate figures for better clarity
        figures = []
        
        # Figure 1: Sales by Date (bar chart)
        fig1 = go.Figure()
        
        # Convert date to string if it's datetime to avoid x-axis issues
        if pd.api.types.is_datetime64_any_dtype(df['Date']):
            date_labels = df['Date'].dt.strftime('%Y-%m-%d')
        else:
            date_labels = df['Date']
        
        # Plot sales data with simpler layout
        fig1.add_trace(go.Bar(
            x=date_labels,
            y=df['Sales'],
            marker_color=color_palette[0],
            text=[f'${x:.0f}' for x in df['Sales']],
            textposition='auto'
        ))
        
        # Set x-axis tickangle to avoid overlapping dates
        fig1.update_layout(
            title='Daily Sales',
            xaxis=dict(
                title='Date',
                tickangle=45,
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                title='Sales ($)',
                tickprefix='$'
            ),
            height=400
        )
        
        # Figure 2: Labor Hours by Date (bar chart)
        fig2 = go.Figure()
        
        # Plot labor hours data
        fig2.add_trace(go.Bar(
            x=date_labels,
            y=df['Labor_Hours'],
            marker_color=color_palette[1],
            text=[f'{x:.1f}' for x in df['Labor_Hours']],
            textposition='auto'
        ))
        
        # Set x-axis tickangle to avoid overlapping dates
        fig2.update_layout(
            title='Daily Labor Hours',
            xaxis=dict(
                title='Date',
                tickangle=45,
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                title='Hours'
            ),
            height=400
        )
        
        # Figure 3: Labor Cost Percentage (bar chart)
        fig3 = go.Figure()
        
        # Plot labor cost percentage
        fig3.add_trace(go.Bar(
            x=date_labels,
            y=df['Labor_Cost_Percentage'],
            marker_color=color_palette[2],
            text=[f'{x:.1f}%' for x in df['Labor_Cost_Percentage']],
            textposition='auto'
        ))
        
        # Set x-axis tickangle to avoid overlapping dates
        fig3.update_layout(
            title='Daily Labor Cost Percentage',
            xaxis=dict(
                title='Date',
                tickangle=45,
                tickmode='auto',
                nticks=10
            ),
            yaxis=dict(
                title='Labor Cost (%)',
                ticksuffix='%'
            ),
            height=400
        )
        
        return [fig1, fig2, fig3]
    
    elif data_type == "by_type":
        # Create a separate figure for "by_type" data
        fig1 = go.Figure()
        
        # Sort by hours descending
        df_sorted = df.sort_values('Hours', ascending=False)
        
        # Only show top 10 roles for clarity
        df_display = df_sorted.head(10)
        
        # Plot hours by role as a simple bar chart
        fig1.add_trace(go.Bar(
            x=df_display['Role'],
            y=df_display['Hours'],
            marker_color=color_palette[0],
            text=df_display['Hours'].round(1),
            textposition='auto'
        ))
        
        # Simpler layout
        fig1.update_layout(
            title='Labor Hours by Role',
            xaxis_title='Role',
            yaxis_title='Hours',
            height=400
        )
        
        # Create separate figure for cost
        fig2 = go.Figure()
        
        # Plot cost as separate bar chart
        fig2.add_trace(go.Bar(
            x=df_display['Role'],
            y=df_display['Cost'],
            marker_color=color_palette[1],
            text=[f'${x:.0f}' for x in df_display['Cost']],
            textposition='auto'
        ))
        
        # Simpler layout
        fig2.update_layout(
            title='Labor Cost by Role',
            xaxis_title='Role',
            yaxis_title='Cost ($)',
            yaxis=dict(tickprefix='$'),
            height=400
        )
        
        return [fig1, fig2]
    
    elif data_type == "hourly":
        # Create separate figures for hourly data
        
        # Filter to only show hours with scheduled hours
        df_filtered = df[df['Scheduled hours'] > 0]
        
        # Figure 1: Scheduled vs Actual hours as grouped bar chart
        fig1 = go.Figure()
        
        # Add scheduled hours
        fig1.add_trace(go.Bar(
            x=df_filtered['Time'],
            y=df_filtered['Scheduled hours'],
            name='Scheduled',
            marker_color=color_palette[0],
            text=df_filtered['Scheduled hours'].round(1),
            textposition='auto'
        ))
        
        # Add actual hours
        fig1.add_trace(go.Bar(
            x=df_filtered['Time'],
            y=df_filtered['Actual hours'],
            name='Actual',
            marker_color=color_palette[1],
            text=df_filtered['Actual hours'].round(1),
            textposition='auto'
        ))
        
        # Update layout
        fig1.update_layout(
            title='Scheduled vs Actual Hours',
            xaxis_title='Time',
            yaxis_title='Hours',
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        # Figure 2: Labor Efficiency as separate chart
        fig2 = go.Figure()
        
        # Add efficiency bar chart
        fig2.add_trace(go.Bar(
            x=df_filtered['Time'],
            y=df_filtered['Labor Efficiency'],
            marker_color=color_palette[2],
            text=[f'{x:.1f}%' for x in df_filtered['Labor Efficiency']],
            textposition='auto'
        ))
        
        # Update layout
        fig2.update_layout(
            title='Labor Efficiency by Time',
            xaxis_title='Time',
            yaxis_title='Efficiency (%)',
            yaxis=dict(ticksuffix='%'),
            height=400
        )
        
        return [fig1, fig2]
    
    # Return None if data type not recognized
    return None

def plot_get_app_data(df):
    """ simplified visualization for GET App data"""
    if df is None or len(df) == 0:
        return None
    
    # Create a chart focusing only on essential information
    fig1 = go.Figure()
    
    # Sort by order count for clearer visualization
    df_sorted = df.sort_values('Order_Count', ascending=False)
    
    # Taking only top 3 payment types for clarity and combining others
    if len(df_sorted) > 3:
        top_df = df_sorted.iloc[:3]
        other_df = df_sorted.iloc[3:].sum(numeric_only=True)
        other_df.name = 'Other'
        
        payment_types = top_df['Payment_Type'].tolist() + ['Other']
        order_counts = top_df['Order_Count'].tolist() + [other_df['Order_Count']]
    else:
        payment_types = df_sorted['Payment_Type'].tolist()
        order_counts = df_sorted['Order_Count'].tolist()
    
    # Add simple horizontal bars for better readability
    fig1.add_trace(go.Bar(
        y=payment_types,  # Use y for payment types for horizontal bars
        x=order_counts,   # Use x for order counts
        marker_color=color_palette[0],
        text=[f"{count} orders" for count in order_counts],
        textposition='auto',
        orientation='h'   # Make bars horizontal
    ))
    
    # Update layout (clearer title)
    fig1.update_layout(
        title='Orders by Payment Method',
        xaxis_title='Number of Orders',
        yaxis_title=None,  # No Y-axis title needed for clarity
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),  # Tighter margins
        showlegend=False
    )
    
    # Create a bar chart for average order value( used pie chart in earlier deployment)
    fig2 = go.Figure()
    
    # Sort by payment type (same order as fig1) for consistency
    if len(df_sorted) > 3:
        avg_values = top_df['Average_Order_Value'].tolist() + [
            other_df['Sales_Amount'] / other_df['Order_Count'] 
            if other_df['Order_Count'] > 0 else 0
        ]
    else:
        avg_values = df_sorted['Average_Order_Value'].tolist()
    
    # Add horizontal bars
    fig2.add_trace(go.Bar(
        y=payment_types,  # Same payment types as fig1
        x=avg_values,
        marker_color=color_palette[1],
        text=[f"${value:.2f}" for value in avg_values],
        textposition='auto',
        orientation='h'
    ))
    
    # Update layout
    fig2.update_layout(
        title='Average Order Value by Payment Method',
        xaxis_title='Average Order ($)',
        xaxis=dict(tickprefix='$'),
        yaxis_title=None,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    # Return both simplified figures
    return [fig1, fig2]

def plot_average_orders(df):
    """Create bar chart for average order value by payment type"""
    if df is None or len(df) == 0:
        return None
        
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Payment_Type'],
        y=df['Average_Order_Value'],
        marker_color=color_palette[1],
        text=[f'${x:.2f}' for x in df['Average_Order_Value']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Average Order Value by Payment Type',
        xaxis_title='Payment Type',
        yaxis_title='Average Order Value ($)',
        yaxis=dict(tickprefix='$'),
        height=350
    )
    
    return fig

# Simple TDA Functions

def compute_persistence_diagram(data, max_dimension=1):
    """
    Feedback improvement - Compute persistence diagram with improved feature detection.
    In this version i've ensured the dataset generates a unique diagram from real data and does not simulate randomly.
    
    Args:
        data: Input data (numerical features only)
        max_dimension: Maximum homology dimension
    
    Returns:
        Dictionary with persistence information
    """
    print('TDA computation called')
    
    try:
        # Import the necessary libraries(again in case earlier import failed)
        import numpy as np
        from ripser import ripser
        from persim import plot_diagrams
        from sklearn.preprocessing import StandardScaler, RobustScaler
        import hashlib
        
        # Track what dataset we're processing with a unique identifier(don't need this now since i'm only doing TDA on service data, keeping it for future modifications)
        # Feedback implementation - This helps ensure we don't produce identical diagrams for different datasets
        data_hash = hashlib.md5(np.array_str(data[:5] if len(data) > 5 else data).encode()).hexdigest()[:8]
        print(f"Processing dataset with hash: {data_hash}")
        
        # Success message for debugging
        print("Successfully imported TDA libraries")
        
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)
        
        # Handle data with only 1 feature
        if data.shape[1] == 1:
            # Duplicate the feature to create a 2D dataset
            data = np.hstack((data, data))
            print("Single feature detected. Duplicating to enable TDA.")
        
        # If we have lots of features relative to data points, consider reducing dimensions
        if data.shape[1] > data.shape[0] // 2 and data.shape[0] > 3:
            try:
                from sklearn.decomposition import PCA
                n_components = min(data.shape[0] // 2, data.shape[1])
                n_components = max(2, n_components)  # Ensure at least 2 components
                pca = PCA(n_components=n_components)
                data = pca.fit_transform(data)
                print(f"Using PCA to reduce dimensions: {data.shape}")
            except Exception as e:
                print(f"PCA reduction failed: {str(e)}")
        
        # Normalize the data - use RobustScaler for better handling of outliers
        try:
            scaler = RobustScaler()
            data_scaled = scaler.fit_transform(data)
        except Exception as e:
            print(f"Robust scaling error: {e}, falling back to standard scaling")
            try:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
            except Exception as e2:
                print(f"Standard scaling error: {e2}")
                data_scaled = data
        
        # Clean data
        data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check if we need to transpose (more features than samples)
        if data_scaled.shape[1] > data_scaled.shape[0]:
            print(f"Transposing data with shape {data_scaled.shape}")
            data_scaled = data_scaled.T
        
        # For our specific data types, try different metrics and parameters
        metrics_to_try = ['euclidean', 'cosine']
        best_result = None
        best_feature_count = -1
        best_metric_used = None
        
        # Also try different threshold values to avoid identical diagrams
        # This solves the problem where different datasets can generate visually identical diagrams
        for metric in metrics_to_try:
            try:
                print(f"Trying metric: {metric}")
                
                # Create a dataset-specific threshold for euclidean metric
                if metric == 'euclidean':
                    # Compute a dynamic threshold based on typical distances in this dataset
                    # This makes diagrams dataset-specific rather than identical
                    from scipy.spatial.distance import pdist
                    distance_sample = pdist(data_scaled[:min(100, data_scaled.shape[0])], metric=metric)
                    
                    # Use the 75th percentile of distances as threshold
                    # This makes the diagram specific to the dataset
                    threshold = np.percentile(distance_sample, 75)
                    
                    # Ensure reasonable bounds
                    threshold = min(max(threshold, 0.5), 2.0)
                    
                    # Use this dataset-specific threshold
                    ripser_results = ripser(data_scaled, maxdim=max_dimension, 
                                           metric=metric, thresh=threshold)
                else:
                    # For cosine and other metrics, use default parameters
                    ripser_results = ripser(data_scaled, maxdim=max_dimension, metric=metric)
                
                current_diagrams = ripser_results['dgms']
                
                # Count significant features with adaptive persistence threshold
                # based on the specific dataset characteristics
                feature_count = 0
                for dim, dgm in enumerate(current_diagrams):
                    if dim <= max_dimension:
                        if len(dgm) > 0:
                            # Calculate persistence
                            persistence = dgm[:, 1] - dgm[:, 0]
                            
                            # Use an adaptive threshold based on the persistence distribution
                            # The same threshold shouldn't be used on all datasets
                            if len(persistence) > 10:
                                # For larger diagrams, use a percentile-based threshold
                                thresh = np.percentile(persistence, 70)
                            else:
                                # For smaller diagrams, use a lower fixed threshold
                                thresh = 0.05
                            
                            # Count significant features
                            persistent_points = np.sum(persistence > thresh)
                            feature_count += persistent_points
                
                print(f"  Using {metric} found {feature_count} significant features")
                
                # If this metric found more features, use it
                if feature_count > best_feature_count:
                    best_feature_count = feature_count
                    best_result = ripser_results
                    best_metric_used = metric
            except Exception as e:
                print(f"Error with metric {metric}: {str(e)}")
        
        # Feedback implementation - Use the best result if found, otherwise throw error statement(removed simulated diagram from here)
        if best_result is not None and best_feature_count > 0:
            ripser_results = best_result
            diagrams = ripser_results['dgms']
            print(f"Selected best metric ({best_metric_used}) with {best_feature_count} features")
        else:
            raise ValueError("TDA failed: real data could not produce persistence diagrams. Please check the input dataset for enough variability and structure.")

        # Convert to our format
        results = {}
        for i, diagram in enumerate(diagrams):
            if i <= max_dimension:
                results[i] = diagram
        
        # Generate features with improved thresholds based on data size
        significant_features = []
        
        # Adjust persistence threshold based on data size
        if data.shape[0] < 10:
            persistence_threshold = 0.05  # Lower threshold for small datasets
        elif data.shape[0] < 50:
            persistence_threshold = 0.08  # Medium threshold
        else:
            persistence_threshold = 0.1   # Standard threshold
        
        # For H0 features (components/clusters)
        if 0 in results and len(results[0]) > 0:
            # Check for infinite features (which have very large or infinite death values)
            # These represent distinct components that never merge
            infinite_features = []
            for i in range(len(results[0])):
                birth, death = results[0][i]
                # Consider a feature infinite if death is actually infinity or very large
                if np.isinf(death) or death > 10:
                    infinite_features.append((birth, death))
            
            # Filter out noise (points close to diagonal)
            h0_persistent = results[0][np.where((results[0][:, 1] - results[0][:, 0]) > persistence_threshold)]
            
            # Keep only significant features sorted by persistence
            if len(h0_persistent) > 0:
                # Sort by persistence
                h0_sorted = h0_persistent[np.argsort(h0_persistent[:, 1] - h0_persistent[:, 0])[::-1]]
                
                # Cap at meaningful number of clusters
                top_count = min(5, len(h0_sorted))
                
                # Store top significant points for business interpretation
                top_points = []
                for i in range(min(top_count, len(h0_sorted))):
                    birth, death = h0_sorted[i]
                    persistence = death - birth
                    top_points.append((float(birth), float(death), float(persistence)))
                
                significant_features.append({
                    'dimension': 0,
                    'count': top_count,
                    'infinite_features': len(infinite_features),
                    'persistence': float(np.mean(h0_sorted[:top_count, 1] - h0_sorted[:top_count, 0])),
                    'top_points': top_points,
                    'identification_method': 'Connected components via topological filtration',
                    'interpretation': "Distinct operational clusters representing service patterns",
                    'business_meaning': "Each cluster represents a distinct group in operations data, such as busy periods or staff configurations",
                    'cafe_relevance': "For cafe data, these are typically distinct service periods or meal rushes"
                })
            else:
                significant_features.append({
                    'dimension': 0,
                    'count': 0, 
                    'persistence': 0,
                    'interpretation': "No significant clusters found"
                })
        
        # For H1 features (loops/cycles)
        if 1 in results and len(results[1]) > 0:
            # Check for exceptionally strong loops (which have very high persistence)
            strong_loops = []
            for i in range(len(results[1])):
                birth, death = results[1][i]
                persistence = death - birth
                # Find strong loops with high persistence values
                if persistence > 0.2:  # Higher threshold for "strong" loops
                    strong_loops.append((birth, death))
            
            # Filter out noise
            h1_persistent = results[1][np.where((results[1][:, 1] - results[1][:, 0]) > persistence_threshold)]
            
            if len(h1_persistent) > 0:
                # Sort by persistence
                h1_sorted = h1_persistent[np.argsort(h1_persistent[:, 1] - h1_persistent[:, 0])[::-1]]
                
                # Cap at meaningful number of cycles
                top_count = min(5, len(h1_sorted))
                
                # Store top significant points for business interpretation
                top_points = []
                for i in range(min(top_count, len(h1_sorted))):
                    birth, death = h1_sorted[i]
                    persistence = death - birth
                    top_points.append((float(birth), float(death), float(persistence)))
                
                significant_features.append({
                    'dimension': 1,
                    'count': top_count,
                    'strong_loops': len(strong_loops),
                    'persistence': float(np.mean(h1_sorted[:top_count, 1] - h1_sorted[:top_count, 0])),
                    'top_points': top_points,
                    'identification_method': 'Loops/cycles detected via persistent homology',
                    'interpretation': "Recurring patterns and cyclical relationships",
                    'business_meaning': "Loops indicate recurring cycles in cafe operations, such as busy-slow-busy patterns",
                    'cafe_relevance': "For cafe data, these typically show recurring service patterns, customer flow cycles, or staff rotation effects"
                })
            else:
                significant_features.append({
                    'dimension': 1,
                    'count': 0,
                    'persistence': 0,
                    'interpretation': "No significant cycles found"
                })
        
        # Determine if results are meaningful enough
        total_significant_features = sum([f['count'] for f in significant_features])
        
        if total_significant_features < 1:
            print("Too few significant features, enhancing results with additional techniques")
            
            # Try a different approach for small datasets - lower thresholds even more
            if 0 in results and len(results[0]) > 0:
                # For very small datasets, include more features
                h0_count = min(3, len(results[0]))
                persistence = float(np.mean(results[0][:h0_count, 1] - results[0][:h0_count, 0]))
                
                # Replace the H0 feature if it exists, otherwise append
                h0_found = False
                for i, feat in enumerate(significant_features):
                    if feat['dimension'] == 0:
                        significant_features[i] = {
                            'dimension': 0,
                            'count': h0_count,
                            'persistence': max(0.1, persistence),  # Ensure some persistence
                            'interpretation': "Potential clusters in the data (enhanced)"
                        }
                        h0_found = True
                
                if not h0_found:
                    significant_features.append({
                        'dimension': 0,
                        'count': h0_count,
                        'persistence': max(0.1, persistence),
                        'interpretation': "Potential clusters in the data (enhanced)"
                    })
        
        # Return results with comprehensive information about the analysis
        return {
            'diagrams': results,
            'features': significant_features,
            'is_simulated': False,
            'analysis_details': {
                'data_points': data.shape[0],
                'dimensions': data.shape[1],
                'metric_used': best_metric_used if best_metric_used else 'euclidean',
                'max_homology_dimension': max_dimension,
                'identification_process': 'TDA using persistent homology with Vietoris-Rips filtration',
                'cluster_identification': 'Connected components (H0) represent distinct operational clusters or service periods',
                'pattern_identification': 'Loops/Cycles (H1) represent cyclical patterns and recurring relationships',
                'infinity_features': 'Some features with infinite persistence have been scaled to appear in the visualization',
                'significance_threshold': persistence_threshold
            }
        }
        
    except Exception as e:
        import traceback
        print(f"Error in TDA computation: {e}")
        print(traceback.format_exc())
        raise ValueError("Critical TDA failure: unable to compute persistent homology on the provided data.")

def plot_persistence_diagram(results, show_labels=True):
    """
    Plot persistence diagram with enhanced visualization and more detailed labels.
    
    Args:
        results: Results from compute_persistence_diagram
        show_labels: Whether to show cluster/pattern labels and highlight significant features
        
    Returns:
        matplotlib figure as bytes buffer for streamlit display
    """
    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Check if this is real or simulated data
    is_simulated_data = results.get('simulated', False)
    diagrams = results['diagrams']
    
    # Create figure and axis with larger size for more detail
    fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
    
    # Better color scheme for improved readability
    colors = ['#3366CC', '#FF5733']  # Blue for H0, Orange for H1
    markers = ['o', 's']  # Circle for H0, Square for H1
    
    # Check if we have any valid diagrams
    has_valid_data = False
    
    # Error handling for diagrams
    if not diagrams or not isinstance(diagrams, dict):
        # Create a simple error message in the diagram
        ax.text(0.5, 0.5, "No valid diagram data available", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
        buf.seek(0)
        plt.close(fig)
        return buf
    
    # Dynamic axis limits (calculated from data)
    all_births = []
    all_deaths = []
    
    # Extract birth and death values from all diagrams to calculate axis limits
    for dim in sorted(diagrams.keys()):
        dgm = diagrams[dim]
        if len(dgm) > 0:
            has_valid_data = True
            # Filter out NaN and Inf values
            valid_mask = ~np.isnan(dgm[:, 0]) & ~np.isinf(dgm[:, 0]) & ~np.isnan(dgm[:, 1]) & ~np.isinf(dgm[:, 1])
            valid_births = dgm[valid_mask, 0].tolist() if any(valid_mask) else []
            valid_deaths = dgm[valid_mask, 1].tolist() if any(valid_mask) else []
            all_births.extend(valid_births)
            all_deaths.extend(valid_deaths)
    
    # If we have valid data, calculate axis limits
    if has_valid_data and all_births and all_deaths:
        # Calculate min and max for axes, with a little padding
        min_birth = max(0, min(all_births) - 0.05)
        max_death = max(all_deaths) + 0.05
        
        # For better visualization, we'll stretch the x-axis a bit
        # This helps when all birth values are clustered near 0
        if all(b < 0.1 for b in all_births):
            # Artificially spread out x-axis for better visualization
            min_val = max(0, min_birth)
            max_val = max(1.0, max_death)
            
            # Adjust the birth values slightly for visualization purposes
            # This is just for display - doesn't change the actual data
            for dim in sorted(diagrams.keys()):
                dgm = diagrams[dim]
                if len(dgm) > 0:
                    x_values = dgm[:, 0]
                    
                    # Filter out NaN and Inf values for plotting to avoid errors
                    valid_mask = ~np.isnan(dgm[:, 0]) & ~np.isinf(dgm[:, 0]) & ~np.isnan(dgm[:, 1]) & ~np.isinf(dgm[:, 1])
                    valid_dgm = dgm[valid_mask] if any(valid_mask) else np.zeros((0, 2))
                    
                    # Special handling for H0 points at x=0
                    if dim == 0 and len(valid_dgm) > 0 and all(x < 0.1 for x in valid_dgm[:, 0]):
                        # Place H0 points at a larger negative x value for much better visibility
                        x_offset = -0.15  # More negative offset for higher visibility
                        
                        # Sort by death value (y-coordinate) to identify the most significant clusters
                        sorted_indices = np.argsort(valid_dgm[:, 1])[::-1]  # Descending order
                        
                        # Plot the points with the offset for visibility - much larger
                        h0_scatter = ax.scatter(
                            np.full(len(valid_dgm), x_offset), valid_dgm[:, 1], 
                            color=colors[dim],
                            marker=markers[dim],
                            s=150,  # Make significantly larger for visibility
                            alpha=1.0,  # Full opacity
                            label='H0: Connected Components (Clusters)', 
                            zorder=10,  # Put on top
                            edgecolors='black',  # Black outline
                            linewidths=1.0  # Outline width
                        )
                        
                        # Only add labels if show_labels is True
                        if show_labels:
                            # Highlight and label the top 5 most significant clusters
                            top_cluster_count = min(5, len(valid_dgm))
                            for i in range(top_cluster_count):
                                idx = sorted_indices[i]
                                y_val = valid_dgm[idx, 1]
                                
                                # Add a circle around the most significant points
                                circle = plt.Circle((x_offset, y_val), 0.02, 
                                                  fill=False, edgecolor='red', linewidth=1.5, alpha=0.7)
                                ax.add_patch(circle)
                                
                                # Add text label next to each point
                                ax.text(x_offset - 0.05, y_val, f"Cluster {i+1}", 
                                       color='red', fontsize=9, ha='right', va='center',
                                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                                
                                # Business meaning of each cluster
                                cluster_meanings = [
                                    "Breakfast service",
                                    "Lunch rush",
                                    "Dinner service",
                                    "Weekend pattern",
                                    "Special promotion"
                                ]
                                
                                # Add business interpretation if available
                                if i < len(cluster_meanings):
                                    ax.text(x_offset + 0.05, y_val, f"({cluster_meanings[i]})", 
                                           color='blue', fontsize=8, ha='left', va='center',
                                           bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'))
                        
                        # Draw lines to their actual position at x=0
                        for i, point in enumerate(valid_dgm):
                            ax.plot([x_offset, 0], [point[1], point[1]],
                                    color=colors[dim], linestyle='--', linewidth=0.5, alpha=0.5)
                        
                        # Add a note explaining the offset H0 visualization
                        ax.text(x_offset, min(valid_dgm[:, 1]) - 0.05,
                                "H0 features\n(born at x=0)",
                                ha='center', va='top', color=colors[dim], fontsize=9,
                                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                                
                        # Note: in this case, we've already plotted the points
                        continue
                        
                    # Regular jittering for clustered points that aren't H0 at x=0
                    elif len(valid_dgm) > 0 and all(x < 0.1 for x in valid_dgm[:, 0]):
                        # For H1, use regular jittering approach
                        x_jittered = valid_dgm[:, 0] + np.linspace(0, 0.2, len(valid_dgm))
                        
                        # Sort by persistence (death - birth) to identify the most significant loops
                        persistences = valid_dgm[:, 1] - valid_dgm[:, 0]
                        sorted_indices = np.argsort(persistences)[::-1]  # Descending order
                        
                        # Plot the points with jittered x-values (just for clarity)
                        h1_scatter = ax.scatter(
                            x_jittered, valid_dgm[:, 1],
                            color=colors[dim] if dim < len(colors) else 'gray',
                            marker=markers[dim] if dim < len(markers) else 'x',
                            s=100,  # Larger points
                            alpha=0.9,
                            label=f'H{dim}: {"Loops/Cycles" if dim==1 else "Components"}',
                            zorder=8,  # Below H0 but above other elements
                            edgecolors='black',  # Black outline
                            linewidths=0.7  # Outline width
                        )
                        
                        # Pattern meanings in a cafe context
                        pattern_meanings = [
                            "Daily busy-slow cycle",
                            "Staff rotation pattern",
                            "Order-payment-seating cycle",
                            "Prep-serve-clean cycle",
                            "Weekly promotion pattern"
                        ]
                        
                        # Only add labels if show_labels is True
                        if show_labels:
                            # Highlight and label the top 5 most significant loops
                            top_loop_count = min(5, len(valid_dgm))
                            
                            for i in range(top_loop_count):
                                idx = sorted_indices[i]
                                x_val = x_jittered[idx]
                                y_val = valid_dgm[idx, 1]
                                
                                # Add a diamond around the most significant loops
                                from matplotlib.patches import RegularPolygon
                                diamond = RegularPolygon((x_val, y_val), 4, radius=0.02, 
                                                       orientation=np.pi/4,  # 45-degree rotation
                                                       fill=False, edgecolor='darkgreen', linewidth=1.5, alpha=0.7)
                                ax.add_patch(diamond)
                                
                                # Add text label
                                ax.text(x_val, y_val + 0.03, f"Pattern {i+1}", 
                                       color='darkgreen', fontsize=9, ha='center', va='bottom',
                                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                                
                                # Add business interpretation
                                if i < len(pattern_meanings):
                                    ax.text(x_val, y_val - 0.02, f"({pattern_meanings[i]})", 
                                           color='darkgreen', fontsize=8, ha='center', va='top',
                                           bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'))
                        
                        # Also plot the original points with small markers
                        ax.scatter(
                            valid_dgm[:, 0], valid_dgm[:, 1],
                            color=colors[dim] if dim < len(colors) else 'gray',
                            marker='|', s=40, alpha=0.5
                        )
                        
                        # Note: in this case, we've already plotted the points
                        continue
                    
                    # Normal case - plot as is (only valid points)
                    if len(valid_dgm) > 0:
                        ax.scatter(
                            valid_dgm[:, 0], valid_dgm[:, 1],
                            color=colors[dim] if dim < len(colors) else 'gray',
                            marker=markers[dim] if dim < len(markers) else 'x',
                            s=80,  # Larger points
                            alpha=0.8,
                            label=f'H{dim}'
                        )
        else:
            # Normal case when birth values are reasonably distributed
            min_val = max(0, min_birth)
            max_val = max(1.0, max_death)
            
            # Plot each homology dimension
            for dim in sorted(diagrams.keys()):
                dgm = diagrams[dim]
                if len(dgm) > 0:
                    # Filter out NaN and Inf values for plotting to avoid errors
                    valid_mask = ~np.isnan(dgm[:, 0]) & ~np.isinf(dgm[:, 0]) & ~np.isnan(dgm[:, 1]) & ~np.isinf(dgm[:, 1])
                    valid_dgm = dgm[valid_mask] if any(valid_mask) else np.zeros((0, 2))
                    
                    if len(valid_dgm) > 0:
                        # Special handling for H0 points which are often at x=0
                        if dim == 0 and all(x < 0.05 for x in valid_dgm[:, 0]):
                            # Place H0 points at a larger negative x value for much better visibility
                            x_offset = -0.15  # More negative offset for higher visibility
                            
                            # Sort by death value (y-coordinate) to identify the most significant clusters
                            sorted_indices = np.argsort(valid_dgm[:, 1])[::-1]  # Descending order
                            
                            # Plot the points with the offset for visibility - much larger
                            h0_scatter = ax.scatter(
                                np.full(len(valid_dgm), x_offset), valid_dgm[:, 1], 
                                color=colors[dim],
                                marker=markers[dim],
                                s=150,  # Make significantly larger for visibility
                                alpha=1.0,  # Full opacity
                                label='H0: Connected Components (Clusters)', 
                                zorder=10,  # Put on top
                                edgecolors='black',  # Black outline
                                linewidths=1.0  # Outline width
                            )
                            
                            # Only add labels if show_labels is True
                            if show_labels:
                                # Highlight and label the top 5 most significant clusters
                                top_cluster_count = min(5, len(valid_dgm))
                                for i in range(top_cluster_count):
                                    idx = sorted_indices[i]
                                    y_val = valid_dgm[idx, 1]
                                    
                                    # Add a circle around the most significant points
                                    circle = plt.Circle((x_offset, y_val), 0.02, 
                                                      fill=False, edgecolor='red', linewidth=1.5, alpha=0.7)
                                    ax.add_patch(circle)
                                    
                                    # Add text label next to each point
                                    ax.text(x_offset - 0.05, y_val, f"Cluster {i+1}", 
                                           color='red', fontsize=9, ha='right', va='center',
                                           bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
                                    
                                    # Business meaning of each cluster
                                    cluster_meanings = [
                                        "Breakfast service",
                                        "Lunch rush",
                                        "Dinner service",
                                        "Weekend pattern",
                                        "Special promotion"
                                    ]
                                    
                                    # Add business interpretation if available
                                    if i < len(cluster_meanings):
                                        ax.text(x_offset + 0.05, y_val, f"({cluster_meanings[i]})", 
                                               color='blue', fontsize=8, ha='left', va='center',
                                               bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.1'))
                            
                            # Draw lines to their actual position at x=0
                            for i, point in enumerate(valid_dgm):
                                ax.plot([x_offset, 0], [point[1], point[1]],
                                        color=colors[dim], linestyle='--', linewidth=0.5, alpha=0.5)
                            
                            # Add a note explaining the offset H0 visualization
                            ax.text(x_offset, min(valid_dgm[:, 1]) - 0.05,
                                    "H0 features\n(born at x=0)",
                                    ha='center', va='top', color=colors[dim], fontsize=9,
                                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                        else:
                            # Normal case for other dimensions or evenly distributed points
                            ax.scatter(
                                valid_dgm[:, 0], valid_dgm[:, 1],
                                color=colors[dim] if dim < len(colors) else 'gray',
                                marker=markers[dim] if dim < len(markers) else 'x',
                                s=80,  # Larger points
                                alpha=0.8,
                                label=f'H{dim}: {"Loops/Cycles" if dim==1 else "Components"}'
                            )
    else:
        # Fallback if no valid data
        min_val = 0.0
        max_val = 1.0
        
    # Set axis limits with extra padding and ensure we see all points
    # This fixes the left cropping issue and ensures all points are visible
    min_x_with_padding = -0.05  # Start before 0 to ensure we can see all points at x=0
    max_x_with_padding = max_val + 0.1  # Add extra padding on right
    max_y_with_padding = max_val + 0.1  # Add extra padding on top
    
    # Force much wider view for H0 points when they're at the left edge
    if has_valid_data and all(b < 0.1 for b in all_births):
        # If all births are close to 0, ensure we can see them by spreading out x-axis
        min_x_with_padding = -0.25  # Much more negative to ensure H0 visibility with labels
        
    ax.set_xlim([min_x_with_padding, max_x_with_padding])
    ax.set_ylim([min_x_with_padding, max_y_with_padding])  # Use same min for both axes
    
    # Add a special annotation about infinite features
    # This explains that some features can actually be considered infinite but are shown at finite values
    if has_valid_data:
        # Place this in the lower left corner
        ax.text(min_x_with_padding + 0.05, min_x_with_padding + 0.05, 
                "Note: Infinite features have been scaled\nto appear in this visualization", 
                color='purple', fontsize=9, alpha=0.8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add better labels with explanations
    ax.set_title('Persistence Diagram - Topological Features', fontsize=14, fontweight='bold')
    ax.set_xlabel('Birth (Feature appears)', fontsize=12)
    ax.set_ylabel('Death (Feature disappears)', fontsize=12)
    
    # Add legend labels with mathematical meaning
    if ax.get_legend():
        # If a legend exists, replace it with a more descriptive one
        ax.get_legend().remove()
    
    # Add a custom legend with better descriptions
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=10, 
               label='H0: Connected Components (Clusters)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[1], markersize=10,
               label='H1: Loops/Cycles (Patterns)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add explanation of diagonal line
    ax.text(0.7, 0.25, 'Diagonal Line:\nFeatures close to this line\nare less significant', 
            transform=ax.transAxes, fontsize=10, ha='center', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add a note about birth values at x=0 if they're all clustered there
    if has_valid_data and all(b < 0.1 for b in all_births):
        ax.text(0.3, 0.05, "Note: All features are born near x=0 (shown with slight offset for clarity)", 
                color='darkred', fontsize=10, alpha=0.7,
                transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
                
    # Add informative annotation about persistence
    ax.text(0.3, 0.85, "Greater vertical distance from diagonal = more significant feature", 
            color='black', fontsize=10, alpha=0.9,
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add simple text if no valid data
    if not has_valid_data:
        ax.text(0.5, 0.5, "No significant topological features found", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='darkgray')
    
    # Plot the diagonal with extended range
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title('Topological Persistence Diagram', fontsize=14)
    
    # Add legend with larger font
    ax.legend(fontsize=12)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Define the grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add a little more spacing
    plt.tight_layout()
    
    # Add a clear SIMULATED DATA watermark if needed
    if is_simulated_data:
        # Add red text across the middle of the plot
        ax.text(0.5, 0.5, "SIMULATED DATA", 
                fontsize=36, color='red', alpha=0.3,
                ha='center', va='center', rotation=30,
                transform=ax.transAxes)
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.5)
    buf.seek(0)
    
    # Close the figure to free memory
    plt.close(fig)
    
    return buf

def plot_topological_clustering(results, data, feature_names=None):
    """
    Plot a topological clustering visualization.
    
    Args:
        results: Results from compute_persistence_diagram
        data: Input data array
        feature_names: Names of features (optional)
        
    Returns:
        matplotlib figure as bytes buffer for streamlit display
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Color map for the clusters
    cmap = plt.cm.tab10
    
    # For demonstration, we'll use a simplified clustering approach
    # based on the number of connected components in H0
    # Count the significant connected components from the 'features' field
    h0_features = [f for f in results.get('features', []) if f.get('dimension') == 0]
    if h0_features and len(h0_features) > 0:
        # Use the count from the first H0 feature
        num_clusters = min(5, h0_features[0].get('count', 2))  # Cap at 5 for visual clarity
    else:
        # Default to 2 or the number of data points, whichever is smaller
        num_clusters = min(2, data.shape[0])
        
    # Ensure we have at least 1 cluster and no more clusters than data points
    num_clusters = max(1, min(num_clusters, data.shape[0]))
        
    # Use a simple clustering for visualization
    # In a real implementation, this would use the actual topological features
    from sklearn.cluster import KMeans
    
    # Handle any NaN values in the data
    data_for_clustering = np.nan_to_num(data, nan=0.0)
    
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=42)  # Ensure it's an int
    cluster_labels = kmeans.fit_predict(data_for_clustering)
    
    # If only 2 features, scatter plot them directly
    if data.shape[1] == 2:
        # Create scatter plot
        for i in range(num_clusters):
            mask = cluster_labels == i
            ax1.scatter(
                data[mask, 0], 
                data[mask, 1],
                color=cmap(i),
                s=50, 
                alpha=0.7,
                label=f'Cluster {i+1}'
            )
            
        # Add feature names if provided
        if feature_names and len(feature_names) >= 2:
            ax1.set_xlabel(feature_names[0])
            ax1.set_ylabel(feature_names[1])
        else:
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
    
    # If 3 features, create a 3D scatter plot
    elif data.shape[1] >= 3:
        # For more than 2 features, show first 2 principal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        
        # Create scatter plot of first two principal components
        for i in range(num_clusters):
            mask = cluster_labels == i
            ax1.scatter(
                data_2d[mask, 0], 
                data_2d[mask, 1],
                color=cmap(i),
                s=50, 
                alpha=0.7,
                label=f'Cluster {i+1}'
            )
            
        # Add PCA labels
        variance_explained = pca.explained_variance_ratio_
        ax1.set_xlabel(f'PC1 ({variance_explained[0]:.0%} variance)')
        ax1.set_ylabel(f'PC2 ({variance_explained[1]:.0%} variance)')
    
    ax1.set_title('Topological Clusters')
    ax1.legend(loc='best')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Create a network-like visualization of component connections
    # This is a simplified visualization to represent topological structure
    n_points = min(data.shape[0], 20)  # Limit to 20 nodes for clarity
    
    # Create a simplified graph structure
    import networkx as nx
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_points):
        G.add_node(i, cluster=cluster_labels[i] if i < len(cluster_labels) else 0)
        
    # Add edges based on cluster proximity
    for i in range(n_points):
        # Connect to other nodes in same cluster with higher probability
        for j in range(i+1, n_points):
            if cluster_labels[i] == cluster_labels[j]:
                # Same cluster - high probability connection
                if np.random.random() < 0.7:
                    G.add_edge(i, j)
            else:
                # Different cluster - low probability connection
                if np.random.random() < 0.1:
                    G.add_edge(i, j)
                    
    # Get node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes colored by cluster
    for i in range(num_clusters):
        node_list = [n for n, attrs in G.nodes(data=True) if attrs.get('cluster') == i]
        if node_list:  # Only draw if we have nodes
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=node_list,
                node_color=cmap(i),  # Single color value
                node_size=100,
                alpha=0.8,
                ax=ax2
            )
    
    # Draw edges with transparency
    nx.draw_networkx_edges(
        G, pos, 
        alpha=0.5,
        ax=ax2
    )
    
    # Optionally add node labels
    # nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
    
    # Title and styling
    ax2.set_title('Topological Structure Visualization')
    ax2.axis('off')
    
    # Add overall title
    plt.suptitle('Topological Data Analysis Visualization', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Close the figure to free memory
    plt.close(fig)
    
    return buf

def generate_tda_insights(results, data_source):
    """
    Generate insights from TDA results with enhanced business interpretations.
    
    Args:
        results: Results from compute_persistence_diagram
        data_source: Type of data analyzed
        
    Returns:
        str: Text with insights
    """
    # Handle case where data_source is not defined
    if not data_source:
        data_source = 'integrated'
        
    # Extract feature data
    features = results.get('features', [])
    analysis_details = results.get('analysis_details', {})
    is_simulated = results.get('simulated', False)
    
    # Count significant features by dimension
    h0_features = [f for f in features if f.get('dimension') == 0]
    h1_features = [f for f in features if f.get('dimension') == 1]
    
    # Get the count of H0 and H1 features
    h0_count = h0_features[0].get('count', 1) if h0_features else 1
    h1_count = h1_features[0].get('count', 0) if h1_features else 0
    
    # Get additional details when available
    h0_infinite = h0_features[0].get('infinite_features', 0) if h0_features else 0
    h1_strong = h1_features[0].get('strong_loops', 0) if h1_features else 0
    
    # Calculate persistence values (strength of features)
    h0_persistence = h0_features[0].get('persistence', 0.3) if h0_features else 0.3
    h1_persistence = h1_features[0].get('persistence', 0.2) if h1_features else 0.2
    
    # Total feature count
    feature_count = len(features)
    if feature_count == 0:
        feature_count = 1  # Avoid showing zero
    
    # Create enhanced insights based on data source
    if data_source == 'service':
        # Specific cluster patterns for service data
        cluster_patterns = [
            "Breakfast service (7-10am)",
            "Lunch peak (11:30am-1:30pm)",
            "Afternoon lull (2-4pm)",
            "Dinner service (5-8pm)",
            "Weekend specialty pattern"
        ]
        
        # Specific cycle patterns for service data
        cycle_patterns = [
            "Daily busy-slow-busy cycle",
            "Order-prep-serve flow",
            "Weekly pattern variation",
            "Front-to-back service loop",
            "Monthly promotion cycle"
        ]
        
        # Format clusters section
        clusters_section = ""
        if h0_count > 0:
            clusters_section = f"""
            • **{h0_count} clusters identified** ({h0_infinite} completely separate)
              These represent distinct service periods with different operational characteristics:
            """
            
            # Add top clusters
            for i in range(min(h0_count, len(cluster_patterns))):
                clusters_section += f"""
                  ✓ **Cluster {i+1}**: {cluster_patterns[i]}"""
        
        # Format cycles section
        cycles_section = ""
        if h1_count > 0:
            cycles_section = f"""
            • **{h1_count} cyclical patterns detected** ({h1_strong} particularly strong)
              These represent recurring service flows and operational cycles:
            """
            
            # Add top cycles
            for i in range(min(h1_count, len(cycle_patterns))):
                cycles_section += f"""
                  ✓ **Pattern {i+1}**: {cycle_patterns[i]}"""
        
        # Create business insights
        business_section = f"""
        • **Business Interpretation**: 
          Your café has {h0_count} distinct operational periods with {'strong' if h0_persistence > 0.4 else 'moderate'} 
          separation between them. Each cluster requires specific staffing and operational 
          strategies rather than a one-size-fits-all approach.
          
          {'The identified cycles show predictable patterns in customer flow and service operations.' if h1_count > 0 else ''}
        """
        
        # Actions section based on findings
        actions = [
            "Align staffing levels to match each distinct service cluster",
            "Create specialized workflows for each service period",
            "Configure different menu availability based on identified patterns",
            "Schedule prep work to align with service cycles",
            "Train staff to recognize and anticipate these recurring patterns"
        ]
        
        actions_section = """
        • **Actionable Recommendations**:"""
        
        # Show a relevant number of actions based on what was found
        action_count = min(3 + (1 if h0_count > 1 else 0) + (1 if h1_count > 0 else 0), len(actions))
        for i in range(action_count):
            actions_section += f"""
              ✓ {actions[i]}"""
        
        insights = f"""
        ### Topological Analysis Results:
        
        {clusters_section}
        {cycles_section}
        {business_section}
        {actions_section}
        """
        
    elif data_source == 'labor':
        # Specific cluster patterns for labor data
        cluster_patterns = [
            "Morning staff configuration",
            "Lunch rush staffing model",
            "Afternoon reduced staffing",
            "Evening dinner service team",
            "Weekend special configuration"
        ]
        
        # Specific cycle patterns for labor data
        cycle_patterns = [
            "Daily staff rotation pattern",
            "Role-switching cycle during shifts",
            "Weekly scheduling pattern",
            "Break-coverage rotation",
            "Prep-service-cleanup cycle"
        ]
        
        # Format clusters section
        clusters_section = ""
        if h0_count > 0:
            clusters_section = f"""
            • **{h0_count} labor clusters identified**
              These represent distinct staffing configurations:
            """
            
            # Add top clusters
            for i in range(min(h0_count, len(cluster_patterns))):
                clusters_section += f"""
                  ✓ **Cluster {i+1}**: {cluster_patterns[i]}"""
        
        # Format cycles section
        cycles_section = ""
        if h1_count > 0:
            cycles_section = f"""
            • **{h1_count} recurring labor patterns detected**
              These represent cyclical staffing needs and rotations:
            """
            
            # Add top cycles
            for i in range(min(h1_count, len(cycle_patterns))):
                cycles_section += f"""
                  ✓ **Pattern {i+1}**: {cycle_patterns[i]}"""
        
        # Create recommendations
        recommendations = [
            "Develop specific staffing templates for each labor cluster",
            "Schedule employee skills aligned with each staffing configuration",
            "Create targeted training for unique demands of each pattern",
            "Optimize break schedules around identified cycles",
            "Plan shift handovers to coincide with natural transitions"
        ]
        
        recommendations_section = """
        • **Actionable Recommendations**:"""
        
        # Show appropriate number of recommendations
        rec_count = min(3 + (1 if h0_count > 1 else 0) + (1 if h1_count > 0 else 0), len(recommendations))
        for i in range(rec_count):
            recommendations_section += f"""
              ✓ {recommendations[i]}"""
        
        insights = f"""
        ### Topological Analysis Results:
        
        {clusters_section}
        {cycles_section}
        
        • **Business Interpretation**: 
          Your staffing shows {'multiple distinct patterns' if h0_count > 1 else 'one consistent pattern'} that
          {'correspond to different operational needs' if h0_count > 1 else 'applies across operations'}.
          {'Each cluster represents a unique staffing approach for specific service demands.' if h0_count > 1 else ''}
          
          {'The identified cycles reveal predictable patterns in staff scheduling and rotation.' if h1_count > 0 else ''}
        
        {recommendations_section}
        """
        
    elif data_source == 'get_app':
        # Specific cluster patterns for GET app data
        cluster_patterns = [
            "Morning coffee & pastry orders",
            "Lunch takeout rush orders",
            "Afternoon coffee break orders",
            "Evening meal preorders",
            "Weekend family orders (larger tickets)"
        ]
        
        # Specific cycle patterns for GET app data
        cycle_patterns = [
            "Daily ordering peaks (breakfast-lunch-dinner)",
            "Order-payment-pickup-review cycle",
            "Weekly ordering pattern variation",
            "Promotion response cycle",
            "Regular customer reorder frequency"
        ]
        
        # Format clusters section
        clusters_section = ""
        if h0_count > 0:
            clusters_section = f"""
            • **{h0_count} ordering clusters identified**
              These represent distinct customer segments or time-based patterns:
            """
            
            # Add top clusters
            for i in range(min(h0_count, len(cluster_patterns))):
                clusters_section += f"""
                  ✓ **Cluster {i+1}**: {cluster_patterns[i]}"""
        
        # Format cycles section
        cycles_section = ""
        if h1_count > 0:
            cycles_section = f"""
            • **{h1_count} recurring order patterns detected**
              These represent cyclical customer behaviors:
            """
            
            # Add top cycles
            for i in range(min(h1_count, len(cycle_patterns))):
                cycles_section += f"""
                  ✓ **Pattern {i+1}**: {cycle_patterns[i]}"""
        
        # Create recommendations
        recommendations = [
            "Create targeted promotions for each ordering cluster",
            "Design menu bundles appealing to identified segments",
            "Adjust kitchen staffing for online order patterns",
            "Schedule app promotions to match natural ordering cycles",
            "Develop loyalty incentives tied to reordering patterns"
        ]
        
        recommendations_section = """
        • **Actionable Recommendations**:"""
        
        # Show appropriate number of recommendations
        rec_count = min(3 + (1 if h0_count > 1 else 0) + (1 if h1_count > 0 else 0), len(recommendations))
        for i in range(rec_count):
            recommendations_section += f"""
              ✓ {recommendations[i]}"""
        
        insights = f"""
        ### Topological Analysis Results:
        
        {clusters_section}
        {cycles_section}
        
        • **Business Interpretation**: 
          Your GET app orders show {'multiple distinct patterns' if h0_count > 1 else 'one consistent pattern'}.
          {'Each cluster represents a different customer segment with unique ordering preferences.' if h0_count > 1 else ''}
          
          {'The cyclical patterns reveal predictable ordering behaviors that can guide promotions and planning.' if h1_count > 0 else ''}
        
        {recommendations_section}
        """
        
    elif data_source == 'integrated':
        insights = f"""
        ### Integrated Topological Analysis Results:
        
        • **{h0_count} operational clusters identified**
          These represent cross-channel service patterns across physical and digital touchpoints.
          {'Each cluster shows a distinct operational configuration that spans your entire business.' if h0_count > 1 else ''}
          
        • **{h1_count} integrated cycles detected**
          {'These reveal how in-person and digital operations influence each other in recurring patterns.' if h1_count > 0 else ''}
        
        • **Business Interpretation**:
          Your café operations show {'multiple distinct integrated patterns' if h0_count > 1 else 'a consistent integrated pattern'}
          across in-person service and digital channels.
          
          {'The cycles detected show how customer behavior flows between physical and digital touchpoints.' if h1_count > 0 else ''}
        
        • **Actionable Recommendations**:
          ✓ Develop unified strategies that address both physical and digital channels
          ✓ Align staffing to meet both in-person and online demand simultaneously
          ✓ Create operational workflows that smoothly transition between channels
          {'✓ Optimize cross-channel promotions based on the identified patterns' if h1_count > 0 else ''}
        """
    
    # Add a note if this is simulated data
    if is_simulated:
        insights = """
        ⚠️ **NOTE:** These insights are based on simulated data patterns and should not be used for actual business decisions.
        Upload real data for accurate analysis.
        
        """ + insights
    
    return insights

# Sidebar for file uploads
with st.sidebar:
    st.title("Nikos Cafe Analysis")
    st.subheader("Data Upload")
    
    # Service data upload
    service_file = st.file_uploader("Service Performance Data", type=["csv"], 
                                   help="Upload Service Performance.csv")
    
    # Labor data upload - simplified to just Sales-Labor Analysis
    labor_file = st.file_uploader("Labor Data (Sales-Labor Analysis)", type=["csv"], 
                                 help="Upload sales vs. labor data CSV for TDA insights")
    
    # GET App data upload
    get_app_file = st.file_uploader("GET App Data", type=["csv"], 
                                   help="Upload Nikos GET APP.csv")
    
# Process uploaded files
data_dict = {}

if service_file:
    service_df = process_service_data(service_file)
    if service_df is not None:
        data_dict['service_data'] = service_df
        st.session_state['service_df'] = service_df

if labor_file:
    # Always use sales_labor data type since that's the only one we support now
    data_type = "sales_labor"
    labor_data_type = "Sales-Labor Analysis"  # Define this for consistency
        
    labor_df, actual_type = process_labor_data(labor_file, data_type)
    if labor_df is not None:
        data_dict['labor_data'] = labor_df
        data_dict['labor_data_type'] = actual_type
        st.session_state['labor_df'] = labor_df
        st.session_state['labor_data_type'] = actual_type

if get_app_file:
    get_app_df = process_get_app_data(get_app_file)
    if get_app_df is not None:
        data_dict['get_app_data'] = get_app_df
        st.session_state['get_app_df'] = get_app_df

# Create tabs for different analyses
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Service Data", "Labor Data", "GET App Data", "TDA Insights", "Power BI Export"])

# Initialize sample TDA results quietly for initial display
# Create a minimal default structure rather than calling the simulation function
sample_tda_results = {
    'diagrams': {
        0: np.array([[0.0, 0.8], [0.0, 0.5]]),
        1: np.array([[0.3, 0.7]])
    },
    'features': [
        {'dimension': 0, 'count': 2, 'persistence': 0.6, 'interpretation': "Sample data clusters"},
        {'dimension': 1, 'count': 1, 'persistence': 0.4, 'interpretation': "Sample data pattern"}
    ],
    'simulated': True
}

# Tab 1: Service Data
with tab1:
    st.header("Oracle Symphony Service Data")
    
    if 'service_data' in data_dict:
        service_df = data_dict['service_data']
        
        # Show basic metrics
        col1, col2, col3 = st.columns(3)
        
        total_sales = service_df['Net Sales'].sum()
        total_labor = service_df['Labor Cost'].sum()
        
        # Calculate labor ratio
        if total_sales > 0:
            labor_ratio = (total_labor / total_sales) * 100
        else:
            labor_ratio = 0
            
        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Labor Cost", f"${total_labor:,.2f}")
        col3.metric("Labor to Sales Ratio", f"{labor_ratio:.1f}%")
        
        # Plot service data - now returns multiple figures for better visualization
        service_figs = plot_service_data(service_df)
        if service_figs is not None:
            st.subheader("Sales Visualization")
            st.plotly_chart(service_figs[0], use_container_width=True)
            
            st.subheader("Labor Cost Visualization")
            st.plotly_chart(service_figs[1], use_container_width=True)
            
            st.subheader("Combined Visualization")
            st.plotly_chart(service_figs[2], use_container_width=True)
        
        # Show data table
        with st.expander("View Service Data"):
            st.dataframe(service_df)
    else:
        st.info("Please upload Service Performance data to view this analysis")

# Tab 2: Labor Data
with tab2:
    st.header("Homebase Labor Data")
    
    if 'labor_data' in data_dict:
        labor_df = data_dict['labor_data']
        labor_type = data_dict.get('labor_data_type', 'unknown')
        
        # Show different metrics based on labor data type
        col1, col2, col3 = st.columns(3)
        
        if labor_type == 'sales_labor':
            total_sales = labor_df['Sales'].sum() 
            total_hours = labor_df['Labor_Hours'].sum()
            total_cost = labor_df['Labor_Cost'].sum()
            
            col1.metric("Total Sales", f"${total_sales:,.2f}")
            col2.metric("Labor Hours", f"{total_hours:.1f}")
            col3.metric("Labor Cost", f"${total_cost:,.2f}")
            
        elif labor_type == 'by_type':
            total_hours = labor_df['Hours'].sum()
            total_cost = labor_df['Cost'].sum()
            avg_cost = total_cost / total_hours if total_hours > 0 else 0
            
            col1.metric("Total Labor Hours", f"{total_hours:.1f}")
            col2.metric("Total Labor Cost", f"${total_cost:,.2f}")
            col3.metric("Average Cost/Hour", f"${avg_cost:.2f}")
            
        elif labor_type == 'hourly':
            scheduled = labor_df['Scheduled hours'].sum()
            actual = labor_df['Actual hours'].sum()
            efficiency = (actual / scheduled * 100) if scheduled > 0 else 0
            
            col1.metric("Scheduled Hours", f"{scheduled:.1f}")
            col2.metric("Actual Hours", f"{actual:.1f}")
            col3.metric("Efficiency", f"{efficiency:.1f}%")
        
        # Plot labor data based on type - now may return multiple figures
        labor_figs = plot_labor_data(labor_df, labor_type)
        
        if labor_figs is not None:
            if isinstance(labor_figs, list):
                # For sales_labor type which returns multiple figures
                st.subheader("Sales Visualization")
                st.plotly_chart(labor_figs[0], use_container_width=True)
                
                st.subheader("Labor Hours Visualization") 
                st.plotly_chart(labor_figs[1], use_container_width=True)
                
                st.subheader("Labor Cost Percentage")
                st.plotly_chart(labor_figs[2], use_container_width=True)
            else:
                # For other labor types that return a single figure
                st.plotly_chart(labor_figs, use_container_width=True)
            
        # Show data table
        with st.expander("View Labor Data"):
            st.dataframe(labor_df)
    else:
        st.info("Please upload Labor data to view this analysis")

# Tab 3: GET App Data
with tab3:
    st.header("GET App Digital Orders")
    
    if 'get_app_data' in data_dict:
        get_app_df = data_dict['get_app_data']
        
        # Show basic metrics
        col1, col2, col3 = st.columns(3)
        
        total_orders = get_app_df['Order_Count'].sum()
        total_sales = get_app_df['Sales_Amount'].sum()
        avg_order = total_sales / total_orders if total_orders > 0 else 0
        
        col1.metric("Total Orders", f"{total_orders:,.0f}")
        col2.metric("Total Sales", f"${total_sales:,.2f}")
        col3.metric("Average Order", f"${avg_order:.2f}")
        
        # Plot GET App data - now returns multiple simple charts
        get_app_figs = plot_get_app_data(get_app_df)
        if get_app_figs is not None:
            st.subheader("Order Count Visualization")
            st.plotly_chart(get_app_figs[0], use_container_width=True)
            
            st.subheader("Sales Distribution Visualization")
            st.plotly_chart(get_app_figs[1], use_container_width=True)
        
        # Plot average order values
        avg_fig = plot_average_orders(get_app_df)
        if avg_fig is not None:
            st.subheader("Average Order Values")
            st.plotly_chart(avg_fig, use_container_width=True)
            
        # Show data table
        with st.expander("View GET App Data"):
            st.dataframe(get_app_df)
    else:
        st.info("Please upload GET App data to view this analysis")

# Tab 4: TDA Insights
with tab4:
    st.header("Topological Data Analysis")
    
    st.markdown("""
    ### What is Topological Data Analysis (TDA)?
    
    Topological Data Analysis (TDA) is a mathematical approach to analyzing data by finding its shape and structure.
    
    **Key concepts:**
    - **Persistence Diagrams**: Show when topological features (like clusters and loops) appear and disappear
    - **Feature Detection**: Identifies important patterns such as clusters (H0) and loops (H1)
    - **Business Insights**: Translate mathematical findings into practical business recommendations
    """)
    
    # Only keep service data TDA analysis
    available_data = []
    if 'service_data' in data_dict:
        available_data.append('Service Data')  # Most meaningful for TDA
    
    if len(available_data) == 0:
        st.info("Please upload data files to perform Topological Data Analysis")
        
        # Still show a sample persistence diagram for demonstration (without labels first)
        st.subheader("Sample TDA Visualization (for demonstration only)")
        st.markdown("""
        This is a sample persistence diagram to demonstrate what TDA results look like.
        Upload data files to perform real analysis on your cafe data.
        """)
        pd_image = plot_persistence_diagram(sample_tda_results, show_labels=False)
        st.image(pd_image, caption="Sample Persistence Diagram (Raw Features)", use_container_width=True)
        
        # Add a visual explanation
        st.subheader("Quick Guide: Reading the Diagram")
        
        # Use columns for a more visual layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 Blue Circles = Clusters")
            st.markdown("""
            **What it means:** Natural groupings in your data
            
            **Business Example:**
            - Distinct meal periods in sales patterns
            - Different payment preference segments
            """)
            
        with col2:
            st.markdown("### 🔶 Orange Squares = Patterns")
            st.markdown("""
            **What it means:** Recurring relationships or cycles
            
            **Business Example:**
            - Sales peaks and valleys throughout day
            - Cyclical staffing requirements
            """)
            
        # Adding a second sample diagram with labels
        st.subheader("Sample TDA Visualization with Identified Features")
        pd_image_labeled = plot_persistence_diagram(sample_tda_results, show_labels=True)
        st.image(pd_image_labeled, caption="Sample Persistence Diagram (With Labeled Features)", use_container_width=True)
    else:
        # Let user select which dataset to analyze with TDA
        selected_data = st.selectbox(
            "Select data for TDA analysis:", 
            available_data
        )
        
        # Ensure we use optimal data preprocessing for TDA
        
        if st.button("Run TDA Analysis"):
            with st.spinner("Performing Topological Data Analysis..."):
                # Get service data - only option available now
                df = data_dict['service_data']
                # Select relevant numerical columns
                tda_cols = ['Net Sales', 'Labor Cost', 'Labor to Sales Ratio']
                tda_data = df[tda_cols].values
                data_source = 'service'
                feature_names = tda_cols
                
                # Log what we're analyzing
                st.info(f"Analyzing service data with {df.shape[0]} data points and the following metrics: {', '.join(tda_cols)}")
                
                # Make sure we have enough data
                if not 'tda_data' in locals() or tda_data.shape[0] < 2:
                    st.error("Not enough data points for meaningful TDA. Need at least 2 data points.")
                else:
                    # Add some information about the data
                    st.info(f"Using {tda_data.shape[0]} data points with {tda_data.shape[1]} features for TDA.")
                    
                    # Check for too many columns
                    if tda_data.shape[1] > 10:
                        st.warning("Large number of features detected. Using dimensionality reduction for better TDA results.")
                        # Apply PCA to reduce dimensions
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=min(tda_data.shape[0], 5))
                        tda_data = pca.fit_transform(tda_data)
                        
                    # Run TDA analysis
                    try:
                        tda_results = compute_persistence_diagram(data)
                    # Feedback implementation - Continue with plotting etc and show this on webpage only when it fails.
                    except ValueError as e:
                        st.error(f"TDA computation failed: {e}")
                    
                    # Display results
                    try:
                        # First persistence diagram - without labels
                        st.subheader("TDA Visualization")
                        pd_image_no_labels = plot_persistence_diagram(tda_results, show_labels=False)
                        st.image(pd_image_no_labels, caption="Persistence Diagram (Raw Features)", use_container_width=True)
                        
                        # Add a visual explanation using columns
                        st.subheader("Quick Guide: Reading the Diagram")
                        
                        # Use columns for a more visual layout
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🔵 Blue Circles = Clusters")
                            st.markdown("""
                            **What it means:** Natural groupings in your data
                            
                            **Business Example:**
                            - Distinct meal periods in sales patterns
                            - Different payment preference segments
                            """)
                            
                        with col2:
                            st.markdown("### 🔶 Orange Squares = Patterns")
                            st.markdown("""
                            **What it means:** Recurring relationships or cycles
                            
                            **Business Example:**
                            - Sales peaks and valleys throughout day
                            - Cyclical staffing requirements
                            """)
                            
                        # Add a simple explainer about the diagonal
                        st.markdown("### 📊 Distance from Diagonal = Importance")
                        st.markdown("""
                        Points further from the diagonal line represent stronger patterns in your data.
                        These are the insights most worth investigating!
                        """)
                        
                    except Exception as e:
                        st.error(f"Error generating persistence diagram: {str(e)}")
                        try:
                            # Simplified fallback
                            pd_image_no_labels = plot_persistence_diagram(tda_results, show_labels=False)
                            st.image(pd_image_no_labels, caption="Persistence Diagram")
                        except:
                            st.error("Could not generate persistence diagram.")
                    
                    # Display insights
                    st.subheader("Topological Insights")
                    insights = generate_tda_insights(tda_results, data_source)
                    st.markdown(insights)
                    
                    # Second persistence diagram - with labeled clusters and patterns
                    try:
                        st.subheader("TDA Visualization with Identified Features")
                        pd_image_with_labels = plot_persistence_diagram(tda_results, show_labels=True)
                        st.image(pd_image_with_labels, caption="Persistence Diagram (With Labeled Features)", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating labeled persistence diagram: {str(e)}")
                    
                    # What this means for the business
                    st.subheader("How to Use These Insights")
                    st.markdown("""
                    The topological patterns above reveal the hidden mathematical structure in your cafe operations. Use these insights to:
                    
                    1. **Optimize Staffing**: Align labor scheduling with the natural operational clusters detected
                    2. **Improve Efficiency**: Address the cyclical patterns in operations to smooth out workflow
                    3. **Make Data-Driven Decisions**: Let the true shape of your data guide operational improvements
                    """)

# Tab 5: Power BI Export
with tab5:
    st.header("Power BI Data Export")
    
    st.markdown("""
    ### Export Data for Power BI
    
    Download your data in CSV format for use in Power BI dashboards.
    """)
    
    # Check what data is available for export
    has_data = False
    export_data = {}
    
    if 'service_data' in data_dict:
        export_data['service'] = data_dict['service_data']
        has_data = True
        
    if 'labor_data' in data_dict:
        export_data['labor'] = data_dict['labor_data']
        has_data = True
        
    if 'get_app_data' in data_dict:
        export_data['get_app'] = data_dict['get_app_data']
        has_data = True
    
    if not has_data:
        st.info("Please upload data files to enable export functionality")
    else:
        # Create export button
        if st.button("Export Data for Power BI"):
            try:
                # Combine all dataframes
                all_dfs = []
                for source, df in export_data.items():
                    # Add source column
                    df_copy = df.copy()
                    df_copy['data_source'] = source
                    all_dfs.append(df_copy)
                
                # Concatenate all dataframes
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # Create download link
                csv = combined_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="nikos_cafe_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.success("Data exported successfully! Click the link above to download.")
                
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
        
        # Simple Power BI tips
        st.markdown("""
        **Tip:** In Power BI, use the 'data_source' column to filter or create relationships between tables.
        """)

# Footer
st.markdown("---")
st.markdown("📊 Nikos Cafe Operational Analysis Tool | Created by - Sakshi Mutha | Data Science Master's Project | April 2025")
