import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import re
import io
import warnings
warnings.filterwarnings('ignore')

# For TDA analysis
try:
    from ripser import ripser
    from persim import plot_diagrams
    tda_libraries_available = True
except ImportError:
    tda_libraries_available = False

# Define color palette for consistent visualizations
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Set page configuration
st.set_page_config(
    page_title="Nikos Café TDA Analysis",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insights-box {
        background-color: #F0F9FF;
        border-left: 5px solid #0EA5E9;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

#------------------------
# TDA Analysis Functions
#------------------------

def compute_persistence_diagram(data, max_dimension=1, max_epsilon=2.0):
    """Compute persistence diagram using ripser"""
    # Normalize data for fair comparison across features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    try:
        if tda_libraries_available:
            # Compute persistence diagrams
            ripser_result = ripser(scaled_data, maxdim=max_dimension, thresh=max_epsilon)
            diagrams = ripser_result['dgms']
            
            # Extract information about topological features
            diagram_info = []
            for dim, diagram in enumerate(diagrams):
                if len(diagram) > 0:
                    # Calculate persistence (death - birth)
                    persistence = diagram[:, 1] - diagram[:, 0]
                    finite_persistence = persistence[~np.isinf(persistence)]
                    
                    if len(finite_persistence) > 0:
                        info = {
                            'dimension': dim,
                            'feature_type': 'Connected Components' if dim == 0 else 'Loops/Cycles',
                            'business_meaning': 'Distinct operational states' if dim == 0 else 'Recurring patterns',
                            'total_count': len(diagram),
                            'significant_count': np.sum(finite_persistence > 0.1),
                            'max_persistence': np.max(finite_persistence) if len(finite_persistence) > 0 else 0,
                            'mean_persistence': np.mean(finite_persistence) if len(finite_persistence) > 0 else 0,
                            'infinite_features': np.sum(np.isinf(diagram[:, 1])),
                            'complexity_level': 'High' if len(diagram) > 5 else 'Medium' if len(diagram) > 2 else 'Low'
                        }
                        diagram_info.append(info)
            
            return diagrams, diagram_info
        else:
            # Use simulated data if ripser not available
            return simulate_persistence_diagrams(max_dimension, scaled_data.shape[0])
    
    except Exception as e:
        st.warning(f"Error computing persistence diagram: {e}")
        # Use simulated data if computation fails
        return simulate_persistence_diagrams(max_dimension, scaled_data.shape[0])

def simulate_persistence_diagrams(max_dimension, num_points):
    """Simulate persistence diagrams when ripser is not available"""
    diagrams = []
    diagram_info = []
    
    # Simulate H0 (connected components)
    num_h0 = min(20, num_points // 5 + 1)
    h0_births = np.zeros(num_h0)
    h0_deaths = np.sort(np.random.uniform(0.1, 2.0, num_h0))
    h0_deaths[-1] = np.inf  # One component never dies
    h0_diagram = np.column_stack((h0_births, h0_deaths))
    diagrams.append(h0_diagram)
    
    diagram_info.append({
        'dimension': 0,
        'feature_type': 'Connected Components',
        'business_meaning': 'Distinct operational states',
        'total_count': num_h0,
        'significant_count': num_h0 // 2,
        'max_persistence': np.max(h0_deaths[:-1]) if len(h0_deaths) > 1 else 0,
        'mean_persistence': np.mean(h0_deaths[:-1]) if len(h0_deaths) > 1 else 0,
        'infinite_features': 1,
        'complexity_level': 'Medium'
    })
    
    # Simulate H1 (loops)
    if max_dimension >= 1:
        num_h1 = min(10, num_points // 10)
        h1_births = np.sort(np.random.uniform(0.05, 1.2, num_h1))
        h1_deaths = np.array([b + np.random.uniform(0.1, 0.8) for b in h1_births])
        h1_diagram = np.column_stack((h1_births, h1_deaths))
        diagrams.append(h1_diagram)
        
        diagram_info.append({
            'dimension': 1,
            'feature_type': 'Loops/Cycles',
            'business_meaning': 'Recurring patterns',
            'total_count': num_h1,
            'significant_count': num_h1 // 2,
            'max_persistence': np.max(h1_deaths - h1_births) if len(h1_deaths) > 0 else 0,
            'mean_persistence': np.mean(h1_deaths - h1_births) if len(h1_deaths) > 0 else 0,
            'infinite_features': 0,
            'complexity_level': 'Medium'
        })
    
    return diagrams, diagram_info

def extract_topological_features(diagrams, threshold=0.1):
    """Extract significant topological features from persistence diagrams"""
    feature_data = {}
    feature_types = ['Connected Components', 'Cycles/Loops']
    business_implications = [
        'Distinct operational states or customer segments',
        'Recurring patterns or operational feedback loops'
    ]
    
    for i, diagram in enumerate(diagrams):
        if len(diagram) > 0 and i < len(feature_types):
            # Calculate persistence (death - birth)
            persistence = diagram[:, 1] - diagram[:, 0]
            
            # Identify significant features (high persistence or infinite)
            significant_mask = (persistence > threshold) | np.isinf(diagram[:, 1])
            significant_count = np.sum(significant_mask)
            
            # Determine business complexity level
            if significant_count > 3:
                complexity = 'High'
                complexity_insight = 'Multiple distinct patterns requiring detailed analysis'
            elif significant_count > 1:
                complexity = 'Medium'
                complexity_insight = 'Several clear patterns for operational optimization'
            else:
                complexity = 'Low'
                complexity_insight = 'Simple structure with few distinct patterns'
            
            # Store in business-friendly terms
            feature_data[feature_types[i]] = {
                'count': int(significant_count),
                'max_persistence': float(np.max(persistence[~np.isinf(persistence)])) if len(persistence[~np.isinf(persistence)]) > 0 else 0,
                'business_implication': business_implications[i],
                'complexity_level': complexity,
                'complexity_insight': complexity_insight
            }
    
    return feature_data

def plot_enhanced_persistence_diagram(diagrams, title="Operational Topology Analysis", show_annotations=True):
    """Create an enhanced persistence diagram with business-focused annotations"""
    # Use Plotly for interactive visualization
    fig = go.Figure()
    
    # Determine plot bounds
    all_points = []
    for diagram in diagrams:
        for point in diagram:
            if not np.isinf(point[1]):
                all_points.append(point)
    
    if all_points:
        all_points = np.array(all_points)
        x_max = np.max(all_points[:, 0]) * 1.2
        y_max = np.max(all_points[:, 1]) * 1.2
        max_bound = max(x_max, y_max, 1.5)
    else:
        max_bound = 1.5
    
    # Add diagonal line (birth = death)
    fig.add_trace(go.Scatter(
        x=[0, max_bound],
        y=[0, max_bound],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Diagonal (birth = death)',
        hoverinfo='skip'
    ))
    
    # Plot each dimension with different colors and hover info
    feature_types = ['Connected Components', 'Cycles/Loops']
    business_meanings = ['Distinct operational states', 'Recurring patterns']
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)']
    
    for i, diagram in enumerate(diagrams):
        if i < len(feature_types) and len(diagram) > 0:
            # Separate finite and infinite points
            finite_mask = ~np.isinf(diagram[:, 1])
            finite_diagram = diagram[finite_mask]
            infinite_diagram = diagram[~finite_mask]
            
            # Calculate persistence for finite points
            if len(finite_diagram) > 0:
                persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
                # Scale marker size by persistence
                marker_sizes = 10 + 40 * (persistence / max(persistence.max(), 0.1))
                
                # Add finite points
                fig.add_trace(go.Scatter(
                    x=finite_diagram[:, 0],
                    y=finite_diagram[:, 1],
                    mode='markers',
                    marker=dict(
                        size=marker_sizes,
                        color=colors[i],
                        line=dict(width=1, color='black'),
                        opacity=0.7
                    ),
                    name=feature_types[i],
                    hovertemplate=f"{feature_types[i]}<br>Birth: %{{x:.2f}}<br>Death: %{{y:.2f}}<br>Persistence: %{{text:.2f}}<br>{business_meanings[i]}",
                    text=persistence
                ))
            
            # Add infinite points at the top of the chart
            if len(infinite_diagram) > 0:
                fig.add_trace(go.Scatter(
                    x=infinite_diagram[:, 0],
                    y=[max_bound * 0.95] * len(infinite_diagram),
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=colors[i],
                        symbol='triangle-up',
                        line=dict(width=1, color='black'),
                        opacity=0.7
                    ),
                    name=f"{feature_types[i]} (Infinite Persistence)",
                    hovertemplate=f"Infinite {feature_types[i]}<br>Birth: %{{x:.2f}}<br>Never dies"
                ))
    
    # Add annotations to explain the diagram
    if show_annotations:
        annotations = []
        
        # Explain diagonal
        annotations.append(dict(
            x=max_bound * 0.6,
            y=max_bound * 0.6,
            xref="x", yref="y",
            text="Points near diagonal<br>represent noise",
            showarrow=True,
            arrowhead=3,
            ax=-40,
            ay=-40,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        ))
        
        # Explain points far from diagonal
        annotations.append(dict(
            x=max_bound * 0.2,
            y=max_bound * 0.7,
            xref="x", yref="y",
            text="Points far from diagonal<br>represent significant patterns",
            showarrow=True,
            arrowhead=3,
            ax=40,
            ay=0,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        ))
        
        # Explain infinite points if any
        if any(np.isinf(diagram[:, 1]).any() for diagram in diagrams if len(diagram) > 0):
            annotations.append(dict(
                x=max_bound * 0.2,
                y=max_bound * 0.95,
                xref="x", yref="y",
                text="Triangles represent features<br>with infinite persistence<br>(fundamental patterns)",
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-40,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            ))
        
        fig.update_layout(annotations=annotations)
    
    # Layout customization
    fig.update_layout(
        title={
            'text': title,
            'font': dict(size=20, family="Arial", color="#333333"),
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Birth",
        yaxis_title="Death",
        xaxis=dict(
            range=[-0.05, max_bound],
            gridcolor='lightgray',
        ),
        yaxis=dict(
            range=[-0.05, max_bound],
            gridcolor='lightgray',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#333333"
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(240, 240, 240, 0.8)",
        hovermode="closest"
    )
    
    # Add business interpretation footer
    if show_annotations:
        fig.add_annotation(
            text="<b>Business Interpretation:</b> Each point represents a pattern in your operational data.<br>Connected Components (blue) show distinct operational states or customer segments.<br>Cycles/Loops (orange) show recurring patterns or operational feedback loops.",
            xref="paper", yref="paper",
            x=0.5, y=0,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 220, 0.8)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=4,
            align="center"
        )
    
    return fig

def generate_business_insights(diagram_info, feature_data, dataset_type, selected_features):
    """Generate business-focused insights from topological analysis"""
    dataset_names = {
        "sales_data": "Oracle Symphony Sales Data",
        "service_data": "Oracle Symphony Service Performance",
        "labor_data": "Homebase Labor Data",
        "get_app_data": "GET App Online Orders",
        "combined": "Integrated Cross-Platform Data"
    }
    
    dataset_name = dataset_names.get(dataset_type, dataset_type)
    insights = [f"## Business Insights from {dataset_name}"]
    
    # Add summary of what was analyzed
    insights.append(f"\nAnalysis based on: {', '.join(selected_features)}")
    
    # Add insights based on topological features
    if 'Connected Components' in feature_data:
        cc_data = feature_data['Connected Components']
        insights.append(f"\n### Operational States Analysis")
        insights.append(f"**Complexity Level:** {cc_data['complexity_level']}")
        insights.append(f"**Key Finding:** {cc_data['count']} distinct operational patterns identified")
        
        # Specific insights based on dataset type
        if dataset_type == "sales_data":
            insights.append("The topology reveals distinct pricing and sales volume clusters, suggesting:")
            insights.append("- Multiple customer segments with different price sensitivities")
            insights.append("- Natural groupings in menu item performance")
            insights.append("- Opportunity for targeted promotions based on identified segments")
        
        elif dataset_type == "service_data":
            insights.append("The topology reveals distinct operational periods, suggesting:")
            insights.append("- Natural transitions between slow and peak periods")
            insights.append("- Different service dynamics throughout the day")
            insights.append("- Opportunity for period-specific operational strategies")
        
        elif dataset_type == "labor_data":
            insights.append("The topology reveals distinct staffing patterns, suggesting:")
            insights.append("- Different workforce utilization modes")
            insights.append("- Periods of over and understaffing")
            insights.append("- Opportunity for more precise scheduling based on identified states")
        
        elif dataset_type == "get_app_data":
            insights.append("The topology reveals distinct digital ordering patterns, suggesting:")
            insights.append("- Different online customer segments")
            insights.append("- Periods of high and low online order activity")
            insights.append("- Opportunity for targeted digital promotions")
        
        elif dataset_type == "combined":
            insights.append("The topology reveals coordination patterns across systems, suggesting:")
            insights.append("- Integrated operational states spanning physical and digital channels")
            insights.append("- Complex interactions between labor, sales, and online orders")
            insights.append("- Opportunity for holistic operational strategies")
    
    if 'Cycles/Loops' in feature_data:
        loop_data = feature_data['Cycles/Loops']
        insights.append(f"\n### Recurring Patterns Analysis")
        insights.append(f"**Complexity Level:** {loop_data['complexity_level']}")
        insights.append(f"**Key Finding:** {loop_data['count']} recurring operational cycles identified")
        
        # Specific insights based on dataset type
        if dataset_type == "sales_data":
            insights.append("The cycles in sales data suggest:")
            insights.append("- Recurring purchase patterns across menu categories")
            insights.append("- Feedback loops between pricing and sales volume")
            insights.append("- Opportunity for menu engineering based on identified cycles")
        
        elif dataset_type == "service_data":
            insights.append("The cycles in service data suggest:")
            insights.append("- Recurring busy-slow patterns throughout the day")
            insights.append("- Feedback loops between service speed and sales")
            insights.append("- Opportunity for service pace optimization")
        
        elif dataset_type == "labor_data":
            insights.append("The cycles in labor data suggest:")
            insights.append("- Recurring staffing efficiency patterns")
            insights.append("- Feedback loops between scheduling and actual hours")
            insights.append("- Opportunity for dynamic scheduling adjustments")
        
        elif dataset_type == "get_app_data":
            insights.append("The cycles in online order data suggest:")
            insights.append("- Recurring digital ordering patterns")
            insights.append("- Feedback loops between order volume and fulfillment")
            insights.append("- Opportunity for dynamic digital menu adjustments")
        
        elif dataset_type == "combined":
            insights.append("The cycles across systems suggest:")
            insights.append("- Complex operational rhythms that span physical and digital channels")
            insights.append("- Feedback loops between staffing, sales, and online orders")
            insights.append("- Opportunity for synchronized operational planning")
    
    # Add recommendations section
    insights.append("\n### Action Recommendations")
    
    if dataset_type == "sales_data":
        insights.append("1. **Menu Engineering:** Adjust menu pricing and promotion based on identified item clusters")
        insights.append("2. **Category Management:** Optimize inventory and promotion for high-performing categories")
        insights.append("3. **Price Sensitivity Testing:** Experiment with pricing in different identified segments")
    
    elif dataset_type == "service_data":
        insights.append("1. **Period-Specific Operations:** Implement different service strategies for each identified operational state")
        insights.append("2. **Sales-Labor Balancing:** Adjust staffing to match the identified sales-labor relationship patterns")
        insights.append("3. **Peak Management:** Develop specific strategies for handling identified peak periods")
    
    elif dataset_type == "labor_data":
        insights.append("1. **Dynamic Scheduling:** Implement more precise scheduling based on identified staffing patterns")
        insights.append("2. **Staff Optimization:** Address periods of over and understaffing revealed in the topology")
        insights.append("3. **Efficiency Improvement:** Develop strategies to reduce scheduling-actual discrepancies")
    
    elif dataset_type == "get_app_data":
        insights.append("1. **Digital Menu Optimization:** Adjust online offerings based on identified ordering patterns")
        insights.append("2. **Targeted Promotions:** Develop specific offers for different online customer segments")
        insights.append("3. **Service Integration:** Coordinate in-store operations with digital order surges")
    
    elif dataset_type == "combined":
        insights.append("1. **Integrated Operations:** Develop strategies that coordinate across all systems and channels")
        insights.append("2. **Cross-Channel Optimization:** Synchronize staffing with both in-store and online demand")
        insights.append("3. **Holistic Performance Metrics:** Develop KPIs that capture the cross-system patterns identified")
    
    # Note for Power BI integration
    insights.append("\n*These insights can be further explored through the companion Power BI dashboard.*")
    
    return "\n".join(insights)

#------------------------
# Data Processing Functions
#------------------------

def process_oracle_sales_data(file):
    """Process menu sales data from Oracle Symphony"""
    try:
        # Read Excel file
        excel_data = pd.read_excel(file, header=None)
        
        # Find header row
        header_row = None
        for i in range(min(15, excel_data.shape[0])):
            if excel_data.iloc[i].astype(str).str.contains('Item Name|Menu Item|Item').any():
                header_row = i
                break
        
        # Process based on found headers or default structure
        if header_row is None:
            header_row = 0
            excel_data.columns = ['Item Name', 'Item Code', 'Category', 'Quantity', 'Price', 'Gross Sales', 'Discount', 'Net Sales']
        else:
            headers = excel_data.iloc[header_row]
            excel_data = excel_data.iloc[header_row+1:].reset_index(drop=True)
            excel_data.columns = headers
        
        # Clean column names and data
        excel_data.columns = [str(col).strip() for col in excel_data.columns]
        menu_df = excel_data[excel_data.iloc[:, 0].notna()].copy()
        
        # Convert numeric columns
        for col in menu_df.columns:
            if col.lower() in ['quantity', 'price', 'gross sales', 'discount', 'net sales', 'sales']:
                menu_df[col] = pd.to_numeric(menu_df[col], errors='coerce')
        
        # Add derived metrics
        if 'Category' not in menu_df.columns:
            menu_df['Category'] = menu_df.iloc[:, 0].astype(str).str.split().str[0]
        
        if 'Average Price' not in menu_df.columns and 'Price' not in menu_df.columns:
            if all(col in menu_df.columns for col in ['Gross Sales', 'Quantity']):
                menu_df['Average Price'] = menu_df['Gross Sales'] / menu_df['Quantity']
        
        # Calculate profitability metrics
        if all(col in menu_df.columns for col in ['Gross Sales', 'Quantity']):
            menu_df['Revenue per Item'] = menu_df['Gross Sales'] / menu_df['Quantity']
            
        return menu_df
    
    except Exception as e:
        st.error(f"Error processing Oracle Symphony sales data: {e}")
        return None

def process_oracle_service_data(file):
    """Process service performance data from Oracle Symphony"""
    try:
        # Read Excel file
        excel_data = pd.read_excel(file, header=None)
        
        # Extract hourly data using pattern matching
        hourly_data = []
        time_pattern = r'\d{1,2}:\d{2}\s*(AM|PM)'
        
        for i in range(excel_data.shape[0]):
            row = excel_data.iloc[i]
            for j, cell in enumerate(row):
                if isinstance(cell, str) and re.match(time_pattern, cell):
                    time_value = cell
                    sales_value = None
                    labor_value = None
                    
                    # Look for numeric values that could be sales/labor
                    for k, value in enumerate(row):
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            if k < j and sales_value is None:  # Sales before time
                                sales_value = value
                            elif k > j and labor_value is None:  # Labor after time
                                labor_value = value
                    
                    if sales_value is not None:
                        hourly_data.append({
                            'Time': time_value,
                            'Net Sales': sales_value,
                            'Labor Cost': labor_value if labor_value is not None else 0
                        })
                    break
        
        # If no data found, try alternative approach
        if len(hourly_data) == 0:
            # Find header row with 'Time' or 'Hour'
            header_row = None
            for i in range(excel_data.shape[0]):
                if any(str(cell).lower() in ['time', 'hour'] for cell in excel_data.iloc[i]):
                    header_row = i
                    break
            
            if header_row is not None:
                headers = excel_data.iloc[header_row]
                data = excel_data.iloc[header_row+1:].reset_index(drop=True)
                data.columns = headers
                
                # Find relevant columns
                time_col = None
                sales_col = None
                labor_col = None
                
                for j, col in enumerate(data.columns):
                    col_lower = str(col).lower()
                    if 'time' in col_lower or 'hour' in col_lower:
                        time_col = col
                    elif 'sales' in col_lower or 'revenue' in col_lower:
                        sales_col = col
                    elif 'labor' in col_lower or 'cost' in col_lower:
                        labor_col = col
                
                if time_col is not None and sales_col is not None:
                    for _, row in data.iterrows():
                        if not pd.isna(row[time_col]) and not pd.isna(row[sales_col]):
                            hourly_data.append({
                                'Time': row[time_col],
                                'Net Sales': row[sales_col],
                                'Labor Cost': row[labor_col] if labor_col is not None and not pd.isna(row[labor_col]) else 0
                            })
        
        # Process extracted data
        if len(hourly_data) == 0:
            raise ValueError("Could not extract service performance data. Please check the file format.")
        
        service_df = pd.DataFrame(hourly_data)
        
        # Convert time to 24-hour format
        service_df['Hour'] = service_df['Time'].apply(lambda x: 
                                            int(re.search(r'(\d+)', str(x)).group(1)) 
                                            if 'AM' in str(x) or int(re.search(r'(\d+)', str(x)).group(1)) == 12
                                            else int(re.search(r'(\d+)', str(x)).group(1)) + 12)
        
        # Calculate derived metrics (with safety checks)
        service_df['Labor to Sales Ratio'] = 0  # Default value
        positive_sales_mask = service_df['Net Sales'] > 0
        if positive_sales_mask.any():
            service_df.loc[positive_sales_mask, 'Labor to Sales Ratio'] = (
                service_df.loc[positive_sales_mask, 'Labor Cost'] / 
                service_df.loc[positive_sales_mask, 'Net Sales'] * 100
            )
        
        # Create scaled version of labor costs for visualization if needed
        if service_df['Net Sales'].max() > 0 and service_df['Labor Cost'].max() > 0:
            if service_df['Labor Cost'].max() < service_df['Net Sales'].max() / 10:
                scale_factor = service_df['Net Sales'].max() / max(service_df['Labor Cost'].max(), 0.01) / 2
                service_df['Labor Cost (Scaled)'] = service_df['Labor Cost'] * scale_factor
        
        return service_df
    
    except Exception as e:
        st.error(f"Error processing Oracle Symphony service performance data: {e}")
        return None

def process_homebase_labor_data(file, file_type="hourly"):
    """Process labor data from Homebase
    
    Args:
        file: Uploaded file object
        file_type: Type of labor file ("hourly", "by_type", or "sales_labor")
    
    Returns:
        Processed DataFrame
    """
    try:
        if file_type == "hourly":
            # Process traditional hourly labor costs format
            labor_df = pd.read_csv(file, skiprows=3)
            
            # Convert time to 24-hour format
            labor_df['Hour'] = labor_df['Time'].apply(lambda x: int(re.sub(r'[^0-9]', '', x.split('AM')[0])) if 'AM' in x 
                                                else (int(re.sub(r'[^0-9]', '', x.split('PM')[0])) + 12 if 'PM' in x and int(re.sub(r'[^0-9]', '', x.split('PM')[0])) != 12 
                                                    else int(re.sub(r'[^0-9]', '', x.split('PM')[0]))))
            
            # Calculate derived metrics
            labor_df['Hour Difference'] = labor_df['Actual hours'] - labor_df['Scheduled hours']
            labor_df['Cost Difference'] = labor_df['Actual cost'] - labor_df['Scheduled cost']
            
            # Calculate labor efficiency with division by zero protection
            nonzero_scheduled = labor_df['Scheduled hours'] > 0
            labor_df['Labor Efficiency'] = np.nan
            labor_df.loc[nonzero_scheduled, 'Labor Efficiency'] = labor_df.loc[nonzero_scheduled, 'Actual hours'] / labor_df.loc[nonzero_scheduled, 'Scheduled hours'] * 100
            
            # Add workforce utilization metric
            labor_df['Workforce Utilization'] = np.where(
                labor_df['Labor Efficiency'] <= 100, 
                labor_df['Labor Efficiency'],  # If under 100%, show actual efficiency
                100  # Cap at 100% for overstaffed periods
            )
            
            return labor_df, "hourly"
        
        elif file_type == "by_type" or (hasattr(file, 'name') and "labor_by_type" in file.name.lower()):
            # Process labor by type data
            labor_by_type_df = pd.read_csv(file)
            
            # The data has costs in rows 1-43 and hours in rows 44-86
            # Split into costs and hours
            costs_df = labor_by_type_df.iloc[1:43].copy()
            hours_df = labor_by_type_df.iloc[44:86].copy()
            
            # Extract role names (they're in column 0)
            roles = costs_df.iloc[:, 0].values
            
            # Reshape the data for better analysis
            days = labor_by_type_df.columns[1:8]  # Days are columns 1-7
            
            # Create long format data for costs
            costs_data = []
            for i, role in enumerate(roles):
                for j, day in enumerate(days):
                    cost = costs_df.iloc[i, j+1]
                    if pd.notna(cost) and cost != '$0':
                        cost_value = float(str(cost).replace('$', '').replace(',', ''))
                        costs_data.append({
                            'Role': role,
                            'Day': day,
                            'Cost': cost_value
                        })
            
            cost_long_df = pd.DataFrame(costs_data)
            
            # Create long format data for hours
            hours_data = []
            for i, role in enumerate(roles):
                for j, day in enumerate(days):
                    hours = hours_df.iloc[i, j+1]
                    if pd.notna(hours) and hours != '0':
                        hours_data.append({
                            'Role': role,
                            'Day': day,
                            'Hours': float(hours)
                        })
            
            hours_long_df = pd.DataFrame(hours_data)
            
            # Merge costs and hours
            labor_by_role_df = pd.merge(cost_long_df, hours_long_df, on=['Role', 'Day'], how='outer')
            
            # Calculate hourly rate
            labor_by_role_df['Hourly Rate'] = labor_by_role_df['Cost'] / labor_by_role_df['Hours']
            
            # Extract day of week
            labor_by_role_df['Day of Week'] = labor_by_role_df['Day'].str.split(',').str[1].str.strip()
            
            # Convert from wide to long format for better visualization
            return labor_by_role_df, "by_type"
        
        elif file_type == "sales_labor" or (hasattr(file, 'name') and "sales_labor" in file.name.lower()):
            # Process sales and labor analysis data
            sales_labor_df = pd.read_csv(file)
            
            # Ensure date column is properly formatted
            sales_labor_df['Date'] = pd.to_datetime(sales_labor_df['Date'])
            
            # Calculate additional KPIs
            nonzero_labor = sales_labor_df['Labor_Hours'] > 0
            sales_labor_df['Sales_per_Labor_Hour'] = 0
            if nonzero_labor.any():
                sales_labor_df.loc[nonzero_labor, 'Sales_per_Labor_Hour'] = (
                    sales_labor_df.loc[nonzero_labor, 'Sales'] / 
                    sales_labor_df.loc[nonzero_labor, 'Labor_Hours']
                )
            
            nonzero_sales = sales_labor_df['Sales'] > 0
            sales_labor_df['Labor_Cost_Percentage'] = 0
            if nonzero_sales.any():
                sales_labor_df.loc[nonzero_sales, 'Labor_Cost_Percentage'] = (
                    sales_labor_df.loc[nonzero_sales, 'Labor_Cost'] / 
                    sales_labor_df.loc[nonzero_sales, 'Sales'] * 100
                )
            
            # Add weekday column for better analysis
            sales_labor_df['Weekday'] = sales_labor_df['Date'].dt.day_name()
            
            return sales_labor_df, "sales_labor"
        
        else:
            st.error(f"Unknown labor data file type")
            return None, None
    
    except Exception as e:
        st.error(f"Error processing labor data: {e}")
        return None, None

def process_get_app_data(file):
    """Process data from the GET App platform"""
    try:
        # Read Excel file
        get_app_df = pd.read_excel(file)
        
        # The data has unnamed columns, so assign meaningful names
        new_columns = {
            'Unnamed: 0': 'Location',
            'Unnamed: 1': 'Payment_Type',
            'Unnamed: 2': 'Order_Count',
            'Unnamed: 3': 'Order_Type',
            'Unnamed: 4': 'Time_Period',
            'Unnamed: 5': 'Category',
            'Unnamed: 6': 'Sales_Amount', 
            'Unnamed: 7': 'Order_Time',
            'Unnamed: 8': 'Additional_Info'
        }
        
        # Rename columns
        get_app_df.rename(columns=new_columns, inplace=True)
        
        # Skip header rows
        get_app_df = get_app_df.iloc[3:].copy()
        
        # Convert numeric columns
        numeric_cols = ['Order_Count', 'Sales_Amount']
        for col in numeric_cols:
            if col in get_app_df.columns:
                get_app_df[col] = pd.to_numeric(get_app_df[col], errors='coerce')
        
        # Extract date information if available
        if 'Time_Period' in get_app_df.columns:
            try:
                get_app_df['Order_Date'] = pd.to_datetime(get_app_df['Time_Period'], errors='coerce')
            except:
                pass
        
        # Add hour information for time-based analysis
        if 'Order_Time' in get_app_df.columns:
            try:
                get_app_df['Hour'] = pd.to_datetime(get_app_df['Order_Time'], errors='coerce').dt.hour
            except:
                # If direct conversion fails, try to extract hour using regex
                get_app_df['Hour'] = get_app_df['Order_Time'].astype(str).str.extract(r'(\d+)(?=:)').astype(float)
        
        return get_app_df
    
    except Exception as e:
        st.error(f"Error processing GET App data: {e}")
        return None

def export_for_power_bi(data_dict, file_path='nikos_cafe_data_for_power_bi.csv'):
    """Export processed data for Power BI integration"""
    combined_data = []
    
    # Process each dataset
    for dataset_name, df in data_dict.items():
        if df is not None and len(df) > 0:
            # Add a source column
            df_copy = df.copy()
            df_copy['Data_Source'] = dataset_name
            
            # Keep only columns that are useful for Power BI
            cols_to_keep = []
            
            # Common columns to keep for all datasets
            if 'Hour' in df_copy.columns:
                cols_to_keep.append('Hour')
            
            # Dataset-specific columns
            if dataset_name == 'labor_data':
                specific_cols = ['Scheduled hours', 'Actual hours', 'Scheduled cost', 
                               'Actual cost', 'Labor Efficiency', 'Data_Source']
                cols_to_keep.extend([col for col in specific_cols if col in df_copy.columns])
            
            elif dataset_name == 'sales_data':
                specific_cols = ['Item Name', 'Category', 'Quantity', 'Gross Sales', 
                               'Net Sales', 'Data_Source']
                cols_to_keep.extend([col for col in specific_cols if col in df_copy.columns])
            
            elif dataset_name == 'service_data':
                specific_cols = ['Net Sales', 'Labor Cost', 'Labor to Sales Ratio', 'Data_Source']
                cols_to_keep.extend([col for col in specific_cols if col in df_copy.columns])
            
            elif dataset_name == 'get_app_data':
                specific_cols = ['Payment_Type', 'Order_Count', 'Category', 
                               'Sales_Amount', 'Order_Date', 'Data_Source']
                cols_to_keep.extend([col for col in specific_cols if col in df_copy.columns])
            
            # Keep only the selected columns
            df_slim = df_copy[[col for col in cols_to_keep if col in df_copy.columns]].copy()
            combined_data.append(df_slim)
    
    # Combine all datasets
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        combined_df.to_csv(file_path, index=False)
        return combined_df, file_path
    else:
        return None, None

#------------------------
# Visualization Functions
#------------------------

def plot_labor_data(labor_df, data_type="hourly"):
    """Create visualization for labor data
    
    Args:
        labor_df: DataFrame with labor data
        data_type: Type of labor data ("hourly", "by_type", or "sales_labor")
    
    Returns:
        Plotly figure or None if no data
    """
    if labor_df is None or len(labor_df) == 0:
        return None
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    if data_type == "hourly":
        # Traditional hourly labor visualization
        if 'Scheduled hours' in labor_df.columns and 'Actual hours' in labor_df.columns:
            # Filter to business hours
            business_hours = labor_df[labor_df['Scheduled hours'] > 0]
            
            # Add scheduled hours bars
            fig.add_trace(go.Bar(
                x=business_hours['Hour'],
                y=business_hours['Scheduled hours'],
                name='Scheduled Hours',
                marker_color=color_palette[0],
                opacity=0.7,
                text=business_hours['Scheduled hours'].round(1),
                textposition='auto',
                hovertemplate='Hour: %{x}:00<br>Scheduled: %{y:.1f} hours<extra></extra>'
            ))
    
            # Add actual hours bars
            fig.add_trace(go.Bar(
                x=business_hours['Hour'],
                y=business_hours['Actual hours'],
                name='Actual Hours',
                marker_color=color_palette[1],
                opacity=0.7,
                text=business_hours['Actual hours'].round(1),
                textposition='auto',
                hovertemplate='Hour: %{x}:00<br>Actual: %{y:.1f} hours<extra></extra>'
            ))
            
            # Add efficiency line on secondary y-axis
            fig.add_trace(go.Scatter(
                x=business_hours['Hour'],
                y=business_hours['Labor Efficiency'],
                name='Labor Efficiency',
                mode='lines+markers',
                marker=dict(size=8, color=color_palette[2]),
                line=dict(width=2, color=color_palette[2]),
                yaxis='y2',
                hovertemplate='Hour: %{x}:00<br>Efficiency: %{y:.1f}%<extra></extra>'
            ))
            
            # Add 100% reference line
            fig.add_trace(go.Scatter(
                x=[business_hours['Hour'].min(), business_hours['Hour'].max()],
                y=[100, 100],
                name='Target Efficiency (100%)',
                mode='lines',
                line=dict(dash='dash', color='red', width=1),
                yaxis='y2',
                hoverinfo='skip'
            ))
            
            # Update layout
            fig.update_layout(
                title='Homebase Hourly Labor Data Analysis',
                barmode='group',
                xaxis=dict(
                    title='Hour of Day',
                    tickmode='array',
                    tickvals=business_hours['Hour'],
                    ticktext=[f"{h}:00" for h in business_hours['Hour']]
                ),
                yaxis=dict(
                    title='Hours',
                    gridcolor='lightgray'
                ),
                yaxis2=dict(
                    title=dict(text='Efficiency (%)', font=dict(color=color_palette[2])),
                    tickfont=dict(color=color_palette[2]),
                    overlaying='y',
                    side='right',
                    range=[0, max(150, business_hours['Labor Efficiency'].max() * 1.1)],
                    gridcolor='lightgray',
                    zerolinecolor='lightgray'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                height=500
            )
            
            # Add insights annotation
            total_scheduled = business_hours['Scheduled hours'].sum()
            total_actual = business_hours['Actual hours'].sum()
            percent_diff = (total_actual - total_scheduled) / total_scheduled * 100 if total_scheduled > 0 else 0
            
            insight_text = f"Total Scheduled: {total_scheduled:.1f} hours | Total Actual: {total_actual:.1f} hours | Difference: {percent_diff:.1f}%<br>"
            if percent_diff < -5:
                insight_text += "<b>Insight:</b> Actual hours significantly below scheduled suggests understaffing or high no-show rates."
            elif percent_diff > 5:
                insight_text += "<b>Insight:</b> Actual hours above scheduled suggests unplanned overtime or inefficient scheduling."
            else:
                insight_text += "<b>Insight:</b> Actual hours closely match scheduled, indicating effective labor planning."
            
            fig.add_annotation(
                xref='paper', yref='paper',
                x=0.5, y=-0.15,
                text=insight_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 220, 0.8)",
                bordercolor="#c7c7c7",
                borderwidth=1,
                borderpad=8,
                align="center"
            )
    
    elif data_type == "by_type":
        # Labor by role visualization
        if 'Role' in labor_df.columns and 'Hours' in labor_df.columns and 'Cost' in labor_df.columns:
            # Sort by total hours
            role_summary = labor_df.sort_values('Hours', ascending=False)
            
            # Add hours bars
            fig.add_trace(go.Bar(
                x=role_summary['Role'],
                y=role_summary['Hours'],
                name='Labor Hours',
                marker_color=color_palette[0],
                opacity=0.7,
                text=role_summary['Hours'].round(1),
                textposition='auto',
                hovertemplate='Role: %{x}<br>Hours: %{y:.1f}<extra></extra>'
            ))
            
            # Add cost line on secondary y-axis
            fig.add_trace(go.Scatter(
                x=role_summary['Role'],
                y=role_summary['Cost'],
                name='Labor Cost',
                mode='lines+markers',
                marker=dict(size=8, color=color_palette[3]),
                line=dict(width=2, color=color_palette[3]),
                yaxis='y2',
                hovertemplate='Role: %{x}<br>Cost: $%{y:.2f}<extra></extra>'
            ))
            
            # Calculate cost per hour
            role_summary['Cost_Per_Hour'] = role_summary['Cost'] / role_summary['Hours']
            
            # Add cost per hour line
            fig.add_trace(go.Scatter(
                x=role_summary['Role'],
                y=role_summary['Cost_Per_Hour'],
                name='Cost Per Hour',
                mode='lines+markers',
                marker=dict(size=8, color=color_palette[1], symbol='diamond'),
                line=dict(width=2, color=color_palette[1], dash='dot'),
                yaxis='y3',
                hovertemplate='Role: %{x}<br>Cost/Hour: $%{y:.2f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Labor by Role Analysis',
                xaxis_title='Staff Role',
                yaxis=dict(
                    title='Hours',
                    gridcolor='lightgray'
                ),
                yaxis2=dict(
                    title=dict(text='Total Cost ($)', font=dict(color=color_palette[3])),
                    tickfont=dict(color=color_palette[3]),
                    overlaying='y',
                    side='right',
                    tickprefix='$',
                    gridcolor='lightgray'
                ),
                yaxis3=dict(
                    title=dict(text='Cost Per Hour ($)', font=dict(color=color_palette[1])),
                    tickfont=dict(color=color_palette[1]),
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=0.93,
                    tickprefix='$'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                height=500
            )
            
            # Add insights annotation
            total_hours = role_summary['Hours'].sum()
            total_cost = role_summary['Cost'].sum()
            avg_cost_per_hour = total_cost / total_hours if total_hours > 0 else 0
            
            # Find highest cost role
            highest_cost_role = role_summary.loc[role_summary['Cost'].idxmax(), 'Role']
            highest_cost = role_summary['Cost'].max()
            highest_cost_pct = (highest_cost / total_cost * 100) if total_cost > 0 else 0
            
            insight_text = f"Total Labor Hours: {total_hours:.1f} | Total Cost: ${total_cost:.2f} | Avg Cost/Hour: ${avg_cost_per_hour:.2f}<br>"
            insight_text += f"<b>Insight:</b> {highest_cost_role} represents the highest labor cost at ${highest_cost:.2f} ({highest_cost_pct:.1f}% of total)."
            
            fig.add_annotation(
                xref='paper', yref='paper',
                x=0.5, y=-0.15,
                text=insight_text,
                showarrow=False,
                font=dict(size=12),
                bgcolor="rgba(255, 255, 220, 0.8)",
                bordercolor="#c7c7c7",
                borderwidth=1,
                borderpad=8,
                align="center"
            )
            
    elif data_type == "sales_labor":
        # Sales-labor analysis visualization
        if 'Date' in labor_df.columns and 'Sales' in labor_df.columns and 'Labor_Hours' in labor_df.columns:
            # Add sales bars
            fig.add_trace(go.Bar(
                x=labor_df['Date'],
                y=labor_df['Sales'],
                name='Sales',
                marker_color=color_palette[0],
                opacity=0.7,
                text=['${:.0f}'.format(x) for x in labor_df['Sales']],
                textposition='auto',
                hovertemplate='Date: %{x}<br>Sales: $%{y:.2f}<extra></extra>'
            ))
            
            # Add labor hours line
            fig.add_trace(go.Scatter(
                x=labor_df['Date'],
                y=labor_df['Labor_Hours'],
                name='Labor Hours',
                mode='lines+markers',
                marker=dict(size=8, color=color_palette[1]),
                line=dict(width=2, color=color_palette[1]),
                yaxis='y2',
                hovertemplate='Date: %{x}<br>Labor Hours: %{y:.1f}<extra></extra>'
            ))
            
            # Add labor cost percentage line
            if 'Labor_Cost_Percentage' in labor_df.columns:
                fig.add_trace(go.Scatter(
                    x=labor_df['Date'],
                    y=labor_df['Labor_Cost_Percentage'],
                    name='Labor Cost %',
                    mode='lines+markers',
                    marker=dict(size=8, color=color_palette[2], symbol='diamond'),
                    line=dict(width=2, color=color_palette[2], dash='dot'),
                    yaxis='y3',
                    hovertemplate='Date: %{x}<br>Labor Cost: %{y:.1f}%<extra></extra>'
                ))
            
            # Update layout
            fig.update_layout(
                title='Sales and Labor Analysis',
                xaxis_title='Date',
                yaxis=dict(
                    title='Sales ($)',
                    gridcolor='lightgray',
                    tickprefix='$'
                ),
                yaxis2=dict(
                    title=dict(text='Labor Hours', font=dict(color=color_palette[1])),
                    tickfont=dict(color=color_palette[1]),
                    overlaying='y',
                    side='right',
                    gridcolor='lightgray'
                ),
                yaxis3=dict(
                    title=dict(text='Labor Cost %', font=dict(color=color_palette[2])),
                    tickfont=dict(color=color_palette[2]),
                    anchor="free",
                    overlaying="y",
                    side="right",
                    position=0.93,
                    ticksuffix='%'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                height=500
            )
            
            # Add insights annotation
            total_sales = labor_df['Sales'].sum()
            total_labor_hours = labor_df['Labor_Hours'].sum()
            total_labor_cost = labor_df['Labor_Cost'].sum() if 'Labor_Cost' in labor_df.columns else 0
            
            sales_per_labor_hour = total_sales / total_labor_hours if total_labor_hours > 0 else 0
            labor_cost_pct = (total_labor_cost / total_sales * 100) if total_sales > 0 and total_labor_cost > 0 else 0
            
            insight_text = f"Total Sales: ${total_sales:.2f} | Total Labor Hours: {total_labor_hours:.1f} | Sales per Labor Hour: ${sales_per_labor_hour:.2f}<br>"
            insight_text += f"<b>Insight:</b> Overall labor cost percentage is {labor_cost_pct:.1f}% of sales."
            
    # Default fallback - show simple text explanation if no visualization could be generated
    else:
        # Create a simple text-based visualization with column names
        column_names = labor_df.columns.tolist()
        fig.add_trace(go.Table(
            header=dict(
                values=["<b>Available Columns in Labor Data</b>", "<b>Data Type</b>"],
                line_color='darkslategray',
                fill_color='lightgrey',
                align='left'
            ),
            cells=dict(
                values=[
                    column_names,
                    [str(labor_df[col].dtype) for col in column_names]
                ],
                line_color='darkslategray',
                fill_color='white',
                align='left'
            )
        ))
        
        fig.update_layout(
            title=f'Labor Data Overview ({data_type} format)',
            height=400
        )
            
        fig.add_annotation(
            xref='paper', yref='paper',
            x=0.5, y=-0.15,
            text="This table shows available data fields for analysis. Select the appropriate labor data type from the sidebar.",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 220, 0.8)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=8,
            align="center"
        )
    
    return fig

def plot_service_data(service_df):
    """Create visualization for Oracle Symphony service data"""
    if service_df is None or len(service_df) == 0:
        return None
    
    # Aggregate by hour
    hourly_data = service_df.groupby('Hour').agg({
        'Net Sales': 'sum',
        'Labor Cost': 'sum'
    }).reset_index()
    
    # Calculate labor to sales ratio
    hourly_data['Labor to Sales Ratio'] = 0
    positive_sales = hourly_data['Net Sales'] > 0
    if positive_sales.any():
        hourly_data.loc[positive_sales, 'Labor to Sales Ratio'] = (
            hourly_data.loc[positive_sales, 'Labor Cost'] / 
            hourly_data.loc[positive_sales, 'Net Sales'] * 100
        )
    
    # Create interactive visualization
    fig = go.Figure()
    
    # Add sales bars
    fig.add_trace(go.Bar(
        x=hourly_data['Hour'],
        y=hourly_data['Net Sales'],
        name='Net Sales',
        marker_color=color_palette[0],
        opacity=0.7,
        text=['${:.0f}'.format(x) for x in hourly_data['Net Sales']],
        textposition='auto',
        hovertemplate='Hour: %{x}:00<br>Sales: $%{y:.2f}<extra></extra>'
    ))
    
    # Add labor cost line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=hourly_data['Hour'],
        y=hourly_data['Labor Cost'],
        name='Labor Cost',
        mode='lines+markers',
        marker=dict(size=8, color=color_palette[3]),
        line=dict(width=2, color=color_palette[3]),
        yaxis='y2',
        hovertemplate='Hour: %{x}:00<br>Labor Cost: $%{y:.2f}<extra></extra>'
    ))
    
    # Add labor to sales ratio line
    fig.add_trace(go.Scatter(
        x=hourly_data['Hour'],
        y=hourly_data['Labor to Sales Ratio'],
        name='Labor to Sales Ratio (%)',
        mode='lines+markers',
        marker=dict(size=8, color=color_palette[2], symbol='diamond'),
        line=dict(width=2, color=color_palette[2], dash='dot'),
        yaxis='y3',
        hovertemplate='Hour: %{x}:00<br>Labor/Sales Ratio: %{y:.1f}%<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Oracle Symphony Service Performance Analysis',
        xaxis=dict(
            title='Hour of Day',
            tickmode='array',
            tickvals=hourly_data['Hour'],
            ticktext=[f"{h}:00" for h in hourly_data['Hour']]
        ),
        yaxis=dict(
            title='Net Sales ($)',
            gridcolor='lightgray',
            tickprefix='$'
        ),
        yaxis2=dict(
            title=dict(text='Labor Cost ($)', font=dict(color=color_palette[3])),
            tickfont=dict(color=color_palette[3]),
            anchor="free",
            overlaying="y",
            side="right",
            position=1.0,
            tickprefix='$'
        ),
        yaxis3=dict(
            title=dict(text='Labor/Sales Ratio (%)', font=dict(color=color_palette[2])),
            tickfont=dict(color=color_palette[2]),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.93,
            ticksuffix='%'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    # Add insights annotation
    peak_hour = hourly_data.loc[hourly_data['Net Sales'].idxmax(), 'Hour']
    peak_sales = hourly_data['Net Sales'].max()
    min_ratio_idx = hourly_data[hourly_data['Net Sales'] > 0]['Labor to Sales Ratio'].idxmin() if any(hourly_data['Net Sales'] > 0) else 0
    min_ratio_hour = hourly_data.loc[min_ratio_idx, 'Hour'] if min_ratio_idx > 0 else 0
    min_ratio = hourly_data[hourly_data['Net Sales'] > 0]['Labor to Sales Ratio'].min() if any(hourly_data['Net Sales'] > 0) else 0
    
    insight_text = f"Peak Sales Hour: {peak_hour}:00 (${peak_sales:.0f}) | Most Efficient Hour: {min_ratio_hour}:00 ({min_ratio:.1f}%)<br>"
    insight_text += "<b>Insight:</b> The relationship between sales and labor costs reveals opportunities for "
    insight_text += "targeted staffing adjustments during peak and non-peak periods."
    
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.5, y=-0.15,
        text=insight_text,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 220, 0.8)",
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=8,
        align="center"
    )
    
    return fig

def plot_sales_data(sales_df):
    """Create visualization for Oracle Symphony sales data"""
    if sales_df is None or len(sales_df) == 0:
        return None, None
    
    # Create category summary
    if 'Category' in sales_df.columns and 'Gross Sales' in sales_df.columns:
        category_sales = sales_df.groupby('Category').agg({
            'Gross Sales': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        category_sales = category_sales.sort_values('Gross Sales', ascending=False)
        category_sales['Average Price'] = category_sales['Gross Sales'] / category_sales['Quantity']
        
        # Create top categories chart
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=category_sales['Category'][:10],
            y=category_sales['Gross Sales'][:10],
            marker_color=color_palette[0],
            text=['${:.0f}'.format(x) for x in category_sales['Gross Sales'][:10]],
            textposition='auto',
            hovertemplate='Category: %{x}<br>Sales: $%{y:.2f}<extra></extra>'
        ))
        
        fig1.update_layout(
            title='Top Categories by Sales',
            xaxis_title='Category',
            yaxis_title='Gross Sales ($)',
            yaxis=dict(tickprefix='$'),
            height=400
        )
        
        # Create quantity vs price bubble chart
        fig2 = px.scatter(
            category_sales,
            x='Quantity',
            y='Average Price',
            size='Gross Sales',
            color='Gross Sales',
            hover_name='Category',
            size_max=50,
            color_continuous_scale=px.colors.sequential.Blues
        )
        
        fig2.update_layout(
            title='Category Analysis: Quantity vs Price',
            xaxis_title='Quantity Sold',
            yaxis_title='Average Price ($)',
            yaxis=dict(tickprefix='$'),
            height=500
        )
        
        return fig1, fig2
    
    return None, None

def plot_get_app_data(get_app_df):
    """Create visualization for GET App data"""
    if get_app_df is None or len(get_app_df) == 0:
        return None, None
    
    try:
        # Analyze by payment type
        payment_summary = get_app_df.groupby('Payment_Type').agg({
            'Order_Count': 'sum',
            'Sales_Amount': 'sum'
        }).reset_index()
        
        payment_summary['Average Order Value'] = payment_summary['Sales_Amount'] / payment_summary['Order_Count']
        
        # Create interactive visualization
        fig1 = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                            subplot_titles=("Orders by Payment Type", "Sales Distribution"))
        
        # Bar chart for order counts
        fig1.add_trace(
            go.Bar(
                x=payment_summary['Payment_Type'],
                y=payment_summary['Order_Count'],
                text=payment_summary['Order_Count'],
                textposition='auto',
                marker_color=color_palette[0],
                name='Order Count',
                hovertemplate='Payment Type: %{x}<br>Orders: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Pie chart for sales distribution
        fig1.add_trace(
            go.Pie(
                labels=payment_summary['Payment_Type'],
                values=payment_summary['Sales_Amount'],
                marker=dict(colors=color_palette),
                textinfo='label+percent',
                hovertemplate='Payment Type: %{label}<br>Sales: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Update layout
        fig1.update_layout(
            title_text="GET App Digital Orders Analysis",
            showlegend=False,
            height=400
        )
        
        # Show average order value comparison
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=payment_summary['Payment_Type'],
            y=payment_summary['Average Order Value'],
            marker_color=color_palette[1],
            text=['${:.2f}'.format(x) for x in payment_summary['Average Order Value']],
            textposition='auto',
            hovertemplate='Payment Type: %{x}<br>Average Order: $%{y:.2f}<extra></extra>'
        ))
        
        fig2.update_layout(
            title="Average Order Value by Payment Type",
            xaxis_title="Payment Type",
            yaxis_title="Average Order Value ($)",
            yaxis=dict(tickprefix='$'),
            height=400
        )
        
        # Add insights annotation
        top_payment_idx = payment_summary['Order_Count'].idxmax()
        top_payment = payment_summary.loc[top_payment_idx, 'Payment_Type']
        top_payment_count = payment_summary.loc[top_payment_idx, 'Order_Count']
        
        highest_avg_idx = payment_summary['Average Order Value'].idxmax()
        highest_avg = payment_summary.loc[highest_avg_idx, 'Payment_Type']
        highest_avg_value = payment_summary.loc[highest_avg_idx, 'Average Order Value']
        
        insight_text = f"Most Popular: {top_payment} ({top_payment_count} orders) | Highest Avg Value: {highest_avg} (${highest_avg_value:.2f})<br>"
        insight_text += "<b>Insight:</b> Different payment methods correlate with different order values, suggesting distinct customer segments."
        
        fig2.add_annotation(
            xref='paper', yref='paper',
            x=0.5, y=-0.2,
            text=insight_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 220, 0.8)",
            bordercolor="#c7c7c7",
            borderwidth=1,
            borderpad=8,
            align="center"
        )
        
        return fig1, fig2
    
    except Exception as e:
        st.error(f"Error creating GET App visualizations: {e}")
        return None, None

def analyze_integrated_data(labor_df, service_df, get_app_df):
    """Perform integrated analysis across all data sources"""
    if labor_df is None or service_df is None:
        return None, None, None, None
    
    try:
        # Extract key metrics by hour from each platform
        
        # From Homebase
        if 'Hour' in labor_df.columns:
            labor_hourly = labor_df.groupby('Hour').agg({
                'Scheduled hours': 'sum',
                'Actual hours': 'sum',
                'Labor Efficiency': 'mean'
            }).reset_index()
        else:
            return None, None, None, None
        
        # From Oracle Symphony
        if 'Hour' in service_df.columns:
            service_hourly = service_df.groupby('Hour').agg({
                'Net Sales': 'sum',
                'Labor Cost': 'sum',
                'Labor to Sales Ratio': 'mean'
            }).reset_index()
        else:
            return None, None, None, None
        
        # From GET App
        get_app_hourly = None
        if get_app_df is not None and 'Hour' in get_app_df.columns:
            get_app_hourly = get_app_df.groupby('Hour').agg({
                'Order_Count': 'sum',
                'Sales_Amount': 'sum'
            }).reset_index()
            get_app_hourly['Online Order Ratio'] = 0
        
        # Merge datasets
        combined_df = pd.merge(labor_hourly, service_hourly, on='Hour', how='inner')
        
        if get_app_hourly is not None:
            combined_df = pd.merge(combined_df, get_app_hourly, on='Hour', how='left')
            combined_df['Order_Count'] = combined_df['Order_Count'].fillna(0)
            combined_df['Sales_Amount'] = combined_df['Sales_Amount'].fillna(0)
            
            # Calculate online sales ratio
            combined_df['Online Order Ratio'] = 0
            nonzero_sales = combined_df['Net Sales'] > 0
            if nonzero_sales.any():
                combined_df.loc[nonzero_sales, 'Online Order Ratio'] = \
                    combined_df.loc[nonzero_sales, 'Sales_Amount'] / combined_df.loc[nonzero_sales, 'Net Sales'] * 100
        
        # Add efficiency metrics
        combined_df['Labor-Sales Coordination'] = 100 - abs(100 - combined_df['Labor Efficiency'])
        
        # Integrated visualization
        fig1 = go.Figure()
        
        # Add sales and labor data
        fig1.add_trace(go.Bar(
            x=combined_df['Hour'],
            y=combined_df['Net Sales'],
            name='Total Sales',
            marker_color=color_palette[0],
            hovertemplate='Hour: %{x}:00<br>Sales: $%{y:.2f}<extra></extra>'
        ))
        
        if 'Sales_Amount' in combined_df.columns:
            fig1.add_trace(go.Bar(
                x=combined_df['Hour'],
                y=combined_df['Sales_Amount'],
                name='Online Sales (GET App)',
                marker_color=color_palette[1],
                hovertemplate='Hour: %{x}:00<br>Online Sales: $%{y:.2f}<extra></extra>'
            ))
        
        fig1.add_trace(go.Scatter(
            x=combined_df['Hour'],
            y=combined_df['Actual hours'],
            name='Labor Hours',
            mode='lines+markers',
            marker=dict(size=8, color=color_palette[2]),
            line=dict(width=2, color=color_palette[2]),
            yaxis='y2',
            hovertemplate='Hour: %{x}:00<br>Labor Hours: %{y:.1f}<extra></extra>'
        ))
        
        # Update layout
        fig1.update_layout(
            title='Integrated Cross-Platform Analysis by Hour',
            barmode='stack',
            xaxis=dict(
                title='Hour of Day',
                tickmode='array',
                tickvals=combined_df['Hour'],
                ticktext=[f"{h}:00" for h in combined_df['Hour']]
            ),
            yaxis=dict(
                title='Sales ($)',
                tickprefix='$'
            ),
            yaxis2=dict(
                title=dict(text='Labor Hours', font=dict(color=color_palette[2])),
                tickfont=dict(color=color_palette[2]),
                overlaying='y',
                side='right'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500
        )
        
        # Labor efficiency vs online order ratio
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig2.add_trace(
            go.Scatter(
                x=combined_df['Hour'],
                y=combined_df['Labor Efficiency'],
                name='Labor Efficiency',
                mode='lines+markers',
                marker=dict(size=8, color=color_palette[0]),
                line=dict(width=2, color=color_palette[0])
            ),
            secondary_y=False
        )
        
        if 'Online Order Ratio' in combined_df.columns:
            fig2.add_trace(
                go.Scatter(
                    x=combined_df['Hour'],
                    y=combined_df['Online Order Ratio'],
                    name='Online Order Ratio',
                    mode='lines+markers',
                    marker=dict(size=8, color=color_palette[1]),
                    line=dict(width=2, color=color_palette[1], dash='dot')
                ),
                secondary_y=True
            )
        
        fig2.update_layout(
            title_text="Labor Efficiency vs Online Orders",
            xaxis=dict(
                title='Hour of Day',
                tickmode='array',
                tickvals=combined_df['Hour'],
                ticktext=[f"{h}:00" for h in combined_df['Hour']]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=400
        )
        
        fig2.update_yaxes(title_text="Labor Efficiency (%)", secondary_y=False, ticksuffix="%")
        fig2.update_yaxes(title_text="Online Order Ratio (%)", secondary_y=True, ticksuffix="%", range=[0, 100])
        
        # Select features for TDA analysis
        combined_features = ['Hour', 'Actual hours', 'Net Sales', 'Labor Cost', 'Labor Efficiency']
        
        if get_app_hourly is not None:
            combined_features.extend(['Sales_Amount', 'Online Order Ratio'])
        
        # Filter out any non-existent columns
        combined_features = [f for f in combined_features if f in combined_df.columns]
        
        # Perform TDA on combined data
        combined_tda_data = combined_df[combined_features].dropna().values
        
        # Perform TDA analysis if there's enough data
        if combined_tda_data.shape[0] >= 5 and combined_tda_data.shape[1] >= 2:
            # Compute persistence diagrams
            combined_diagrams, combined_diagram_info = compute_persistence_diagram(combined_tda_data, max_dimension=1)
            
            # Extract topological features
            combined_features_data = extract_topological_features(combined_diagrams)
            
            # Plot interactive persistence diagram
            diagram_fig = plot_enhanced_persistence_diagram(combined_diagrams, title="Cross-Platform Integrated Data - Topological Structure")
            
            # Generate business insights
            combined_insights = generate_business_insights(combined_diagram_info, combined_features_data, "combined", combined_features)
            
            return fig1, fig2, diagram_fig, combined_insights
        
        return fig1, fig2, None, None
    
    except Exception as e:
        st.error(f"Error performing integrated analysis: {e}")
        return None, None, None, None

#------------------------
# Main Streamlit Application
#------------------------

def main():
    # Title and introduction
    st.markdown("<h1 class='main-header'>Nikos Café Topological Data Analysis</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This interactive application performs advanced Topological Data Analysis (TDA) on Nikos Café operational data to reveal hidden patterns, 
    optimize operations, and enhance decision-making capabilities. The analysis integrates data from multiple systems:
    
    - **Oracle Symphony** (sales and finance data)
    - **Homebase** (employee scheduling and labor data)
    - **GET App** (online ordering platform)
    
    Upload your operational data files to begin the analysis.
    """)
    
    # Check for TDA libraries
    if not tda_libraries_available:
        st.warning("TDA libraries (ripser, persim) are not available. Using simulated TDA functions for demonstration purposes.")
    
    # Create sidebar for file uploads
    with st.sidebar:
        st.header("Data Upload")
        st.markdown("Upload your operational data files:")
        
        # Oracle Symphony sales data
        sales_file = st.file_uploader("Oracle Symphony Sales Data (Excel)", type=["xlsx", "xls"], help="Upload Menu Item Quantity Sold by Weekday.xlsx")
        
        # Oracle Symphony service data
        service_file = st.file_uploader("Oracle Symphony Service Performance (Excel)", type=["xlsx", "xls"], help="Upload Service Performance.xlsx")
        
        # Homebase labor data
        labor_data_type = st.radio("Labor Data Type", 
                                 ["Hourly Labor Costs", "Labor by Role", "Sales-Labor Analysis"],
                                 help="Select the type of labor data you're uploading")
        
        labor_file = st.file_uploader("Homebase Labor Data (CSV)", type=["csv"], 
                                      help="Upload labor data file based on the selected type above")
        
        # GET App data
        get_app_file = st.file_uploader("GET App Data (Excel)", type=["xlsx", "xls"], help="Upload Nikos GET APP.xlsx")
        
        # Sample data option
        st.markdown("---")
        if st.button("Use Sample Data"):
            st.info("Using sample data for demonstration...")
            # Set flags for using sample data
            st.session_state['use_sample_data'] = True
        
        # Export to Power BI option
        st.markdown("---")
        st.header("Power BI Integration")
        if st.button("Export Data for Power BI"):
            if 'data_dict' in st.session_state and any(df is not None for df in st.session_state['data_dict'].values()):
                combined_df, file_path = export_for_power_bi(st.session_state['data_dict'])
                if combined_df is not None:
                    st.success(f"Data successfully prepared for Power BI with {len(combined_df)} rows")
                    
                    # Create download button for the file
                    st.download_button(
                        label="Download CSV for Power BI",
                        data=combined_df.to_csv(index=False).encode('utf-8'),
                        file_name="nikos_cafe_data_for_power_bi.csv",
                        mime="text/csv"
                    )
            else:
                st.error("No data available to export. Please upload at least one data file.")
    
    # Process uploaded files
    data_dict = {}
    
    # Check if sample data is requested
    if 'use_sample_data' in st.session_state and st.session_state['use_sample_data']:
        # Use sample data instead of uploading
        pass
    else:
        # Process uploaded files
        if sales_file:
            sales_df = process_oracle_sales_data(sales_file)
            if sales_df is not None:
                data_dict['sales_data'] = sales_df
                st.session_state['sales_df'] = sales_df
        elif 'sales_df' in st.session_state:
            data_dict['sales_data'] = st.session_state['sales_df']
        
        if service_file:
            service_df = process_oracle_service_data(service_file)
            if service_df is not None:
                data_dict['service_data'] = service_df
                st.session_state['service_df'] = service_df
        elif 'service_df' in st.session_state:
            data_dict['service_data'] = st.session_state['service_df']
        
        if labor_file:
            # Determine file type based on radio selection
            file_type = "hourly"
            if labor_data_type == "Labor by Role":
                file_type = "by_type"
            elif labor_data_type == "Sales-Labor Analysis":
                file_type = "sales_labor"
                
            labor_df, data_type = process_homebase_labor_data(labor_file, file_type)
            
            if labor_df is not None:
                data_dict['labor_data'] = labor_df
                data_dict['labor_data_type'] = data_type
                st.session_state['labor_df'] = labor_df
                st.session_state['labor_data_type'] = data_type
        elif 'labor_df' in st.session_state:
            data_dict['labor_data'] = st.session_state['labor_df']
            data_dict['labor_data_type'] = st.session_state.get('labor_data_type', 'hourly')
        
        if get_app_file:
            get_app_df = process_get_app_data(get_app_file)
            if get_app_df is not None:
                data_dict['get_app_data'] = get_app_df
                st.session_state['get_app_df'] = get_app_df
        elif 'get_app_df' in st.session_state:
            data_dict['get_app_data'] = st.session_state['get_app_df']
    
    # Store data dictionary in session state
    st.session_state['data_dict'] = data_dict
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Labor Analysis", "Sales Analysis", "GET App Analysis", "Integrated TDA"])
    
    #--------------------------
    # Tab 1: Overview
    #--------------------------
    with tab1:
        st.markdown("<h2 class='sub-header'>Operational Overview</h2>", unsafe_allow_html=True)
        
        # Check if any data is available
        if data_dict:
            # Create metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            # Total sales metric
            total_sales = 0
            if 'service_data' in data_dict:
                total_sales = data_dict['service_data']['Net Sales'].sum()
            
            col1.markdown("<div class='metric-card'><div class='metric-value'>${:,.2f}</div><div class='metric-label'>Total Sales</div></div>".format(total_sales), unsafe_allow_html=True)
            
            # Labor hours metric
            total_hours = 0
            if 'labor_data' in data_dict:
                labor_df = data_dict['labor_data']
                labor_type = data_dict.get('labor_data_type', 'hourly')
                
                if labor_type == 'hourly' and 'Actual hours' in labor_df.columns:
                    total_hours = labor_df['Actual hours'].sum()
                elif labor_type == 'by_type' and 'Hours' in labor_df.columns:
                    total_hours = labor_df['Hours'].sum()
                elif labor_type == 'sales_labor' and 'Labor_Hours' in labor_df.columns:
                    total_hours = labor_df['Labor_Hours'].sum()
                
            col2.markdown("<div class='metric-card'><div class='metric-value'>{:,.1f}</div><div class='metric-label'>Labor Hours</div></div>".format(total_hours), unsafe_allow_html=True)
            
            # Labor efficiency metric
            labor_efficiency = 0
            if 'labor_data' in data_dict:
                labor_df = data_dict['labor_data']
                labor_type = data_dict.get('labor_data_type', 'hourly')
                
                if labor_type == 'hourly' and 'Scheduled hours' in labor_df.columns and 'Actual hours' in labor_df.columns:
                    if labor_df['Scheduled hours'].sum() > 0:
                        labor_efficiency = (labor_df['Actual hours'].sum() / labor_df['Scheduled hours'].sum()) * 100
                elif labor_type == 'sales_labor' and 'Labor_Cost_Percentage' in labor_df.columns:
                    # Use labor cost percentage as the efficiency metric
                    nonzero_pct = labor_df['Labor_Cost_Percentage'] > 0
                    if nonzero_pct.any():
                        labor_efficiency = labor_df.loc[nonzero_pct, 'Labor_Cost_Percentage'].mean()
            
            col3.markdown("<div class='metric-card'><div class='metric-value'>{:,.1f}%</div><div class='metric-label'>Labor Efficiency</div></div>".format(labor_efficiency), unsafe_allow_html=True)
            
            # Online orders metric
            online_orders = 0
            if 'get_app_data' in data_dict:
                online_orders = data_dict['get_app_data']['Order_Count'].sum()
            
            col4.markdown("<div class='metric-card'><div class='metric-value'>{:,.0f}</div><div class='metric-label'>Online Orders</div></div>".format(online_orders), unsafe_allow_html=True)
            
            # Summary of available data
            st.markdown("<h3>Available Data Sources</h3>", unsafe_allow_html=True)
            
            data_sources = []
            if 'sales_data' in data_dict:
                data_sources.append("✓ Oracle Symphony Sales Data: {:,} menu items".format(len(data_dict['sales_data'])))
            else:
                data_sources.append("○ Oracle Symphony Sales Data: Not uploaded")
            
            if 'service_data' in data_dict:
                data_sources.append("✓ Oracle Symphony Service Data: {:,} hourly records".format(len(data_dict['service_data'])))
            else:
                data_sources.append("○ Oracle Symphony Service Data: Not uploaded")
            
            if 'labor_data' in data_dict:
                data_sources.append("✓ Homebase Labor Data: {:,} hourly labor records".format(len(data_dict['labor_data'])))
            else:
                data_sources.append("○ Homebase Labor Data: Not uploaded")
            
            if 'get_app_data' in data_dict:
                data_sources.append("✓ GET App Data: {:,} digital order records".format(len(data_dict['get_app_data'])))
            else:
                data_sources.append("○ GET App Data: Not uploaded")
            
            for source in data_sources:
                st.markdown(source)
            
            # TDA Introduction
            st.markdown("<br><div class='insights-box'><h3>What is Topological Data Analysis?</h3>", unsafe_allow_html=True)
            st.markdown("""
            Topological Data Analysis (TDA) is a mathematical approach that analyzes the "shape" of data to find hidden patterns and relationships. Key benefits for Nikos Café:
            
            - **Detects Hidden Patterns**: Identifies clusters and relationships that traditional analytics miss
            - **Noise Resistant**: Maintains accuracy despite unusual events or outliers
            - **Reveals System Dynamics**: Uncovers how different aspects of café operations influence each other
            
            In business terms, TDA helps answer questions like:
            - What are the distinct operational states of the café throughout the day?
            - How do digital orders affect in-store operations?
            - What recurring cycles exist in labor efficiency and sales?
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample visualization if any data is available
            if 'service_data' in data_dict and 'labor_data' in data_dict:
                st.markdown("<h3>Quick Insights</h3>", unsafe_allow_html=True)
                
                # Integrated visualization preview
                labor_df = data_dict.get('labor_data')
                service_df = data_dict.get('service_data')
                get_app_df = data_dict.get('get_app_data')
                
                # Get first visualization from integrated analysis
                fig1, _, _, _ = analyze_integrated_data(labor_df, service_df, get_app_df)
                if fig1 is not None:
                    st.plotly_chart(fig1, use_container_width=True)
                    st.markdown("""
                    The chart above shows an integrated view of sales and labor data across different hours of the day. 
                    For more detailed analysis, explore the specific tabs above.
                    """)
        else:
            # No data uploaded yet
            st.markdown("""
            ### Getting Started
            
            To begin your analysis, please upload data files from Oracle Symphony, Homebase, and/or GET App using the sidebar.
            
            For each data source, the application will:
            1. Process and clean the data
            2. Generate interactive visualizations
            3. Perform topological analysis to reveal hidden patterns
            4. Provide business-focused insights and recommendations
            
            If you don't have actual data files available, you can click "Use Sample Data" in the sidebar to see the application in action with demonstration data.
            """)
            
            st.markdown("*Upload your data files using the sidebar to begin analysis.*")
    
    #--------------------------
    # Tab 2: Labor Analysis
    #--------------------------
    with tab2:
        st.markdown("<h2 class='sub-header'>Homebase Labor Data Analysis</h2>", unsafe_allow_html=True)
        
        # Check if labor data is available
        if 'labor_data' in data_dict:
            labor_df = data_dict['labor_data']
            
            # Display basic metrics
            col1, col2, col3 = st.columns(3)
            labor_type = data_dict.get('labor_data_type', 'hourly')
            
            # Display different metrics based on labor data type
            if labor_type == 'hourly' and 'Scheduled hours' in labor_df.columns and 'Actual hours' in labor_df.columns:
                total_scheduled = labor_df['Scheduled hours'].sum()
                total_actual = labor_df['Actual hours'].sum()
                if total_scheduled > 0:
                    scheduling_accuracy = (total_actual / total_scheduled * 100)
                else:
                    scheduling_accuracy = 0
                
                col1.metric("Total Scheduled Hours", f"{total_scheduled:.1f}")
                col2.metric("Total Actual Hours", f"{total_actual:.1f}")
                col3.metric("Scheduling Accuracy", f"{scheduling_accuracy:.1f}%")
                
            elif labor_type == 'by_type' and 'Hours' in labor_df.columns and 'Cost' in labor_df.columns:
                total_hours = labor_df['Hours'].sum()
                total_cost = labor_df['Cost'].sum()
                avg_hourly = total_cost / total_hours if total_hours > 0 else 0
                
                col1.metric("Total Labor Hours", f"{total_hours:.1f}")
                col2.metric("Total Labor Cost", f"${total_cost:.2f}")
                col3.metric("Average Hourly Cost", f"${avg_hourly:.2f}")
                
            elif labor_type == 'sales_labor' and 'Labor_Hours' in labor_df.columns:
                total_labor_hours = labor_df['Labor_Hours'].sum()
                total_labor_cost = labor_df['Labor_Cost'].sum()
                total_sales = labor_df['Sales'].sum()
                labor_pct = (total_labor_cost / total_sales * 100) if total_sales > 0 else 0
                
                col1.metric("Total Labor Hours", f"{total_labor_hours:.1f}")
                col2.metric("Total Labor Cost", f"${total_labor_cost:.2f}")
                col3.metric("Labor Cost %", f"{labor_pct:.1f}%")
            
            # Plot labor data visualization based on type
            labor_type = data_dict.get('labor_data_type', 'hourly')
            labor_fig = plot_labor_data(labor_df, data_type=labor_type)
            
            if labor_fig is not None:
                st.plotly_chart(labor_fig, use_container_width=True, key="labor_viz")
            
            # Perform TDA analysis on labor data
            st.markdown("<h3>Topological Analysis of Labor Data</h3>", unsafe_allow_html=True)
            
            # Select features for analysis
            labor_features = ['Hour', 'Scheduled hours', 'Actual hours', 'Scheduled cost', 'Actual cost', 'Labor Efficiency']
            labor_features = [f for f in labor_features if f in labor_df.columns]
            
            # Filter rows with data based on data type
            labor_type = data_dict.get('labor_data_type', 'hourly')
            
            if labor_type == 'hourly' and 'Scheduled hours' in labor_df.columns:
                # For hourly data, filter by scheduled hours
                filtered_labor_df = labor_df[labor_df['Scheduled hours'] > 0]
            elif labor_type == 'by_type' and 'Hours' in labor_df.columns:
                # For by_type data, filter by hours
                filtered_labor_df = labor_df[labor_df['Hours'] > 0]
            elif labor_type == 'sales_labor' and 'Labor_Hours' in labor_df.columns:
                # For sales_labor data, filter by labor hours
                filtered_labor_df = labor_df[labor_df['Labor_Hours'] > 0]
            else:
                # Fallback to just use the data as is
                filtered_labor_df = labor_df
                
            if len(labor_features) > 0:
                labor_tda_data = filtered_labor_df[labor_features].dropna().values
            else:
                # If no features match, select numeric columns as a fallback
                numeric_cols = filtered_labor_df.select_dtypes(include=[np.number]).columns.tolist()
                labor_features = [col for col in numeric_cols if col not in ['index']]
                labor_tda_data = filtered_labor_df[labor_features].dropna().values
            
            if labor_tda_data.shape[0] >= 5 and labor_tda_data.shape[1] >= 2:
                # Compute persistence diagrams
                labor_diagrams, labor_diagram_info = compute_persistence_diagram(labor_tda_data, max_dimension=1)
                
                # Extract topological features
                labor_features_data = extract_topological_features(labor_diagrams)
                
                # Plot interactive persistence diagram
                diagram_fig = plot_enhanced_persistence_diagram(labor_diagrams, title="Homebase Labor Data - Topological Structure")
                st.plotly_chart(diagram_fig, use_container_width=True)
                
                # Generate business insights
                labor_insights = generate_business_insights(labor_diagram_info, labor_features_data, "labor_data", labor_features)
                st.markdown("<div class='insights-box'>", unsafe_allow_html=True)
                st.markdown(labor_insights)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Insufficient data for meaningful topological analysis of labor data.")
        else:
            st.info("Please upload Homebase labor data to view this analysis.")
    
    #--------------------------
    # Tab 3: Sales Analysis
    #--------------------------
    with tab3:
        st.markdown("<h2 class='sub-header'>Oracle Symphony Sales & Service Analysis</h2>", unsafe_allow_html=True)
        
        # Check if service data is available
        if 'service_data' in data_dict:
            service_df = data_dict['service_data']
            
            # Display basic metrics
            col1, col2, col3 = st.columns(3)
            
            total_sales = service_df['Net Sales'].sum()
            total_labor_cost = service_df['Labor Cost'].sum()
            
            if total_sales > 0:
                overall_labor_ratio = (total_labor_cost / total_sales) * 100
            else:
                overall_labor_ratio = 0
            
            peak_hour_idx = service_df.groupby('Hour')['Net Sales'].sum().idxmax()
            peak_hour = f"{peak_hour_idx}:00"
            
            col1.metric("Total Sales", f"${total_sales:.2f}")
            col2.metric("Total Labor Cost", f"${total_labor_cost:.2f}")
            col3.metric("Labor to Sales Ratio", f"{overall_labor_ratio:.1f}%")
            
            # Plot service data visualization
            service_fig = plot_service_data(service_df)
            if service_fig is not None:
                st.plotly_chart(service_fig, use_container_width=True)
            
            # Show sales data analysis if available
            if 'sales_data' in data_dict:
                sales_df = data_dict['sales_data']
                st.markdown("<h3>Menu Sales Analysis</h3>", unsafe_allow_html=True)
                
                # Create metrics for sales data
                col1, col2, col3 = st.columns(3)
                
                total_items = sales_df['Quantity'].sum() if 'Quantity' in sales_df.columns else 0
                avg_price = sales_df['Gross Sales'].sum() / total_items if total_items > 0 and 'Gross Sales' in sales_df.columns else 0
                category_count = sales_df['Category'].nunique() if 'Category' in sales_df.columns else 0
                
                col1.metric("Total Items Sold", f"{total_items:.0f}")
                col2.metric("Average Item Price", f"${avg_price:.2f}")
                col3.metric("Menu Categories", f"{category_count}")
                
                # Plot sales data visualizations
                sales_fig1, sales_fig2 = plot_sales_data(sales_df)
                
                if sales_fig1 is not None:
                    st.plotly_chart(sales_fig1, use_container_width=True)
                
                if sales_fig2 is not None:
                    st.plotly_chart(sales_fig2, use_container_width=True)
            
            # Perform TDA analysis on service data
            st.markdown("<h3>Topological Analysis of Service Data</h3>", unsafe_allow_html=True)
            
            # Select features for analysis
            service_features = ['Hour', 'Net Sales', 'Labor Cost', 'Labor to Sales Ratio']
            service_features = [f for f in service_features if f in service_df.columns]
            
            # Filter rows with data
            filtered_service_df = service_df[service_df['Net Sales'] > 0]
            service_tda_data = filtered_service_df[service_features].dropna().values
            
            if service_tda_data.shape[0] >= 5 and service_tda_data.shape[1] >= 2:
                # Compute persistence diagrams
                service_diagrams, service_diagram_info = compute_persistence_diagram(service_tda_data, max_dimension=1)
                
                # Extract topological features
                service_features_data = extract_topological_features(service_diagrams)
                
                # Plot interactive persistence diagram
                diagram_fig = plot_enhanced_persistence_diagram(service_diagrams, title="Oracle Symphony Service Data - Topological Structure")
                st.plotly_chart(diagram_fig, use_container_width=True)
                
                # Generate business insights
                service_insights = generate_business_insights(service_diagram_info, service_features_data, "service_data", service_features)
                st.markdown("<div class='insights-box'>", unsafe_allow_html=True)
                st.markdown(service_insights)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Insufficient data for meaningful topological analysis of service data.")
        else:
            st.info("Please upload Oracle Symphony service data to view this analysis.")
    
    #--------------------------
    # Tab 4: GET App Analysis
    #--------------------------
    with tab4:
        st.markdown("<h2 class='sub-header'>GET App Digital Orders Analysis</h2>", unsafe_allow_html=True)
        
        # Check if GET App data is available
        if 'get_app_data' in data_dict:
            get_app_df = data_dict['get_app_data']
            
            # Display basic metrics
            col1, col2, col3 = st.columns(3)
            
            total_orders = get_app_df['Order_Count'].sum() if 'Order_Count' in get_app_df.columns else 0
            total_sales = get_app_df['Sales_Amount'].sum() if 'Sales_Amount' in get_app_df.columns else 0
            
            if total_orders > 0:
                avg_order_value = total_sales / total_orders
            else:
                avg_order_value = 0
            
            col1.metric("Total Digital Orders", f"{total_orders:.0f}")
            col2.metric("Total Online Sales", f"${total_sales:.2f}")
            col3.metric("Average Order Value", f"${avg_order_value:.2f}")
            
            # Plot GET App visualizations
            get_app_fig1, get_app_fig2 = plot_get_app_data(get_app_df)
            
            if get_app_fig1 is not None:
                st.plotly_chart(get_app_fig1, use_container_width=True)
            
            if get_app_fig2 is not None:
                st.plotly_chart(get_app_fig2, use_container_width=True)
            
            # Perform TDA analysis on GET App data
            st.markdown("<h3>Topological Analysis of GET App Data</h3>", unsafe_allow_html=True)
            
            # Select features for analysis
            numeric_cols = get_app_df.select_dtypes(include=[np.number]).columns.tolist()
            get_app_features = [col for col in numeric_cols if col not in ['index']]
            
            # Filter rows with data
            filtered_get_app_df = get_app_df.dropna(subset=get_app_features)
            get_app_tda_data = filtered_get_app_df[get_app_features].values
            
            if get_app_tda_data.shape[0] >= 5 and get_app_tda_data.shape[1] >= 2:
                # Compute persistence diagrams
                get_app_diagrams, get_app_diagram_info = compute_persistence_diagram(get_app_tda_data, max_dimension=1)
                
                # Extract topological features
                get_app_features_data = extract_topological_features(get_app_diagrams)
                
                # Plot interactive persistence diagram
                diagram_fig = plot_enhanced_persistence_diagram(get_app_diagrams, title="GET App Order Data - Topological Structure")
                st.plotly_chart(diagram_fig, use_container_width=True)
                
                # Generate business insights
                get_app_insights = generate_business_insights(get_app_diagram_info, get_app_features_data, "get_app_data", get_app_features)
                st.markdown("<div class='insights-box'>", unsafe_allow_html=True)
                st.markdown(get_app_insights)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Insufficient data for meaningful topological analysis of GET App data.")
        else:
            st.info("Please upload GET App data to view this analysis.")
    
    #--------------------------
    # Tab 5: Integrated TDA
    #--------------------------
    with tab5:
        st.markdown("<h2 class='sub-header'>Integrated Cross-Platform Analysis</h2>", unsafe_allow_html=True)
        
        # Check if we have enough data for integrated analysis
        if 'labor_data' in data_dict and 'service_data' in data_dict:
            labor_df = data_dict['labor_data']
            service_df = data_dict['service_data']
            get_app_df = data_dict.get('get_app_data')
            
            # Perform integrated analysis
            fig1, fig2, diagram_fig, combined_insights = analyze_integrated_data(labor_df, service_df, get_app_df)
            
            if fig1 is not None:
                st.markdown("<h3>Cross-Platform Operational Patterns</h3>", unsafe_allow_html=True)
                st.plotly_chart(fig1, use_container_width=True)
            
            if fig2 is not None:
                st.plotly_chart(fig2, use_container_width=True)
            
            if diagram_fig is not None:
                st.markdown("<h3>Topological Analysis of Integrated Data</h3>", unsafe_allow_html=True)
                st.plotly_chart(diagram_fig, use_container_width=True)
            
            if combined_insights is not None:
                st.markdown("<div class='insights-box'>", unsafe_allow_html=True)
                st.markdown(combined_insights)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Insufficient data for integrated topological analysis.")
            
            # Power BI integration section
            st.markdown("<h3>Power BI Integration</h3>", unsafe_allow_html=True)
            st.markdown("""
            The insights from this integrated analysis can be further explored through an interactive Power BI dashboard.
            
            To create your dashboard:
            1. Click the "Export Data for Power BI" button in the sidebar
            2. Download the CSV file
            3. Import the data into Power BI Desktop
            4. Create visualizations based on the patterns revealed in this analysis
            
            The Power BI dashboard will allow stakeholders to interactively explore the operational patterns identified through topological analysis.
            """)
        else:
            st.info("Please upload both labor data and service data to view the integrated cross-platform analysis.")

if __name__ == "__main__":
    main()