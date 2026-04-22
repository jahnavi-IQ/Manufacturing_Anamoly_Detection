"""
Visualization Components for Streamlit UI
==========================================

Plotly charts and visualizations for pump anomaly detection interface.
Simplified color scheme: Black, White, Blue
"""

import plotly.graph_objects as go
from typing import List, Dict


def create_feature_deviation_chart(
    deviations: List[Dict], 
    top_n: int = 10
) -> go.Figure:
    """
    Create bar chart showing feature deviations with detailed hover information
    
    Args:
        deviations: List of deviation dictionaries
        top_n: Number of features to show
    
    Returns:
        Plotly figure
    """
    # Get top N deviations
    top_devs = sorted(deviations, key=lambda x: x['abs_z_score'], reverse=True)[:top_n]
    
    # Prepare data
    features = [d['feature_name'] for d in top_devs]
    z_scores = [d['z_score'] for d in top_devs]
    abs_z_scores = [d['abs_z_score'] for d in top_devs]
    
    # Simplified color scheme - only blue shades
    bar_colors = ['#1f77b4' if abs(z) > 2 else '#4a9eff' if abs(z) > 1 else '#80b9ff' for z in z_scores]
    
    # Prepare detailed hover information with actual values and boundaries
    hover_texts = []
    for d in top_devs:
        # Extract values from deviation dict
        normal_mean = d['normal_mean']
        normal_std = d['normal_std']
        test_value = d['user_value']
        severity = d['severity']
        z_score = d['z_score']
        
        # Calculate boundary values
        # Normal range: mean ± 1σ
        normal_lower = normal_mean - normal_std
        normal_upper = normal_mean + normal_std
        
        # Moderate range: mean ± 2σ  
        moderate_lower = normal_mean - 2 * normal_std
        moderate_upper = normal_mean + 2 * normal_std
        
        # Build hover text with actual values
        hover_text = (
            f"<b>{d['feature_name']}</b><br>"
            f"<br>"
            f"<b>Test Sample Value:</b> {test_value:.6f}<br>"
            f"<b>Normal Mean:</b> {normal_mean:.6f}<br>"
            f"<br>"
            f"<b>Normal Range (±1σ):</b><br>"
            f"  {normal_lower:.6f} to {normal_upper:.6f}<br>"
            f"<br>"
            f"<b>Moderate Range (±2σ):</b><br>"
            f"  {moderate_lower:.6f} to {moderate_upper:.6f}<br>"
            f"<br>"
            f"<b>Z-Score:</b> {z_score:.2f}σ<br>"
            f"<b>Severity:</b> <b>{severity}</b>"
        )
        hover_texts.append(hover_text)
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=z_scores,
        y=features,
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=1)
        ),
        text=[f"{z:.2f}σ" for z in z_scores],
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    ))
    
    # Add reference lines
    fig.add_vline(x=-2, line_dash="dash", line_color="#4a9eff", opacity=0.5)
    fig.add_vline(x=2, line_dash="dash", line_color="#4a9eff", opacity=0.5)
    fig.add_vline(x=-1, line_dash="dash", line_color="#80b9ff", opacity=0.3)
    fig.add_vline(x=1, line_dash="dash", line_color="#80b9ff", opacity=0.3)
    fig.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Feature Deviation from Normal (Z-Scores)",
        xaxis_title="Standard Deviations from Normal Mean",
        yaxis_title="Features",
        height=500,
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray',
            gridcolor='#333'
        ),
        yaxis=dict(
            gridcolor='#333'
        )
    )
    
    return fig


def create_feature_importance_chart(
    chart_data: List[Dict],
    top_n: int = 10
) -> go.Figure:
    """
    Create feature importance bar chart
    
    Args:
        chart_data: List of feature importance dictionaries
        top_n: Number of features to show
    
    Returns:
        Plotly figure
    """
    if not chart_data:
        # Return an empty Plotly figure instead of None
        fig = go.Figure()
        fig.update_layout(
            title="No Feature Importance Data Available",
            height=300,
            paper_bgcolor='#0e1117',
            font=dict(color='white')
        )
        return fig
    
    # Sort and limit
    chart_data = sorted(chart_data, key=lambda x: x['z_score'], reverse=True)[:top_n]
    
    features = [d['feature'] for d in chart_data]
    z_scores = [d['z_score'] for d in chart_data]
    
    # Simplified color scheme - only blue shades
    colors = ['#1f77b4' if abs(z) > 2 else '#4a9eff' if abs(z) > 1 else '#80b9ff' for z in z_scores]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=z_scores,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f"{z:.2f}σ" for z in z_scores],
        textposition='outside',
        textfont=dict(color='white'),
        hovertemplate='<b>%{y}</b><br>Deviation: %{x:.2f}σ<br><extra></extra>'
    ))
    
    fig.update_layout(
        title="Top Contributing Features",
        xaxis_title="Deviation (Standard Deviations)",
        yaxis_title="Features",
        height=450,
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#333'),
        yaxis=dict(gridcolor='#333')
    )
    
    return fig

def create_severity_distribution(deviations: List[Dict]) -> go.Figure:
    """
    Create pie chart of deviation severity distribution
    
    Args:
        deviations: List of deviation dictionaries
    
    Returns:
        Plotly figure
    """
    # Count by severity
    severe = sum(1 for d in deviations if d['severity'] == 'Severe')
    moderate = sum(1 for d in deviations if d['severity'] == 'Moderate')
    normal = sum(1 for d in deviations if d['severity'] == 'Normal')
    
    # Simplified blue shades
    fig = go.Figure(data=[go.Pie(
        labels=['Severe', 'Moderate', 'Normal'],
        values=[severe, moderate, normal],
        hole=0.3,
        marker=dict(
            colors=['#1f77b4', '#4a9eff', '#80b9ff'],
            line=dict(color='white', width=2)
        ),
        textinfo='label+value',
        textfont=dict(size=14, color='white')
    )])
    
    fig.update_layout(
        title="Deviation Severity Distribution",
        height=300,
        showlegend=True,
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        legend=dict(font=dict(color='white'))
    )
    
    return fig


# Example usage
if __name__ == "__main__":
    print("Visualization Components Test")
    print("=" * 60)
    
    # Test deviation chart
    test_deviations = [
        {
            'feature_name': 'rms_energy',
            'z_score': 2.5,
            'abs_z_score': 2.5,
            'user_value': 0.045,
            'normal_mean': 0.025,
            'normal_std': 0.008,
            'color': 'red',
            'severity': 'Severe'
        },
        {
            'feature_name': 'spectral_centroid',
            'z_score': 1.8,
            'abs_z_score': 1.8,
            'user_value': 1200.5,
            'normal_mean': 1000.0,
            'normal_std': 111.4,
            'color': 'orange',
            'severity': 'Moderate'
        },
        {
            'feature_name': 'zero_crossing_rate',
            'z_score': 0.5,
            'abs_z_score': 0.5,
            'user_value': 0.052,
            'normal_mean': 0.050,
            'normal_std': 0.004,
            'color': 'green',
            'severity': 'Normal'
        }
    ]
    
    fig = create_feature_deviation_chart(test_deviations)
    print("✅ Feature deviation chart created with enhanced hover information")
    print("\nHover tooltip now includes:")
    print("  - Test Sample Value")
    print("  - Normal Range (±1σ)")
    print("  - Moderate Range (±2σ)")
    print("  - Normal Mean and Std Dev")
    print("  - Z-Score")
    print("  - Severity Label")