"""
Custom CSS Styling for Streamlit UI
====================================

Custom styling and themes for pump anomaly detection interface.
Simplified color scheme: Black, White, Blue
"""


# Main CSS styling
CUSTOM_CSS = """
<style>
    /* ========== GLOBAL STYLES ========== */
    
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #999;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* ========== ALERT BOXES ========== */
    
    .alert-normal {
        background-color: #000000;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(31,119,180,0.3);
    }
    
    .alert-abnormal {
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #ff6b6b;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(255,107,107,0.3);
    }
    
    .alert-warning {
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4a9eff;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(74,158,255,0.3);
    }
    
    .alert-info {
        background-color: #000000;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(31,119,180,0.3);
    }
    
    /* ========== PREDICTION RESULT CARD ========== */
    
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .prediction-card-normal {
        background-color: #000000;
        border: 3px solid #1f77b4;
    }
    
    .prediction-card-abnormal {
        background-color: #1a1a1a;
        border: 3px solid #ff6b6b;
    }
    
    .prediction-label {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    .confidence-score {
        font-size: 1.8rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    
    /* ========== METRIC CARDS ========== */
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #999 !important;
    }
    
    /* ========== HEADERS ========== */
    
    .stMarkdown h1 {
        color: #1f77b4 !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
    }
    
    .stMarkdown h2 {
        color: #4a9eff !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
    }
    
    .stMarkdown h3 {
        color: #4a9eff !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
    }
    
    .stMarkdown h4 {
        color: #1f77b4 !important;
        font-weight: 500 !important;
    }
    
    /* ========== TABS ========== */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000000;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #1a1a1a;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 500;
        color: #999;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white !important;
    }
    
    /* ========== SIDEBAR - BLACK BACKGROUND ========== */
    
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1f77b4 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #999 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
    }
    
    /* ========== BUTTONS ========== */
    
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(31,119,180,0.3);
    }
    
    .stButton>button:hover {
        background-color: #4a9eff;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(74,158,255,0.4);
    }
    
    /* ========== FILE UPLOADER - BLACK BACKGROUND ========== */
    
    [data-testid="stFileUploader"] {
        background-color: #000000 !important;
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background-color: #000000 !important;
    }
    
    /* ========== EXPANDER ========== */
    
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border-radius: 8px;
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* ========== DATA FRAMES ========== */
    
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* ========== MAIN CONTAINER LAYOUT ========== */
    
    /* Default: Center content with 20% margins when sidebar is closed */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 5% !important;
        padding-right: 5% !important;
        max-width: none !important;
    }
    
    /* When sidebar is closed: 20% empty | 60% content | 20% empty */
    @media (min-width: 768px) {
        body:not([data-sidebar-open="true"]) .main .block-container {
            width: 60% !important;
            max-width: 1200px !important;
            margin-left: 20% !important;
            margin-right: 20% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
    }
    
    /* When sidebar is open: Use 100% available width after sidebar */
    [data-testid="stSidebar"][aria-expanded="true"] ~ .main .block-container {
        width: 100% !important;
        max-width: 1400px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Additional handling for when sidebar is collapsed */
    [data-testid="collapsedControl"] ~ .main .block-container {
        width: 60% !important;
        max-width: 1200px !important;
        margin-left: 20% !important;
        margin-right: 20% !important;
    }
    
    /* ========== DIVIDERS (HIDDEN) ========== */
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 0px solid transparent;
        display: none;
    }
    
    /* ========== CUSTOM BADGES ========== */
    
    .badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        font-size: 0.875rem;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        margin: 0.25rem;
    }
    
    .badge-success {
        background-color: #1f77b4;
        color: white;
    }
    
    .badge-danger {
        background-color: #ff6b6b;
        color: white;
    }
    
    .badge-warning {
        background-color: #4a9eff;
        color: white;
    }
    
    .badge-info {
        background-color: #1f77b4;
        color: white;
    }
    
    /* ========== LOADING SPINNER ========== */
    
    .stSpinner > div {
        border-top-color: #1f77b4 !important;
    }
    
    /* ========== TOOLTIPS ========== */
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted #999;
    }
    
    /* ========== FEATURE TABLE ========== */
    
    .feature-table {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* ========== STATUS INDICATORS ========== */
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background-color: #28a745;
        box-shadow: 0 0 8px rgba(40, 167, 69, 0.5);
    }
    
    .status-offline {
        background-color: #dc3545;
        box-shadow: 0 0 8px rgba(220, 53, 69, 0.5);
    }
    
    /* ========== PROGRESS BAR ========== */
    
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    
    /* ========== CODE BLOCKS ========== */
    
    .stCodeBlock {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
    }
    
    /* ========== JSON VIEWER ========== */
    
    .stJson {
        background-color: #1a1a1a !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* ========== CUSTOM METRIC CARD ========== */
    
    .metric-card {
        background-color: #000000;
        border: 2px solid #1f77b4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(31,119,180,0.3);
    }
    
    .metric-card h3 {
        color: #999 !important;
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        font-weight: 500;
    }
    
    .metric-card p {
        color: #1f77b4 !important;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
</style>
"""

# API status indicator
def get_api_status_html(is_connected: bool) -> str:
    """
    Get HTML for API status indicator
    
    Args:
        is_connected: Whether API is connected
    
    Returns:
        HTML string
    """
    status_class = "status-online" if is_connected else "status-offline"
    status_text = " Connected" if is_connected else " Disconnected"
    status_color = "#28a745" if is_connected else "#dc3545"  
    
    html = f"""
    <div style="display: flex; align-items: center; padding: 0.5rem;">
        <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: {status_color}; margin-right: 8px;"></span>
        <span style="color: #ffffff; font-weight: 600;">API Status: </span>
        <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
    </div>
    """
    
    return html

# Model summary display 
def get_model_summary_html(training_report: dict) -> str:
    """
    Get HTML for data summary display from training_report.json
    Including confusion matrix
    
    Args:
        training_report: Training report dictionary from training_report.json
    
    Returns:
        HTML string
    """
    if not training_report:
        return ""
    
    # Extract data from correct JSON structure
    dataset_info = training_report.get('dataset_info', {})
    total_samples = dataset_info.get('total_samples', 0)
    class_dist = dataset_info.get('class_distribution', {}).get('total', {})
    normal = class_dist.get('normal', 0)
    abnormal = class_dist.get('abnormal', 0)
    
    # Get test performance metrics
    test_performance = training_report.get('test_performance', {})
    metrics = test_performance.get('metrics', {})
    f1_score = metrics.get('f1_score', {}).get('percentage', 'N/A')  
    recall = metrics.get('recall', {}).get('percentage', 'N/A')
    precision = metrics.get('precision', {}).get('percentage', 'N/A')
    
    # Get model info
    metadata = training_report.get('metadata', {})
    model_version = metadata.get('version', 'N/A')
    
    # Get confusion matrix
    confusion_matrix = test_performance.get('confusion_matrix', {})
    tn = confusion_matrix.get('true_negatives', 0)
    fp = confusion_matrix.get('false_positives', 0)
    fn = confusion_matrix.get('false_negatives', 0)
    tp = confusion_matrix.get('true_positives', 0)
    
    html = f"""
    <div style="background-color: #000000; padding: 1rem; border-radius: 8px; border: 1px solid #1f77b4; margin-bottom: 1rem;">
        <div style="color: #1f77b4; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem;">
            Dataset
        </div>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">Total Samples:</span>
                <span style="color: #1f77b4; font-weight: 600; font-size: 0.85rem;">{total_samples:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">Normal:</span>
                <span style="color: #1f77b4; font-weight: 600; font-size: 0.85rem;">{normal:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">Abnormal:</span>
                <span style="color: #1f77b4; font-weight: 600; font-size: 0.85rem;">{abnormal:,}</span>
            </div>
        </div>
    </div>
    
    <div style="background-color: #000000; padding: 1rem; border-radius: 8px; border: 1px solid #1f77b4; margin-bottom: 1rem;">
        <div style="color: #1f77b4; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem;">
            Model Performance
        </div>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">F1 Score:</span>
                <span style="color: #1f77b4; font-weight: 600; font-size: 0.85rem;">{f1_score}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">Recall:</span>
                <span style="color: #1f77b4; font-weight: 600; font-size: 0.85rem;">{recall}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">Precision:</span>
                <span style="color: #1f77b4; font-weight: 600; font-size: 0.85rem;">{precision}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #999; font-size: 0.85rem;">Version:</span>
                <span style="color: #999; font-weight: 600; font-size: 0.75rem;">{model_version}</span>
            </div>
        </div>
    </div>
    
    <div style="background-color: #000000; padding: 1rem; border-radius: 8px; border: 1px solid #1f77b4;">
        <div style="color: #1f77b4; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.8rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem;">
            Confusion Matrix
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0;">
            <div style="padding: 0.75rem; border-right: 1px solid #333; border-bottom: 1px solid #333; text-align: center;">
                <div style="color: #999; font-size: 0.75rem; margin-bottom: 0.3rem;">TP</div>
                <div style="color: #1f77b4; font-weight: 600; font-size: 1.2rem;">{tp}</div>
            </div>
            <div style="padding: 0.75rem; border-bottom: 1px solid #333; text-align: center;">
                <div style="color: #999; font-size: 0.75rem; margin-bottom: 0.3rem;">FP</div>
                <div style="color: #1f77b4; font-weight: 600; font-size: 1.2rem;">{fp}</div>
            </div>
            <div style="padding: 0.75rem; border-right: 1px solid #333; text-align: center;">
                <div style="color: #999; font-size: 0.75rem; margin-bottom: 0.3rem;">FN</div>
                <div style="color: #1f77b4; font-weight: 600; font-size: 1.2rem;">{fn}</div>
            </div>
            <div style="padding: 0.75rem; text-align: center;">
                <div style="color: #999; font-size: 0.75rem; margin-bottom: 0.3rem;">TN</div>
                <div style="color: #1f77b4; font-weight: 600; font-size: 1.2rem;">{tn}</div>
            </div>
        </div>
    </div>
    """
    
    return html

# Feature severity badge
def get_severity_badge_html(severity: str) -> str:
    """
    Get HTML for severity badge
    
    Args:
        severity: 'Severe', 'Moderate', or 'Normal'
    
    Returns:
        HTML string
    """
    badge_map = {
        'Severe': ('badge-danger', '🔴'),
        'Moderate': ('badge-warning', '🟡'),
        'Normal': ('badge-success', '🟢')
    }
    
    badge_class, icon = badge_map.get(severity, ('badge-info', '⚪'))
    
    html = f"""
    <span class="badge {badge_class}">
        {icon} {severity}
    </span>
    """
    
    return html

# New function for simple metric cards
def create_simple_metric_card_html(metrics: dict) -> str:
    """
    Create HTML for 3 simple metric cards (Precision, Recall, F1 Score)
    
    Args:
        metrics: Dictionary containing precision, recall, f1_score
    
    Returns:
        HTML string
    """
    precision = metrics.get('precision', {}).get('percentage', 'N/A')
    recall = metrics.get('recall', {}).get('percentage', 'N/A')
    f1_score = metrics.get('f1_score', {}).get('percentage', 'N/A')
    
    html = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0;">
        <div class="metric-card">
            <h3>PRECISION</h3>
            <p>{precision}</p>
        </div>
        <div class="metric-card">
            <h3>RECALL</h3>
            <p>{recall}</p>
        </div>
        <div class="metric-card">
            <h3>F1 SCORE</h3>
            <p>{f1_score}</p>
        </div>
    </div>
    """
    
    return html


# Example usage
if __name__ == "__main__":
    print("CSS Styling Module Test")
    print("=" * 60)
    
    # Test API status
    status_html = get_api_status_html(True)
    print("✅ API status HTML generated")
    
    # Test severity badge
    badge_html = get_severity_badge_html("Severe")
    print("✅ Severity badge HTML generated")
    
    # Test metric cards
    test_metrics = {
        'precision': {'percentage': '95.5%'},
        'recall': {'percentage': '92.3%'},
        'f1_score': {'percentage': '83.9%'}
    }
    metric_html = create_simple_metric_card_html(test_metrics)
    print("✅ Simple metric cards HTML generated")
    
    print("\n✅ CSS styling module working!")