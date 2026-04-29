"""
Streamlit UI for Pump Anomaly Detection System
===============================================

Interactive web interface for pump sound anomaly detection with explainability.

Usage:
    streamlit run app.py 
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import requests
import pandas as pd
import librosa
import tempfile
import numpy as np
import json
from datetime import datetime

# Import custom modules
from src.config import config
from components.styles import CUSTOM_CSS, get_model_summary_html
from components.visualizations import (
    create_feature_deviation_chart,
    create_feature_importance_chart,
    create_severity_distribution
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=config.UI_TITLE,
    page_icon=config.UI_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{config.API_URL}/health", timeout=5)
        # Accept any response - if Lambda responds at all, it's connected
        return True, None
    except Exception as e:
        return False, None


def get_model_info():
    """Get model information from API"""
    try:
        response = requests.get(f"{config.API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_training_report():
    try:
        import boto3
        s3 = boto3.client(
            's3',
            region_name='us-east-1',
            aws_access_key_id=st.secrets["aws"]["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
        )
        response = s3.get_object(
            Bucket='pump-anomaly-models-prod-2026',
            Key='training_report.json'
        )
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        return None


def predict_audio(audio_bytes, filename):
    """
    Send audio file to API for prediction
    """
    try:
        import base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        response = requests.post(
            f"{config.API_URL}/predict",
            data=audio_bytes, 
            headers={"Content-Type": "audio/wav"},
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            error_msg = response.json().get('message', 'Unknown error')
            return None, f"API Error: {error_msg}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API server. Is it running?"
    except requests.exceptions.Timeout:
        return None, "Request timeout. Audio processing took too long."
    except Exception as e:
        return None, str(e)

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with API status and model info"""
    with st.sidebar:
        st.markdown("## System Status")
        
        # Data Summary
        st.markdown("####")
        training_report = get_training_report()
        if training_report:
            st.markdown(get_model_summary_html(training_report), unsafe_allow_html=True)
        else:
            st.warning("Data summary unavailable")
        
        # About
        st.markdown("## About")
        st.markdown("""
        **Pump Anomaly Detection System**
        
        Uses XGBoost machine learning model to detect abnormal pump sounds.
        
        **Features:**
        - Real-time audio analysis
        - Explainable AI predictions
        - Interactive visualizations
        - 42 acoustic features
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown('<h1 class="main-header">Pump Anomaly Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Acoustic Analysis with Explainability</p>', unsafe_allow_html=True)
    
    # Check API connection
    is_healthy, _ = check_api_health()
    
    if not is_healthy:
        st.error("⚠️ **API Server Not Connected!** Please start the API server before using this interface.")
        st.info("Start the server with: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`")
        return
    
    # File uploader
    st.markdown("### Upload Pump Audio")
    
    uploaded_file = st.file_uploader(
        "Upload a .wav audio file of pump sound",
        type=['wav'],
        help="Select a WAV audio file for analysis"
    )
    
    if uploaded_file is not None:
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        

        # Predict button
        if st.button("Analyze Pump Audio", type="primary", width='stretch'):
            with st.spinner("Analyzing audio... This may take a few seconds..."):
                
                start_time = datetime.now()
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load audio for visualization
                audio, sr_raw = librosa.load(tmp_path, sr=None)
                sr = int(sr_raw)
                
                # Make prediction
                result, error = predict_audio(uploaded_file.getvalue(), uploaded_file.name)
                
                if error:
                    st.error(f"❌ Prediction failed: {error}")
                    return
                
                if result:                    
                    # Extract key info
                    prediction = result['prediction']
                    confidence = result['confidence']
                    prob_normal = result['probability_normal']
                    prob_abnormal = result['probability_abnormal']
                    explainability = result.get('explainability', {})
                                        
                    # ========================================
                    # METRICS
                    # ========================================
                    st.markdown("---")
                    st.markdown("### Prediction Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Prediction",
                            prediction,
                            delta="Normal" if prediction == 'Normal' else "Anomaly"
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence",
                            f"{confidence*100:.2f}%",
                            delta="High" if result['is_confident'] else "Low"
                        )
                    
                    with col3:
                        st.metric(
                            "Normal Probability",
                            f"{prob_normal*100:.2f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "Abnormal Probability",
                            f"{prob_abnormal*100:.2f}%"
                        )
                    
                    # ========================================
                    # TABS FOR DETAILED ANALYSIS
                    # ========================================
                    
                    st.markdown("---")
                    
                    # Create tabs
                    tab1, tab2, tab3 = st.tabs([
                        "Explainability",
                        "Feature Analysis",
                        "Recommendations"
                    ])
                    
                    # TAB 1: EXPLAINABILITY
                    with tab1:
                        st.markdown("## Why This Prediction?")
                        
                        if explainability.get('stats_available'):
                            # Natural language explanation
                            st.markdown("### Explanation")
                            explanation_text = explainability.get('explanation_text', '')
                            st.markdown(explanation_text)
                            
                            st.markdown("---")
                            
                            # Feature deviations
                            deviations = explainability.get('deviations', {})
                            all_deviations = deviations.get('all', [])
                            
                            if all_deviations:
                                st.markdown("### Feature Deviation Analysis")
                                
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Deviation chart
                                    fig = create_feature_deviation_chart(all_deviations, top_n=10)
                                    st.plotly_chart(fig, width='stretch')
                                
                                with col2:
                                    # Severity distribution
                                    fig = create_severity_distribution(all_deviations)
                                    st.plotly_chart(fig, width='stretch')
                                
                                # Top deviating features table
                                st.markdown("### Top 10 Deviating Features")
                                
                                top_10 = all_deviations[:10]
                                
                                df = pd.DataFrame([
                                    {
                                        'Feature': d['feature_name'],
                                        'Z-Score': f"{d['z_score']:.2f}σ",
                                        'Severity': d['severity'],
                                        'Your Value': f"{d['user_value']:.4f}",
                                        'Normal Mean': f"{d['normal_mean']:.4f}",
                                        'Description': d['description']
                                    }
                                    for d in top_10
                                ])
                                
                                st.dataframe(df, width='stretch', hide_index=True)
                            
                            st.markdown("---")
                            
                            # Interpretation guide
                            with st.expander("How to Interpret These Results"):
                                st.markdown("""
                                **Z-Score (Standard Deviations):**
                                - Shows how far each feature is from normal training data
                                - 🔴 **|Z| > 2.0**: Severe deviation (high alert)
                                - 🟠 **1.0 < |Z| < 2.0**: Moderate deviation (warning)
                                - 🟢 **|Z| < 1.0**: Within normal range
                                
                                **Severity Distribution:**
                                - Shows the proportion of features in each severity category
                                - More severe deviations → Higher likelihood of anomaly
                                
                                **Feature Descriptions:**
                                - Each feature captures specific acoustic properties
                                - Deviations indicate which properties are abnormal
                                
                                ---
                                
                                ### Why Use 2σ Instead of 3σ?
                                
                                **Short Answer:** This is a safety-critical design decision for industrial equipment monitoring.
                                
                                **Here's Why:**
                                
                                **1. Cost of Errors is Asymmetric**
                                """)
                                
                                # Create comparison table
                                comparison_data = {
                                    'Error Type': ['False Positive', 'False Negative'],
                                    'In 3σ System': ['Lower', 'Higher'],
                                    'In 2σ System': ['Higher', 'Lower'],
                                    'Cost': [
                                        'Unnecessary inspection = $1,000',
                                        'Missed failure = $100,000+ (damage, safety risk, downtime)'
                                    ]
                                }
                                
                                df_comparison = pd.DataFrame(comparison_data)
                                st.dataframe(df_comparison, width='stretch', hide_index=True)
                                
                                st.markdown("""
                                **In pump monitoring:** It's better to inspect 10 times unnecessarily than miss 1 real failure.
                                
                                **2. Industrial Standards**
                                
                                Many industrial monitoring systems use **2σ as the action threshold**:
                                
                                ```
                                Statistical Interpretation:
                                - 2σ threshold = 95% confidence
                                - Only 5% of truly normal samples exceed this
                                - If your pump exceeds 2σ, there's a 95% chance something is wrong
                                ```
                                
                                **Bottom Line:** The 2σ threshold balances sensitivity (catching failures early) with specificity (avoiding too many false alarms), making it ideal for safety-critical pump monitoring.
                                """)
                        
                        else:
                            st.warning("""
                            ⚠️ **Limited Explainability**
                            
                            Training statistics not available. Run `training_stats_generator.py` to enable full explainability features.
                            
                            Basic explanation:
                            """ + explainability.get('explanation_text', ''))
                    
                    # TAB 2: FEATURE ANALYSIS
                    with tab2:
                        st.markdown("## Extracted Features")
                        
                        # ===================================================================
                        # SECTION 1: ALL EXTRACTED FEATURES (Grouped)
                        # ===================================================================
                        
                        st.markdown("### All Extracted Features")
                        st.markdown("Complete list of 42 acoustic features extracted from your audio.")
    
                        features = result.get('features', {})
                        feature_values = features.get('values', [])
                        feature_names = features.get('names', [])
                        
                        if feature_values and feature_names:
                            # Create DataFrame
                            df_features = pd.DataFrame({
                                'Feature Name': feature_names,
                                'Value': feature_values
                            })
                            
                            # Add feature index
                            df_features.insert(0, 'Index', range(1, len(df_features) + 1))
                            
                            # Feature groups with descriptions
                            feature_groups = {
                                'MFCCs': {
                                    'indices': list(range(0, 13)),
                                    'description': 'Represent the short-term power spectrum of sound, capturing timbre and texture characteristics similar to human hearing perception.'
                                },
                                'Spectral': {
                                    'indices': list(range(13, 19)),
                                    'description': 'Measure frequency distribution properties like brightness, energy concentration, and tonal balance in the audio signal over time.'
                                },
                                'Temporal': {
                                    'indices': list(range(19, 23)),
                                    'description': 'Capture time-domain characteristics including signal energy variations, zero-crossing patterns, and amplitude envelope dynamics over duration.'
                                },
                                'Advanced': {
                                    'indices': list(range(23, 35)),
                                    'description': 'Complex acoustic properties including spectral entropy, harmonic content, and energy distribution patterns for detailed sound characterization.'
                                },
                                'New Features': {
                                    'indices': list(range(35, 42)),
                                    'description': 'Discriminative attributes like pitch tracking, flatness, crest factor, and kurtosis measuring signal impulsiveness and statistical distribution shape.'
                                }
                            }
                            
                            # Display by groups
                            for group_name, group_data in feature_groups.items():
                                indices = group_data['indices']
                                expander_label = f"**{group_name}:**"
                                with st.expander(expander_label, expanded=(group_name == 'MFCCs')):
                                    # Display description with styling
                                    st.markdown(
                                        f'<p style="color: #ffffff; margin-bottom: 1rem; font-size: 0.9rem;">{group_data["description"]}</p>',
                                        unsafe_allow_html=True
                                    )
                                    group_df = df_features.iloc[indices].copy()
                                    
                                    # Display table
                                    st.dataframe(
                                        group_df,
                                        width='stretch',
                                        hide_index=True
                                    )
                                    
                                    # Show count
                                    st.caption(f"{len(indices)} features in this group")
                            
                            st.markdown("---")
                            
                            # ===================================================================
                            # SECTION 2: DOWNLOAD OPTIONS
                            # ===================================================================
                            
                            st.markdown("### Download Feature Data")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Download as CSV
                                csv = df_features.to_csv(index=False)
                                st.download_button(
                                    label=" Download Features (CSV)",
                                    data=csv,
                                    file_name=f"pump_features_{uploaded_file.name.replace('.wav', '')}.csv",
                                    mime="text/csv",
                                    width='stretch'
                                )
                            
                            with col2:
                                # Download as JSON
                                json_data = json.dumps({
                                    'filename': uploaded_file.name,
                                    'timestamp': start_time.isoformat(),
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'features': {name: float(val) for name, val in zip(feature_names, feature_values)}
                                }, indent=2)
                                
                                st.download_button(
                                    label=" Download Full Report (JSON)",
                                    data=json_data,
                                    file_name=f"pump_analysis_{uploaded_file.name.replace('.wav', '')}.json",
                                    mime="application/json",
                                    width='stretch'
                                )
                        
                        else:
                            st.warning("Feature data not available")
                    
                    # TAB 3: RECOMMENDATIONS
                    with tab3:
                        st.markdown("## Recommendations")
                        
                        recommendations = explainability.get('recommendations', [])
                        
                        if recommendations:
                            for rec in recommendations:
                                if "⚠️ **URGENT**" in rec or "⚠️ **Immediate" in rec:
                                    st.error(rec)
                                elif "⚠️" in rec:
                                    st.warning(rec)
                                else:
                                    st.success(rec)
                        else:
                            if prediction == 'Normal':
                                st.success("""
                                 **Continue Normal Operations**
                                - Monitor regularly for any changes
                                - Maintain scheduled maintenance
                                - Keep logs for trend analysis
                                """)
                            else:
                                st.warning("""
                                ⚠️ **Immediate Action Required**
                                - Schedule inspection immediately
                                - Check features with high deviation scores
                                - Review maintenance records
                                - Consider backup pump activation
                                """)
                        
                        st.markdown("---")
                        
                        # Next steps
                        st.markdown("### Next Steps")
                        st.info("""
                        1. **Document the results** - Save this report for records
                        2. **Compare with history** - Look at trends over time
                        3. **Take action** - Follow the recommendations above
                        4. **Re-test** - Monitor and re-test after any interventions
                        5. **Contact support** - If unsure, consult maintenance team
                        """)
                    
                    # ========================================
                    # RAW RESULTS (EXPANDABLE)
                    # ========================================
                    
                    st.markdown("---")
                    
                    with st.expander("View Raw API Response"):
                        st.json(result)
                
                # Clean up temp file
                try:
                    Path(tmp_path).unlink()
                except:
                    pass
    
    else:
        # Instructions when no file uploaded
        st.markdown("### How to Use This System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 1️⃣ Upload Audio
            
            Click the upload button above and select a **.wav** file of your pump sound recording.
            """)
        
        with col2:
            st.markdown("""
            #### 2️⃣ AI Analysis
            
            Click "Analyze" to process your audio. 
            """)
        
        with col3:
            st.markdown("""
            #### 3️⃣ Get Insights
            
            Review the results!
            """)


if __name__ == "__main__":
    main()
