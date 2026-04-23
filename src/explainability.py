"""
Explainability Module for Pump Anomaly Detection
=================================================

Generates human-readable explanations for model predictions using:
1. Feature deviation analysis (z-scores)
2. Feature importance ranking
3. Natural language reasoning
4. Confidence analysis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle

from .config import config, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from .utils import calculate_z_scores, get_severity_color, get_severity_label, logger


class ExplainabilityEngine:
    """
    Generate explanations for model predictions
    """
    
    def __init__(self, training_stats_path: Optional[str] = None):
        """
        Initialize explainability engine
        
        Args:
            training_stats_path: Path to training statistics pickle file
        """
        self.training_stats = None
        self.stats_loaded = False
        
        if training_stats_path:
            self.load_training_stats(training_stats_path)
    
    def load_training_stats(self, stats_path: str):
        """
        Load training statistics for deviation analysis
        
        Args:
            stats_path: Path to statistics pickle file
        """
        try:
            with open(stats_path, 'rb') as f:
                self.training_stats = pickle.load(f)
            
            self.stats_loaded = True
            logger.info(f"Loaded training statistics!")
        
        except Exception as e:
            logger.warning(f"Could not load training statistics: {e}")
            logger.warning("Explainability will be limited")
            self.stats_loaded = False
    
    def calculate_feature_deviations(self, user_features: np.ndarray) -> List[Dict]:
        """
        Calculate how much each feature deviates from training normal
        
        Args:
            user_features: Extracted features from user's audio
        
        Returns:
            List of deviation dictionaries sorted by severity
        """
        if not self.stats_loaded or self.training_stats is None:
            logger.warning("Training statistics not loaded — skipping deviation calculation")
            return []

        # Calculate z-scores (deviations from normal mean)
        normal_stats = {
            'mean': self.training_stats['normal']['mean'],
            'std': self.training_stats['normal']['std']
        }
        
        z_scores = calculate_z_scores(user_features, normal_stats)
        
        # Create deviation info for each feature
        deviations = []
        for i, (feature_name, z_score, feature_value) in enumerate(
            zip(FEATURE_NAMES, z_scores, user_features)
        ):
            deviation = {
                'feature_name': feature_name,
                'feature_index': i,
                'z_score': float(z_score),
                'abs_z_score': float(abs(z_score)),
                'user_value': float(feature_value),
                'normal_mean': float(normal_stats['mean'][i]),
                'normal_std': float(normal_stats['std'][i]),
                'severity': get_severity_label(z_score),
                'color': get_severity_color(z_score),
                'description': FEATURE_DESCRIPTIONS.get(feature_name, 'Unknown feature')
            }
            deviations.append(deviation)
        
        # Sort by absolute z-score (most deviant first)
        deviations.sort(key=lambda x: x['abs_z_score'], reverse=True)
        
        return deviations
    
    def get_top_deviating_features(
        self, 
        deviations: List[Dict], 
        n: int = 10
    ) -> List[Dict]:
        """
        Get top N most deviating features
        
        Args:
            deviations: List of deviation dictionaries
            n: Number of top features to return
        
        Returns:
            List of top N deviations
        """
        return deviations[:n]
    
    def categorize_deviations(self, deviations: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize deviations by severity
        
        Args:
            deviations: List of deviation dictionaries
        
        Returns:
            Dictionary with severity categories
        """
        categorized = {
            'severe': [],   # |z| >= 2.0
            'moderate': [], # 1.0 <= |z| < 2.0
            'normal': []    # |z| < 1.0
        }
        
        for dev in deviations:
            if dev['abs_z_score'] >= config.SEVERE_DEVIATION_THRESHOLD:
                categorized['severe'].append(dev)
            elif dev['abs_z_score'] >= config.MODERATE_DEVIATION_THRESHOLD:
                categorized['moderate'].append(dev)
            else:
                categorized['normal'].append(dev)
        
        return categorized
    
    def generate_natural_language_explanation(
        self,
        prediction: str,
        confidence: float,
        deviations: List[Dict],
        top_n: int = 5
    ) -> str:
        """
        Generate human-readable explanation
        
        Args:
            prediction: 'Normal' or 'Abnormal'
            confidence: Confidence score (0-1)
            deviations: List of feature deviations
            top_n: Number of top features to mention
        
        Returns:
            Natural language explanation string
        """
        if not self.stats_loaded or len(deviations) == 0:
            return self._generate_basic_explanation(prediction, confidence)
        
        # Get top deviating features
        top_deviations = self.get_top_deviating_features(deviations, n=top_n)
        
        # Count severe deviations
        categorized = self.categorize_deviations(deviations)
        n_severe = len(categorized['severe'])
        n_moderate = len(categorized['moderate'])
        
        # Build explanation
        explanation_parts = []
        
        # Opening statement
        if prediction == 'Abnormal':
            explanation_parts.append(
                f"The pump was classified as **ABNORMAL** with {confidence*100:.1f}% confidence."
            )
        else:
            explanation_parts.append(
                f"The pump was classified as **NORMAL** with {confidence*100:.1f}% confidence."
            )
        
        # Deviation summary
        if n_severe > 0:
            explanation_parts.append(
                f"\n**Key Findings:**\n"
                f"- {n_severe} features show severe deviations (>2σ from normal)\n"
                f"- {n_moderate} features show moderate deviations (1-2σ from normal)"
            )
        elif n_moderate > 0:
            explanation_parts.append(
                f"\n**Key Findings:**\n"
                f"- {n_moderate} features show moderate deviations (1-2σ from normal)\n"
                f"- Most features are within normal range"
            )
        else:
            explanation_parts.append(
                "\n**Key Findings:**\n"
                "- All features are within normal statistical range\n"
                "- No significant deviations detected"
            )
        
        # Top deviating features
        if prediction == 'Abnormal' and len(top_deviations) > 0:
            explanation_parts.append("\n**Most Abnormal Features:**")
            
            for i, dev in enumerate(top_deviations[:3], 1):
                direction = "higher" if dev['z_score'] > 0 else "lower"
                explanation_parts.append(
                    f"{i}. **{dev['feature_name']}**: "
                    f"{abs(dev['z_score']):.1f}σ {direction} than normal "
                    f"({dev['description']})"
                )
        
        # Interpretation
        explanation_parts.append(self._add_interpretation(prediction, categorized))
        
        return "\n".join(explanation_parts)
    
    def _generate_basic_explanation(self, prediction: str, confidence: float) -> str:
        """
        Generate basic explanation without deviation analysis
        
        Args:
            prediction: 'Normal' or 'Abnormal'
            confidence: Confidence score
        
        Returns:
            Basic explanation string
        """
        if prediction == 'Abnormal':
            return (
                f"The pump was classified as **ABNORMAL** with {confidence*100:.1f}% confidence.\n\n"
                "The model detected acoustic patterns that differ from normal pump operation. "
                "This may indicate mechanical wear, bearing issues, or other anomalies."
            )
        else:
            return (
                f"The pump was classified as **NORMAL** with {confidence*100:.1f}% confidence.\n\n"
                "The acoustic patterns match those of healthy pump operation. "
                "Continue regular monitoring and maintenance schedules."
            )
    
    def _add_interpretation(self, prediction: str, categorized: Dict) -> str:
        """
        Add interpretation based on deviation patterns
        
        Args:
            prediction: 'Normal' or 'Abnormal'
            categorized: Categorized deviations
        
        Returns:
            Interpretation string
        """
        n_severe = len(categorized['severe'])
        
        if prediction == 'Abnormal':
            if n_severe >= 5:
                return (
                    "\n**Interpretation:**\n"
                    "Multiple acoustic features show significant abnormalities. "
                    "This pattern is consistent with mechanical issues such as:\n"
                    "- Bearing wear or damage\n"
                    "- Cavitation\n"
                    "- Misalignment\n"
                    "- Impeller issues\n\n"
                    "**Recommendation:** Immediate inspection recommended."
                )
            elif n_severe >= 2:
                return (
                    "\n**Interpretation:**\n"
                    "Several features deviate from normal patterns. "
                    "Early signs of potential issues detected.\n\n"
                    "**Recommendation:** Schedule inspection and monitor closely."
                )
            else:
                return (
                    "\n**Interpretation:**\n"
                    "Subtle acoustic anomalies detected. "
                    "May indicate early-stage degradation.\n\n"
                    "**Recommendation:** Monitor and compare with historical data."
                )
        else:
            return (
                "\n**Interpretation:**\n"
                "Acoustic signature matches healthy pump operation patterns. "
                "No immediate concerns detected.\n\n"
                "**Recommendation:** Continue normal operations and maintenance."
            )
    
    def create_feature_importance_chart_data(
        self,
        deviations: List[Dict],
        top_n: int = 10
    ) -> List[Dict]:
        """
        Prepare data for feature importance visualization
        
        Args:
            deviations: List of deviation dictionaries
            top_n: Number of top features
        
        Returns:
            List of chart data dictionaries
        """
        top_devs = self.get_top_deviating_features(deviations, n=top_n)
        
        chart_data = []
        for dev in top_devs:
            chart_data.append({
                'feature': dev['feature_name'],
                'z_score': dev['abs_z_score'],
                'color': dev['color'],
                'severity': dev['severity'],
                'description': dev['description']
            })
        
        return chart_data
    
    def generate_recommendations(
        self,
        prediction: str,
        confidence: float,
        deviations: List[Dict]
    ) -> List[str]:
        """
        Generate actionable recommendations
        
        Args:
            prediction: 'Normal' or 'Abnormal'
            confidence: Confidence score
            deviations: Feature deviations
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if prediction == 'Abnormal':
            # High confidence abnormal
            if confidence >= 0.90:
                recommendations.extend([
                    "⚠️ **URGENT**: Immediate inspection required",
                    "Check mechanical components (bearings, impeller, seals)",
                    "Review maintenance history for patterns",
                    "Consider backup pump activation",
                    "Document current operating conditions"
                ])
            
            # Medium confidence abnormal
            elif confidence >= 0.70:
                recommendations.extend([
                    "⚠️ Schedule inspection within 24-48 hours",
                    "Monitor closely for changes",
                    "Check features with highest deviations",
                    "Log current operating parameters",
                    "Prepare backup systems"
                ])
            
            # Low confidence abnormal
            else:
                recommendations.extend([
                    "⚠️ Potential anomaly detected (low confidence)",
                    "Compare with historical baseline",
                    "Investigate top deviating features",
                    "Increase monitoring frequency",
                    "Re-test if conditions change"
                ])
        
        else:  # Normal
            if confidence >= 0.90:
                recommendations.extend([
                    "✅ Continue normal operations",
                    "Maintain regular maintenance schedule",
                    "Keep logs for trend analysis",
                    "Periodic re-testing recommended"
                ])
            else:
                recommendations.extend([
                    "✅ Operation appears normal",
                    "Monitor for changes (lower confidence)",
                    "Compare with recent recordings",
                    "Re-test if unusual sounds noticed"
                ])
        
        return recommendations
    
    def create_full_explanation(
        self,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        user_features: np.ndarray
    ) -> Dict:
        """
        Create complete explanation package
        
        Args:
            prediction: 'Normal' or 'Abnormal'
            confidence: Confidence score
            probabilities: Dict with 'normal' and 'abnormal' probabilities
            user_features: Extracted features
        
        Returns:
            Complete explanation dictionary
        """
        # Calculate deviations
        deviations = self.calculate_feature_deviations(user_features)
        
        # Generate natural language explanation
        nl_explanation = self.generate_natural_language_explanation(
            prediction, confidence, deviations, top_n=5
        )
        
        # Get categorized deviations
        categorized = self.categorize_deviations(deviations) if deviations else {}
        
        # Generate recommendations
        recommendations = self.generate_recommendations(prediction, confidence, deviations)
        
        # Prepare chart data
        chart_data = self.create_feature_importance_chart_data(deviations, top_n=10)
        
        return {
            'explanation_text': nl_explanation,
            'deviations': {
                'all': deviations,
                'top_10': deviations[:10] if deviations else [],
                'categorized': categorized
            },
            'chart_data': chart_data,
            'recommendations': recommendations,
            'stats_available': self.stats_loaded
        }


# Example usage
if __name__ == "__main__":
    print("Explainability Engine Test")
    print("=" * 60)
    
    # Test with synthetic data
    engine = ExplainabilityEngine()
    
    # Simulate features and stats
    test_features = np.random.randn(42)
    test_stats = {
        'normal': {
            'mean': np.zeros(42),
            'std': np.ones(42),
            'n_samples': 500
        },
        'abnormal': {
            'n_samples': 100
        },
        'feature_names': FEATURE_NAMES,
        'n_features': 42
    }
    
    engine.training_stats = test_stats
    engine.stats_loaded = True
    
    # Generate explanation
    explanation = engine.create_full_explanation(
        prediction='Abnormal',
        confidence=0.95,
        probabilities={'normal': 0.05, 'abnormal': 0.95},
        user_features=test_features
    )
    
    print("\n" + explanation['explanation_text'])
    print("\n" + "=" * 60)
    print("✅ Explainability engine working!")