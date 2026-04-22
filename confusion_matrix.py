#!/usr/bin/env python3
"""
Confusion Matrix Visualization Generator
Generates confusion matrix visualization from training_report.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path

# File paths 
JSON_FILE = r"C:\Users\chana\OneDrive - IQuest Solutions Corp\Sound_classification\Hitachi\models_ml\training_report.json"
OUTPUT_DIR = r"C:\Users\chana\OneDrive - IQuest Solutions Corp\Sound_classification\Hitachi\visualizations"

def load_training_report(json_path):
    """Load training report from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_confusion_matrix_visualization(report_data, output_dir):
    """
    Create confusion matrix visualization with numbers and percentages
    Standard layout: Columns=Predicted, Rows=Actual
    [[TN, FP], [FN, TP]] - matching standard conventions
    """
    # Extract confusion matrix data from JSON
    cm_data = report_data['test_performance']['confusion_matrix']
    
    # Get values from JSON
    tn = cm_data['true_negatives']
    fp = cm_data['false_positives']
    fn = cm_data['false_negatives']
    tp = cm_data['true_positives']
    
    # Create confusion matrix array with STANDARD layout:
    # Rows = Actual, Columns = Predicted
    # Row 0 (Actual Negative): TN, FP
    # Row 1 (Actual Positive): FN, TP
    cm = np.array([[tn, fp], 
                   [fn, tp]])
    
    # Calculate total samples
    total_samples = cm.sum()
    
    # Calculate percentages
    cm_percentages = (cm / total_samples) * 100
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define color scheme - Dark Blue and Ice Blue checkerboard pattern
    dark_blue = '#1e3a8a'   # Dark blue
    light_blue = '#93c5fd'  # Ice/Light blue
    
    # Create color matrix for checkerboard pattern
    # [[TN, FP], [FN, TP]]
    color_matrix = np.array([[dark_blue, light_blue],    # Row 0: TN, FP
                             [light_blue, dark_blue]])    # Row 1: FN, TP
    
    # Draw cells manually with colors
    for i in range(2):
        for j in range(2):
            color = color_matrix[i, j]
            rect = Rectangle((j, i), 1, 1, facecolor=color, 
                            edgecolor='white', linewidth=3)
            ax.add_patch(rect)
    
    # Add custom annotations with numbers and percentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percentage = cm_percentages[i, j]
            
            # Format text with number and percentage
            text = f'{count}\n({percentage:.2f}%)'
            
            # Determine text color based on background (dark cells get white text)
            is_dark_cell = color_matrix[i, j] == dark_blue
            text_color = 'white' if is_dark_cell else 'black'
            
            # Add text to cell
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=18, fontweight='bold',
                   color=text_color)
    
    # Set axis limits and aspect
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()  # Invert so row 0 is at top
    ax.set_aspect('equal')
    
    # Turn off axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # CORRECTED LABELS: Columns=Predicted, Rows=Actual
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold')
    ax.set_title(f'Confusion Matrix (Safety-ML)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set tick positions at center of cells
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    
    # Set tick labels - Normal first (negative class), then Abnormal (positive class)
    # This matches standard convention where class 0 (Normal) comes before class 1 (Abnormal)
    class_labels = ['Normal', 'Abnormal']
    ax.set_xticklabels(class_labels, fontsize=12)
    ax.set_yticklabels(class_labels, fontsize=12, rotation=90, va='center')
    
    # Remove minor ticks
    ax.tick_params(which='both', length=0)
    
    # Add cell labels for clarity
    # TN at [0,0] - dark blue background, white text
    ax.text(0.1, 0.9, 'TN', fontsize=10, color='white', 
            fontweight='bold', alpha=0.7)
    # FP at [0,1] - light blue background, black text
    ax.text(1.1, 0.9, 'FP', fontsize=10, color='black', 
            fontweight='bold', alpha=0.7)
    # FN at [1,0] - light blue background, black text
    ax.text(0.1, 1.9, 'FN', fontsize=10, color='black', 
            fontweight='bold', alpha=0.7)
    # TP at [1,1] - dark blue background, white text
    ax.text(1.1, 1.9, 'TP', fontsize=10, color='white', 
            fontweight='bold', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'confusion_matrix_ML.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    
    plt.close('all')
    
    return output_path

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("Confusion Matrix Visualization Generator")
    print("="*60)
    
    # Load training report from JSON
    report_data = load_training_report(JSON_FILE)
    print("✓ Training report loaded successfully")
    
    # Create visualization
    print("Generating confusion matrix visualization...")
    output_path = create_confusion_matrix_visualization(report_data, OUTPUT_DIR)
    
    print(f"\nGenerated file: {output_path}")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()