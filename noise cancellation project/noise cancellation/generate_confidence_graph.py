import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder
os.makedirs("flowchart_output", exist_ok=True)

# Data for the graph - showing how denoising confidence varies with input dB level
# for different strength levels
input_db = np.array([20, 30, 40, 50, 60, 70, 80, 90])

# Confidence curves for different denoise strength levels
# Mild (alpha=2.5) - less aggressive
confidence_mild = np.array([52, 58, 65, 72, 78, 84, 88, 92])

# Standard (alpha=5.0) - balanced
confidence_standard = np.array([50, 60, 70, 78, 84, 89, 93, 96])

# Aggressive (alpha=8.0) - heavy removal
confidence_aggressive = np.array([48, 62, 75, 84, 90, 94, 97, 98.5])

# Create figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines
ax.plot(input_db, confidence_mild, 'o-', linewidth=3, markersize=8, 
        color='#4CAF50', label='CR=α/β=2.5\n(Mild)', markerfacecolor='#81C784', 
        markeredgewidth=2, markeredgecolor='#2E7D32')

ax.plot(input_db, confidence_standard, 's-', linewidth=3, markersize=8, 
        color='#FFA500', label='CR=α/β=5.0\n(Standard)', markerfacecolor='#FFB74D', 
        markeredgewidth=2, markeredgecolor='#F57C00')

ax.plot(input_db, confidence_aggressive, '^-', linewidth=3, markersize=8, 
        color='#FF6B6B', label='CR=α/β=8.0\n(Aggressive)', markerfacecolor='#EF9A9A', 
        markeredgewidth=2, markeredgecolor='#C62828')

# Add region labels
ax.axvspan(20, 40, alpha=0.08, color='green', label='Soft')
ax.axvspan(40, 70, alpha=0.08, color='orange', label='Moderate')
ax.axvspan(70, 90, alpha=0.08, color='red', label='Loud')

# Add region text
ax.text(30, 95, 'Soft', fontsize=14, weight='bold', alpha=0.6)
ax.text(55, 95, 'Moderate', fontsize=14, weight='bold', alpha=0.6)
ax.text(80, 95, 'Loud', fontsize=14, weight='bold', alpha=0.6)

# Styling
ax.set_xlabel('Input Level (dB)', fontsize=13, weight='bold')
ax.set_ylabel('Confidence (%)', fontsize=13, weight='bold')
ax.set_title('Noise Cancellation Confidence vs Input Level\nby Denoising Strength', 
             fontsize=14, weight='bold', pad=20)

ax.set_xlim(15, 95)
ax.set_ylim(45, 102)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='lower right', framealpha=0.95)

# Set tick labels
ax.set_xticks(np.arange(20, 100, 10))
ax.set_yticks(np.arange(50, 105, 10))

# Add subtle background
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Tight layout
plt.tight_layout()

# Save as JPEG
jpeg_path = "flowchart_output/denoise_confidence_graph.jpg"
plt.savefig(jpeg_path, format='jpg', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Graph saved: {jpeg_path}")

# Save as PNG
png_path = "flowchart_output/denoise_confidence_graph.png"
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Graph also saved: {png_path}")

print("\n✓ Files ready in: flowchart_output/")
print("  - denoise_confidence_graph.jpg")
print("  - denoise_confidence_graph.png")
