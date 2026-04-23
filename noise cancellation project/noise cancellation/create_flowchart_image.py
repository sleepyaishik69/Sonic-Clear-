import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Create flowchart folder
os.makedirs("flowchart_output", exist_ok=True)

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 24))
ax.set_xlim(0, 10)
ax.set_ylim(0, 30)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, color='#87CEEB', style='round'):
    """Draw a box with text"""
    if style == 'round':
        box = FancyBboxPatch((x - width/2, y - height/2), width, height, 
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color, linewidth=2)
    elif style == 'diamond':
        # Draw diamond shape
        points = [(x, y + height/2), (x + width/2, y), 
                  (x, y - height/2), (x - width/2, y)]
        diamond = mpatches.Polygon(points, fill=True, edgecolor='black', 
                                   facecolor=color, linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', wrap=True)
        return
    
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Draw flowchart
y_pos = 29

# Start
draw_box(ax, 5, y_pos, 2.5, 0.8, "🎤 Audio File Input\nmp3, wav, ogg, flac, m4a", '#87CEEB')
y_pos -= 1.2
draw_arrow(ax, 5, 29.4, 5, y_pos + 0.4)

# Check extension
draw_box(ax, 5, y_pos, 2.5, 0.8, "Check File\nExtension Valid?", '#FFE4B5', 'diamond')
y_pos -= 1.2
draw_arrow(ax, 5, 27.2, 5, y_pos + 0.4)

# Read audio
draw_box(ax, 5, y_pos, 2.5, 0.8, "Read Audio File\nwith soundfile", '#90EE90')
y_pos -= 1.2
draw_arrow(ax, 5, 25, 5, y_pos + 0.4)

# Convert to mono
draw_box(ax, 5, y_pos, 2.5, 0.8, "Convert Stereo\nto Mono", '#90EE90')
y_pos -= 1.2
draw_arrow(ax, 5, 22.8, 5, y_pos + 0.4)

# Normalize
draw_box(ax, 5, y_pos, 2.5, 0.8, "Normalize Audio\nMax Value = 1.0", '#90EE90')
y_pos -= 1.2
draw_arrow(ax, 5, 20.6, 5, y_pos + 0.4)

# Store original
draw_box(ax, 5, y_pos, 2.5, 0.8, "Store Original\nAudio Copy", '#90EE90')
y_pos -= 1.2
draw_arrow(ax, 5, 18.4, 5, y_pos + 0.4)

# VAD
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 1: Voice Activity\nDetection (VAD)", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, 16.2, 5, y_pos + 0.4)

# Silent regions
draw_box(ax, 5, y_pos, 2.5, 0.8, "Silent Regions\nFound?", '#FFE4B5', 'diamond')
silent_y = y_pos
y_pos -= 1.2

# Left arrow - Yes
draw_arrow(ax, 3.75, silent_y, 2, silent_y - 0.8, "Yes")
draw_box(ax, 2, silent_y - 1.6, 2, 0.8, "Profile Noise from\nSilent Regions", '#FFB6C6')

# Right arrow - No
draw_arrow(ax, 6.25, silent_y, 8, silent_y - 0.8, "No")
draw_box(ax, 8, silent_y - 1.6, 2, 0.8, "Use Quiet Frames\nas Noise Estimate", '#FFB6C6')

# Converge
y_pos -= 2.2
draw_arrow(ax, 2, silent_y - 1.2, 5, y_pos + 0.4)
draw_arrow(ax, 8, silent_y - 1.2, 5, y_pos + 0.4)

# Strength level
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 2: Determine\nStrength Level", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, 11, 5, y_pos + 0.4)

# Strength decision
draw_box(ax, 5, y_pos, 2.5, 0.8, "Mild/Standard/\nAggressive?", '#FFE4B5', 'diamond')
strength_y = y_pos
y_pos -= 1.5

# Continue
draw_arrow(ax, 5, strength_y - 0.4, 5, y_pos + 0.4)

# Spectral Subtraction
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 3: Spectral\nSubtraction", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, 8.2, 5, y_pos + 0.4)

# Noisereduce
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 4: noisereduce\nLibrary", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, 6, 5, y_pos + 0.4)

# Wiener Filter
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 5: Wiener\nFilter", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, 3.8, 5, y_pos + 0.4)

# Voice Protection
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 6: Voice Region\nProtection", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, 1.6, 5, y_pos + 0.4)

# Noise Gate
draw_box(ax, 5, y_pos, 2.5, 0.8, "Step 7: Noise Gate\nThreshold -28dB", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, -0.6, 5, y_pos + 0.4)

# Output
draw_box(ax, 5, y_pos, 2.5, 0.8, "Normalize Output\nPrevent Clipping", '#90EE90')
y_pos -= 1.2
draw_arrow(ax, 5, -2.8, 5, y_pos + 0.4)

# Generate graphs
draw_box(ax, 5, y_pos, 2.5, 0.8, "Generate 3 Graphs\n• Time • Frequency\n• Spectrogram", '#FFD700')
y_pos -= 1.2
draw_arrow(ax, 5, -5, 5, y_pos + 0.4)

# Return
draw_box(ax, 5, y_pos, 2.5, 0.8, "Return JSON with\nAudio + 3 Graphs", '#90EE90')

# Error handling
draw_box(ax, 8.5, 26.5, 2, 0.8, "❌ Error Handler\nReturn 400/500", '#FFB6C6')
draw_arrow(ax, 6.25, 27.2, 7.5, 26.9, "Invalid/Exception")

plt.title("Noise Cancellation Audio Processing Flowchart", fontsize=16, weight='bold', pad=20)

# Save as JPEG
output_path = "flowchart_output/noise_cancellation_flowchart.jpg"
plt.savefig(output_path, format='jpg', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Flowchart saved: {output_path}")

# Also save as PNG for better quality
png_path = "flowchart_output/noise_cancellation_flowchart.png"
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Flowchart also saved: {png_path}")

print(f"\n✓ Folder created: flowchart_output/")
print("✓ Ready to download!")
