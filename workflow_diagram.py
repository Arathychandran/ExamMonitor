import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with white background
fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_setup = '#3498db'      # Blue
color_monitor = '#e74c3c'    # Red
color_verify = '#2ecc71'     # Green
color_report = '#f39c12'     # Orange
color_text = '#2c3e50'       # Dark gray
color_arrow = '#34495e'      # Medium gray

# Title
ax.text(5, 9.5, 'Exam Monitoring System Workflow', 
        fontsize=28, fontweight='bold', ha='center', color=color_text)

# ==================== PHASE 1: SETUP ====================
# Box 1
box1 = FancyBboxPatch((0.3, 7.5), 2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor=color_setup, facecolor=color_setup, alpha=0.8, linewidth=2)
ax.add_patch(box1)
ax.text(1.3, 8.3, 'PHASE 1', fontsize=11, fontweight='bold', ha='center', color='white')
ax.text(1.3, 7.9, 'SETUP', fontsize=11, fontweight='bold', ha='center', color='white')

# Details
ax.text(1.3, 7.4, '• Configure Classroom', fontsize=9, ha='center', color=color_text)
ax.text(1.3, 7.0, '• Upload Student Photos', fontsize=9, ha='center', color=color_text)
ax.text(1.3, 6.6, '• Assign Seats', fontsize=9, ha='center', color=color_text)

# ==================== PHASE 2: VERIFICATION ====================
box2 = FancyBboxPatch((3.5, 7.5), 2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor=color_verify, facecolor=color_verify, alpha=0.8, linewidth=2)
ax.add_patch(box2)
ax.text(4.5, 8.3, 'PHASE 2', fontsize=11, fontweight='bold', ha='center', color='white')
ax.text(4.5, 7.9, 'VERIFICATION', fontsize=11, fontweight='bold', ha='center', color='white')

# Details
ax.text(4.5, 7.4, '• Start Exam', fontsize=9, ha='center', color=color_text)
ax.text(4.5, 7.0, '• Verify Student ID', fontsize=9, ha='center', color=color_text)
ax.text(4.5, 6.6, '• DeepFace Match', fontsize=9, ha='center', color=color_text)

# ==================== PHASE 3: MONITORING ====================
box3 = FancyBboxPatch((6.7, 7.5), 2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor=color_monitor, facecolor=color_monitor, alpha=0.8, linewidth=2)
ax.add_patch(box3)
ax.text(7.7, 8.3, 'PHASE 3', fontsize=11, fontweight='bold', ha='center', color='white')
ax.text(7.7, 7.9, 'MONITORING', fontsize=11, fontweight='bold', ha='center', color='white')

# Details
ax.text(7.7, 7.4, '• Analyze Behavior', fontsize=9, ha='center', color=color_text)
ax.text(7.7, 7.0, '• Detect Cheating', fontsize=9, ha='center', color=color_text)
ax.text(7.7, 6.6, '• Real-time Analysis', fontsize=9, ha='center', color=color_text)

# ==================== PHASE 4: REPORTING ====================
box4 = FancyBboxPatch((9.2, 7.5), 0.5, 1.2, boxstyle="round,pad=0.05", 
                       edgecolor=color_report, facecolor=color_report, alpha=0.8, linewidth=2)
ax.add_patch(box4)

# Arrows between main phases
arrow1 = FancyArrowPatch((2.3, 8.1), (3.5, 8.1), 
                        arrowstyle='->', lw=2.5, color=color_arrow, mutation_scale=25)
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((5.5, 8.1), (6.7, 8.1), 
                        arrowstyle='->', lw=2.5, color=color_arrow, mutation_scale=25)
ax.add_patch(arrow2)

arrow3 = FancyArrowPatch((8.7, 8.1), (9.2, 8.1), 
                        arrowstyle='->', lw=2.5, color=color_arrow, mutation_scale=25)
ax.add_patch(arrow3)

# ==================== DETAILED SECTION ====================
# Horizontal line
ax.plot([0.3, 9.7], [6.3, 6.3], 'k-', linewidth=1, alpha=0.3)

# === SETUP DETAILS ===
ax.text(1.3, 5.9, 'SETUP', fontsize=12, fontweight='bold', ha='center', color=color_setup)
setup_items = [
    '1. Seat Detection\n   (YOLOv8)',
    '2. Upload Photos\n   (Student images)',
    '3. Assign Seats\n   (Roll → Seat mapping)'
]
y_pos = 5.4
for item in setup_items:
    ax.text(1.3, y_pos, item, fontsize=8, ha='center', color=color_text, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))
    y_pos -= 0.8

# === VERIFICATION DETAILS ===
ax.text(4.5, 5.9, 'VERIFICATION', fontsize=12, fontweight='bold', ha='center', color=color_verify)
verify_items = [
    '1. Capture Face\n   (Crop from bbox)',
    '2. DeepFace.verify()\n   (Face matching)',
    '3. Store Result\n   (Verified/Mismatch)'
]
y_pos = 5.4
for item in verify_items:
    ax.text(4.5, y_pos, item, fontsize=8, ha='center', color=color_text,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))
    y_pos -= 0.8

# === MONITORING DETAILS ===
ax.text(7.7, 5.9, 'MONITORING', fontsize=12, fontweight='bold', ha='center', color=color_monitor)
monitor_items = [
    'Head Movement\nDetection',
    'Lip Movement\nDetection',
    'Phone Detection\n(Optional)',
    'Hand Activity\n(Optional)'
]
y_pos = 5.4
for item in monitor_items:
    ax.text(7.7, y_pos, item, fontsize=7.5, ha='center', color=color_text,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.2))
    y_pos -= 0.7

# === REPORTING ===
ax.text(9.45, 5.9, 'REPORT', fontsize=11, fontweight='bold', ha='center', color=color_report, rotation=-90, va='bottom')
ax.text(9.45, 3, 'Export\nResults', fontsize=8, ha='center', color=color_text, rotation=-90,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))

# ==================== KEY FEATURES BOX ====================
features_box = FancyBboxPatch((0.3, 0.3), 9.4, 1.5, boxstyle="round,pad=0.15", 
                              edgecolor='#95a5a6', facecolor='#ecf0f1', alpha=0.8, linewidth=2)
ax.add_patch(features_box)

ax.text(5, 1.65, 'Key AI Models & Technologies', fontsize=12, fontweight='bold', ha='center', color=color_text)

features = [
    '🎯 YOLOv8 - Seat & Object Detection',
    '👤 DeepFace - Face Verification',
    '🎭 MediaPipe - Head & Mouth Tracking',
    '💾 SQLAlchemy - Database Management',
    '🌐 Flask - Web Interface'
]

x_positions = [1.2, 3.1, 5.0, 6.9, 8.8]
for feature, x_pos in zip(features, x_positions):
    ax.text(x_pos, 1.15, feature, fontsize=8.5, ha='center', color=color_text,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.25))

# Add database icon at bottom
ax.text(0.5, 0.05, '🗄️ SQLite Database', fontsize=9, color=color_text, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/Desktop/exam_monitoring_system/workflow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Workflow diagram saved: workflow_diagram.png")

# Also create a simpler, more minimalist version
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')

# Title
ax2.text(5, 9.3, 'Exam Monitoring System - Workflow', 
        fontsize=26, fontweight='bold', ha='center', color=color_text)

# Main flow (simplified)
phases = [
    {'name': '1. SETUP', 'x': 1.5, 'color': color_setup, 
     'desc': 'Configure\nClassroom'},
    {'name': '2. VERIFY', 'x': 3.75, 'color': color_verify,
     'desc': 'Student ID\nVerification'},
    {'name': '3. MONITOR', 'x': 6, 'color': color_monitor,
     'desc': 'Detect\nCheating'},
    {'name': '4. REPORT', 'x': 8.25, 'color': color_report,
     'desc': 'Generate\nReports'}
]

# Draw main flow boxes
for i, phase in enumerate(phases):
    # Main box
    box = FancyBboxPatch((phase['x']-0.6, 6.5), 1.2, 1.8, 
                         boxstyle="round,pad=0.1", 
                         edgecolor=phase['color'], facecolor=phase['color'], 
                         alpha=0.85, linewidth=2.5)
    ax2.add_patch(box)
    
    # Phase name
    ax2.text(phase['x'], 8.0, phase['name'], fontsize=12, fontweight='bold', 
            ha='center', color='white')
    
    # Description
    ax2.text(phase['x'], 7.1, phase['desc'], fontsize=9, ha='center', 
            color='white', fontweight='bold')
    
    # Arrow to next
    if i < len(phases) - 1:
        next_x = phases[i+1]['x'] - 0.6
        arrow = FancyArrowPatch((phase['x']+0.6, 7.4), (next_x, 7.4), 
                               arrowstyle='->', lw=2.5, color=color_arrow, mutation_scale=20)
        ax2.add_patch(arrow)

# Detailed breakdown boxes
breakdown = [
    {
        'title': 'SETUP',
        'items': [
            '• Detect seats (YOLOv8)',
            '• Upload student photos',
            '• Map seats to roll numbers'
        ],
        'x': 1.5, 'color': color_setup
    },
    {
        'title': 'VERIFY',
        'items': [
            '• Capture face from camera',
            '• Compare with student photo',
            '• Log verification status'
        ],
        'x': 3.75, 'color': color_verify
    },
    {
        'title': 'MONITOR',
        'items': [
            '• Head movement detection',
            '• Lip movement analysis',
            '• Phone detection (optional)'
        ],
        'x': 6, 'color': color_monitor
    },
    {
        'title': 'REPORT',
        'items': [
            '• Dashboard visualization',
            '• PDF/Excel export',
            '• Flagged student list'
        ],
        'x': 8.25, 'color': color_report
    }
]

for section in breakdown:
    # Box
    detail_box = FancyBboxPatch((section['x']-0.75, 3.5), 1.5, 2.3,
                               boxstyle="round,pad=0.1",
                               edgecolor=section['color'], facecolor=section['color'],
                               alpha=0.15, linewidth=1.5)
    ax2.add_patch(detail_box)
    
    # Title
    ax2.text(section['x'], 5.6, section['title'], fontsize=11, fontweight='bold',
            ha='center', color=section['color'])
    
    # Items
    y = 5.1
    for item in section['items']:
        ax2.text(section['x'], y, item, fontsize=8, ha='center', 
                color=color_text, multialignment='center')
        y -= 0.55

# Bottom tech stack
tech_box = FancyBboxPatch((0.3, 0.3), 9.4, 2.5,
                         boxstyle="round,pad=0.15",
                         edgecolor='#bdc3c7', facecolor='#f8f9fa', 
                         linewidth=2)
ax2.add_patch(tech_box)

ax2.text(5, 2.5, 'Technology Stack', fontsize=13, fontweight='bold',
        ha='center', color=color_text)

tech_stack = [
    'YOLOv8: Seat & Object Detection',
    'DeepFace: Face Verification',
    'MediaPipe: Motion Analysis',
    'Flask: Web Interface',
    'SQLite: Database'
]

y_pos = 1.9
for tech in tech_stack:
    ax2.text(5, y_pos, tech, fontsize=9, ha='center', color=color_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    y_pos -= 0.35

plt.tight_layout()
plt.savefig('/home/ubuntu/Desktop/exam_monitoring_system/workflow_simple.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Simplified workflow saved: workflow_simple.png")

print("\n✓ Both workflow diagrams created successfully!")
print("  - workflow_diagram.png (detailed)")
print("  - workflow_simple.png (clean & simple for PPT)")
