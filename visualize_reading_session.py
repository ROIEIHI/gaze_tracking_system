"""
Reading Session Data Visualization Tool
Creates comprehensive visualizations from CSV eye tracking data using
concepts from existing visualization functions in EyeMovementAnalyzer:
- Spatial heatmap (based on get_fixation_heatmap_data concept)
- Reading path (based on get_reading_path concept) 
- Velocity analysis (duration-based)
- Temporal timeline 
- Hebrew text support with proper RTL display
- Word grouping to avoid counting same word multiple times
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from datetime import datetime
import os
import sys

# Set matplotlib to support Hebrew fonts and proper RTL display
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Tahoma', 'DejaVu Sans', 'Arial']
plt.rcParams['font.size'] = 16  # Increase default font size
plt.rcParams['axes.unicode_minus'] = False

def fix_hebrew_display(text):
    """Fix Hebrew text display to prevent mirroring by reversing character order"""
    if pd.isna(text) or text == 'Unknown':
        return text
    
    # For Hebrew text, reverse the character order to fix mirroring in matplotlib
    if is_hebrew_text(text):
        # Reverse the string to fix the character order issue
        return text[::-1]
    return text

def is_hebrew_text(text):
    """Check if text contains Hebrew characters"""
    if pd.isna(text) or text == 'Unknown':
        return False
    hebrew_chars = any('\u0590' <= char <= '\u05FF' for char in str(text))
    return hebrew_chars

# Grouping function removed - CSV data is already properly grouped by prediction.py

def load_reading_data(csv_path):
    """Load and prepare the reading session data with Hebrew support"""
    try:
        data = pd.read_csv(csv_path, encoding='utf-8')
        print(f"Loaded {len(data)} fixations from {csv_path}")
        
        # Data is already grouped and cleaned by prediction.py - no additional processing needed
        print(f"Processing {len(data)} reading events...")
        
        return data  # Return the already clean data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_fixation_heatmap_data_from_csv(data):
    """Recreate get_fixation_heatmap_data() functionality from CSV"""
    heatmap_data = []
    for _, row in data.iterrows():
        x = row['Fixation_X_Screen']
        y = row['Fixation_Y_Screen'] 
        duration = row['Fixation_Duration']
        # Normalize intensity based on duration (like original function)
        intensity = min(duration / 100.0, 1.0)  # Normalize to 0-1
        heatmap_data.append((x, y, intensity))
    return heatmap_data

def get_reading_path_from_csv(data):
    """Recreate get_reading_path() functionality from CSV"""
    # Return path of all fixation points (since we have the full session)
    path = [(row['Fixation_X_Screen'], row['Fixation_Y_Screen']) 
            for _, row in data.iterrows()]
    return path

def get_analysis_summary_from_csv(data):
    """Recreate get_analysis_summary() functionality from CSV"""
    durations = data['Fixation_Duration'].values
    session_duration = data['Time_from_Stimulus_Onset'].max() / 1000.0  # Convert to seconds
    
    summary = {
        "session_duration": session_duration,
        "total_fixations": len(data),
        "average_fixation_duration": np.mean(durations) / 1000.0,  # Convert to seconds
        "reading_speed_wpm": 0,  # Would need word count to calculate
        "average_velocity": 0,   # Would need velocity data
        "current_velocity": 0,
        "movement_type": "Reading session complete",
        "lines_read_estimate": 4,  # Estimate based on text layout
        "fixation_rate": len(data) / session_duration if session_duration > 0 else 0
    }
    return summary

def create_spatial_heatmap(data):
    """Create spatial heatmap in separate window with larger fonts"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use data for visualization
    heatmap_data = get_fixation_heatmap_data_from_csv(data)
    if heatmap_data:
        x_coords = [point[0] for point in heatmap_data]
        y_coords = [point[1] for point in heatmap_data]
        intensities = [point[2] for point in heatmap_data]
        
        # Create smooth heatmap background
        x_orig = data['Fixation_X_Screen'].values
        y_orig = data['Fixation_Y_Screen'].values  
        durations_orig = data['Fixation_Duration'].values
        
        hist, xedges, yedges = np.histogram2d(x_orig, y_orig, bins=40, 
                                             weights=durations_orig,
                                             range=[[0, 1080], [0, 720]])
        
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        im = ax.imshow(hist.T, extent=extent, cmap='hot', interpolation='gaussian', alpha=0.6)
        
        # Overlay fixations
        scatter = ax.scatter(x_coords, y_coords, c=intensities, s=200, 
                             cmap='viridis', alpha=0.8, edgecolors='white', linewidth=2)
        
        # Add text regions
        text_regions = [(100, 200, 880, 50), (100, 280, 880, 50), 
                        (100, 360, 880, 50), (100, 440, 880, 50)]
        for x_rect, y_rect, width, height in text_regions:
            rect = Rectangle((x_rect, y_rect), width, height, 
                            linewidth=2, edgecolor='cyan', facecolor='none', alpha=0.4)
            ax.add_patch(rect)
        
        plt.colorbar(im, label='Heat Intensity (Duration)', ax=ax)
        plt.colorbar(scatter, label='Fixation Duration (ms)', ax=ax)
    
    ax.set_xlabel('Screen X Position (pixels)', fontsize=16)
    ax.set_ylabel('Screen Y Position (pixels)', fontsize=16)
    ax.set_title('Spatial Heatmap: Reading Fixation Clusters', 
                 fontsize=18, pad=20)
    ax.tick_params(labelsize=14)
    ax.set_xlim(0, 1080)
    ax.set_ylim(720, 0)
    
    plt.tight_layout()
    return fig

def create_reading_path(data):
    """Create reading path visualization in separate window"""
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Reading path
    x_path = data['Fixation_X_Screen'].values
    y_path = data['Fixation_Y_Screen'].values
    
    for i in range(len(x_path)-1):
        ax.arrow(x_path[i], y_path[i], 
                 x_path[i+1]-x_path[i], y_path[i+1]-y_path[i],
                 head_width=15, head_length=20, fc='blue', ec='blue', alpha=0.7)
    
    scatter = ax.scatter(x_path, y_path, c=range(len(x_path)), 
                        cmap='plasma', s=120, alpha=0.8, edgecolors='white', linewidth=1)
    plt.colorbar(scatter, ax=ax, label='Reading Sequence')
    ax.set_xlabel('X Position (pixels)', fontsize=16)
    ax.set_ylabel('Y Position (pixels)', fontsize=16)
    ax.set_title('Reading Path: Sequential Eye Movement Pattern', fontsize=18, pad=20) 
    ax.invert_yaxis()
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    return fig

def create_duration_histogram(data):
    """Create duration histogram in separate window"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    durations = data['Fixation_Duration'].values

    ax.hist(durations, bins=30, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1.5)
    ax.axvline(np.mean(durations), color='red', linestyle='--', linewidth=3, 
                label=f'Mean: {np.mean(durations):.1f}ms')
    ax.axvline(np.median(durations), color='orange', linestyle='--', linewidth=3,
                label=f'Median: {np.median(durations):.1f}ms')
    
    # Add reference lines for reading patterns
    ax.axvline(200, color='green', linestyle=':', alpha=0.7, linewidth=2, label='Quick reading threshold')
    ax.axvline(500, color='purple', linestyle=':', alpha=0.7, linewidth=2, label='Careful reading threshold')
    
    ax.set_xlabel('Fixation Duration (ms)', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_title('Fixation Duration Distribution', fontsize=22, pad=20)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=16)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Total Events: {len(durations)}
Mean: {np.mean(durations):.1f} ms
Median: {np.median(durations):.1f} ms
Std Dev: {np.std(durations):.1f} ms
Range: {durations.min():.0f} - {durations.max():.0f} ms

Reading Patterns:
Quick (<200ms): {np.sum(durations < 200)} ({100*np.sum(durations < 200)/len(durations):.1f}%)
Normal (200-500ms): {np.sum((durations >= 200) & (durations <= 500))} ({100*np.sum((durations >= 200) & (durations <= 500))/len(durations):.1f}%)
Careful (>500ms): {np.sum(durations > 500)} ({100*np.sum(durations > 500)/len(durations):.1f}%)"""
    
    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    return fig

def create_temporal_timeline(data):
    """Create temporal timeline in separate window"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    times = data['Time_from_Stimulus_Onset'].values
    durations = data['Fixation_Duration'].values
    
    # Plot with larger markers and lines
    ax.plot(times, durations, 'o-', color='green', alpha=0.8, markersize=8, linewidth=2)
    ax.axhline(np.mean(durations), color='red', linestyle='--', alpha=0.8, linewidth=3,
               label=f'Average: {np.mean(durations):.1f}ms')
    
    # Add trend line
    z = np.polyfit(times, durations, 1)
    p = np.poly1d(z)
    ax.plot(times, p(times), "r--", alpha=0.5, linewidth=2, label='Trend')
    
    ax.set_xlabel('Time from Session Start (ms)', fontsize=18)
    ax.set_ylabel('Fixation Duration (ms)', fontsize=18)
    ax.set_title('Temporal Pattern: Reading Duration Over Time', fontsize=22, pad=20)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    return fig

def create_hebrew_word_frequency(data):
    """Create Hebrew word frequency analysis in separate window"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Hebrew word frequency
    word_counts = data['Fixated_Word'].value_counts()
    hebrew_words = word_counts[word_counts.index != 'Unknown'].head(15)
    
    if len(hebrew_words) > 0:
        # Create horizontal bar chart for better Hebrew display
        bars = ax.barh(range(len(hebrew_words)), hebrew_words.values, color='lightblue', alpha=0.8)
        ax.set_yticks(range(len(hebrew_words)))
        
        # Fix Hebrew text display to prevent mirroring
        hebrew_labels = []
        for word in hebrew_words.index:
            formatted_word = fix_hebrew_display(word)
            hebrew_labels.append(formatted_word)
        
        ax.set_yticklabels(hebrew_labels, fontsize=14)
        ax.set_xlabel('Number of Reading Events', fontsize=18)
        ax.set_title('Most Frequently Read Hebrew Words', fontsize=22, pad=20)
        ax.tick_params(labelsize=16)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Hebrew words found in data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=18)
        ax.set_title('Hebrew Word Analysis', fontsize=22)
    
    plt.tight_layout()
    return fig

def create_word_duration_analysis(data):
    """Create word duration analysis in separate window"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Word duration analysis  
    word_durations = data.groupby('Fixated_Word')['Fixation_Duration'].mean()
    word_durations = word_durations[word_durations.index != 'Unknown'].head(12)
    
    if len(word_durations) > 0:
        bars = ax.barh(range(len(word_durations)), word_durations.values, color='lightgreen', alpha=0.8)
        ax.set_yticks(range(len(word_durations)))
        
        # Fix Hebrew text display for duration chart
        duration_labels = []
        for word in word_durations.index:
            formatted_word = fix_hebrew_display(word)
            duration_labels.append(formatted_word)
        
        ax.set_yticklabels(duration_labels, fontsize=14)
        ax.set_xlabel('Average Reading Duration (ms)', fontsize=18)
        ax.set_title('Average Reading Duration by Hebrew Word', fontsize=22, pad=20)
        ax.tick_params(labelsize=16)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 10, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}ms', ha='left', va='center', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No word duration data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=18)
        ax.set_title('Word Duration Analysis', fontsize=22)
    
    plt.tight_layout()
    return fig

def create_summary_statistics(data):
    """Create comprehensive summary statistics in separate window"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    # Calculate statistics
    durations = data['Fixation_Duration'].values
    session_duration = data['Time_from_Stimulus_Onset'].max() / 1000.0
    unique_words = len(data[data['Fixated_Word'] != 'Unknown'])
    total_reading_time = np.sum(durations) / 1000.0  # Convert to seconds
    
    stats_text = f"""
    📊 READING SESSION ANALYSIS SUMMARY 📊
    
    ═══════════════════════════════════════════════════════════════
    
    📋 SESSION OVERVIEW:
    • Total Session Duration: {session_duration:.1f} seconds
    • Total Reading Time: {total_reading_time:.1f} seconds ({100*total_reading_time/session_duration:.1f}% of session)
    • Reading Events Analyzed: {len(data)}
    • Start Time: {data['Time_from_Stimulus_Onset'].min()/1000:.1f}s
    • End Time: {data['Time_from_Stimulus_Onset'].max()/1000:.1f}s
    
    ═══════════════════════════════════════════════════════════════
    
    ⏱️ DURATION STATISTICS:
    • Mean Duration: {np.mean(durations):.1f} ms
    • Median Duration: {np.median(durations):.1f} ms
    • Standard Deviation: {np.std(durations):.1f} ms
    • Duration Range: {durations.min():.0f} - {durations.max():.0f} ms
    
    ═══════════════════════════════════════════════════════════════
    
    📖 READING PATTERNS:
    
    Duration Categories:
    • Quick reading (<200ms): {np.sum(durations < 200)} events ({100*np.sum(durations < 200)/len(durations):.1f}%)
    • Normal reading (200-500ms): {np.sum((durations >= 200) & (durations <= 500))} events ({100*np.sum((durations >= 200) & (durations <= 500))/len(durations):.1f}%)
    • Careful reading (>500ms): {np.sum(durations > 500)} events ({100*np.sum(durations > 500)/len(durations):.1f}%)
    
    ═══════════════════════════════════════════════════════════════
    
    🔤 WORD ANALYSIS:
    • Recognized Hebrew Words: {unique_words}
    • Unknown Positions: {(data['Fixated_Word'] == 'Unknown').sum()}
    
    ═══════════════════════════════════════════════════════════════
    
    📈 READING EFFICIENCY:
    • Estimated Reading Speed: {60*unique_words/total_reading_time:.1f} words/minute
    • Average Time per Word: {total_reading_time*1000/unique_words:.0f} ms/word
    • Reading Event Rate: {len(data)/session_duration:.1f} events/second
    • Word Processing Rate: {unique_words/session_duration:.1f} words/second
    
    ═══════════════════════════════════════════════════════════════
    
    📊 PERFORMANCE INDICATORS:
    • Reading Fluency: {'High' if 60*unique_words/total_reading_time > 200 else 'Moderate' if 60*unique_words/total_reading_time > 150 else 'Careful'}
    • Comprehension Style: {'Quick scan' if np.mean(durations) < 250 else 'Thorough reading' if np.mean(durations) > 400 else 'Balanced reading'}
    • Text Engagement: {'High' if total_reading_time/session_duration > 0.8 else 'Moderate' if total_reading_time/session_duration > 0.6 else 'Selective'}
    
    ═══════════════════════════════════════════════════════════════
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    return fig

def find_latest_csv_file():
    """Find the most recent eye_movement_data CSV file"""
    import glob
    import os
    from datetime import datetime
    
    # Base path to user data
    base_path = r"c:\Users\emilo\OneDrive - Holon Institute of Technology\מעבדה לעיבוד תמונה\gaze_tracking_system-1\user_data"
    
    # Search pattern for CSV files
    search_pattern = os.path.join(base_path, "*", "analysis", "eye_movement_data_*.csv")
    
    # Find all matching CSV files
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("No eye_movement_data CSV files found!")
        return None
    
    # Extract timestamp from filename and find the most recent
    latest_file = None
    latest_timestamp = None
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Extract timestamp from filename like "eye_movement_data_20251024_180351.csv"
        try:
            timestamp_str = filename.replace("eye_movement_data_", "").replace(".csv", "")
            # Parse timestamp: YYYYMMDD_HHMMSS
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_file = file_path
        except ValueError:
            # Skip files that don't match the expected timestamp format
            continue
    
    if latest_file:
        print(f"Found latest CSV file: {os.path.basename(latest_file)}")
        print(f"Timestamp: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        return latest_file
    else:
        print("No valid CSV files with timestamp found!")
        return None

def main():
    """Main function to create all visualizations in separate windows with Hebrew support"""
    # Automatically find the most recent CSV file
    csv_path = find_latest_csv_file()
    if csv_path is None:
        print("❌ No CSV file found. Please run an eye tracking session first.")
        return
    
    # Load data (grouping happens silently in background)
    grouped_data = load_reading_data(csv_path)
    if grouped_data is None:
        return
    
    # Create output directory
    output_dir = "reading_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating reading analysis visualizations...")
    
    # Enable interactive mode for separate windows
    plt.ion()
    
    # 1. Spatial Heatmap
    print("\n1. Creating spatial heatmap...")
    fig1 = create_spatial_heatmap(grouped_data)
    fig1.savefig(f"{output_dir}/01_spatial_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()
    input("Press Enter to continue to next visualization...")
    
    # 2. Reading Path Analysis
    print("\n2. Creating reading path analysis...")
    fig2 = create_reading_path(grouped_data)
    fig2.savefig(f"{output_dir}/02_reading_path.png", dpi=300, bbox_inches='tight')
    plt.show()
    input("Press Enter to continue to next visualization...")
    
    # 3. Duration Analysis - Histogram
    print("\n3. Creating duration histogram...")
    fig3 = create_duration_histogram(grouped_data)
    fig3.savefig(f"{output_dir}/03_duration_histogram.png", dpi=300, bbox_inches='tight')
    plt.show()
    input("Press Enter to continue to next visualization...")
    
    # 4. Duration Analysis - Timeline
    print("\n4. Creating temporal timeline...")
    fig4 = create_temporal_timeline(grouped_data)
    fig4.savefig(f"{output_dir}/04_temporal_timeline.png", dpi=300, bbox_inches='tight')
    plt.show()
    input("Press Enter to continue to next visualization...")
    
    # 5. Hebrew Word Frequency
    print("\n5. Creating Hebrew word frequency analysis...")
    fig5 = create_hebrew_word_frequency(grouped_data)
    fig5.savefig(f"{output_dir}/05_hebrew_word_frequency.png", dpi=300, bbox_inches='tight')
    plt.show()
    input("Press Enter to continue to next visualization...")
    
    # 6. Word Duration Analysis
    print("\n6. Creating word duration analysis...")
    fig6 = create_word_duration_analysis(grouped_data)
    fig6.savefig(f"{output_dir}/06_word_duration_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    input("Press Enter to continue to next visualization...")
    
    # 7. Summary Statistics
    print("\n7. Creating summary statistics...")
    fig7 = create_summary_statistics(grouped_data)
    fig7.savefig(f"{output_dir}/07_summary_stats.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Disable interactive mode
    plt.ioff()
    
    print(f"\n✅ All visualizations completed!")
    print(f"📁 Files saved to '{output_dir}' directory:")
    print(f"   01_spatial_heatmap.png")
    print(f"   02_reading_path.png") 
    print(f"   03_duration_histogram.png")
    print(f"   04_temporal_timeline.png")
    print(f"   05_hebrew_word_frequency.png")
    print(f"   06_word_duration_analysis.png")
    print(f"   07_summary_stats.png")
    
    print(f"\n📊 Analysis Summary:")
    print(f"   • Reading events analyzed: {len(grouped_data)}")
    print(f"   • Hebrew words recognized: {(grouped_data['Fixated_Word'] != 'Unknown').sum()}")
    print(f"   • Session duration: {grouped_data['Time_from_Stimulus_Onset'].max()/1000:.1f} seconds")

if __name__ == "__main__":
    main()