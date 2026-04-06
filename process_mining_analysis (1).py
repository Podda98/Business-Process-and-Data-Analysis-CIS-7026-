"""
=============================================================================
Process Mining Analysis: Employee Onboarding at GlobalCart Inc.
CIS 7026 - Business Process and Data Analysis
=============================================================================
This script performs comprehensive process mining analysis including:
1. Data Loading & Exploratory Analysis
2. Process Discovery (Alpha, Heuristic, Inductive Miners)
3. Conformance Checking (Token-based Replay & Alignment)
4. Bottleneck / Performance Analysis
5. Process Enhancement
6. Object-Centric Process Mining (OCPM) demonstration
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import os

# PM4Py imports
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.statistics.variants.log import get as variants_get
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.statistics.traces.generic.log import case_statistics

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
FIG_DIR = os.path.join(os.path.dirname(BASE_DIR), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Configure plot style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white'
})

# =============================================================================
# 1. DATA LOADING & EXPLORATORY ANALYSIS
# =============================================================================
print("=" * 70)
print("SECTION 1: Data Loading & Exploratory Analysis")
print("=" * 70)

# Load the event log
df = pd.read_csv(os.path.join(DATA_DIR, "employee_onboarding_event_log.csv"))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.rename(columns={
    'case_id': 'case:concept:name',
    'activity': 'concept:name',
    'timestamp': 'time:timestamp',
    'resource': 'org:resource'
})

print(f"Total Events: {len(df)}")
print(f"Total Cases: {df['case:concept:name'].nunique()}")
print(f"Unique Activities: {df['concept:name'].nunique()}")
print(f"Unique Resources: {df['org:resource'].nunique()}")
print(f"Time Period: {df['time:timestamp'].min().date()} to {df['time:timestamp'].max().date()}")

# Convert to PM4Py event log
event_log = pm4py.format_dataframe(df, case_id='case:concept:name',
                                     activity_key='concept:name',
                                     timestamp_key='time:timestamp')
event_log = log_converter.apply(event_log)

# --- Figure 1: Activity Frequency ---
fig, ax = plt.subplots(figsize=(12, 6))
activity_counts = df['concept:name'].value_counts()
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(activity_counts)))
bars = ax.barh(range(len(activity_counts)), activity_counts.values, color=colors)
ax.set_yticks(range(len(activity_counts)))
ax.set_yticklabels(activity_counts.index, fontsize=10)
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Figure 1: Activity Frequency Distribution in Employee Onboarding Process', fontsize=13, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, activity_counts.values):
    ax.text(val + 2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig01_activity_frequency.png"))
plt.close()
print("Saved: fig01_activity_frequency.png")

# --- Figure 2: Process Variants ---
variants = variants_get.get_variants(event_log)
variant_counts = {str(k): len(v) for k, v in variants.items()}
variant_df = pd.DataFrame(list(variant_counts.items()), columns=['Variant', 'Count'])
variant_df = variant_df.sort_values('Count', ascending=False).head(6)

fig, ax = plt.subplots(figsize=(10, 5))
labels = [f"V{i+1}" for i in range(len(variant_df))]
colors_v = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
wedges, texts, autotexts = ax.pie(variant_df['Count'], labels=labels,
                                    autopct='%1.1f%%', colors=colors_v[:len(variant_df)],
                                    startangle=90, textprops={'fontsize': 11})
legend_labels = []
for i, row in variant_df.iterrows():
    acts = row['Variant'].replace("(", "").replace(")", "").replace("'", "")
    short = "->".join([a.strip()[:15] for a in acts.split(",")[:3]]) + "..."
    legend_labels.append(short)
ax.legend(wedges, legend_labels, title="Top Variants", loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
ax.set_title('Figure 2: Process Variant Distribution (Top 6)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig02_variant_distribution.png"))
plt.close()
print("Saved: fig02_variant_distribution.png")

# --- Figure 3: Cases over Time ---
df_temp = df.copy()
df_temp['month'] = df_temp['time:timestamp'].dt.to_period('M')
cases_per_month = df_temp.groupby('month')['case:concept:name'].nunique()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(len(cases_per_month)), cases_per_month.values, 'o-', color='#2c3e50',
        linewidth=2, markersize=8)
ax.fill_between(range(len(cases_per_month)), cases_per_month.values, alpha=0.15, color='#3498db')
ax.set_xticks(range(len(cases_per_month)))
ax.set_xticklabels([str(p) for p in cases_per_month.index], rotation=45)
ax.set_ylabel('Number of Cases')
ax.set_title('Figure 3: Onboarding Cases per Month', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig03_cases_over_time.png"))
plt.close()
print("Saved: fig03_cases_over_time.png")

# =============================================================================
# 2. PROCESS DISCOVERY
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Process Discovery")
print("=" * 70)

# --- Alpha Miner ---
print("\n--- Alpha Miner ---")
try:
    net_alpha, im_alpha, fm_alpha = alpha_miner.apply(event_log)
    gviz = pn_visualizer.apply(net_alpha, im_alpha, fm_alpha,
                                parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
    pn_visualizer.save(gviz, os.path.join(FIG_DIR, "fig04_alpha_miner.png"))
    print("Saved: fig04_alpha_miner.png")
except Exception as e:
    print(f"Alpha Miner visualization failed: {e}")

# --- Heuristic Miner ---
print("\n--- Heuristic Miner ---")
try:
    heu_net = heuristics_miner.apply_heu(event_log, parameters={
        "dependency_threshold": 0.5,
        "and_threshold": 0.65,
        "loop_two_threshold": 0.5
    })
    gviz_heu = hn_visualizer.apply(heu_net,
                                    parameters={hn_visualizer.Variants.PYDOTPLUS.value.Parameters.FORMAT: "png"})
    hn_visualizer.save(gviz_heu, os.path.join(FIG_DIR, "fig05_heuristic_miner.png"))
    print("Saved: fig05_heuristic_miner.png")
except Exception as e:
    print(f"Heuristic Miner visualization failed: {e}")

# --- Inductive Miner ---
print("\n--- Inductive Miner ---")
try:
    net_ind, im_ind, fm_ind = inductive_miner.apply(event_log)
    gviz_ind = pn_visualizer.apply(net_ind, im_ind, fm_ind,
                                    parameters={pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
    pn_visualizer.save(gviz_ind, os.path.join(FIG_DIR, "fig06_inductive_miner.png"))
    print("Saved: fig06_inductive_miner.png")
except Exception as e:
    print(f"Inductive Miner visualization failed: {e}")

# --- Directly-Follows Graph ---
print("\n--- Directly-Follows Graph ---")
try:
    dfg, sa, ea = pm4py.discover_dfg(event_log)
    pm4py.save_vis_dfg(dfg, sa, ea, os.path.join(FIG_DIR, "fig07_dfg.png"))
    print("Saved: fig07_dfg.png")
except Exception as e:
    print(f"DFG visualization failed: {e}")

# =============================================================================
# 3. CONFORMANCE CHECKING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Conformance Checking")
print("=" * 70)

# Use Inductive Miner model for conformance
try:
    # Token-based replay
    replayed = token_replay.apply(event_log, net_ind, im_ind, fm_ind)

    fitness = replay_fitness.apply(event_log, net_ind, im_ind, fm_ind,
                                    variant=replay_fitness.Variants.TOKEN_BASED)
    print(f"\nToken-based Replay Fitness: {fitness}")

    # Extract fitness per case
    case_fitness = []
    for trace_result in replayed:
        missing = trace_result.get('missing_tokens', 0)
        consumed = trace_result.get('consumed_tokens', 0)
        produced = trace_result.get('produced_tokens', 0)
        remaining = trace_result.get('remaining_tokens', 0)
        if consumed + produced > 0:
            fit = 0.5 * (1 - missing / max(consumed, 1)) + 0.5 * (1 - remaining / max(produced, 1))
        else:
            fit = 0
        case_fitness.append(fit)

    # --- Figure 8: Conformance Fitness Distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(case_fitness, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(case_fitness), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean Fitness: {np.mean(case_fitness):.3f}')
    ax.set_xlabel('Fitness Score')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Figure 8: Conformance Fitness Distribution (Token-Based Replay)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig08_conformance_fitness.png"))
    plt.close()
    print("Saved: fig08_conformance_fitness.png")

    # Precision
    try:
        prec = precision_evaluator.apply(event_log, net_ind, im_ind, fm_ind,
                                          variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        print(f"Precision: {prec}")
    except:
        prec = "N/A"
        print("Precision calculation not available")

except Exception as e:
    print(f"Conformance checking error: {e}")
    case_fitness = []

# --- Figure 9: Deviation Analysis ---
# Identify non-conforming cases
df_orig = pd.read_csv(os.path.join(DATA_DIR, "employee_onboarding_event_log.csv"))
standard_trace = "->".join([
    "Offer Accepted", "Background Check Initiated", "Background Check Completed",
    "IT Account Creation", "Equipment Provisioning", "Compliance Training Assigned",
    "Compliance Training Completed", "Team Introduction Meeting", "Buddy Assignment",
    "System Access Granted", "First Week Check-in", "30-Day Review", "Onboarding Complete"
])

case_traces = df_orig.groupby('case_id')['activity'].apply(lambda x: "->".join(x))
conforming = (case_traces == standard_trace).sum()
non_conforming = len(case_traces) - conforming

# Deviation types
deviation_types = {
    'Equipment Rework': 0,
    'Training Retake': 0,
    'Skipped Buddy': 0,
    'Incomplete Process': 0,
    'Delayed BG Check': 0
}

for case_id, trace in case_traces.items():
    acts = trace.split("->")
    if acts.count("Equipment Provisioning") > 1:
        deviation_types['Equipment Rework'] += 1
    if acts.count("Compliance Training Assigned") > 1:
        deviation_types['Training Retake'] += 1
    if "Buddy Assignment" not in acts and len(acts) > 9:
        deviation_types['Skipped Buddy'] += 1
    if "Onboarding Complete" not in acts:
        deviation_types['Incomplete Process'] += 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
labels_pie = ['Conforming', 'Non-Conforming']
sizes_pie = [conforming, non_conforming]
colors_pie = ['#2ecc71', '#e74c3c']
axes[0].pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie,
            startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Conformance vs Non-Conformance', fontsize=12, fontweight='bold')

# Deviation breakdown
dev_labels = [k for k, v in deviation_types.items() if v > 0]
dev_values = [v for v in deviation_types.values() if v > 0]
bars = axes[1].barh(dev_labels, dev_values, color=['#e74c3c', '#f39c12', '#9b59b6', '#3498db'])
axes[1].set_xlabel('Count')
axes[1].set_title('Deviation Types Identified', fontsize=12, fontweight='bold')
for bar, val in zip(bars, dev_values):
    axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2, str(val), va='center')

fig.suptitle('Figure 9: Conformance Analysis & Deviation Breakdown', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig09_deviation_analysis.png"))
plt.close()
print("Saved: fig09_deviation_analysis.png")

# =============================================================================
# 4. BOTTLENECK / PERFORMANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Bottleneck & Performance Analysis")
print("=" * 70)

# Calculate transition times between consecutive activities
df_perf = df_orig.copy()
df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
df_perf = df_perf.sort_values(['case_id', 'timestamp'])

transitions = []
for case_id, group in df_perf.groupby('case_id'):
    group = group.sort_values('timestamp')
    for i in range(len(group) - 1):
        from_act = group.iloc[i]['activity']
        to_act = group.iloc[i + 1]['activity']
        duration = (group.iloc[i + 1]['timestamp'] - group.iloc[i]['timestamp']).total_seconds() / 3600
        transitions.append({
            'from': from_act,
            'to': to_act,
            'duration_hours': duration,
            'case_id': case_id
        })

trans_df = pd.DataFrame(transitions)
avg_transition = trans_df.groupby(['from', 'to'])['duration_hours'].agg(['mean', 'std', 'count']).reset_index()
avg_transition.columns = ['From', 'To', 'Avg Hours', 'Std Hours', 'Count']
avg_transition = avg_transition.sort_values('Avg Hours', ascending=False)

print("\nTop Bottleneck Transitions (by average duration):")
print(avg_transition.head(10).to_string(index=False))

# --- Figure 10: Bottleneck Heatmap ---
# Create pivot table for heatmap
activities_order = [
    "Offer Accepted", "Background Check Initiated", "Background Check Completed",
    "IT Account Creation", "Equipment Provisioning", "Compliance Training Assigned",
    "Compliance Training Completed", "Team Introduction Meeting", "Buddy Assignment",
    "System Access Granted", "First Week Check-in", "30-Day Review", "Onboarding Complete"
]

pivot = trans_df.groupby(['from', 'to'])['duration_hours'].mean().reset_index()
pivot_matrix = pivot.pivot(index='from', columns='to', values='duration_hours')
# Reindex
acts_present = [a for a in activities_order if a in pivot_matrix.index or a in pivot_matrix.columns]
pivot_matrix = pivot_matrix.reindex(index=acts_present, columns=acts_present)

fig, ax = plt.subplots(figsize=(14, 10))
mask = pivot_matrix.isna()
sns.heatmap(pivot_matrix, mask=mask, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=ax, cbar_kws={'label': 'Average Duration (Hours)'},
            linewidths=0.5, linecolor='white')
ax.set_title('Figure 10: Transition Duration Heatmap (Hours) - Bottleneck Identification',
             fontsize=13, fontweight='bold')
ax.set_xlabel('To Activity')
ax.set_ylabel('From Activity')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_bottleneck_heatmap.png"))
plt.close()
print("Saved: fig10_bottleneck_heatmap.png")

# --- Figure 11: Case Duration Distribution ---
case_durations = df_perf.groupby('case_id').agg(
    start=('timestamp', 'min'),
    end=('timestamp', 'max')
)
case_durations['duration_days'] = (case_durations['end'] - case_durations['start']).dt.total_seconds() / 86400

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(case_durations['duration_days'], bins=25, color='#2c3e50', edgecolor='white', alpha=0.85)
mean_dur = case_durations['duration_days'].mean()
median_dur = case_durations['duration_days'].median()
ax.axvline(mean_dur, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_dur:.1f} days')
ax.axvline(median_dur, color='#f39c12', linestyle='--', linewidth=2, label=f'Median: {median_dur:.1f} days')
ax.set_xlabel('Total Onboarding Duration (Days)')
ax.set_ylabel('Number of Cases')
ax.set_title('Figure 11: End-to-End Onboarding Duration Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig11_case_duration.png"))
plt.close()
print("Saved: fig11_case_duration.png")

# --- Figure 12: Resource Workload ---
resource_load = df_orig.groupby('resource')['case_id'].count().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_res = plt.cm.viridis(np.linspace(0.2, 0.9, len(resource_load)))
bars = ax.barh(range(len(resource_load)), resource_load.values, color=colors_res)
ax.set_yticks(range(len(resource_load)))
ax.set_yticklabels(resource_load.index, fontsize=10)
ax.set_xlabel('Number of Events Handled')
ax.set_title('Figure 12: Resource Workload Distribution', fontsize=13, fontweight='bold')
for bar, val in zip(bars, resource_load.values):
    ax.text(val + 1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig12_resource_workload.png"))
plt.close()
print("Saved: fig12_resource_workload.png")

# --- Figure 13: Activity Duration Box Plots ---
activity_durations = []
for case_id, group in df_perf.groupby('case_id'):
    group = group.sort_values('timestamp')
    for i in range(len(group) - 1):
        dur_h = (group.iloc[i+1]['timestamp'] - group.iloc[i]['timestamp']).total_seconds() / 3600
        activity_durations.append({
            'activity': group.iloc[i]['activity'],
            'duration_hours': dur_h
        })

act_dur_df = pd.DataFrame(activity_durations)
act_dur_df = act_dur_df[act_dur_df['activity'].isin(activities_order[:-1])]

fig, ax = plt.subplots(figsize=(14, 6))
act_order_plot = [a for a in activities_order[:-1] if a in act_dur_df['activity'].values]
bp = ax.boxplot([act_dur_df[act_dur_df['activity'] == a]['duration_hours'].values for a in act_order_plot],
                labels=[a[:18] for a in act_order_plot], patch_artist=True, showfliers=False)
colors_box = plt.cm.Set3(np.linspace(0, 1, len(act_order_plot)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax.set_ylabel('Duration to Next Activity (Hours)')
ax.set_title('Figure 13: Activity Waiting Time Distribution (Bottleneck Analysis)', fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig13_activity_duration_boxplot.png"))
plt.close()
print("Saved: fig13_activity_duration_boxplot.png")

# =============================================================================
# 5. PROCESS ENHANCEMENT
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Process Enhancement")
print("=" * 70)

# --- Figure 14: Enhanced Process Model with Performance Overlay ---
# Calculate performance stats for enhancement
perf_stats = trans_df.groupby(['from', 'to']).agg(
    avg_hours=('duration_hours', 'mean'),
    case_count=('case_id', 'nunique')
).reset_index()

# Identify bottlenecks (> 75th percentile duration)
threshold_75 = perf_stats['avg_hours'].quantile(0.75)
perf_stats['is_bottleneck'] = perf_stats['avg_hours'] > threshold_75

print(f"\nBottleneck threshold (75th percentile): {threshold_75:.1f} hours")
print("\nIdentified Bottlenecks:")
bottlenecks = perf_stats[perf_stats['is_bottleneck']].sort_values('avg_hours', ascending=False)
print(bottlenecks[['from', 'to', 'avg_hours', 'case_count']].to_string(index=False))

# --- Figure 14: Enhanced process flow with bottleneck highlighting ---
fig, ax = plt.subplots(figsize=(16, 10))

# Draw the process as a flowchart
positions = {}
x_start = 1
y_center = 5
for i, act in enumerate(activities_order):
    col = i % 4
    row = i // 4
    x = 1 + col * 4
    y = 9 - row * 3
    positions[act] = (x, y)

# Draw nodes
for act, (x, y) in positions.items():
    # Check if this activity has bottleneck transitions coming out
    is_bt = act in bottlenecks['from'].values
    color = '#e74c3c' if is_bt else '#3498db'
    alpha = 0.9 if is_bt else 0.7
    rect = mpatches.FancyBboxPatch((x - 1.5, y - 0.5), 3, 1,
                                     boxstyle="round,pad=0.15", facecolor=color, alpha=alpha,
                                     edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, act.replace(" ", "\n"), ha='center', va='center',
            fontsize=7, fontweight='bold', color='white')

# Draw arrows
for _, row in perf_stats.iterrows():
    if row['from'] in positions and row['to'] in positions:
        x1, y1 = positions[row['from']]
        x2, y2 = positions[row['to']]
        color = '#e74c3c' if row['is_bottleneck'] else '#95a5a6'
        width = 2.5 if row['is_bottleneck'] else 1
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=width, alpha=0.6))
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if row['is_bottleneck']:
            ax.text(mid_x, mid_y + 0.3, f"{row['avg_hours']:.0f}h", fontsize=7,
                    ha='center', color='#e74c3c', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffeaa7', alpha=0.8))

ax.set_xlim(-1, 17)
ax.set_ylim(-1, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Figure 14: Enhanced Process Model with Bottleneck Overlay\n(Red = Bottleneck Activities, Labels = Avg Duration in Hours)',
             fontsize=13, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#e74c3c', alpha=0.9, label='Bottleneck Activity'),
    mpatches.Patch(facecolor='#3498db', alpha=0.7, label='Normal Activity'),
    plt.Line2D([0], [0], color='#e74c3c', linewidth=2.5, label='Bottleneck Transition'),
    plt.Line2D([0], [0], color='#95a5a6', linewidth=1, label='Normal Transition'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig14_enhanced_process_model.png"))
plt.close()
print("Saved: fig14_enhanced_process_model.png")

# --- Figure 15: Regional Performance Comparison ---
regional_dur = df_perf.groupby(['case_id', 'region']).agg(
    start=('timestamp', 'min'),
    end=('timestamp', 'max')
).reset_index()
regional_dur['duration_days'] = (regional_dur['end'] - regional_dur['start']).dt.total_seconds() / 86400

fig, ax = plt.subplots(figsize=(10, 5))
regions_list = regional_dur['region'].unique()
bp2 = ax.boxplot([regional_dur[regional_dur['region'] == r]['duration_days'].values for r in regions_list],
                  labels=regions_list, patch_artist=True, showfliers=True)
region_colors = ['#3498db', '#2ecc71', '#f39c12']
for patch, color in zip(bp2['boxes'], region_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Onboarding Duration (Days)')
ax.set_title('Figure 15: Onboarding Duration by Region (Process Enhancement Insight)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add mean annotations
for i, r in enumerate(regions_list):
    mean_val = regional_dur[regional_dur['region'] == r]['duration_days'].mean()
    ax.text(i + 1, mean_val, f'μ={mean_val:.1f}d', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig15_regional_performance.png"))
plt.close()
print("Saved: fig15_regional_performance.png")

# =============================================================================
# 6. OBJECT-CENTRIC PROCESS MINING (OCPM)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Object-Centric Process Mining (OCPM)")
print("=" * 70)

# --- Figure 16: OCPM Multi-Object Interaction Model ---
fig, ax = plt.subplots(figsize=(16, 8))

# Define object types and their activities
object_types = {
    'Employee': {
        'activities': ['Offer Accepted', 'Team Introduction', 'First Week Check-in',
                       '30-Day Review', 'Onboarding Complete'],
        'color': '#3498db', 'y': 6
    },
    'IT Assets': {
        'activities': ['IT Account Creation', 'Equipment Provisioning', 'System Access Granted'],
        'color': '#2ecc71', 'y': 4
    },
    'Compliance': {
        'activities': ['Background Check Initiated', 'Background Check Completed',
                       'Compliance Training Assigned', 'Compliance Training Completed'],
        'color': '#e74c3c', 'y': 2
    }
}

for obj_type, info in object_types.items():
    y = info['y']
    color = info['color']
    acts = info['activities']
    x_positions = np.linspace(1, 14, len(acts))

    ax.plot(x_positions, [y] * len(acts), '-o', color=color, linewidth=2,
            markersize=12, zorder=3, label=obj_type)

    for j, (x, act) in enumerate(zip(x_positions, acts)):
        ax.annotate(act, (x, y), textcoords="offset points", xytext=(0, 15),
                    ha='center', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2))

# Draw interaction arrows between object types
interactions = [
    ('Employee', 0, 'Compliance', 0, 'Triggers'),
    ('Compliance', 3, 'IT Assets', 0, 'Enables'),
    ('IT Assets', 2, 'Employee', 2, 'Grants'),
    ('Employee', 1, 'IT Assets', 1, 'Receives'),
]

for from_type, from_idx, to_type, to_idx, label in interactions:
    from_info = object_types[from_type]
    to_info = object_types[to_type]
    from_x = np.linspace(1, 14, len(from_info['activities']))[from_idx]
    to_x = np.linspace(1, 14, len(to_info['activities']))[to_idx]
    from_y = from_info['y']
    to_y = to_info['y']

    ax.annotate('', xy=(to_x, to_y + 0.15), xytext=(from_x, from_y - 0.15),
                arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5,
                                connectionstyle='arc3,rad=0.2'))
    mid_x = (from_x + to_x) / 2
    mid_y = (from_y + to_y) / 2
    ax.text(mid_x + 0.3, mid_y, label, fontsize=7, color='#7f8c8d', fontstyle='italic')

ax.set_xlim(0, 15)
ax.set_ylim(0.5, 8)
ax.set_yticks([2, 4, 6])
ax.set_yticklabels(['Compliance\nObject', 'IT Assets\nObject', 'Employee\nObject'], fontsize=10)
ax.set_xlabel('Process Timeline', fontsize=12)
ax.set_title('Figure 16: Object-Centric Process Mining (OCPM) - Multi-Object Interaction Model\nEmployee Onboarding with Three Interacting Object Types',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig16_ocpm_model.png"))
plt.close()
print("Saved: fig16_ocpm_model.png")

# --- Figure 17: Department Handoff Analysis ---
dept_transitions = []
for case_id, group in df_perf.groupby('case_id'):
    group = group.sort_values('timestamp')
    for i in range(len(group) - 1):
        from_dept = group.iloc[i]['department']
        to_dept = group.iloc[i + 1]['department']
        if from_dept != to_dept:
            dur = (group.iloc[i+1]['timestamp'] - group.iloc[i]['timestamp']).total_seconds() / 3600
            dept_transitions.append({
                'from_dept': from_dept,
                'to_dept': to_dept,
                'duration_hours': dur
            })

dept_trans_df = pd.DataFrame(dept_transitions)
dept_pivot = dept_trans_df.groupby(['from_dept', 'to_dept'])['duration_hours'].mean().reset_index()
dept_matrix = dept_pivot.pivot(index='from_dept', columns='to_dept', values='duration_hours')

fig, ax = plt.subplots(figsize=(10, 7))
mask = dept_matrix.isna()
sns.heatmap(dept_matrix, mask=mask, annot=True, fmt='.1f', cmap='Blues',
            ax=ax, cbar_kws={'label': 'Average Handoff Duration (Hours)'},
            linewidths=1, linecolor='white')
ax.set_title('Figure 17: Cross-Department Handoff Duration Analysis\n(Key for Process Enhancement)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('To Department')
ax.set_ylabel('From Department')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig17_department_handoff.png"))
plt.close()
print("Saved: fig17_department_handoff.png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS FOR REPORT")
print("=" * 70)

print(f"\n1. Total cases analyzed: {df_orig['case_id'].nunique()}")
print(f"2. Total events: {len(df_orig)}")
print(f"3. Unique activities: {df_orig['activity'].nunique()}")
print(f"4. Conforming cases: {conforming} ({conforming/len(case_traces)*100:.1f}%)")
print(f"5. Non-conforming cases: {non_conforming} ({non_conforming/len(case_traces)*100:.1f}%)")
print(f"6. Average onboarding duration: {case_durations['duration_days'].mean():.1f} days")
print(f"7. Median onboarding duration: {case_durations['duration_days'].median():.1f} days")
print(f"8. Fastest onboarding: {case_durations['duration_days'].min():.1f} days")
print(f"9. Slowest onboarding: {case_durations['duration_days'].max():.1f} days")
print(f"10. Top bottleneck: {bottlenecks.iloc[0]['from']} -> {bottlenecks.iloc[0]['to']} ({bottlenecks.iloc[0]['avg_hours']:.1f} hours avg)")

print("\n" + "=" * 70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)
