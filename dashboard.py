import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ── Colour palette ────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
ACCENT1  = "#4f8ef7"   # blue
ACCENT2  = "#f7934f"   # orange
ACCENT3  = "#4fc97f"   # green
ACCENT4  = "#f74f7e"   # red/pink
TEXT     = "#e8eaf6"
SUBTEXT  = "#8b90a8"
GRID_C   = "#2a2d3e"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID_C,
    "axes.labelcolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "axes.titlecolor":   TEXT,
    "grid.color":        GRID_C,
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── 1. Load & prepare data ────────────────────────────────────────────────────
print("Loading data and training model...")
df_raw = pd.read_csv('waste bin data.csv')
df = df_raw.copy()

df['avg_daily_waste_kg']    = df['avg_daily_waste_kg'].fillna(df['avg_daily_waste_kg'].median())
df['days_since_last_pickup'] = df['days_since_last_pickup'].fillna(df['days_since_last_pickup'].median())
df['estimated_current_waste_kg'] = df['avg_daily_waste_kg'] * df['days_since_last_pickup']
df['fill_ratio_estimate']   = df['estimated_current_waste_kg'] / df['bin_capacity_kg']
df['adjusted_fill_ratio']   = np.where(df['festival_week'] == 1,
                                        df['fill_ratio_estimate'] * 1.2,
                                        df['fill_ratio_estimate'])
df = df.drop('bin_id', axis=1)

X = df.drop('is_full', axis=1)
y = df['is_full']

numerical_cols   = ['avg_daily_waste_kg', 'days_since_last_pickup', 'festival_week',
                     'bin_capacity_kg', 'estimated_current_waste_kg',
                     'fill_ratio_estimate', 'adjusted_fill_ratio']
categorical_cols = ['location_type', 'weather']

num_tf = Pipeline([('imp', SimpleImputer(strategy='median')),
                   ('sc',  StandardScaler())])
cat_tf = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([('num', num_tf, numerical_cols),
                                  ('cat', cat_tf, categorical_cols)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf_pipeline = Pipeline([('preprocessor', preprocessor),
                         ('classifier',   RandomForestClassifier(random_state=42))])

param_grid = {
    'classifier__n_estimators':    [50, 100, 200, 300],
    'classifier__max_depth':       [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf':  [1, 2, 4],
    'classifier__max_features':    ['sqrt', 'log2'],
}
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5,
                           scoring='f1', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
best_cv_f1 = grid_search.best_score_

# Feature importances from the RF inside the best pipeline
rf_clf = best_model.named_steps['classifier']
ohe_features = (best_model.named_steps['preprocessor']
                .named_transformers_['cat']
                .named_steps['ohe']
                .get_feature_names_out(categorical_cols).tolist())
all_features = numerical_cols + ohe_features
importances  = rf_clf.feature_importances_
feat_df = (pd.DataFrame({'feature': all_features, 'importance': importances})
           .sort_values('importance', ascending=True)
           .tail(10))

print("Training done. Building dashboard...")

# ── 2. Dashboard layout ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle("Waste Bin Fill Prediction  -  Dashboard",
             fontsize=20, fontweight='bold', color=TEXT, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.48, wspace=0.38,
                       top=0.93, bottom=0.07,
                       left=0.06, right=0.97)

# ── Panel 1  Target class distribution ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
counts  = df['is_full'].value_counts().sort_index()
labels  = ['Not Full (0)', 'Full (1)']
colors  = [ACCENT1, ACCENT2]
bars = ax1.bar(labels, counts.values, color=colors, width=0.5,
               edgecolor=BG, linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
             str(val), ha='center', va='bottom', fontsize=11, fontweight='bold', color=TEXT)
ax1.set_title("Target Distribution", fontsize=12, fontweight='bold', pad=8)
ax1.set_ylabel("Count", fontsize=10)
ax1.yaxis.grid(True); ax1.set_axisbelow(True)

# ── Panel 2  Fill ratio by location type ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
loc_types = df_raw['location_type'].dropna().unique()
palette   = [ACCENT1, ACCENT2, ACCENT3]
for i, loc in enumerate(sorted(loc_types)):
    subset = df[df['location_type'] == loc]['fill_ratio_estimate'].dropna()
    ax2.hist(subset, bins=20, alpha=0.75, label=loc, color=palette[i % len(palette)], edgecolor=BG)
ax2.set_title("Fill Ratio by Location Type", fontsize=12, fontweight='bold', pad=8)
ax2.set_xlabel("Fill Ratio Estimate", fontsize=10)
ax2.set_ylabel("Frequency", fontsize=10)
ax2.legend(fontsize=8, framealpha=0, labelcolor=TEXT)
ax2.yaxis.grid(True); ax2.set_axisbelow(True)

# ── Panel 3  Festival week vs is_full (stacked bar) ──────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
grp = df.groupby(['festival_week', 'is_full']).size().unstack(fill_value=0)
x   = np.arange(len(grp))
w   = 0.4
ax3.bar(x, grp[0], width=w, label='Not Full', color=ACCENT1, edgecolor=BG)
ax3.bar(x, grp[1], width=w, bottom=grp[0], label='Full', color=ACCENT2, edgecolor=BG)
ax3.set_xticks(x)
ax3.set_xticklabels(['Normal Week', 'Festival Week'], fontsize=10)
ax3.set_title("Festival Week vs Bin Status", fontsize=12, fontweight='bold', pad=8)
ax3.set_ylabel("Count", fontsize=10)
ax3.legend(fontsize=9, framealpha=0, labelcolor=TEXT)
ax3.yaxis.grid(True); ax3.set_axisbelow(True)

# ── Panel 4  Feature importances (horizontal bar) ────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
bar_colors = [ACCENT3 if v >= feat_df['importance'].median() else ACCENT1
              for v in feat_df['importance']]
bars4 = ax4.barh(feat_df['feature'], feat_df['importance'],
                 color=bar_colors, edgecolor=BG, height=0.65)
for bar, val in zip(bars4, feat_df['importance']):
    ax4.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va='center', fontsize=9, color=TEXT)
ax4.set_title("Top 10 Feature Importances (Random Forest)", fontsize=12, fontweight='bold', pad=8)
ax4.set_xlabel("Importance", fontsize=10)
ax4.xaxis.grid(True); ax4.set_axisbelow(True)

# ── Panel 5  Confusion matrix ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
cm   = confusion_matrix(y_test, y_pred)
cmap_vals = [[PANEL, ACCENT1], [ACCENT4, ACCENT3]]
for i in range(2):
    for j in range(2):
        color = cmap_vals[i][j] if cm[i, j] > 0 else PANEL
        rect  = FancyBboxPatch((j - 0.4, i - 0.4), 0.8, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=BG, linewidth=2)
        ax5.add_patch(rect)
        ax5.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=20, fontweight='bold',
                 color=BG if color != PANEL else TEXT)

ax5.set_xlim(-0.6, 1.6); ax5.set_ylim(-0.6, 1.6)
ax5.set_xticks([0, 1]); ax5.set_yticks([0, 1])
ax5.set_xticklabels(['Pred: 0', 'Pred: 1'], fontsize=10)
ax5.set_yticklabels(['True: 0', 'True: 1'], fontsize=10)
ax5.set_title("Confusion Matrix", fontsize=12, fontweight='bold', pad=8)
ax5.invert_yaxis()

# ── Panel 6  Model metrics scorecard ─────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Best CV F1']
metric_values = [acc, prec, rec, f1, best_cv_f1]
metric_colors = [ACCENT3, ACCENT1, ACCENT2, ACCENT4, "#b47ef7"]

n     = len(metric_labels)
xs    = np.linspace(0.05, 0.95, n)
panel_w, panel_h = 0.16, 0.72
for xi, label, val, color in zip(xs, metric_labels, metric_values, metric_colors):
    # card background
    ax6.add_patch(FancyBboxPatch((xi - panel_w / 2, 0.08), panel_w, panel_h,
                                  boxstyle="round,pad=0.02",
                                  transform=ax6.transAxes,
                                  facecolor=PANEL, edgecolor=color,
                                  linewidth=2, clip_on=False))
    ax6.text(xi, 0.67, f"{val:.4f}", ha='center', va='center',
             transform=ax6.transAxes,
             fontsize=22, fontweight='bold', color=color)
    ax6.text(xi, 0.22, label, ha='center', va='center',
             transform=ax6.transAxes,
             fontsize=11, color=SUBTEXT)

ax6.text(0.5, 0.95, "Model Performance Summary",
         ha='center', va='center', transform=ax6.transAxes,
         fontsize=13, fontweight='bold', color=TEXT)

# ── Best params footer ────────────────────────────────────────────────────────
bp = grid_search.best_params_
bp_str = "  |  ".join(f"{k.replace('classifier__', '')}: {v}" for k, v in bp.items())
fig.text(0.5, 0.015, f"Best Hyperparameters  ►  {bp_str}",
         ha='center', fontsize=8.5, color=SUBTEXT, style='italic')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "dashboard.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nDashboard saved to: {out_path}")
plt.close(fig)
