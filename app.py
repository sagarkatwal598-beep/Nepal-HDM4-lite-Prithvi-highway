import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

st.set_page_config(
    page_title="Nepal HDM-4 Lite | Prithvi Highway PMS",
    page_icon="🛣️",
    layout="wide",
)

st.title("🛣️ Nepal HDM-4 Lite: Prithvi Highway Pavement Management System")
st.caption(
    "Naubise – Mugling – Pokhara (~173.9 km | 348 × 500 m Blocks) • "
    "DoR Flexible Pavement Design Guideline (2021) & NRS 2070 • HiGHS MILP Budget Optimizer"
)

@st.cache_data
def load_data():
    gdf = gpd.read_file("prithvi_corridor_wgs84.geojson")
    df = pd.read_parquet("prithvi_hdm4_phase3.parquet")
    return gdf, df

gdf, df = load_data()

# Identify key column names safely
seg_col = next((c for c in ["segment_id", "seg_id", "block_id", "id"] if c in df.columns), df.columns[0])
scen_col = next((c for c in ["scenario", "traffic_scenario", "regime"] if c in df.columns), None)
treat_col = next((c for c in ["treatment", "strategy", "option", "intervention"] if c in df.columns), None)
cost_col = next((c for c in ["cost_mnpr", "capex_mnpr", "agency_cost_mnpr", "cost"] if c in df.columns), None)
npv_col = next((c for c in ["npv_mnpr", "net_benefit_mnpr", "voc_savings_mnpr", "npv"] if c in df.columns), None)
iri0_col = next((c for c in ["iri_0", "iri0", "initial_iri"] if c in df.columns), None)
iri10_col = next((c for c in ["iri_yr10", "iri_10", "final_iri"] if c in df.columns), None)

# Sidebar Controls
st.sidebar.header("⚙️ DoR Annual Maintenance Parameters")
budget_mnpr = st.sidebar.slider(
    "Annual RBN / DoR Capital Budget (Million NPR)",
    min_value=150.0,
    max_value=1200.0,
    value=450.0,
    step=25.0,
)

scenarios = sorted(df[scen_col].unique().tolist()) if scen_col else ["baseline"]
selected_scenario = st.sidebar.selectbox(
    "Traffic & Axle Loading Regime",
    options=scenarios,
    format_func=lambda x: "Baseline Freight (Standard DoR ESALs)"
    if "base" in str(x).lower()
    else "Hydropower Heavy-Haul Convoy Stress (Fourth Power Law)",
)

safety_iri = st.sidebar.number_input(
    "Mandatory Safety Floor Trigger (IRI₀ ≥ m/km)",
    min_value=3.5,
    max_value=7.0,
    value=4.5,
    step=0.5,
)

sub = df[df[scen_col] == selected_scenario].copy() if scen_col else df.copy()
sub = sub.sort_values([seg_col, cost_col]).reset_index(drop=True)

segments = sub[seg_col].unique()
n_seg = len(segments)
n_vars = len(sub)

# Build HiGHS MILP (0-1 Multiple-Choice Knapsack + Mandatory Safety Constraint)
c_obj = -sub[npv_col].to_numpy(dtype=float)
costs = sub[cost_col].to_numpy(dtype=float)

rows = []
lbs = []
ubs = []

# Row 0: Total budget constraint
rows.append(costs)
lbs.append(0.0)
ubs.append(float(budget_mnpr))

# One treatment per 500m segment + safety floor for IRI0 >= safety_iri
for seg in segments:
    mask = (sub[seg_col] == seg).to_numpy(dtype=float)
    rows.append(mask)
    lbs.append(1.0)
    ubs.append(1.0)

    if iri0_col and cost_col:
        seg_rows = sub[sub[seg_col] == seg]
        if seg_rows[iri0_col].iloc[0] >= safety_iri and budget_mnpr >= 250.0:
            active_mask = ((sub[seg_col] == seg) & (sub[cost_col] > 0)).to_numpy(dtype=float)
            if active_mask.sum() > 0:
                rows.append(active_mask)
                lbs.append(1.0)
                ubs.append(1.0)

A = np.vstack(rows)
constraints = LinearConstraint(A, lbs, ubs)
integrality = np.ones(n_vars)
bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))

res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

if res.success and res.x is not None:
    chosen_idx = np.where(np.round(res.x) == 1)[0]
    opt_plan = sub.iloc[chosen_idx].copy()
else:
    opt_plan = sub.groupby(seg_col, as_index=False).first()

# Worst-First Heuristic Benchmark
wf_spent = 0.0
wf_choices = []
seg_order = (
    sub.groupby(seg_col)[iri0_col].first().sort_values(ascending=False).index
    if iri0_col
    else segments
)
for seg in seg_order:
    opts = sub[sub[seg_col] == seg].sort_values(cost_col, ascending=False)
    picked = opts.iloc[-1]
    for _, r in opts.iterrows():
        if r[cost_col] > 0 and wf_spent + r[cost_col] <= budget_mnpr:
            picked = r
            wf_spent += r[cost_col]
            break
    wf_choices.append(picked)
wf_plan = pd.DataFrame(wf_choices)

milp_spent = opt_plan[cost_col].sum()
milp_npv = opt_plan[npv_col].sum()
wf_npv = wf_plan[npv_col].sum()
gain_npv = milp_npv - wf_npv
avg_yr10_iri = opt_plan[iri10_col].mean() if iri10_col else np.nan

# Top KPI Metrics
k1, k2, k3, k4 = st.columns(4)
k1.metric("MILP Budget Utilized", f"NPR {milp_spent:,.2f} M", f"of NPR {budget_mnpr:,.0f} M cap")
k2.metric("10-Yr Economic NPV (MILP)", f"NPR {milp_npv:,.2f} M", f"+NPR {gain_npv:,.2f} M vs Worst-First")
k3.metric("Worst-First Heuristic NPV", f"NPR {wf_npv:,.2f} M", "Traditional reactive baseline")
k4.metric("Corridor Yr-10 Mean IRI", f"{avg_yr10_iri:.2f} m/km" if not np.isnan(avg_yr10_iri) else "N/A", "Target < 4.5 m/km")

# Merge optimized plan with corridor geometry
geo_seg_col = next((c for c in ["segment_id", "seg_id", "block_id", "id"] if c in gdf.columns), gdf.columns[0])
map_gdf = gdf.merge(opt_plan, left_on=geo_seg_col, right_on=seg_col, how="left", suffixes=("", "_plan"))

col_map, col_chart = st.columns([1.35, 1.0])

with col_map:
    st.subheader("🗺️ Optimized Corridor Treatment Map (348 Blocks)")
    color_map = {
        "Do Nothing": "#6c757d",
        "Routine Patching": "#17a2b8",
        "40 mm AC Overlay": "#28a745",
        "Full Reconstruction": "#dc3545",
    }
    m = folium.Map(location=[27.95, 84.55], zoom_start=9, tiles="CartoDB positron")

    def style_fn(feature):
        tr = str(feature["properties"].get(treat_col, "Do Nothing"))
        col = "#28a745"
        for k, v in color_map.items():
            if k.lower() in tr.lower():
                col = v
                break
        if "nothing" in tr.lower():
            col = "#6c757d"
        return {"color": col, "weight": 5, "opacity": 0.9}

    tooltip_fields = [c for c in [geo_seg_col, "zone", treat_col, cost_col, npv_col, iri0_col, iri10_col] if c in map_gdf.columns]
    folium.GeoJson(
        map_gdf,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields),
    ).add_to(m)
    st_folium(m, width=None, height=460, returned_objects=[])

with col_chart:
    st.subheader("📊 Treatment Allocation & Budget Split")
    summary = (
        opt_plan.groupby(treat_col)
        .agg(Blocks=(seg_col, "count"), Length_km=(seg_col, lambda s: len(s) * 0.5), Cost_MNPR=(cost_col, "sum"), NPV_MNPR=(npv_col, "sum"))
        .reset_index()
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=summary[treat_col],
                y=summary["Length_km"],
                text=summary["Length_km"].apply(lambda x: f"{x:.1f} km"),
                textposition="auto",
                marker_color=["#6c757d", "#17a2b8", "#28a745", "#dc3545"][: len(summary)],
            )
        ]
    )
    fig.update_layout(
        yaxis_title="Corridor Length Treated (km)",
        xaxis_title="Selected Engineering Treatment",
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("📋 Procurement-Ready Annual Road Maintenance Plan (ARMP)")
display_cols = [c for c in [seg_col, "zone", "start_km", "end_km", treat_col, cost_col, npv_col, iri0_col, iri10_col] if c in opt_plan.columns]
st.dataframe(opt_plan[display_cols], use_container_width=True, height=280)

csv_bytes = opt_plan[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download DoR Procurement Schedule (CSV)",
    data=csv_bytes,
    file_name=f"prithvi_armp_{selected_scenario}_{int(budget_mnpr)}MNPR.csv",
    mime="text/csv",
)
