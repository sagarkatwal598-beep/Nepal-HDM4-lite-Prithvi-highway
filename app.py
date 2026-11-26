 import json
import streamlit as st
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
    with open("prithvi_corridor_wgs84.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    df = pd.read_parquet("prithvi_hdm4_phase3.parquet")
    return geojson_data, df

geojson_data, df = load_data()

def find_col(columns, keywords, fallback=None):
    cols_lower = {c.lower(): c for c in columns}
    for kw in keywords:
        if kw in cols_lower:
            return cols_lower[kw]
    for kw in keywords:
        for cl, orig in cols_lower.items():
            if kw in cl:
                return orig
    return fallback

cols = list(df.columns)
num_cols = list(df.select_dtypes(include=[np.number]).columns)

seg_col = find_col(cols, ["segment_id", "seg_id", "block_id", "chainage_id", "id"], cols[0])
scen_col = find_col(cols, ["scenario", "regime", "traffic_case", "case"])
treat_col = find_col(cols, ["treatment", "strategy", "intervention", "option", "action"])
cost_col = find_col(num_cols, ["cost_mnpr", "capex_mnpr", "treatment_cost", "agency_cost", "capex", "cost"])
npv_col = find_col(num_cols, ["npv_mnpr", "net_npv", "total_npv", "npv", "net_benefit", "voc_sav", "benefit"])
iri0_col = find_col(num_cols, ["iri_0", "iri0", "iri_yr0", "initial_iri", "start_iri"])
iri10_col = find_col(num_cols, ["iri_yr10", "iri_10", "iri10", "final_iri", "end_iri"])

# Fallback for cost/npv if custom names were used
if cost_col is None or npv_col is None:
    st.error(f"Could not auto-detect Cost/NPV columns. Found columns: {cols}")
    st.stop()

# Sidebar Controls
st.sidebar.header("⚙️ DoR Annual Maintenance Parameters")
budget_mnpr = st.sidebar.slider(
    "Annual RBN / DoR Capital Budget (Million NPR)",
    min_value=150.0,
    max_value=1200.0,
    value=450.0,
    step=25.0,
)

if scen_col:
    scenarios = sorted(df[scen_col].astype(str).unique().tolist())
    selected_scenario = st.sidebar.selectbox(
        "Traffic & Axle Loading Regime",
        options=scenarios,
        format_func=lambda x: "Baseline Freight (Standard DoR ESALs)"
        if "base" in x.lower()
        else "Hydropower Heavy-Haul Convoy Stress (Fourth Power Law)",
    )
    sub = df[df[scen_col].astype(str) == str(selected_scenario)].copy()
else:
    selected_scenario = "baseline"
    sub = df.copy()

safety_iri = st.sidebar.number_input(
    "Mandatory Safety Floor Trigger (IRI₀ ≥ m/km)",
    min_value=3.5,
    max_value=7.0,
    value=4.5,
    step=0.5,
)

sub = sub.sort_values([seg_col, cost_col]).reset_index(drop=True)

segments = sub[seg_col].unique()
n_vars = len(sub)

# Build HiGHS MILP (0-1 Multiple-Choice Knapsack + Mandatory Safety Constraint)
c_obj = -sub[npv_col].to_numpy(dtype=float)
costs = sub[cost_col].to_numpy(dtype=float)

rows = [costs]
lbs = [0.0]
ubs = [float(budget_mnpr)]

for seg in segments:
    mask = (sub[seg_col] == seg).to_numpy(dtype=float)
    rows.append(mask)
    lbs.append(1.0)
    ubs.append(1.0)

    if iri0_col:
        seg_rows = sub[sub[seg_col] == seg]
        if seg_rows[iri0_col].iloc[0] >= safety_iri and budget_mnpr >= 250.0:
            active_mask = ((sub[seg_col] == seg) & (sub[cost_col] > 0)).to_numpy(dtype=float)
            if active_mask.sum() > 0:
                rows.append(active_mask)
                lbs.append(1.0)
                ubs.append(1.0)

A = np.vstack(rows)
res = milp(
    c=c_obj,
    integrality=np.ones(n_vars),
    bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
    constraints=LinearConstraint(A, lbs, ubs),
)

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

# Attach optimized plan properties directly to GeoJSON features
plan_lookup = opt_plan.set_index(seg_col).to_dict(orient="index")
sample_props = geojson_data["features"][0]["properties"] if geojson_data.get("features") else {}
geo_seg_col = find_col(list(sample_props.keys()), ["segment_id", "seg_id", "block_id", "chainage_id", "id"], list(sample_props.keys())[0])

enriched_features = []
for feat in geojson_data["features"]:
    props = dict(feat["properties"])
    sid = props.get(geo_seg_col)
    if sid in plan_lookup:
        for k, v in plan_lookup[sid].items():
            if pd.notna(v):
                props[k] = float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else str(v))
    enriched_features.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})

enriched_geojson = {"type": "FeatureCollection", "features": enriched_features}

col_map, col_chart = st.columns([1.35, 1.0])

with col_map:
    st.subheader("🗺️ Optimized Corridor Treatment Map (348 Blocks)")
    color_map = {
        "nothing": "#6c757d",
        "patch": "#17a2b8",
        "routine": "#17a2b8",
        "overlay": "#28a745",
        "40": "#28a745",
        "reconstruct": "#dc3545",
    }
    m = folium.Map(location=[27.95, 84.55], zoom_start=9, tiles="CartoDB positron")

    def style_fn(feature):
        tr = str(feature["properties"].get(treat_col, "Do Nothing")).lower() if treat_col else "overlay"
        col = "#28a745"
        for k, v in color_map.items():
            if k in tr:
                col = v
                break
        return {"color": col, "weight": 5, "opacity": 0.9}

    avail_props = list(enriched_features[0]["properties"].keys()) if enriched_features else []
    zone_col = find_col(avail_props, ["zone", "corridor_zone", "section"])
    tooltip_fields = [c for c in [geo_seg_col, zone_col, treat_col, cost_col, npv_col, iri0_col, iri10_col] if c and c in avail_props]

    folium.GeoJson(
        enriched_geojson,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
    ).add_to(m)
    st_folium(m, width=None, height=460, returned_objects=[])

with col_chart:
    st.subheader("📊 Treatment Allocation & Budget Split")
    group_col = treat_col if treat_col else seg_col
    summary = (
        opt_plan.groupby(group_col)
        .agg(
            Blocks=(seg_col, "count"),
            Length_km=(seg_col, lambda s: len(s) * 0.5),
            Cost_MNPR=(cost_col, "sum"),
            NPV_MNPR=(npv_col, "sum"),
        )
        .reset_index()
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=summary[group_col],
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
zone_df_col = find_col(cols, ["zone", "corridor_zone"])
start_col = find_col(cols, ["start_km", "from_km", "chainage_start"])
end_col = find_col(cols, ["end_km", "to_km", "chainage_end"])
display_cols = [c for c in [seg_col, zone_df_col, start_col, end_col, treat_col, cost_col, npv_col, iri0_col, iri10_col] if c and c in opt_plan.columns]
st.dataframe(opt_plan[display_cols], use_container_width=True, height=280)

csv_bytes = opt_plan[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download DoR Procurement Schedule (CSV)",
    data=csv_bytes,
    file_name=f"prithvi_armp_{selected_scenario}_{int(budget_mnpr)}MNPR.csv",
    mime="text/csv",
)
