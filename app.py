import json
import urllib.request
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

st.title("🛣️ Nepal HDM-4 Lite: Prithvi Highway Pavement Management Studio")
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
zone_df_col = find_col(cols, ["zone", "corridor_zone", "section"])
start_col = find_col(num_cols, ["start_km", "from_km", "chainage_start", "km_start"])
end_col = find_col(num_cols, ["end_km", "to_km", "chainage_end", "km_end"])
slope_col = find_col(num_cols, ["slope", "grade", "gradient"])
ndwi_col = find_col(num_cols, ["ndwi", "wetness", "drainage", "moisture"])
msa_col = find_col(num_cols, ["msa", "esal", "traffic"])

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

st.sidebar.markdown("---")
st.sidebar.info(
    "**Corridor Zones (NRS 2070):**\n"
    "- **Zone A:** Naubise – Mugling (2-Lane Gorge)\n"
    "- **Zone B:** Mugling – Abukhaireni (Bottleneck)\n"
    "- **Zone C:** Abukhaireni – Pokhara (4-Lane SASEC)"
)

sub = sub.sort_values([seg_col, cost_col]).reset_index(drop=True)
segments = sub[seg_col].unique()


def solve_milp_and_wf(data_sub, b_cap, s_iri):
    segs = data_sub[seg_col].unique()
    n_v = len(data_sub)
    c_obj = -data_sub[npv_col].to_numpy(dtype=float)
    costs = data_sub[cost_col].to_numpy(dtype=float)

    rows = [costs]
    lbs = [0.0]
    ubs = [float(b_cap)]

    for seg in segs:
        mask = (data_sub[seg_col] == seg).to_numpy(dtype=float)
        rows.append(mask)
        lbs.append(1.0)
        ubs.append(1.0)

        if iri0_col:
            seg_rows = data_sub[data_sub[seg_col] == seg]
            if seg_rows[iri0_col].iloc[0] >= s_iri and b_cap >= 250.0:
                active_mask = ((data_sub[seg_col] == seg) & (data_sub[cost_col] > 0)).to_numpy(dtype=float)
                if active_mask.sum() > 0:
                    rows.append(active_mask)
                    lbs.append(1.0)
                    ubs.append(1.0)

    A = np.vstack(rows)
    res = milp(
        c=c_obj,
        integrality=np.ones(n_v),
        bounds=Bounds(np.zeros(n_v), np.ones(n_v)),
        constraints=LinearConstraint(A, lbs, ubs),
    )

    if res.success and res.x is not None:
        chosen_idx = np.where(np.round(res.x) == 1)[0]
        m_plan = data_sub.iloc[chosen_idx].copy()
    else:
        m_plan = data_sub.groupby(seg_col, as_index=False).first()

    # Worst-First Heuristic
    wf_spent = 0.0
    wf_choices = []
    seg_order = (
        data_sub.groupby(seg_col)[iri0_col].first().sort_values(ascending=False).index
        if iri0_col
        else segs
    )
    for seg in seg_order:
        opts = data_sub[data_sub[seg_col] == seg].sort_values(cost_col, ascending=False)
        picked = opts.iloc[-1]
        for _, r in opts.iterrows():
            if r[cost_col] > 0 and wf_spent + r[cost_col] <= b_cap:
                picked = r
                wf_spent += r[cost_col]
                break
        wf_choices.append(picked)
    w_plan = pd.DataFrame(wf_choices)
    return m_plan, w_plan


opt_plan, wf_plan = solve_milp_and_wf(sub, budget_mnpr, safety_iri)

# Ensure chainage exists for plotting
if start_col is None:
    seg_rank = {s: i * 0.5 for i, s in enumerate(sorted(segments))}
    opt_plan["chainage_km"] = opt_plan[seg_col].map(seg_rank)
    wf_plan["chainage_km"] = wf_plan[seg_col].map(seg_rank)
    chain_col = "chainage_km"
else:
    chain_col = start_col

opt_plan = opt_plan.sort_values(chain_col).reset_index(drop=True)
wf_plan = wf_plan.sort_values(chain_col).reset_index(drop=True)

milp_spent = opt_plan[cost_col].sum()
milp_npv = opt_plan[npv_col].sum()
wf_npv = wf_plan[npv_col].sum()
gain_npv = milp_npv - wf_npv
avg_yr10_iri = opt_plan[iri10_col].mean() if iri10_col else np.nan

# Top KPI Ribbon
k1, k2, k3, k4 = st.columns(4)
k1.metric("MILP Budget Utilized", f"NPR {milp_spent:,.2f} M", f"of NPR {budget_mnpr:,.0f} M cap")
k2.metric("10-Yr Economic NPV (MILP)", f"NPR {milp_npv:,.2f} M", f"+NPR {gain_npv:,.2f} M vs Worst-First")
k3.metric("Worst-First Heuristic NPV", f"NPR {wf_npv:,.2f} M", "Traditional reactive baseline")
k4.metric("Corridor Yr-10 Mean IRI", f"{avg_yr10_iri:.2f} m/km" if not np.isnan(avg_yr10_iri) else "N/A", f"Safety Floor < {safety_iri:.1f} m/km")

# 4-Tab Engineering Studio
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ 1. Map & Strategy Duel",
    "📈 2. Longitudinal IRI & Budget Frontier",
    "🔍 3. Block Inspector & Zone Equity",
    "🌧️ 4. Live Monsoon Watch & Procurement",
])

# ==============================================================================
# TAB 1: MAP & MILP VS WORST-FIRST DUEL
# ==============================================================================
with tab1:
    map_mode = st.radio(
        "Select Allocation Strategy to Visualize on Corridor Map:",
        options=["MILP Optimal Plan (HiGHS Solver)", "Traditional Worst-First Heuristic (Reactive Baseline)"],
        horizontal=True,
    )
    active_plan = opt_plan if "MILP" in map_mode else wf_plan

    plan_lookup = active_plan.set_index(seg_col).to_dict(orient="index")
    sample_props = geojson_data["features"][0]["properties"] if geojson_data.get("features") else {}
    geo_seg_col = find_col(
        list(sample_props.keys()),
        ["segment_id", "seg_id", "block_id", "chainage_id", "id"],
        list(sample_props.keys())[0],
    )

    enriched_features = []
    for feat in geojson_data["features"]:
        props = dict(feat["properties"])
        sid = props.get(geo_seg_col)
        if sid in plan_lookup:
            for k, v in plan_lookup[sid].items():
                if pd.notna(v):
                    props[k] = (
                        float(v)
                        if isinstance(v, (np.floating, float))
                        else (int(v) if isinstance(v, (np.integer, int)) else str(v))
                    )
        enriched_features.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})

    enriched_geojson = {"type": "FeatureCollection", "features": enriched_features}

    col_map, col_chart = st.columns([1.35, 1.0])
    with col_map:
        st.subheader(f"Corridor Map — {map_mode.split(' (')[0]}")
        color_map = {
            "nothing": "#6c757d",
            "patch": "#17a2b8",
            "routine": "#17a2b8",
            "overlay": "#28a745",
            "40": "#28a745",
            "reconstruct": "#dc3545",
        }
        m = folium.Map(location=[27.95, 84.55], zoom_start=9, tiles="OpenStreetMap")
        folium.TileLayer("Esri.WorldImagery", name="Satellite View", attr="Esri").add_to(m)

        def style_fn(feature):
            tr = str(feature["properties"].get(treat_col, "Do Nothing")).lower() if treat_col else "overlay"
            col = "#28a745"
            for k, v in color_map.items():
                if k in tr:
                    col = v
                    break
            return {"color": col, "weight": 5, "opacity": 0.9}

        avail_props = list(enriched_features[0]["properties"].keys()) if enriched_features else []
        z_prop = find_col(avail_props, ["zone", "corridor_zone", "section"])
        tooltip_fields = [
            c for c in [geo_seg_col, z_prop, treat_col, cost_col, npv_col, iri0_col, iri10_col] if c and c in avail_props
        ]

        folium.GeoJson(
            enriched_geojson,
            name="Prithvi Highway Treatments",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        ).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=None, height=450, returned_objects=[])
        st.caption("Legend: ⬜ Grey = Do Nothing | 🟦 Teal = Routine Patching | 🟩 Green = 40 mm AC Overlay | 🟥 Red = Full Reconstruction")

    with col_chart:
        st.subheader("Treatment Allocation & Budget Split")
        group_col = treat_col if treat_col else seg_col
        summary = (
            active_plan.groupby(group_col)
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
            xaxis_title="Engineering Treatment",
            height=240,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: LONGITUDINAL IRI PROFILE & BUDGET FRONTIER
# ==============================================================================
with tab2:
    st.subheader("Longitudinal Roughness Profile (0.0 km Naubise → 173.9 km Pokhara)")
    if iri0_col and iri10_col:
        do_nothing_df = sub[sub[cost_col] == 0].sort_values(chain_col if chain_col in sub.columns else seg_col)
        fig_prof = go.Figure()
        fig_prof.add_trace(
            go.Scatter(
                x=opt_plan[chain_col],
                y=opt_plan[iri0_col],
                mode="lines",
                name="Year-0 Satellite Baseline (IRI₀)",
                line=dict(color="#6c757d", width=1.5, dash="dot"),
            )
        )
        if len(do_nothing_df) == len(opt_plan):
            fig_prof.add_trace(
                go.Scatter(
                    x=opt_plan[chain_col],
                    y=do_nothing_df[iri10_col].values,
                    mode="lines",
                    name="Year-10 Unmanaged (Do Nothing)",
                    line=dict(color="#dc3545", width=1.8),
                    opacity=0.75,
                )
            )
        fig_prof.add_trace(
            go.Scatter(
                x=opt_plan[chain_col],
                y=opt_plan[iri10_col],
                mode="lines",
                name="Year-10 MILP Optimized Plan",
                line=dict(color="#28a745", width=2.3),
            )
        )
        fig_prof.add_hline(
            y=safety_iri,
            line_dash="dash",
            line_color="#ff9800",
            annotation_text=f"DoR Safety Trigger ({safety_iri:.1f} m/km)",
        )
        fig_prof.add_vrect(x0=0, x1=96, fillcolor="rgba(255,193,7,0.08)", line_width=0, annotation_text="Zone A: Naubise–Mugling", annotation_position="top left")
        fig_prof.add_vrect(x0=96, x1=104, fillcolor="rgba(220,53,69,0.08)", line_width=0, annotation_text="Zone B", annotation_position="top left")
        fig_prof.add_vrect(x0=104, x1=174, fillcolor="rgba(40,167,69,0.06)", line_width=0, annotation_text="Zone C: Abukhaireni–Pokhara (4-Lane)", annotation_position="top left")
        fig_prof.update_layout(
            xaxis_title="Chainage from Naubise (km)",
            yaxis_title="International Roughness Index — IRI (m/km)",
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_prof, use_container_width=True)
    else:
        st.info("IRI columns not detected for longitudinal profile.")

    st.subheader("RBN / DoR Budget Efficiency Frontier (NPR 150M → NPR 1,200M)")
    budget_grid = [150, 250, 350, 450, 600, 750, 900, 1100]
    frontier_rows = []
    for b in budget_grid:
        mp, wp = solve_milp_and_wf(sub, b, safety_iri)
        frontier_rows.append({
            "Budget_MNPR": b,
            "MILP_NPV": mp[npv_col].sum(),
            "WorstFirst_NPV": wp[npv_col].sum(),
            "Gain_MNPR": mp[npv_col].sum() - wp[npv_col].sum(),
        })
    f_df = pd.DataFrame(frontier_rows)
    fig_front = go.Figure()
    fig_front.add_trace(go.Scatter(x=f_df["Budget_MNPR"], y=f_df["MILP_NPV"], mode="lines+markers", name="HiGHS MILP Optimal NPV", line=dict(color="#28a745", width=3)))
    fig_front.add_trace(go.Scatter(x=f_df["Budget_MNPR"], y=f_df["WorstFirst_NPV"], mode="lines+markers", name="Worst-First Heuristic NPV", line=dict(color="#dc3545", width=2, dash="dash")))
    fig_front.add_vline(x=budget_mnpr, line_dash="dot", line_color="#17a2b8", annotation_text=f"Current Budget: NPR {int(budget_mnpr)}M")
    fig_front.update_layout(
        xaxis_title="Annual Capital Budget Ceiling (Million NPR)",
        yaxis_title="10-Year Net Present Value (Million NPR)",
        height=310,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_front, use_container_width=True)

# ==============================================================================
# TAB 3: SINGLE-BLOCK ENGINEERING INSPECTOR & ZONE EQUITY
# ==============================================================================
with tab3:
    col_insp, col_zone = st.columns([1.2, 1.0])

    def nearest_landmark(km_val):
        landmarks = [
            (0.0, "Naubise Zero-Point"),
            (22.0, "Galchhi Junction"),
            (43.0, "Malekhu Bazaar"),
            (72.0, "Krishnabhir Slide Zone"),
            (96.0, "Mugling Bridge"),
            (104.0, "Abukhaireni"),
            (126.0, "Damauli (Vyas)"),
            (152.0, "Khairenitar / Dulegauda"),
            (173.9, "Pokhara (Prithvi Chowk)"),
        ]
        closest = min(landmarks, key=lambda x: abs(x[0] - km_val))
        return closest[1]

    with col_insp:
        st.subheader("Single 500 m Block Engineering Inspector")
        block_labels = {}
        for _, r in opt_plan.iterrows():
            km_v = float(r[chain_col])
            sid = r[seg_col]
            block_labels[f"Block {sid} | Km {km_v:.1f} (Near {nearest_landmark(km_v)})"] = sid

        chosen_label = st.selectbox("Select 500 m Highway Block to Inspect:", options=list(block_labels.keys()))
        chosen_sid = block_labels[chosen_label]

        block_opts = sub[sub[seg_col] == chosen_sid].sort_values(cost_col).copy()
        selected_row = opt_plan[opt_plan[seg_col] == chosen_sid].iloc[0]

        b_c1, b_c2, b_c3 = st.columns(3)
        b_c1.metric("Yr-0 Satellite IRI₀", f"{selected_row[iri0_col]:.2f} m/km" if iri0_col else "N/A")
        b_c2.metric("Annual Fatigue (MSA)", f"{selected_row[msa_col]:.2f}" if msa_col else "Zone Calibrated")
        b_c3.metric("MILP Selected Action", str(selected_row[treat_col]) if treat_col else "Optimal")

        # Show 4 treatment alternatives for this block
        insp_cols = [c for c in [treat_col, cost_col, npv_col, iri10_col] if c]
        block_table = block_opts[insp_cols].copy()
        block_table["BCR (NPV / Cost)"] = np.where(
            block_opts[cost_col] > 0,
            (block_opts[npv_col] / block_opts[cost_col]).round(2),
            np.inf,
        )
        st.dataframe(block_table, use_container_width=True, hide_index=True)

        st.success(
            f"**Solver Rationale:** Out of 4 candidate interventions on Block `{chosen_sid}`, the HiGHS MILP solver "
            f"selected **{selected_row[treat_col]}** (Capex: `NPR {selected_row[cost_col]:.2f} M`, "
            f"10-Yr NPV: `NPR {selected_row[npv_col]:.2f} M`) to maximize corridor-wide user VOC savings under the "
            f"`NPR {int(budget_mnpr)} M` budget constraint."
        )

    with col_zone:
        st.subheader("DoR Zone & Division Equity Breakdown")
        if zone_df_col and zone_df_col in opt_plan.columns:
            z_summary = (
                opt_plan.groupby(zone_df_col)
                .agg(
                    Blocks=(seg_col, "count"),
                    Corridor_km=(seg_col, lambda s: len(s) * 0.5),
                    Budget_MNPR=(cost_col, "sum"),
                    NPV_MNPR=(npv_col, "sum"),
                    Mean_Yr10_IRI=(iri10_col, "mean") if iri10_col else (cost_col, "count"),
                )
                .reset_index()
            )
            fig_z = go.Figure(
                data=[
                    go.Pie(
                        labels=z_summary[zone_df_col],
                        values=z_summary["Budget_MNPR"],
                        hole=0.45,
                        textinfo="label+percent",
                    )
                ]
            )
            fig_z.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_z, use_container_width=True)
            st.dataframe(z_summary, use_container_width=True, hide_index=True)
        else:
            st.info("Zone column not present in Parquet summary.")

# ==============================================================================
# TAB 4: LIVE OPEN-METEO MONSOON WATCH & PROCUREMENT SCHEDULE
# ==============================================================================
with tab4:
    st.subheader("🌧️ Live 7-Day Corridor Rainfall & Subgrade Drainage Watch (Open-Meteo API)")

    @st.cache_data(ttl=3600)
    def fetch_corridor_weather():
        stations = [
            {"Station": "Naubise (Km 0)", "lat": 27.716, "lon": 85.150},
            {"Station": "Malekhu (Km 43)", "lat": 27.808, "lon": 84.825},
            {"Station": "Mugling (Km 96)", "lat": 27.855, "lon": 84.560},
            {"Station": "Damauli (Km 126)", "lat": 27.973, "lon": 84.268},
            {"Station": "Pokhara (Km 174)", "lat": 28.209, "lon": 83.985},
        ]
        results = []
        for s in stations:
            url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={s['lat']}&longitude={s['lon']}"
                f"&daily=precipitation_sum,temperature_2m_max&timezone=Asia%2FKathmandu"
            )
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                rain_7d = sum(payload["daily"]["precipitation_sum"])
                max_temp = max(payload["daily"]["temperature_2m_max"])
                risk = "HIGH (Side-Drain Alert)" if rain_7d >= 50 else ("MODERATE" if rain_7d >= 15 else "LOW (Dry Window)")
                results.append({
                    "Corridor Node": s["Station"],
                    "7-Day Forecast Rain (mm)": round(rain_7d, 1),
                    "Peak Temp (°C)": round(max_temp, 1),
                    "Subgrade Moisture Status": risk,
                })
            except Exception:
                results.append({
                    "Corridor Node": s["Station"],
                    "7-Day Forecast Rain (mm)": 0.0,
                    "Peak Temp (°C)": 24.0,
                    "Subgrade Moisture Status": "Offline Fallback",
                })
        return pd.DataFrame(results)

    wx_df = fetch_corridor_weather()
    st.dataframe(wx_df, use_container_width=True, hide_index=True)

    st.subheader("📋 Procurement-Ready Annual Road Maintenance Plan (ARMP)")
    display_cols = [
        c for c in [seg_col, zone_df_col, chain_col, end_col, treat_col, cost_col, npv_col, iri0_col, iri10_col]
        if c and c in opt_plan.columns
    ]
    st.dataframe(opt_plan[display_cols], use_container_width=True, height=280)

    csv_bytes = opt_plan[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download DoR Procurement Schedule (CSV)",
        data=csv_bytes,
        file_name=f"prithvi_armp_{selected_scenario}_{int(budget_mnpr)}MNPR.csv",
        mime="text/csv",
    )
