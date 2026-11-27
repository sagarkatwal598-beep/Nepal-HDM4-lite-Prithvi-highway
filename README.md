# 🛣️ Nepal HDM-4 Lite: Prithvi Highway Pavement Management System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nepal-hdm4-lite-prithvi-highway-jp7bfitxuwoiatd9bjhjqp.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Standards](https://img.shields.io/badge/Standards-DoR%202021%20%7C%20NRS%202070-green)
![Solver](https://img.shields.io/badge/Optimizer-SciPy%20HiGHS%20MILP-orange)

An open-source, data-driven **Pavement Management System (PMS)** and **Annual Road Maintenance Plan (ARMP)** decision-support engine built for Nepal's **Prithvi Highway (`Naubise – Mugling – Pokhara`)**.

The platform integrates satellite remote sensing (**Google Earth Engine**), mechanistic-empirical pavement deterioration (**HDM-4 / DoR Flexible Pavement Design Guideline 2021**), and **Mixed-Integer Linear Programming (`HiGHS MILP`)** to maximize 10-year Vehicle Operating Cost (`VOC`) savings under constrained Roads Board Nepal (`RBN`) / Department of Roads (`DoR`) maintenance budgets.

🔗 **Live Interactive Web App:** [Launch Nepal HDM-4 Lite Dashboard](https://nepal-hdm4-lite-prithvi-highway-jp7bfitxuwoiatd9bjhjqp.streamlit.app)

---

## 📌 Engineering Problem & Context

The **173.9 km Prithvi Highway** is Nepal's primary economic lifeline connecting the Kathmandu Valley to Pokhara and the western Terai. Traditional road maintenance in Nepal frequently relies on a reactive **"Worst-First"** heuristic—patching severe post-monsoon failures only after structural collapse has occurred.

**Nepal HDM-4 Lite** demonstrates how proactive, lifecycle-optimized preservation outperforms reactive repairs, while accounting for:
1. **Corridor Capacity & Lane-Distribution Transitions:**
   * **Zone A (`Naubise – Mugling`):** 2-lane river-gorge highway ($D = 0.75$) experiencing severe freight fatigue (`~3.49 MSA/yr`).
   * **Zone B (`Mugling – Abukhaireni`):** High-stress bottleneck link ($D = 0.75$).
   * **Zone C (`Abukhaireni – Pokhara`):** Upgraded 4-lane SASEC divided highway ($D = 0.50$).
2. **Hydropower Heavy-Haul Axle Stress:** Models multi-axle modular trailer convoys transporting electromechanical equipment (turbines/transformers) along the Trishuli and Marsyangdi cascades using the AASHTO/DoR **Fourth Power Law** $(W / 8.16)^4$.

---

## 🏗️ System Architecture & Methodology

### Phase 1: Corridor Extraction & Homogeneous Segmentation (`OSMnx` + `GeoPandas`)
* Extracted the arterial centerline of the Prithvi Highway from OpenStreetMap and projected it to **WGS 84 / UTM Zone 45N (`EPSG:32645`)**.
* Partitioned the **173.9 km** alignment into **348 homogeneous 500 m management blocks** with zone-specific structural numbers (`SNP`) and cumulative `ESAL` / `MSA` traffic growth projections per **NRS 2070** and **DoR (2021)**.

### Phase 2: Satellite Condition Baseline (`Google Earth Engine`)
* **Sentinel-1 C-Band SAR (VV/VH Backscatter):** Derived surface macro-texture and initial roughness proxy ($\text{IRI}_0$), classifying the network into `160` Good (`80.0 km`), `28` Fair (`14.0 km`), `150` Poor (`75.0 km`), and `10` Critical (`5.0 km`) blocks.
* **SRTM 30 m DEM:** Computed longitudinal gradients to capture heavy-truck climbing/braking shear stress.
* **Sentinel-2 ($\Delta\text{NDWI}$) & CHIRPS Rainfall:** Quantified monsoon surface ponding and subgrade moisture susceptibility, applying effective `SNP` reduction factors on poorly drained blocks.

### Phase 3: 10-Year Lifecycle Deterioration & Economics (`HDM-4 Lite`)
* Simulated **2,784 roughness (`IRI`) and economic trajectories** across **4 maintenance interventions** and **2 traffic regimes** (`Baseline Freight` vs. `Hydropower Heavy-Haul Convoys`) at an 8% social discount rate:
  1. *Do Nothing* (Routine drainage/safety only)
  2. *Routine Patching & Surface Sealing*
  3. *40 mm Asphalt Concrete (AC) Structural Overlay*
  4. *Full-Depth Pavement Reconstruction*

### Phase 4: Constrained Budget Optimization (`SciPy HiGHS MILP`)
* Formulated a **0-1 Multiple-Choice Knapsack Problem** maximizing 10-year Net Present Value (`NPV` of user `VOC` savings minus agency capital cost) subject to an annual `NPR` budget ceiling and a mandatory DoR safety trigger ($\text{IRI}_0 \ge 4.5\text{ m/km}$).

---

## 📊 Key Quantitative Results (MILP vs. Worst-First Benchmark)

| Traffic Regime | Annual Budget (`MNPR`) | MILP Spent (`MNPR`) | MILP 10-Yr NPV (`MNPR`) | Worst-First NPV (`MNPR`) | Optimization Gain (`MNPR`) | Yr-10 Mean `IRI` (`m/km`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Freight** | 250.00 | 249.90 | 8,164.87 | 7,602.94 | **+561.92** | 6.49 |
| **Baseline Freight** | 450.00 | 450.00 | 13,649.71 | 13,163.67 | **+486.04** | 5.50 |
| **Baseline Freight** | 650.00 | 649.65 | 18,719.02 | 18,452.06 | **+266.96** | 4.55 |
| **Baseline Freight** | 1,000.00 | 999.65 | 21,670.79 | 21,341.44 | **+329.35** | 3.77 |
| **Hydropower Heavy-Haul** | 250.00 | 249.90 | 8,869.55 | 8,621.53 | **+248.02** | 6.81 |
| **Hydropower Heavy-Haul** | 450.00 | 450.00 | 15,340.06 | 15,042.74 | **+297.31** | 5.91 |
| **Hydropower Heavy-Haul** | 650.00 | 649.65 | 21,435.92 | 21,167.47 | **+268.44** | 4.98 |
| **Hydropower Heavy-Haul** | 1,000.00 | 999.65 | 25,082.60 | 24,623.98 | **+458.62** | 4.14 |

> **Key Takeaway:** At a realistic **NPR 450 Million** annual budget, mathematical optimization unlocks **NPR 486.04 Million** in additional 10-year road-user savings compared to Worst-First allocation without spending a single extra rupee.

---

## 📁 Repository Structure

```text
├── app.py                            # Interactive Streamlit web dashboard & HiGHS MILP solver
├── requirements.txt                  # Python dependencies for Streamlit Community Cloud
├── prithvi_corridor_wgs84.geojson    # 348 segmented 500 m blocks with GEE & DoR traffic attributes
├── prithvi_hdm4_phase3.parquet       # 2,784 pre-computed 10-year HDM-4 deterioration & NPR NPV trajectories
└── README.md                         # Project documentation
