import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# =============================
# CONFIG
# =============================
DEFAULT_VEHICLE_CAPACITY = 4500
SERVICE_TIME_MIN = 10
PREP_TIME_MIN = 45
MAX_TRIPS_PER_VEHICLE = 3
HORIZON_MINUTES = 24 * 60
WEEKDAY_FORECAST_WEIGHT = 0.2

# =============================
# HELPER FUNCTIONS
# =============================
def clean_id(val):
    """Normalize IDs for matching."""
    return str(val).strip().upper()

def find_col(df, keywords):
    """Find column in df whose name matches keywords (case/space insensitive)."""
    normalized_keywords = [re.sub(r'[^a-z0-9]', '', k.lower()) for k in keywords]
    for col in df.columns:
        norm_col = re.sub(r'[^a-z0-9]', '', col.lower())
        if any(k in norm_col for k in normalized_keywords):
            return col
    return None

def build_bidirectional_dict(df, from_col, to_col, dist_col):
    """Builds a distance/duration dict with both directions filled."""
    d = {}
    for _, r in df.iterrows():
        a, b = clean_id(r[from_col]), clean_id(r[to_col])
        dist = pd.to_numeric(r[dist_col], errors='coerce')
        if pd.notna(dist):
            d[(a, b)] = float(dist)
            d[(b, a)] = float(dist)  # Ensure both directions exist
    return d

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df_st = pd.read_csv("station_wise_unified.csv")
    df_routes = pd.read_csv("luag_dbs_to_mgs_routes.csv")
    try:
        df_alloc = pd.read_csv("luag_lcv_allocation_to_dbs_request.csv")
    except FileNotFoundError:
        df_alloc = None
    return df_st, df_routes, df_alloc

# =============================
# VRP SOLVER FUNCTION
# =============================
def run_vrp_with_prep(df_st, df_routes, df_alloc):
    # Detect columns
    station_id_col = find_col(df_st, ["dbs", "station", "station_id"])
    demand_col = find_col(df_st, ["demand", "quantity", "order", "sales", "cng"])
    rop_col = find_col(df_st, ["rop", "reorder"])
    time_col = find_col(df_st, ["time", "transaction", "datetime", "date"])
    time_weight_col = find_col(df_st, ["time_weight"])
    weekday_forecast_col = find_col(df_st, ["weekday_forecast", "forecast"])

    route_from_col = find_col(df_routes, ["from", "from_station", "mgs"])
    route_to_col   = find_col(df_routes, ["to", "to_station", "dbs"])
    dist_col       = find_col(df_routes, ["distance"])
    dur_col        = find_col(df_routes, ["duration", "travel_time"])

    # Filter latest fortnight
    if time_col:
        df_st[time_col] = pd.to_datetime(df_st[time_col], errors='coerce')
        latest = df_st[time_col].max()
        cutoff = latest - pd.Timedelta(days=15)
        df_st = df_st[df_st[time_col] >= cutoff]

    # Prepare demand with clean IDs
    df_st['station_clean'] = df_st[station_id_col].apply(clean_id)
    df_st['demand_val'] = pd.to_numeric(df_st.get(demand_col), errors='coerce').fillna(0.0)
    if rop_col:
        df_st['rop_flag'] = df_st.get(rop_col).astype(bool)
    else:
        df_st['rop_flag'] = False

    customers_df = df_st[df_st['demand_val'] > 0]
    customers = customers_df['station_clean'].unique().tolist()

    # Depots
    depots = df_routes[route_from_col].apply(clean_id).unique().tolist()
    if not depots:
        depots = ["MGS1"]

    # Distances & Durations
    dist_dict = build_bidirectional_dict(df_routes, route_from_col, route_to_col, dist_col) if dist_col else {}
    dur_dict  = build_bidirectional_dict(df_routes, route_from_col, route_to_col, dur_col) if dur_col else {}

    avg_dist = np.mean(list(dist_dict.values())) if dist_dict else 50

    # Nodes
    node_list = list(depots) + list(customers)
    node_index = {loc: i for i, loc in enumerate(node_list)}
    n_nodes = len(node_list)

    # Build distance matrix
    big = int(1e6)
    distance_matrix = np.full((n_nodes, n_nodes), big, dtype=int)
    for i, a in enumerate(node_list):
        for j, b in enumerate(node_list):
            if a == b:
                distance_matrix[i, j] = 0
            else:
                if (a, b) in dur_dict:
                    distance_matrix[i, j] = int(round(dur_dict[(a, b)]))
                elif (a, b) in dist_dict:
                    distance_matrix[i, j] = int(round(dist_dict[(a, b)] * 2))
                else:
                    distance_matrix[i, j] = int(round(avg_dist * 2))

    # Demands
    demands = []
    for loc in node_list:
        if loc in depots:
            demands.append(0)
        else:
            row = customers_df[customers_df['station_clean'] == loc]
            base = float(row['demand_val'].sum())
            time_w = float(row.get(time_weight_col, pd.Series([1.0])).values[0]) if time_weight_col else 1.0
            wf = float(row.get(weekday_forecast_col, pd.Series([0])).values[0]) if weekday_forecast_col else 0.0
            demands.append(int(math.ceil(base * time_w + WEEKDAY_FORECAST_WEIGHT * wf)))

    # Vehicles
    vehicles_physical = []
    if df_alloc is not None:
        veh_col = find_col(df_alloc, ["lcv", "vehicle"])
        start_col = find_col(df_alloc, ["mgs", "from", "depot"])
        stage_col_alloc = find_col(df_alloc, ["stage"])
        if veh_col:
            vids = df_alloc[veh_col].astype(str).unique().tolist()
            for vid in vids:
                vehicles_physical.append({
                    "id": vid,
                    "start": depots[0],
                    "capacity": DEFAULT_VEHICLE_CAPACITY,
                    "stage": 2
                })
    else:
        for i, d in enumerate(depots):
            vehicles_physical.append({"id": f"VEH_{i+1}", "start": d, "capacity": DEFAULT_VEHICLE_CAPACITY, "stage": 2})

    vehicles_physical = [v for v in vehicles_physical if v.get('stage', 2) == 2]

    vehicles = []
    for v in vehicles_physical:
        for k in range(MAX_TRIPS_PER_VEHICLE):
            vehicles.append({
                "phys_id": v['id'],
                "copy_idx": k,
                "start": v['start'],
                "capacity": v['capacity']
            })

    starts = [node_index[clean_id(v['start'])] for v in vehicles]
    ends = starts[:]

    # OR-Tools setup
    manager = pywrapcp.RoutingIndexManager(n_nodes, len(vehicles), starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index, to_index):
        return int(distance_matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])
    transit_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_cb(from_index):
        return int(demands[manager.IndexToNode(from_index)])
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [v['capacity'] for v in vehicles], True, "Capacity")

    def time_cb(from_index, to_index):
        travel = int(distance_matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])
        return travel + (SERVICE_TIME_MIN if manager.IndexToNode(to_index) >= len(depots) else 0)
    time_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(time_idx, 60, HORIZON_MINUTES, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    for v_idx, veh in enumerate(vehicles):
        start_node = routing.Start(v_idx)
        time_dim.CumulVar(start_node).SetRange(int(veh['copy_idx'] * PREP_TIME_MIN), HORIZON_MINUTES)

    for node in range(len(depots), n_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], 100000)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(params)

    rows = []
    if solution:
        for v_idx, veh in enumerate(vehicles):
            index = routing.Start(v_idx)
            if routing.IsEnd(index):
                continue
            route_nodes = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_nodes.append(node_list[node])
                index = solution.Value(routing.NextVar(index))
            route_nodes.append(node_list[manager.IndexToNode(index)])

            # Compute actual totals from dist_dict & demand mapping
            total_distance = 0
            total_load = 0
            for i in range(len(route_nodes) - 1):
                a, b = clean_id(route_nodes[i]), clean_id(route_nodes[i+1])
                total_distance += dist_dict.get((a, b), avg_dist)
                if b not in depots:
                    total_load += demands[node_index[b]]

            rows.append({
                "Physical_Vehicle": veh['phys_id'],
                "Trip_Copy": veh['copy_idx'] + 1,
                "Route": " -> ".join(route_nodes),
                "Distance_min": round(total_distance, 2),
                "Load_units": total_load
            })
    return pd.DataFrame(rows)

# =============================
# STREAMLIT UI
# =============================
st.title("🚛 CNG LCV Scheduling Optimization - Fixed Distance & Load Units")

df_st, df_routes, df_alloc = load_data()

if st.button("Run Optimization"):
    result_df = run_vrp_with_prep(df_st, df_routes, df_alloc)
    st.dataframe(result_df)
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Optimized Routes CSV", csv, "optimized_lcv_routes.csv", "text/csv")
