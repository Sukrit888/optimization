import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import timedelta
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# =============================
# CONFIG
# =============================
DEFAULT_VEHICLE_CAPACITY = 4500
SERVICE_TIME_MIN = 10
PREP_TIME_MIN = 45
MAX_TRIPS_PER_VEHICLE = 3
TIME_WINDOW_SLACK = 120
HORIZON_MINUTES = 24 * 60
WEEKDAY_FORECAST_WEIGHT = 0.2

# =============================
# HELPER FUNCTIONS
# =============================
def find_col(df, keywords):
    keys = [k.lower() for k in keywords]
    for c in df.columns:
        if any(k in c.lower() for k in keys):
            return c
    return None

def symmetrize_distance(df, from_col, to_col, dist_col):
    d = {}
    for _, r in df.iterrows():
        a, b, dist = r[from_col], r[to_col], r[dist_col]
        if pd.isna(a) or pd.isna(b) or pd.isna(dist):
            continue
        d[(str(a), str(b))] = float(dist)
        d[(str(b), str(a))] = float(dist)
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
    station_id_col = find_col(df_st, ["dbs", "station", "station_id"])
    demand_col = find_col(df_st, ["demand", "quantity"])
    rop_col = find_col(df_st, ["rop", "reorder"])
    time_col = find_col(df_st, ["time", "transaction", "datetime"])
    time_weight_col = find_col(df_st, ["time_weight"])
    weekday_forecast_col = find_col(df_st, ["weekday_forecast", "forecast"])

    route_from_col = find_col(df_routes, ["from", "from_station", "mgs"])
    route_to_col   = find_col(df_routes, ["to", "to_station", "dbs"])
    dist_col       = find_col(df_routes, ["distance"])
    dur_col        = find_col(df_routes, ["duration", "travel_time"])

    # Fortnight filter
    if time_col:
        df_st[time_col] = pd.to_datetime(df_st[time_col], errors='coerce')
        latest = df_st[time_col].max()
        cutoff = latest - pd.Timedelta(days=15)
        df_st = df_st[df_st[time_col] >= cutoff]

    # Prepare customers
    df_st['demand_val'] = pd.to_numeric(df_st.get(demand_col), errors='coerce').fillna(0.0)
    if rop_col:
        df_st['rop_flag'] = df_st.get(rop_col).astype(bool)
    else:
        df_st['rop_flag'] = False

    customers_df = df_st[(df_st['demand_val'] > 0)]
    customers = customers_df[station_id_col].astype(str).unique().tolist()

    # Depots
    depots = df_routes[route_from_col].astype(str).unique().tolist()
    if not depots:
        depots = ["MGS1"]

    # Distances
    dist_dict = symmetrize_distance(df_routes, route_from_col, route_to_col, dist_col) if dist_col else {}
    dur_dict = symmetrize_distance(df_routes, route_from_col, route_to_col, dur_col) if dur_col else {}

    node_list = list(depots) + list(customers)
    node_index = {loc: i for i, loc in enumerate(node_list)}
    n_nodes = len(node_list)

    big = int(1e6)
    distance_matrix = np.full((n_nodes, n_nodes), big, dtype=int)
    for i, a in enumerate(node_list):
        for j, b in enumerate(node_list):
            if a == b:
                distance_matrix[i, j] = 0
            else:
                k = (str(a), str(b))
                if k in dur_dict:
                    distance_matrix[i, j] = int(round(dur_dict[k]))
                elif k in dist_dict:
                    distance_matrix[i, j] = int(round(dist_dict[k] * 2))
                else:
                    distance_matrix[i, j] = big

    demands = []
    for loc in node_list:
        if loc in depots:
            demands.append(0)
        else:
            row = customers_df[customers_df[station_id_col].astype(str) == str(loc)]
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

    starts = [node_index[v['start']] for v in vehicles]
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
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route_nodes.append(node_list[node])
                prev = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev, index, v_idx)
                route_load += demands[manager.IndexToNode(prev)]
            route_nodes.append(node_list[manager.IndexToNode(index)])
            rows.append({
                "Physical_Vehicle": veh['phys_id'],
                "Trip_Copy": veh['copy_idx'] + 1,
                "Route": " -> ".join(route_nodes),
                "Distance_min": route_distance,
                "Load_units": route_load
            })
    return pd.DataFrame(rows)

# =============================
# STREAMLIT UI
# =============================
st.title("🚛 CNG LCV Scheduling Optimization with 45-min Prep Time")

df_st, df_routes, df_alloc = load_data()

if st.button("Run Optimization"):
    result_df = run_vrp_with_prep(df_st, df_routes, df_alloc)
    st.dataframe(result_df)
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Optimized Routes CSV", csv, "optimized_lcv_routes_with_prep.csv", "text/csv")
