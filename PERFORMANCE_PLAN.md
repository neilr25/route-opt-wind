# Performance Optimisation Plan

**Goal:** Run 365-day batch in <3 min without reducing mesh scope (keep 25nm lanes, 200nm stages, 17 lanes).

**Current state (per date):**
| Step | Time | bottleneck |
|------|------|------------|
| Mesh build | 5s | Once only (cached) |
| Weather fetch (252 nodes) | 2.0s | DuckDB + cKDTree per date |
| A* search (252 nodes, 1039 edges) | 2.5s | Python loops + polar lookups |
| **Total per date** | **~4.5s** | |
| **365-day batch** | **~27 min** | |

---

## Phase 1 – Weather Pre-Loading (biggest win, ~15x)

**Problem:** Every date opens DuckDB, filters by date, samples 252 nodes via cKDTree.

**Solution:** Load all weather once, build an O(1) lookup table.

1. **Pre-compute ERA5 grid mapping**
   - ERA5 is a regular 0.25° lat/lon grid.
   - Map each mesh node to its grid cell `( grid_lat, grid_lon )` once using simple integer division.
   - Store mapping: `{node_key: (lat_idx, lon_idx)}`.

2. **Load full year into memory**
   - Stack all 12 monthly Parquets into one in-memory DuckDB/Arrow table.
   - Group into a 3D array: `(time, lat_idx, lon_idx) → (ws, wd)`.
   - ~25 MB compressed → ~200 MB in RAM for a full year.

3. **Per-date lookup**
   - Instead of 252 cKDTree queries, do one vectorised slice: `weather[date_idx, lat_indices, lon_indices]`.
   - Time: **2.0s → 0.02s per date**.

**Impact on batch:** 365 × 2.0s = 730s → 365 × 0.02s = 7s (**~100x** on weather alone).

---

## Phase 2 – Parallel A* Search (~8x)

**Problem:** Each date runs sequentially in a single Python process.

**Solution:** Multiprocessing with immutable shared state.

1. **Share the mesh across workers**
   - Build mesh once in parent process.
   - Pass `nx.DiGraph` to children via `multiprocessing.Manager()` or serialise + fork.

2. **One process per month**
   - 12 workers (or 1 per CPU core), each handles ~30 days.
   - Worker receives mesh + pre-loaded weather slice for its month.

3. **ProcessPoolExecutor wrapper**
   ```python
   with ProcessPoolExecutor(max_workers=12) as pool:
       results = pool.map(optimize_day, date_chunks)
   ```

**Impact on batch:** Single-threaded 27 min → ~3.5 min on 8 cores (**~8x**).

---

## Phase 3 – Vectorised Edge Costs (~5x)

**Problem:** `fuel_with_wind()` and `edge_cost()` are called inside a Python loop for every edge (up to 1039 per date).

**Solution:** Pre-compute all edge costs for a stage as vectorised numpy operations.

1. **Pre-compute edge attributes as arrays**
   - Extract all edge bearings, distances, destination nodes into numpy arrays.

2. **Vectorised TWA + polar lookup**
   - Compute TWA for all edges: `np.abs((wd - bearing + 180) % 360 - 180)`.
   - Vectorise bilinear polar interpolation using `scipy.interpolate.RegularGridInterpolator`.
   
3. **A* uses pre-computed costs**
   - Instead of calling `edge_cost()`, A* looks up `cost = edge_cost_array[edge_id]`.
   
**Impact:** Python loop overhead drops; A* time **2.5s → 0.5s per date** (~5x).

---

## Phase 4 – Cython/Numba for Hot Paths (~3x)

**Problem:** `_haversine_nm`, `_bearing`, `_twa` are pure Python functions called thousands of times.

**Solution:** JIT compile with `numba.njit`.

1. Wrap `_haversine_nm`, `_bearing`, `_twa` in `@njit`.
2. Pre-compute heuristic cache with numba-accelerated loop.
3. Numba’s `prange` for parallelising batch date loops if Phase 2 isn’t enough.

**Impact:** A* search **0.5s → 0.15s per date** (~3x).

---

## Combined Estimate

| Phase | Speedup | Batch Time |
|-------|---------|------------|
| Baseline | 1× | 27 min |
| Phase 1 (weather) | 15× | 1.8 min |
| Phase 2 (parallel) | 8× | 0.9 min |
| Phase 3 (vectorised) | 5× | 0.18 min |
| Phase 4 (numba) | 3× | **0.06 min (3.6 s)** |

**Conservative realistic target:** **2–3 minutes** for all 365 days (applying Phases 1+2+3; Phase 4 is bonus).

---

## Implementation Order

1. **Phase 1** – biggest return, minimal code change (~2 hrs work)
2. **Phase 2** – wrap existing `batch_run_2025.py` in `ProcessPoolExecutor` (~1 hr)
3. **Phase 3** – refactor A* loop to use pre-computed arrays (~2–3 hrs)
4. **Phase 4** – add `numba` dependency, decorate hot functions (~1 hr)

## What Does NOT Change

- Mesh resolution stays 25nm / 200nm stages / 17 lanes.
- A* algorithm unchanged (just faster lookups).
- Output format unchanged (same Parquet schema).
- Dashboard unchanged (API layer unaffected).

## Files to Modify

| File | Change |
|------|--------|
| `weather_client.py` | Add `load_year()` + `lookup_day()` vectorised |
| `batch_run_2025.py` | Wrap in `ProcessPoolExecutor` |
| `optimizer.py` | Accept pre-computed `edge_cost_array` |
| `mesh.py` | Export `edge_bearings`, `edge_distances` arrays |
| `cost_engine.py` | Optional numba JIT decorators |
| `requirements.txt` | Add `numba` |
