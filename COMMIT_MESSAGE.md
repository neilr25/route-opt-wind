feat: wind-assisted shipping route optimiser v1.0

Implements corridor-based A* search around ATOBVIAC trade routes using
real Hadnymax-2FR35 polar data and ERA5 gridded weather.

Key features:
- ATOBVIAC baseline + corridor mesh (25nm lateral, 200nm stages)
- DuckDB-backed Parquet weather lookup (0.25deg ERA5)
- Centre-lane maritime guarantee + hard land blocks on offset lanes
- FastAPI dashboard with interactive Leaflet map
- 365-day batch runner outputs Parquet results

Repo includes 2025 annual analysis (Rotterdam->New York):
- Mean optimisation savings: 18.5t (10.0%)
- Best day: 61.2t saved (2025-10-03)
