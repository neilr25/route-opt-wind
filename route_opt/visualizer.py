"""Plotly map + savings table visualiser."""

from pathlib import Path
from typing import List, Tuple, Optional

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_routes(
    baseline: List[Tuple[float, float]],
    optimized: List[Tuple[float, float]],
    title: str = "Route Optimisation",
    out_html: Path | None = None,
    G: Optional[nx.DiGraph] = None,
) -> go.Figure:
    """Plot baseline and optimised routes on a map."""
    # DataFrames for Plotly
    df_base = pd.DataFrame(baseline, columns=["lat", "lon"])
    df_base["type"] = "Great-circle / standard"
    df_opt = pd.DataFrame(optimized, columns=["lat", "lon"])
    df_opt["type"] = "Optimised"
    df = pd.concat([df_base, df_opt], ignore_index=True)

    fig = px.line_geo(
        df,
        lat="lat",
        lon="lon",
        color="type",
        color_discrete_map={"Great-circle / standard": "#EF553B", "Optimised": "#00CC96"},
        projection="natural earth",
        title=title,
    )

    # --- Overlay mesh if provided ---
    if G is not None:
        # Nodes
        node_lats, node_lons, node_colors, node_sizes = [], [], [], []
        for n, data in G.nodes(data=True):
            node_lats.append(data["lat"])
            node_lons.append(data["lon"])
            if data.get("is_land", False):
                node_colors.append("red")
                node_sizes.append(6)
            else:
                node_colors.append("grey")
                node_sizes.append(3)

        fig.add_trace(
            go.Scattergeo(
                lat=node_lats,
                lon=node_lons,
                mode="markers",
                marker=dict(color=node_colors, size=node_sizes, opacity=0.6),
                name="Mesh nodes",
                hovertemplate="lat=%{lat:.3f}<br>lon=%{lon:.3f}<extra></extra>",
            )
        )

        # Edges
        land_edge_lats, land_edge_lons = [], []
        sea_edge_lats, sea_edge_lons = [], []
        for u, v, data in G.edges(data=True):
            u_lat, u_lon = G.nodes[u]["lat"], G.nodes[u]["lon"]
            v_lat, v_lon = G.nodes[v]["lat"], G.nodes[v]["lon"]
            if data.get("crosses_land", False):
                land_edge_lats.extend([u_lat, v_lat, None])
                land_edge_lons.extend([u_lon, v_lon, None])
            else:
                sea_edge_lats.extend([u_lat, v_lat, None])
                sea_edge_lons.extend([u_lon, v_lon, None])

        if sea_edge_lats:
            fig.add_trace(
                go.Scattergeo(
                    lat=sea_edge_lats,
                    lon=sea_edge_lons,
                    mode="lines",
                    line=dict(color="lightgrey", width=0.5),
                    opacity=0.4,
                    name="Mesh edges (sea)",
                    hoverinfo="skip",
                )
            )
        if land_edge_lats:
            fig.add_trace(
                go.Scattergeo(
                    lat=land_edge_lats,
                    lon=land_edge_lons,
                    mode="lines",
                    line=dict(color="red", width=1.5, dash="dot"),
                    opacity=0.7,
                    name="Mesh edges (land crossing)",
                    hoverinfo="skip",
                )
            )

    fig.update_geos(
        showcoastlines=True, coastlinecolor="RebeccaPurple",
        showland=True, landcolor="LightGreen",
        showocean=True, oceancolor="LightBlue",
    )
    fig.update_layout(height=700, margin={"r": 0, "t": 40, "l": 0, "b": 0})

    if out_html:
        fig.write_html(str(out_html))
    return fig


def print_savings(cost_standard_no_wind: float, cost_standard_wind: float, cost_optimised: float) -> None:
    """Print three-way comparison to console."""
    print("\n=== Route Comparison ===")
    print(f"{'Metric':<32} {'Value':>12}")
    print(f"{'Standard route (no wind)':<32} {cost_standard_no_wind:>12.2f} t")
    print(f"{'Standard route (with wind)':<32} {cost_standard_wind:>12.2f} t")
    print(f"{'Optimised route (with wind)':<32} {cost_optimised:>12.2f} t")
    saved_vs_standard = cost_standard_no_wind - cost_standard_wind
    pct_vs_standard = (saved_vs_standard / cost_standard_no_wind * 100) if cost_standard_no_wind > 0 else 0
    saved_vs_opt = cost_standard_no_wind - cost_optimised
    pct_vs_opt = (saved_vs_opt / cost_standard_no_wind * 100) if cost_standard_no_wind > 0 else 0
    print(f"{'Wind assist on standard':<32} {saved_vs_standard:>12.2f} t ({pct_vs_standard:.1f}%)")
    print(f"{'Optimisation total savings':<32} {saved_vs_opt:>12.2f} t ({pct_vs_opt:.1f}%)")
    print("========================\n")
