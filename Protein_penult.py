import streamlit as st
import os
from Bio.PDB import PDBParser, PDBList
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import tempfile
from scipy.spatial.distance import cdist
from collections import defaultdict
import base64
from PIL import Image

# Get the absolute path of the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(current_dir, "icon.png")

# Load the image only if it exists, otherwise fallback to emoji
if os.path.exists(icon_path):
    im = Image.open(icon_path)
    st.set_page_config(page_title="Protein Contact Network Explorer", layout="wide", page_icon=im)
else:
    st.set_page_config(page_title="Protein Contact Network Explorer", layout="wide", page_icon="🧬")

# -----------------------------------------------------------------------------
# CUSTOM CSS (Optimized for Readability & Professional UI)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global App Background */
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%); }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* Standard Section Box */
    .section-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
        margin-bottom: 15px;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* --- UPLOADER STYLING --- */
    [data-testid="stFileUploader"] {
        margin: 0 auto;
        width: 90%; 
        margin-top: 10px; 
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent; 
        border: none;
        padding: 10px 0;
    }

    /* --- GENERAL UI --- */
    .stButton>button {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(42, 82, 152, 0.4);
    }
    .hub-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin-bottom: 10px;
        border-left: 4px solid #2a5298;
        transition: all 0.2s ease;
    }
    .hub-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateX(5px);
    }
    .metric-box {
        background: #f0f4f8;
        padding: 12px;
        border-radius: 6px;
        text-align: center;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] { justify-content: flex-end; gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 0px 20px;
        border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        border: 2px solid #1e3c72;
        color: #1e3c72;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS & DATA
# -----------------------------------------------------------------------------

aa_1_letter = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "UNK": "X"
}

res_type_map = {
    "ALA": "hydrophobic", "VAL": "hydrophobic", "ILE": "hydrophobic",
    "LEU": "hydrophobic", "MET": "hydrophobic", "PHE": "hydrophobic",
    "TYR": "hydrophobic", "TRP": "hydrophobic", "PRO": "hydrophobic",
    "GLY": "polar", "SER": "polar", "THR": "polar", "CYS": "polar",
    "ASN": "polar", "GLN": "polar",
    "ASP": "negative", "GLU": "negative",
    "LYS": "positive", "ARG": "positive", "HIS": "positive"
}
res_type_map["UNK"] = "polar"

residue_type_colors = {
    "hydrophobic": "#D94E1E",
    "polar": "#003B6F",
    "positive": "#007A55",
    "negative": "#B32630",
}
default_color = "#8da0cb"


def compute_enhanced_metrics(adj_np, dist_matrix):
    N = len(adj_np)
    degrees = np.sum(adj_np, axis=0)
    
    betweenness = np.zeros(N)
    # Simplified calculation for performance
    for i in range(N):
        for j in range(N):
            if i != j and adj_np[i, j] == 1:
                betweenness[i] += 1
    betweenness = betweenness / max(betweenness.max(), 1)
    
    closeness = np.zeros(N)
    for i in range(N):
        distances = dist_matrix[i]
        valid_distances = distances[distances > 0]
        if len(valid_distances) > 0:
            closeness[i] = len(valid_distances) / np.sum(valid_distances)
    
    clustering = np.zeros(N)
    for i in range(N):
        neighbors = np.where(adj_np[i] == 1)[0]
        k = len(neighbors)
        if k >= 2:
            subgraph = adj_np[np.ix_(neighbors, neighbors)]
            actual_edges = np.sum(subgraph) / 2
            possible_edges = k * (k - 1) / 2
            clustering[i] = actual_edges / possible_edges
    
    return {
        'degree': degrees,
        'betweenness': betweenness,
        'closeness': closeness,
        'clustering': clustering
    }


def identify_hub_communities(adj_np, hub_indices, residues):
    communities = {}
    for hub_idx in hub_indices:
        neighbors = np.where(adj_np[hub_idx] == 1)[0]
        res_types = defaultdict(int)
        for n_idx in neighbors:
            res_name = residues[n_idx].get_resname().strip().upper()
            res_type = res_type_map.get(res_name, "polar")
            res_types[res_type] += 1
        
        communities[hub_idx] = {
            'size': len(neighbors),
            'composition': dict(res_types),
            'neighbors': neighbors.tolist()
        }
    return communities


def compute_pcn_df(structure, model_id, chain_id, threshold, progress=None, progress_label=None):
    model = structure[int(model_id) - 1]
    chain = model[chain_id]

    residues = []
    coords = []

    for i, res in enumerate(chain.get_residues()):
        if "CA" in res:
            residues.append(res)
            coords.append(res["CA"].get_coord())
        if progress is not None and i % 5 == 0:
            val = min(20 + (len(residues) % 100), 50)
            progress.progress(val)
            if progress_label is not None:
                progress_label.text(f"{val}% complete — scanning residues")

    if len(coords) == 0:
        return pd.DataFrame(), pd.DataFrame(), [], np.empty((0, 3)), [], {}

    coords = np.vstack(coords)
    if progress is not None:
        progress.progress(60)
        if progress_label is not None:
            progress_label.text("60% complete — building coordinate matrix")
    
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
    adjacency = (dist_matrix <= threshold).astype(int)
    
    if progress is not None:
        progress.progress(90)
        if progress_label is not None:
            progress_label.text("90% complete — computing adjacency")
    
    np.fill_diagonal(adjacency, 0)
    
    labels = [f"{res.get_resname().strip()}-{res.get_id()[1]}" for res in residues]
    
    vis_labels = []
    for res in residues:
        rname = res.get_resname().strip().upper()
        rcode = aa_1_letter.get(rname, "X")
        vis_labels.append(f"{rcode}-{res.get_id()[1]}")
    
    metrics = compute_enhanced_metrics(adjacency, dist_matrix)

    if progress is not None:
        progress.progress(100)
        if progress_label is not None:
            progress_label.text("100% complete — done computing PCN")

    return (
        pd.DataFrame(adjacency, index=labels, columns=labels),
        pd.DataFrame(dist_matrix, index=labels, columns=labels),
        labels,
        vis_labels, 
        coords,
        residues,
        metrics
    )


def render_distance_heatmap(dist_df):
    """
    Renders an interactive, strictly square heatmap for the distance matrix.
    Color Scheme: Fluorescent Red (Close) -> Yellow -> Green -> Cyan -> Blue (Far).
    """
    # Extract residue numbers for cleaner axes
    try:
        axis_labels = [int(label.split('-')[-1]) for label in dist_df.index]
    except:
        axis_labels = dist_df.index

    # 5-Step Fluorescent Scale to ensure vibrant colors throughout
    custom_colorscale = [
        [0.00, 'rgb(255, 0, 0)'],    # Red (Close)
        [0.25, 'rgb(255, 255, 0)'],  # Yellow (Bright)
        [0.50, 'rgb(0, 255, 0)'],    # Green (Fluorescent)
        [0.75, 'rgb(0, 255, 255)'],  # Cyan (Bright)
        [1.00, 'rgb(0, 0, 255)']     # Blue (Far)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=dist_df.values,
        x=axis_labels,
        y=axis_labels,
        colorscale=custom_colorscale,
        colorbar=dict(
            title='Distance (Å)',
            tickmode='auto', 
            nticks=6
        ),
        hovertemplate=(
            '<b>Residue i:</b> %{y}<br>'
            '<b>Residue j:</b> %{x}<br>'
            '<b>Distance:</b> %{z:.2f} Å<extra></extra>'
        )
    ))

    fig.update_layout(
        title={
            'text': 'Pairwise Distance Matrix Heatmap',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Residue Number",
        yaxis_title="Residue Number",
        width=700,
        height=700,
        autosize=False,
        # Force strict square aspect ratio
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        yaxis=dict(
            autorange='reversed', # Index 0 at top
            scaleanchor='x',      
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_adjacency_heatmap(adj_df):
    """
    Renders a strictly square binary heatmap for the adjacency matrix.
    Black = Connected, White = Not Connected.
    """
    # Extract residue numbers
    try:
        axis_labels = [int(label.split('-')[-1]) for label in adj_df.index]
    except:
        axis_labels = adj_df.index

    z_vals = adj_df.values
    hover_text = np.where(z_vals == 1, "Connected", "Not Connected")

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=axis_labels,
        y=axis_labels,
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False, # No colorbar needed for binary
        xgap=0.5, # Small gaps to visualize grid
        ygap=0.5,
        customdata=hover_text,
        hovertemplate=(
            '<b>Residue i:</b> %{y}<br>'
            '<b>Residue j:</b> %{x}<br>'
            '<b>Status:</b> %{customdata}<extra></extra>'
        )
    ))

    fig.update_layout(
        title={
            'text': 'Binary Adjacency Map',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Residue Number",
        yaxis_title="Residue Number",
        width=700,
        height=700,
        autosize=False,
        # Force strict square aspect ratio
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        yaxis=dict(
            autorange='reversed',
            scaleanchor='x',
            scaleratio=1,
            constrain='domain',
            showgrid=False
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def build_3d_figure_enhanced(labels, vis_labels, coords, adj_np, dist_matrix, node_colors, node_sizes, 
                            residues, hub_indices_global, metrics, highlight_communities=False, view_mode="Show All"):
    N = len(labels)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    hover_texts = []
    for i in range(N):
        res = residues[i]
        name = res.get_resname().strip().upper()
        if name not in res_type_map: name = "UNK"
        hover_text = (
            f"<b>{labels[i]}</b><br>" 
            f"Type: {res_type_map[name].capitalize()}<br>"
            f"Degree: {int(metrics['degree'][i])}<br>"
            f"Clustering: {metrics['clustering'][i]:.3f}<br>"
            f"Closeness: {metrics['closeness'][i]:.3f}"
        )
        hover_texts.append(hover_text)

    traces = []
    norm_edge_x, norm_edge_y, norm_edge_z = [], [], []
    hub_edge_x, hub_edge_y, hub_edge_z = [], [], []
    
    for i in range(N):
        for j in range(i + 1, N):
            if adj_np[i, j] == 1:
                # Visibility check
                if node_sizes[i] == 0 or node_sizes[j] == 0:
                    norm_edge_x.extend([coords[i][0], coords[j][0], None])
                    norm_edge_y.extend([coords[i][1], coords[j][1], None])
                    norm_edge_z.extend([coords[i][2], coords[j][2], None])
                else:
                    is_hub_edge = highlight_communities and (i in hub_indices_global or j in hub_indices_global)
                    if is_hub_edge:
                        hub_edge_x.extend([coords[i][0], coords[j][0], None])
                        hub_edge_y.extend([coords[i][1], coords[j][1], None])
                        hub_edge_z.extend([coords[i][2], coords[j][2], None])
                    else:
                        norm_edge_x.extend([coords[i][0], coords[j][0], None])
                        norm_edge_y.extend([coords[i][1], coords[j][1], None])
                        norm_edge_z.extend([coords[i][2], coords[j][2], None])

    traces.append(go.Scatter3d(
        x=norm_edge_x if norm_edge_x else [], y=norm_edge_y if norm_edge_y else [], z=norm_edge_z if norm_edge_z else [],
        mode="lines",
        line=dict(color='rgba(0,0,0,0.2)', width=2),
        hoverinfo="none",
        showlegend=False
    ))

    traces.append(go.Scatter3d(
        x=hub_edge_x if hub_edge_x else [], y=hub_edge_y if hub_edge_y else [], z=hub_edge_z if hub_edge_z else [],
        mode="lines",
        line=dict(color='rgba(220, 50, 50, 0.6)', width=5),
        hoverinfo="none",
        showlegend=False
    ))

    node_trace_idx = 2
    traces.append(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text", 
            marker=dict(symbol='circle', size=node_sizes, color=node_colors, line=dict(width=1.5, color="white")),
            text=vis_labels, 
            textposition="middle center",
            textfont=dict(family="Arial", size=10, color="black", weight="bold"), 
            hovertext=hover_texts, 
            hoverinfo="text",
            showlegend=False,
        )
    )

    dummy_traces = []
    legend_items = [("Hydrophobic", "#D94E1E"), ("Polar", "#003B6F"), ("Positive", "#007A55"), ("Negative", "#B32630")]
    for name, color in legend_items:
        dummy_traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(size=15, color=color), name=name, showlegend=True))
    
    fig = go.Figure(data=traces + dummy_traces)

    updatemenus = []
    if view_mode == "Show Hubs Only" and len(hub_indices_global) > 0:
        buttons = [dict(
            label="Hub View",
            method="update", 
            args=[
                {"marker.color": [node_colors], "marker.size": [node_sizes]}, 
                {"scene.camera": None, "scene.annotations": []},
                [node_trace_idx] 
            ]
        )]
        
        sorted_hubs = sorted(hub_indices_global, key=lambda idx: metrics['degree'][idx], reverse=True)
        for idx in sorted_hubs:
            label = labels[idx] 
            degree = int(metrics['degree'][idx])
            cx, cy, cz = coords[idx][0], coords[idx][1], coords[idx][2]
            
            new_colors = list(node_colors)
            new_colors[idx] = "#FFFF00"
            new_sizes = list(node_sizes)
            if new_sizes[idx] > 0:
                new_sizes[idx] = new_sizes[idx] + 5 
            
            buttons.append(dict(
                label=f"Target: {label}",
                method="update",
                args=[
                    {"marker.color": [new_colors], "marker.size": [new_sizes]},
                    {
                        "scene.camera.center": {"x": 0, "y": 0, "z": 0}, 
                        "scene.camera.eye": {"x": cx/40 + 0.5, "y": cy/40 + 0.5, "z": cz/40 + 0.5},
                        "scene.annotations": [] 
                    },
                    [node_trace_idx]
                ]
            ))
            
        updatemenus = [dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0, xanchor="left",
            y=1.0, yanchor="top",
            bgcolor="white", bordercolor="#2a5298", borderwidth=2,
            pad={"r": 10, "t": 10},
            font=dict(size=12, color="black", family="Arial")
        )]

    fig.update_layout(
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        updatemenus=updatemenus,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#e0e0e0", borderwidth=1),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data", bgcolor="#f8f9fa", dragmode="orbit"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=900 
    )
    return fig


def render_hub_analysis_panel(labels, residues, metrics, hub_indices, communities, adj_np):
    # --- Custom Hub SVG Icon ---
    hub_icon_svg = """<svg width="80" height="80" viewBox="-20 -20 140 140" xmlns="http://www.w3.org/2000/svg" style="display: block;">
    <g stroke="#1e3c72" stroke-width="8" stroke-linecap="round">
        <line x1="50" y1="50" x2="50" y2="10" />
        <line x1="50" y1="50" x2="90" y2="50" />
        <line x1="50" y1="50" x2="50" y2="90" />
        <line x1="50" y1="50" x2="10" y2="50" />
        <line x1="50" y1="50" x2="22" y2="22" />
        <line x1="50" y1="50" x2="78" y2="22" />
        <line x1="50" y1="50" x2="78" y2="78" />
        <line x1="50" y1="50" x2="22" y2="78" />
    </g>
    <circle cx="50" cy="50" r="18" fill="#1e3c72" />
    <circle cx="50" cy="10" r="8" fill="#1e3c72" />
    <circle cx="90" cy="50" r="8" fill="#1e3c72" />
    <circle cx="50" cy="90" r="8" fill="#1e3c72" />
    <circle cx="10" cy="50" r="8" fill="#1e3c72" />
    <circle cx="22" cy="22" r="8" fill="#1e3c72" />
    <circle cx="78" cy="22" r="8" fill="#1e3c72" />
    <circle cx="78" cy="78" r="8" fill="#1e3c72" />
    <circle cx="22" cy="78" r="8" fill="#1e3c72" />
</svg>"""

    # --- Header Box ---
    header_html = f"""<div style="background: white; padding: 20px; border-radius: 12px; margin: 30px 0 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 6px solid #1e3c72; display: flex; align-items: center; gap: 25px;">
    <div style="flex-shrink: 0;">{hub_icon_svg}</div>
    <div><h3 style="margin: 0; color: #1e3c72; font-size: 38px !important; font-weight: 800 !important; line-height: 1.2;">Hub Analysis</h3></div>
</div>"""

    st.markdown(header_html, unsafe_allow_html=True)
    
    if len(hub_indices) == 0:
        st.info("No hubs detected with current criteria.")
        return
    
    sorted_hubs = sorted(hub_indices, key=lambda i: metrics['degree'][i], reverse=True)
    top_n = 7
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="metric-box"><div style="font-size: 24px; font-weight: 700; color: #2a5298;">{len(hub_indices)}</div><div style="font-size: 12px; color: #666;">Total Hubs</div></div>""", unsafe_allow_html=True)
    with col2:
        avg_degree = np.mean([metrics['degree'][i] for i in hub_indices])
        st.markdown(f"""<div class="metric-box"><div style="font-size: 24px; font-weight: 700; color: #2a5298;">{avg_degree:.1f}</div><div style="font-size: 12px; color: #666;">Avg Hub Degree</div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if len(sorted_hubs) > 0:
        csv_data = []
        for idx in sorted_hubs:
            label = labels[idx]
            res_name = residues[idx].get_resname().strip()
            res_type = res_type_map.get(res_name, "polar")
            csv_data.append({
                "Residue Label": label, "Residue Name": res_name, "Residue Type": res_type,
                "Degree": int(metrics['degree'][idx]), "Clustering Coeff": f"{metrics['clustering'][idx]:.4f}", "Closeness Centrality": f"{metrics['closeness'][idx]:.4f}"
            })
        
        df_hubs = pd.DataFrame(csv_data)
        csv_string = df_hubs.to_csv(index=False).encode('utf-8')
        
        if len(sorted_hubs) > top_n:
            st.info(f"Showing top {top_n} of {len(sorted_hubs)} hubs. Download CSV for full list.")
        
        st.download_button(label="📥 Download Full Hub Report (CSV)", data=csv_string, file_name="hub_analysis_report.csv", mime="text/csv")
    
    st.markdown("---")
    
    # --- DYNAMIC HEADER LOGIC ---
    if len(sorted_hubs) <= top_n:
        st.markdown("#### 🔍 Hub Details")
    else:
        st.markdown(f"#### 🔍 Hub Details (Top {top_n})")
    
    display_hubs = sorted_hubs[:top_n]
    for rank, idx in enumerate(display_hubs, 1):
        label = labels[idx]
        res_name = residues[idx].get_resname().strip()
        res_type = res_type_map.get(res_name, "polar")
        color = residue_type_colors[res_type]
        degree = int(metrics['degree'][idx])
        clustering = metrics['clustering'][idx]
        closeness = metrics['closeness'][idx]
        community = communities.get(idx, {})
        community_size = community.get('size', 0)
        composition = community.get('composition', {})
        comp_str = ", ".join([f"{k}: {v}" for k, v in composition.items()])
        
        st.markdown(f"""
        <div class="hub-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div><span style="font-size: 18px; font-weight: 700; color: {color};">#{rank} {label}</span><span style="color: #666; font-size: 14px; margin-left: 10px;">({res_name} - {res_type})</span></div>
                <div style="background: {color}; color: white; padding: 5px 12px; border-radius: 20px; font-weight: 600;">Deg: {degree}</div>
            </div>
            <div style="margin-top: 10px; font-size: 13px; color: #555;">
                <div style="display: flex; gap: 20px;"><div>• Clustering: <b>{clustering:.3f}</b></div><div>• Closeness: <b>{closeness:.3f}</b></div></div>
                <div style="margin-top:5px;"><b>Community:</b> {community_size} neighbors</div>
                <div style="font-size: 11px; color: #777;">{comp_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def draw_pcn_plot_enhanced(labels, vis_labels, coords, adjacency, dist_matrix, residues, threshold, metrics):
    if len(labels) == 0 or coords.size == 0:
        st.warning("No residues available for visualization.")
        return

    N = len(labels)
    adj_np = np.array(adjacency)
    degrees = metrics['degree']
    deg_min = degrees.min() if len(degrees) else 0
    deg_ptp = np.ptp(degrees) if np.ptp(degrees) > 0 else 1
    
    # --- Visualization Header (Emoji Removed) ---
    st.markdown("""<div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"><h3 style="color: #1e3c72; margin-bottom: 10px;">Interactive Network Explorer</h3></div>""", unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        # Renamed "Centrality" -> "Closeness Centrality"
        # Added "Clustering Coefficient"
        view_mode = st.radio(
            "View Filter:", 
            ["Show All", "Show Hubs Only", "Hydrophobic Core", "Closeness Centrality", "Clustering Coefficient", "Custom Degree Filter"]
        )
        # Set value=True by default
        strict_filter = st.checkbox("Hide Unmatched Nodes", value=True, help="If checked, unmatched nodes appear as faint 'ghosts' with no labels.")
    
    hub_percentile = 10
    custom_min_degree = 0
    highlight_communities = False
    centrality_threshold = 0.1
    clustering_threshold = 0.0
    exact_degree_match = False
    max_closeness = np.max(metrics['closeness']) if len(metrics['closeness']) > 0 else 1.0

    # --- Filter Inputs ---
    with filter_col2:
        if view_mode == "Custom Degree Filter":
            max_deg = int(degrees.max()) if N > 0 else 0
            custom_min_degree = st.slider("Degree Value", 0, max_deg, 0)
            exact_degree_match = st.checkbox("Match Exact Degree Only", value=False)
        elif view_mode == "Show Hubs Only":
            hub_percentile = st.number_input("Top % (Percentile):", min_value=1, max_value=50, value=10, step=1)
            highlight_communities = st.checkbox("Highlight Communities", value=True)
        elif view_mode == "Closeness Centrality":
            centrality_threshold = st.slider("Relative Centrality (Normalized)", 0.0, 1.0, 0.1, 0.001)
        elif view_mode == "Clustering Coefficient":
            clustering_threshold = st.slider("Min Clustering Coefficient", 0.0, 1.0, 0.5, 0.01)
    
    # --- Hub Logic ---
    raw_threshold = np.percentile(degrees, 100 - hub_percentile)
    degree_threshold_hub = int(raw_threshold) 
    hub_indices_global = np.where(degrees >= degree_threshold_hub)[0]

    with filter_col2:
        if view_mode == "Show Hubs Only":
            st.info(f"Showing top {hub_percentile}% (Degree ≥ {degree_threshold_hub})")

    final_colors, final_sizes, final_text_labels = [], [], []

    for i in range(N):
        deg = degrees[i]
        norm_deg = (deg - deg_min) / deg_ptp
        is_selected = False
        
        # --- Selection Logic ---
        if view_mode == "Show All": 
            is_selected = True
        elif view_mode == "Show Hubs Only" and i in hub_indices_global: 
            is_selected = True
        elif view_mode == "Custom Degree Filter":
            if exact_degree_match and deg == custom_min_degree: is_selected = True
            elif not exact_degree_match and deg >= custom_min_degree: is_selected = True
        elif view_mode == "Closeness Centrality" and metrics['closeness'][i] >= centrality_threshold * max_closeness: 
            is_selected = True
        elif view_mode == "Clustering Coefficient" and metrics['clustering'][i] >= clustering_threshold:
            is_selected = True
        elif view_mode == "Hydrophobic Core":
            name = residues[i].get_resname().strip().upper()
            if res_type_map.get(name, "UNK") == "hydrophobic": is_selected = True
        
        # --- Styling ---
        if is_selected:
            name = residues[i].get_resname().strip().upper()
            if name not in res_type_map: name = "UNK"
            base_hex = residue_type_colors.get(res_type_map[name], default_color)
            opacity_val = 0.8 + (norm_deg * 0.2)
            h = base_hex.lstrip('#')
            rgb = tuple(int(h[x:x+2], 16) for x in (0, 2, 4))
            final_colors.append(f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{float(opacity_val):.2f})")
            final_sizes.append(15 + (norm_deg * 25))
            final_text_labels.append(vis_labels[i]) 
        else:
            if strict_filter:
                final_colors.append("rgba(200, 200, 200, 0.05)")
                final_sizes.append(5)
                final_text_labels.append("") 
            else:
                final_colors.append("rgba(200, 200, 200, 0.15)")
                final_sizes.append(8)
                final_text_labels.append(vis_labels[i])
            
    communities = identify_hub_communities(adj_np, hub_indices_global, residues)
    
    fig = build_3d_figure_enhanced(
        labels, final_text_labels, coords, adj_np, dist_matrix,
        final_colors, final_sizes, residues,
        hub_indices_global, metrics,
        highlight_communities=(view_mode == "Show Hubs Only" and highlight_communities),
        view_mode=view_mode
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'toImageButtonOptions': {
            'format': 'png', 'filename': 'protein_network_hd',
            'height': 1200, 'width': 1600, 'scale': 3 
        },
        'displayModeBar': True,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    })
    
    render_hub_analysis_panel(labels, residues, metrics, hub_indices_global, communities, adj_np)


def load_structure_from_upload(uploaded_file, progress=None, progress_label=None):
    # CRITICAL FIX: Ensure file pointer is at the beginning
    uploaded_file.seek(0)
    
    if progress is not None:
        progress.progress(5)
        if progress_label is not None:
            progress_label.text("5% complete — starting to read file")
    
    text = uploaded_file.read().decode("utf-8")
    
    if progress is not None:
        progress.progress(25)
        if progress_label is not None:
            progress_label.text("25% complete — file read into memory")
            
    structure = PDBParser(QUIET=True).get_structure("uploaded", StringIO(text))
    
    if progress is not None:
        progress.progress(45)
        if progress_label is not None:
            progress_label.text("45% complete — parsed structure")
    return structure


def load_demo_structure(pdb_id, progress=None, progress_label=None):
    local_demo_path = os.path.join(os.path.dirname(__file__), "demo_data", f"{pdb_id}.pdb")
    if os.path.exists(local_demo_path):
        if progress is not None:
            progress.progress(50)
            if progress_label is not None:
                progress_label.text("50% complete — loading local demo file")
        return PDBParser(QUIET=True).get_structure(pdb_id.lower(), local_demo_path)

    pdbl = PDBList()
    temp = tempfile.gettempdir()
    if progress is not None:
        progress.progress(30)
        if progress_label is not None:
            progress_label.text("30% complete — fetching demo from RCSB")
    path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=temp)
    if progress is not None:
        progress.progress(70)
        if progress_label is not None:
            progress_label.text("70% complete — demo file downloaded")
    structure = PDBParser(QUIET=True).get_structure(pdb_id.lower(), path)
    if progress is not None:
        progress.progress(95)
        if progress_label is not None:
            progress_label.text("95% complete — parsing demo file")
    return structure


def process_and_render_pcn(structure, model_choice=None, chain_choice=None, progress_bar=None, progress_label=None):
    # Safety check for empty structure
    if not structure:
        st.error("Structure file could not be parsed. Please check if it's a valid PDB.")
        return

    model_ids = list(range(1, len(structure) + 1))
    
    if model_choice is None:
        if len(model_ids) > 1:
            st.info(f"Multi-model structure detected ({len(model_ids)} models).")
            model_choice = st.selectbox("Select NMR Model", model_ids)
        else:
           st.info("Single structural model detected (no NMR ensemble). Using Model 1.")
            model_choice = 1
    
    chains = list(structure[model_choice - 1].get_chains())
    chain_ids = [c.id for c in chains]
    
    if chain_choice is None:
        if len(chain_ids) > 1:
            st.info(f"Multiple chains detected ({len(chain_ids)} chains).")
            chain_choice = st.selectbox("Select Chain", chain_ids)
        elif len(chain_ids) == 1:
            st.info(f"Single chain detected (Chain {chain_ids[0]}).")
            chain_choice = chain_ids[0]
        else:
            st.error("No chains found in this structure.")
            return

    if progress_bar:
            progress_bar.progress(55)
            progress_label.text("55% complete — preparing computation")
    
    # --- COMPUTATION ---
    adj_df, dist_df, labels, vis_labels, coords, residues, metrics = compute_pcn_df(
        structure, model_choice, chain_choice, threshold, 
        progress=progress_bar, 
        progress_label=progress_label
    )

    if labels is None or len(labels) == 0:
        st.error("No valid C-alpha atoms found in the selected chain.")
        return

    num_nodes = len(labels)
    num_edges = int(np.sum(adj_df.values) // 2) if num_nodes else 0
    density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0

    # --- EXTRACT METADATA ---
    header = structure.header
    
    def get_compound_info(key, default="Unknown"):
        if 'compound' in header and header['compound']:
            first_mol = next(iter(header['compound'].values()), {})
            return first_mol.get(key, default)
        return default

    def get_source_info(key, default="Unknown"):
        if 'source' in header and header['source']:
            first_src = next(iter(header['source'].values()), {})
            return first_src.get(key, default)
        return default

    meta_pdb_id = header.get('idcode', 'USER UPLOAD').upper()
    meta_name = get_compound_info('molecule', 'Unknown Protein').capitalize()
    meta_class = header.get('head', 'Unclassified').capitalize()
    meta_organism = get_source_info('organism_scientific', 'Unknown Organism').capitalize()
    meta_method = header.get('structure_method', 'unknown').upper()
    
    is_engineered = get_compound_info('engineered', '').lower() == 'yes' or 'mutation' in get_compound_info('other_details', '').lower()
    meta_engineered = "Yes" if is_engineered else "No"
    
    if len(meta_pdb_id) == 4 and meta_pdb_id != "USER UPLOAD":
        pdb_display = f'<a href="https://www.rcsb.org/structure/{meta_pdb_id}" target="_blank" style="text-decoration:none; color:#1e3c72; border-bottom: 2px solid #1e3c72;">{meta_pdb_id} ↗</a>'
    else:
        pdb_display = meta_pdb_id

    # --- STRUCTURE OVERVIEW (Clean Version - No SVG) ---
    structure_card_html = f"""
    <div style="background: white; padding: 30px; border-radius: 12px; margin: 30px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-left: 6px solid #2a5298;">
        <h3 style="color: #1e3c72; margin-top: 0; margin-bottom: 25px; font-size: 28px; font-weight: 700; border-bottom: 2px solid #eee; padding-bottom: 15px;">
            Structure Overview
        </h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; font-size: 20px; line-height: 1.8; color: #333;">
            <div>
                <p style="margin: 10px 0;"><b>PDB ID:</b> {pdb_display}</p>
                <p style="margin: 10px 0;"><b>Protein:</b> {meta_name}</p>
                <p style="margin: 10px 0;"><b>Functional Class:</b> {meta_class}</p>
                <p style="margin: 10px 0;"><b>Organism:</b> <i>{meta_organism}</i></p>
            </div>
            <div>
                <p style="margin: 10px 0;"><b>Chain Analyzed:</b> <span style="background: #fff3e0; padding: 4px 12px; border-radius: 6px; color: #e65100; font-weight:bold;">{chain_choice}</span></p>
                <p style="margin: 10px 0;"><b>Method:</b> {meta_method}</p>
                <p style="margin: 10px 0;"><b>Engineered:</b> {meta_engineered}</p>
            </div>
        </div>
    </div>
    """
    
    st.markdown(structure_card_html, unsafe_allow_html=True)

    # --- NETWORK SUMMARY SECTION ---
    st.markdown(
        f"""
        <div style="background: white; padding: 20px; border-radius: 10px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <h3 style="color: #1e3c72; margin-bottom: 15px;">Network Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 8px;">
                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px;">
                        <circle cx="24" cy="24" r="8" fill="#2a5298" stroke="#1e3c72" stroke-width="2"/>
                        <circle cx="24" cy="24" r="3" fill="white"/>
                    </svg>
                    <div style="font-size: 32px; font-weight: 700; color: #2a5298;">{num_nodes}</div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">Nodes</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 8px;">
                    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px;">
                        <circle cx="12" cy="24" r="4" fill="#2a5298"/>
                        <circle cx="36" cy="24" r="4" fill="#2a5298"/>
                        <line x1="16" y1="24" x2="32" y2="24" stroke="#1e3c72" stroke-width="3" stroke-linecap="round"/>
                    </svg>
                    <div style="font-size: 32px; font-weight: 700; color: #2a5298;">{num_edges}</div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">Edges</div>
                </div>
                <div style="text-align: center; padding: 15px; background: #f0f4f8; border-radius: 8px;">
                    <svg width="48" height="48" viewBox="0 0 256 256" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px;">
                        <circle cx="64" cy="48" r="16" fill="none" stroke="#1e3c72" stroke-width="8"/>
                        <circle cx="176" cy="64" r="16" fill="none" stroke="#1e3c72" stroke-width="8"/>
                        <circle cx="64" cy="160" r="16" fill="none" stroke="#1e3c72" stroke-width="8"/>
                        <circle cx="128" cy="128" r="16" fill="none" stroke="#1e3c72" stroke-width="8"/>
                        <circle cx="176" cy="192" r="16" fill="none" stroke="#1e3c72" stroke-width="8"/>
                        <circle cx="96" cy="208" r="16" fill="none" stroke="#1e3c72" stroke-width="8"/>
                        <line x1="75" y1="56" x2="118" y2="120" stroke="#2a5298" stroke-width="6"/>
                        <line x1="165" y1="72" x2="138" y2="120" stroke="#2a5298" stroke-width="6"/>
                        <line x1="75" y1="152" x2="118" y2="135" stroke="#2a5298" stroke-width="6"/>
                        <line x1="138" y1="135" x2="167" y2="184" stroke="#2a5298" stroke-width="6"/>
                        <line x1="105" y1="201" x2="167" y2="199" stroke="#2a5298" stroke-width="6"/>
                        <line x1="75" y1="168" x2="88" y2="200" stroke="#2a5298" stroke-width="6"/>
                    </svg>
                    <div style="font-size: 32px; font-weight: 700; color: #2a5298;">{density:.4f}</div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">Graph Density</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- DISTANCE MATRIX & HEATMAP ---
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Distance Matrix (Preview)</h3>", unsafe_allow_html=True)
    st.dataframe(dist_df.iloc[:10, :10])
    st.download_button("Download Distance Matrix (CSV)", dist_df.to_csv().encode(), "distance.csv")
    
    with st.expander("Values Visualization (Heatmap)", expanded=True):
        render_distance_heatmap(dist_df)

    # --- ADJACENCY MATRIX & BINARY MAP ---
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Adjacency Matrix (Preview)</h3>", unsafe_allow_html=True)
    st.dataframe(adj_df.iloc[:10, :10])
    st.download_button("Download Adjacency Matrix (CSV)", adj_df.to_csv().encode(), "adjacency.csv")
    
    with st.expander("Binary Visualization (Adjacency Map)", expanded=True):
        render_adjacency_heatmap(adj_df)
    
    # --- NETWORK FILES ---
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Download Network Files</h3>", unsafe_allow_html=True)
    
    edges_sif = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_df.values[i][j] == 1:
                edges_sif.append(f"{labels[i]} pp {labels[j]}")
    sif_text = "\n".join(edges_sif)
    
    edges_txt = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_df.values[i][j] == 1:
                edges_txt.append(f"{labels[i]} {labels[j]}")
    txt_text = "\n".join(edges_txt)

    col_sif, col_edge = st.columns(2)
    
    with col_sif:
            st.markdown("""
            **SIF (Cytoscape)**
            A Single Interaction Format (SIF) file containing the list of interactions between each residue in the following format:
                        
            *Residue_A pp Residue_B*
            """)
            st.download_button("Download SIF (Cytoscape)", sif_text, "network.sif")
            
    with col_edge:
            st.markdown("""
            **Edge List (Text)**
            A simple text file containing the list of edges between each residue in the following format:
                        
            *Residue_A Residue_B*
            """)
            st.download_button("Download Edge List (.txt)", txt_text, "edges.txt")

    draw_pcn_plot_enhanced(labels, vis_labels, coords, adj_df.values, dist_df.values, residues, threshold, metrics)
    
    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Residue Degree Distribution</h3>", unsafe_allow_html=True)
    if len(labels) > 0:
        degree_counts = pd.Series(metrics['degree']).value_counts().sort_index()
        fig_hist = go.Figure(go.Bar(
            x=degree_counts.index, 
            y=degree_counts.values,
            text=degree_counts.values,
            textposition='outside',
            marker_color="#2a5298"
        ))
        fig_hist.update_layout(height=360, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    if progress_bar:
        progress_bar.progress(100)
        progress_label.text("Processing complete")


# -----------------------------------------------------------------------------
# MAIN LAYOUT
# -----------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1 style='text-align:center; font-size:52px; color: white; margin: 0; font-weight: 700; letter-spacing: -0.5px;'>
        Protein Contact Network Explorer
    </h1>
    <p style='text-align:center; font-size:18px; color: #e3f2fd; margin-top: 10px; font-weight: 400;'>
        Analyze residue contacts • Visualize networks • Advanced hub analysis • Export data
    </p>
</div>
""", unsafe_allow_html=True)

tab_analysis, tab_help, tab_about = st.tabs(["🧪 Analysis Tool", "❓ Help Guide", "🔍 About"])

with tab_analysis:
    left_col, mid_col, right_col = st.columns([2, 5, 2])

    with left_col:
        st.markdown("""
        <div class="section-box">
            <div style="text-align:center;">
                <svg width="40" height="40" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" stroke="#2a5298" stroke-width="2" fill="none" stroke-linecap="round"/>
                    <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" stroke="#2a5298" stroke-width="2" fill="none" stroke-linecap="round"/>
                </svg>
            </div>
            <h3 style="text-align:center; color:#1e3c72;">Try with Demo</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write("Try a preloaded PDB to see example networks quickly.")
        demo_choices = ["None", "1CRN", "1UBQ", "4HHB"]
        demo_selection = st.selectbox("Demo protein", demo_choices, index=0, key="demo_selection")

    with mid_col:
        st.markdown("""
        <div class="section-box" style="text-align: center; min-height: 200px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <div style="margin-bottom: 10px;">
                <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto;">
                    <line x1="50" y1="20" x2="20" y2="75" stroke="#2a5298" stroke-width="4" stroke-linecap="round"/>
                    <line x1="50" y1="20" x2="80" y2="75" stroke="#2a5298" stroke-width="4" stroke-linecap="round"/>
                    <line x1="20" y1="75" x2="80" y2="75" stroke="#2a5298" stroke-width="4" stroke-linecap="round"/>
                    <circle cx="50" cy="20" r="10" fill="#FF5252" stroke="white" stroke-width="2"/>
                    <circle cx="20" cy="75" r="10" fill="#2196F3" stroke="white" stroke-width="2"/>
                    <circle cx="80" cy="75" r="10" fill="#4CAF50" stroke="white" stroke-width="2"/>
                </svg>
            </div>
            <h3 style="text-align:center; color: #1e3c72; margin: 0; padding: 0;">
                Upload PDB Files
            </h3>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDB file", type=["pdb"], label_visibility="collapsed")

    with right_col:
        st.markdown("""
        <div class="section-box">
            <div style="text-align:center;">
                <svg width="40" height="40" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <line x1="30" y1="70" x2="70" y2="30" stroke="#000" stroke-width="3"/>
                    <circle cx="30" cy="70" r="18" fill="#FF5252"/>
                    <circle cx="70" cy="30" r="18" fill="#2196F3"/>
                </svg>
            </div>
            <h3 style="text-align:center; color:#1e3c72;">Contact Threshold</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write("Set the distance threshold (Å).")
        threshold = st.number_input("Contact threshold (Å)", min_value=1.0, max_value=20.0, value=5.0, step=0.1)

    st.divider()

    if uploaded_file:
        st.session_state["is_demo"] = False
        progress_bar = mid_col.progress(0)
        progress_label = mid_col.empty()
        progress_label.text("0% complete — waiting to start")
        with st.spinner("Processing uploaded PDB file..."):
            structure = load_structure_from_upload(uploaded_file, progress_bar, progress_label)
            process_and_render_pcn(structure, progress_bar=progress_bar, progress_label=progress_label)

    else:
        st.session_state["is_demo"] = (demo_selection != "None")
        demo_active = st.session_state.get("is_demo", False)
        demo_id = st.session_state.get("demo_selection", demo_selection)

        if demo_active and demo_id and demo_id != "None":
            progress_bar = mid_col.progress(0)
            progress_label = mid_col.empty()
            with st.spinner(f"Loading demo {demo_id}..."):
                structure = load_demo_structure(demo_id, progress_bar, progress_label)
                process_and_render_pcn(structure, progress_bar=progress_bar, progress_label=progress_label)
        else:
            st.info("Upload a PDB file or try the demo to begin.")

# --- TAB 2: Help ---
with tab_help:
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); border-left: 5px solid #1e3c72;">
        <h3 style="color: #1e3c72; margin-top: 0;">User Guide</h3>
        <p style="font-size: 16px;">Follow these steps to analyze your protein structures.</p>
    </div>
    """, unsafe_allow_html=True)

    col_guide_1, col_guide_2 = st.columns(2)

    with col_guide_1:
        st.markdown("""
        #### 1. Uploading a PDB File
        Upload a valid `.pdb` file using the sidebar file uploader. The application automatically parses:
        * Protein chains
        * NMR models (if present)
        * Residue coordinates (Cα atoms)
        
        *Alternatively, select **Try Demo (1CRN)** to load a reference structure.*

        ---
        #### 2. Selecting Chain & Model
        * **Chains:** If multiple chains exist, select exactly one.
        * **NMR Models:** If multiple models exist, select one for analysis.
        
        *Note: Only a single chain–model combination can be analyzed at a time.*
        """)

    with col_guide_2:
        st.markdown("""
        #### 3. Contact Threshold
        Defines which residues are interacting.
        * **Default:** 5.0 Å
        * **Logic:** Pairs with Cα–Cα distance ≤ threshold = **1**, else **0**.

        ---
        #### 4. Output Matrices
        PCN Explorer generates two key matrices available for preview & download:
        * **Distance Matrix (CSV):** Pairwise Cα distances.
        * **Adjacency Matrix (CSV):** Binary contact matrix.
        """)

    st.markdown("---")
    
    st.markdown("""
    #### 5. Network Visualization
    The interactive 3D viewer displays residues as nodes and contacts as edges.
    
    * **Hover:** See residue name, index, and centrality metrics.
    * **Filters:** Toggle between "Show All", "Hubs Only", or "Hydrophobic Core".
    * **Colors:**
        * <span style="color:#D94E1E"><b>● Hydrophobic</b></span>
        * <span style="color:#003B6F"><b>● Polar</b></span>
        * <span style="color:#007A55"><b>● Positive</b></span>
        * <span style="color:#B32630"><b>● Negative</b></span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Troubleshooting & Common Errors")

    with st.expander("Error: No Cα atoms found"):
        st.markdown("""
        **Possible Causes:**
        - Missing or incomplete residues in the PDB file.
        - Non-standard residue naming.
        - The chain contains only heteroatoms (DNA, RNA, Ligands).

        **Solution:**
        - Verify that the selected chain contains standard amino acids with valid Cα atoms.
        """)

    with st.expander("Error: Model index out of range"):
        st.markdown("""
        **Cause:**
        - Selected an NMR model number that is not present in the uploaded structure.

        **Solution:**
        - Choose a model within the available range shown in the selector dropdown.
        """)

    with st.expander("Issue: Blank or empty visualization"):
        st.markdown("""
        **Possible Causes:**
        - No residues were parsed.
        - Contact threshold is set too low (resulting in 0 edges).
        - Selected chain contains missing coordinates.

        **Solution:**
        - Increase the contact threshold (e.g., to 6.0 Å or 7.0 Å).
        - Recheck chain and model selection.
        - Verify PDB file integrity.
        """)

# --- TAB 3: About ---
with tab_about:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #1e3c72; font-weight: 800;">About PCN Explorer</h2>
        <p style="font-size: 18px; color: #555;">An advanced tool for Protein Contact Network Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #2a5298; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #1e3c72;">Overview</h4>
        <p>
            <b>Protein Contact Networks (PCNs)</b> represent amino acid residues as nodes and residue–residue contacts as edges, 
            providing a graph-theoretic abstraction of protein tertiary structure.
        </p>
        <p>
            PCN Explorer constructs these networks using <b>Cα–Cα geometric distances</b> and integrates quantitative 
            matrix representations with interactive 3D visualizations to facilitate structural and network-based analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_feat, col_meth = st.columns(2)

    with col_feat:
        st.markdown("""
        ### Key Features
        * **Input Flexibility:** Support for X-ray (.pdb) and multi-model NMR ensembles.
        * **Matrix Generation:** Automatic computation of Distance and Binary Adjacency matrices.
        * **Interactive Viz:** 3D Network explorer with zoom, orbit, and hub highlighting.
        * **Export Data:** Download CSV matrices, SIF files (Cytoscape), and Edge lists.
        * **Residue Coloring:** Analysis based on Hydrophobic cores.
        * **Graph Metrics:** Degree distribution, clustering coefficients, and centrality.
        * **Distance Matrix Visualization:** Heatmaps and clustering analyses.
        * **Adjacency Matrix Visualization:** Binary contact maps for network insights           
        """)

    with col_meth:
        st.markdown("""
        ### Network Definition
        1.  **Extraction:** Cα atomic coordinates are extracted for the selected chain.
        2.  **Distance Calculation:** Pairwise Euclidean distances generate an $N \\times N$ distance matrix.
        3.  **Adjacency:**
            $$ A_{ij} = 1 \\text{ if } dist(i, j) \leq threshold $$
            $$ A_{ij} = 0 \\text{ otherwise } $$
        """)

    st.markdown("---")
    
    st.markdown("### Mathematical Formulations")
    
    st.markdown("""
    #### 1. Node Degree (Hub Identification)
    The degree $k_i$ of a residue $i$ is the number of other residues it is in contact with. Residues with a degree in the top percentile (e.g., top 10%) are identified as **Hubs**.
    
    $$ k_i = \\sum_{j} A_{ij} $$
    """)
    
    st.markdown("""
    #### 2. Local Clustering Coefficient
    Measures the degree to which a residue's neighbors are also connected to each other (local cohesiveness). For a residue $i$ with degree $k_i$:
    
    $$ C_i = \\frac{2 e_i}{k_i (k_i - 1)} $$
    
    Where $e_i$ is the number of actual edges between the neighbors of residue $i$.
    """)
    
    st.markdown("""
    #### 3. Closeness Centrality
    Represents how close a residue is to all other residues in the network. It is the reciprocal of the sum of shortest path distances $d(i, j)$ from node $i$ to all other nodes $j$.
    
    $$ C_{close}(i) = \\frac{N - 1}{\\sum_{j \\neq i} d(i, j)} $$
    """)
    
    st.markdown("""
    #### 4. Betweenness Centrality
    Quantifies the influence of a residue on the flow of information through the network. It is defined as the fraction of all shortest paths $\\sigma_{st}$ between any pair of nodes $(s, t)$ that pass through node $i$.
    
    $$ C_{between}(i) = \\sum_{s \\neq i \\neq t} \\frac{\\sigma_{st}(i)}{\\sigma_{st}} $$
    """)

    st.markdown("---")

    st.markdown("### Installation & Usage")
    st.code("""
# 1. Clone the repository
git clone https://github.com/<your-username>/PCN-Explorer
cd PCN-Explorer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
streamlit run Protein_penult.py
    """, language="bash")

    st.markdown("""
    <div style="margin-top: 40px; text-align: center; color: #888; font-size: 14px;">
        <hr>
        Developed by <b>Akhurath Ganapathy</b> and <b>Sanjana Vijay Krishnan</b> (2025)
    </div>
    """, unsafe_allow_html=True)
