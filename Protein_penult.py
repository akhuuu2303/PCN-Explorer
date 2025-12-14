import streamlit as st
import os
from Bio.PDB import PDBParser, PDBList
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import tempfile

st.set_page_config(page_title="Protein Contact Network Explorer", layout="wide")

# Custom CSS for modern scientific theme
st.markdown("""
<style>
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    /* Section boxes */
    .section-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298;
        margin-bottom: 15px;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        border-radius: 8px;
    }
    
    /* Button styling */
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
    
    /* Number input styling */
    .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Header with Help link
st.markdown("""
<div class="main-header">
    <div style='text-align: right; margin-top: -10px; margin-bottom: 10px;'>
        <a href="https://github.com/akhuuu2303/PCN-Explorer/blob/main/HELP.md"
           target="_blank"
           style="color: #ffffff; text-decoration: none; font-weight: 600; font-size: 16px; 
                  background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 6px;
                  transition: all 0.3s ease;">
           Help
        </a>
    </div>
    <h1 style='text-align:center; font-size:52px; color: white; margin: 0; font-weight: 700; letter-spacing: -0.5px;'>
        Protein Contact Network Explorer
    </h1>
    <p style='text-align:center; font-size:18px; color: #e3f2fd; margin-top: 10px; font-weight: 400;'>
        Analyze residue contacts • Visualize networks • Export data
    </p>
</div>
""", unsafe_allow_html=True)

# Three column layout for input controls
left_col, mid_col, right_col = st.columns([2, 5, 2])

with left_col:
    st.markdown("""
    <div class="section-box">
        <div style="text-align: center;">
            <svg width="40" height="40" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px;">
                <rect x="8" y="12" width="48" height="36" rx="4" fill="none" stroke="#2a5298" stroke-width="3"/>
                <rect x="16" y="20" width="32" height="20" rx="2" fill="#e3f2fd" stroke="#2a5298" stroke-width="2"/>
                <path d="M24 30 L28 34 L36 26" stroke="#2a5298" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                <line x1="20" y1="52" x2="44" y2="52" stroke="#2a5298" stroke-width="2" stroke-linecap="round"/>
                <circle cx="32" cy="56" r="2" fill="#2a5298"/>
            </svg>
        </div>
        <h3 style='text-align:center; color: #1e3c72; margin-bottom: 15px;'>Try with Demo</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write("Try a preloaded PDB to see example networks quickly.")
    demo_choices = ["None", "1CRN", "1UBQ", "4HHB"]
    demo_selection = st.selectbox("Demo protein", demo_choices, index=0, key="demo_selection")

with mid_col:
    st.markdown("""
    <div class="section-box">
        <div style="text-align: center;">
            <svg width="40" height="40" viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px;">
                <path d="M128 0c-17.6 0-32 14.4-32 32v448c0 17.6 14.4 32 32 32h320c17.6 0 32-14.4 32-32V128L352 0H128z" fill="#2a5298"/>
                <path d="M384 128h96L352 0v96c0 17.6 14.4 32 32 32z" fill="#1e3c72"/>
                <rect x="96" y="384" width="320" height="128" fill="#1e3c72"/>
                <text x="140" y="470" font-size="90" font-weight="bold" fill="white">PDB</text>
                <ellipse cx="220" cy="180" rx="60" ry="25" fill="white"/>
                <ellipse cx="220" cy="205" rx="60" ry="25" fill="white"/>
                <ellipse cx="220" cy="230" rx="60" ry="25" fill="white"/>
                <path d="M160 180 v75 q0,25 60,25 q60,0 60,-25 v-75" fill="none" stroke="white" stroke-width="8"/>
                <rect x="285" y="190" width="70" height="90" rx="5" fill="white"/>
                <path d="M330 210 L320 225 L340 225 Z M305 235 L315 250 L325 250 M335 235 L345 250 L355 250" stroke="#1e3c72" stroke-width="5" stroke-linecap="round" fill="none"/>
            </svg>
        </div>
        <h3 style='text-align:center; color: #1e3c72; margin-bottom: 15px;'>Upload PDB Files</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write("Upload your own PDB file for analysis. Accepted file type: .pdb")
    uploaded_file = st.file_uploader("Upload a PDB file", type=["pdb"])        
    st.divider()

with right_col:
    st.markdown("""
    <div class="section-box">
        <div style="text-align: center;">
            <svg width="40" height="40" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 10px;">
                <circle cx="32" cy="32" r="20" fill="none" stroke="#2a5298" stroke-width="3"/>
                <line x1="32" y1="32" x2="32" y2="18" stroke="#2a5298" stroke-width="2.5" stroke-linecap="round"/>
                <line x1="32" y1="32" x2="42" y2="38" stroke="#2a5298" stroke-width="2.5" stroke-linecap="round"/>
                <circle cx="32" cy="32" r="3" fill="#2a5298"/>
                <path d="M48 48 L58 58" stroke="#2a5298" stroke-width="3" stroke-linecap="round"/>
                <circle cx="58" cy="58" r="4" fill="none" stroke="#2a5298" stroke-width="3"/>
            </svg>
        </div>
        <h3 style='text-align:center; color: #1e3c72; margin-bottom: 15px;'>Contact Threshold</h3>
    </div>
    """, unsafe_allow_html=True)
    st.write("Set the distance threshold (Å) for residue contacts.")
    threshold = st.number_input(
        "Contact threshold (Å)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.1,
    )

# Residue type mapping
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
    "hydrophobic": "#FF6B35",
    "polar": "#004E89",
    "positive": "#00A676",
    "negative": "#E63946",
}

default_color = "#8da0cb"

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
        return pd.DataFrame(), pd.DataFrame(), [], np.empty((0, 3)), []

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

    labels = [f"{res.get_resname().strip()}_{res.get_id()[1]}" for res in residues]

    if progress is not None:
        progress.progress(100)
        if progress_label is not None:
            progress_label.text("100% complete — done computing PCN")

    return (
        pd.DataFrame(adjacency, index=labels, columns=labels),
        pd.DataFrame(dist_matrix, index=labels, columns=labels),
        labels,
        coords,
        residues
    )

def draw_pcn_plot_enhanced(labels, coords, adjacency, dist_matrix, residues):
    """
    Enhanced dual-plot PCN visualizer with modern scientific styling
    """

    if len(labels) == 0 or coords.size == 0:
        st.warning("No residues available for visualization.")
        return

    N = len(labels)
    adj_np = np.array(adjacency)

    degrees = np.sum(adj_np, axis=0)
    deg_min = degrees.min() if len(degrees) else 0
    deg_ptp = np.ptp(degrees) if np.ptp(degrees) > 0 else 1

    min_size, max_size = 6, 18
    node_sizes = min_size + ((degrees - deg_min) / deg_ptp) * (max_size - min_size)

    st.markdown(
        """
        <div style="background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
            <h3 style="color: #1e3c72; margin-bottom: 10px;">Residue Type Legend</h3>
            <div style="font-size:15px; display: flex; justify-content: center; gap: 25px; flex-wrap: wrap;">
                <span style="color:#FF6B35; font-weight: 600;">■ Hydrophobic</span>
                <span style="color:#004E89; font-weight: 600;">■ Polar</span>
                <span style="color:#00A676; font-weight: 600;">■ Positive</span>
                <span style="color:#E63946; font-weight: 600;">■ Negative</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color: #1e3c72;'>Contact Highlight View</h3>", unsafe_allow_html=True)
        max_degree = int(degrees.max()) if N > 0 else 0
        degree_filter = st.selectbox(
                "Highlight residues with degree:",
                ["None"] + [str(i) for i in range(max_degree + 1)],
                index=0,
                key="degree_filter_left"
            )

        node_colors_left = []
        for i, res in enumerate(residues):
            name = res.get_resname().strip().upper()
            if name not in res_type_map:
                name = "UNK"
            base_color = residue_type_colors.get(res_type_map[name], default_color)

            if degree_filter != "None":
                if degrees[i] == int(degree_filter):
                    node_colors_left.append(base_color)
                else:
                    node_colors_left.append("rgba(200,200,200,0.2)")
            else:
                node_colors_left.append(base_color)

        fig_left = build_3d_figure(
            labels, coords, adj_np, dist_matrix, node_colors_left, node_sizes
        )
        st.plotly_chart(
            fig_left,
              use_container_width=True,
              key="contact_plot")

    with col2:
        st.markdown("<h3 style='color: #1e3c72;'>Residue Viewer</h3>", unsafe_allow_html=True)

        all_resnames = sorted(set([res.get_resname().strip().upper() for res in residues]))
        residue_filter = st.selectbox(
            "Show only this residue type:",
            ["All"] + all_resnames,
            index=0,
            key="residue_filter_right"
        )

        node_colors_right = []
        for i, res in enumerate(residues):
            name = res.get_resname().strip().upper()
            if name not in res_type_map:
                name = "UNK"
            base_color = residue_type_colors.get(res_type_map[name], default_color)

            if residue_filter != "All" and name != residue_filter:
                node_colors_right.append("rgba(200,200,200,0.1)")
            else:
                node_colors_right.append(base_color)

        fig_right = build_3d_figure(
            labels, coords, adj_np, dist_matrix, node_colors_right, node_sizes
        )
        st.plotly_chart(fig_right,
                         use_container_width=True,
                         key="residue_plot")


def build_3d_figure(labels, coords, adj_np, dist_matrix, node_colors, node_sizes):

    N = len(labels)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    base_edges = []
    hover_edges = []

    for i in range(N):
        for j in range(i + 1, N):
            if adj_np[i, j] != 1:
                continue

            d = dist_matrix[i, j]
            rel = max(0.01, (threshold - d) / threshold)
            width = 1 + 3 * rel
            opacity = 0.25 + 0.60 * rel

            x0, y0, z0 = coords[i]
            x1, y1, z1 = coords[j]

            base_edges.append(
                go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode="lines",
                    line=dict(color=f"rgba(0,0,0,{opacity:.2f})", width=width),
                    hoverinfo="none",
                    showlegend=False
                )
            )

            hover_edges.append(
                go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode="lines",
                    line=dict(color="#2a5298", width=width + 5),
                    hoverinfo="text",
                    hovertext=f"{labels[i]} — {labels[j]}<br>{d:.2f} Å",
                    opacity=0.0,
                    showlegend=False
                )
            )

    glow = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=[s * 1.8 for s in node_sizes],
            color=node_colors,
            opacity=0.20
        ),
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1.5, color="#333333")),
        text=labels,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(data=base_edges + hover_edges + [glow, node_trace])

    fig.update_layout(
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            bgcolor="#f8f9fa"
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=500
    )

    return fig


def load_structure_from_upload(progress=None, progress_label=None):
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

def run_pcn_app(structure, progress=None, progress_label=None):

    model_ids = list(range(1, len(structure) + 1))
    model_choice = st.selectbox("Select NMR Model", model_ids)

    chain_ids = [c.id for c in structure[model_choice - 1].get_chains()]
    chain_choice = st.selectbox("Select Chain", chain_ids)

    if progress is not None:
        progress.progress(55)
        if progress_label is not None:
            progress_label.text("55% complete — preparing computation")
    adj_df, dist_df, labels, coords, residues = compute_pcn_df(
        structure, model_choice, chain_choice, threshold, progress=progress, progress_label=progress_label
    )

    num_nodes = len(labels)
    num_edges = int(np.sum(adj_df.values) // 2) if num_nodes else 0
    max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
    density = num_edges / max_edges

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

    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Distance Matrix (Preview)</h3>", unsafe_allow_html=True)
    st.dataframe(dist_df.iloc[:10, :10])

    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Adjacency Matrix (Preview)</h3>", unsafe_allow_html=True)
    st.dataframe(adj_df.iloc[:10, :10])

    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Download Data</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Distance Matrix (CSV)", dist_df.to_csv().encode(), "distance.csv")
    with col2:
        st.download_button("Adjacency Matrix (CSV)", adj_df.to_csv().encode(), "adjacency.csv")
    
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_df.values[i][j] == 1:
                edges.append(f"{labels[i]} pp {labels[j]}")

    sif_text = "\n".join(edges)
    with col3:
        st.download_button("SIF (Cytoscape)", sif_text, "network.sif")
    
    draw_pcn_plot_enhanced(
        labels,
        coords,
        adj_df.values,
        dist_df.values,
        residues
    )

    st.markdown("<h3 style='color: #1e3c72; margin-top: 30px;'>Residue Degree Distribution</h3>", unsafe_allow_html=True)

    if len(labels) > 0:
        degrees = np.sum(adj_df.values, axis=0)
        degree_counts = pd.Series(degrees).value_counts().sort_index()

        fig = go.Figure(
            go.Bar(
                x=degree_counts.index, 
                y=degree_counts.values, 
                marker_color="#2a5298",
                marker_line_color="#1e3c72",
                marker_line_width=1.5,
                hovertemplate="Degree: %{x}<br>Residues: %{y}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Degree (number of contacts)",
            yaxis_title="Residue count",
            margin=dict(l=0, r=0, t=10, b=0),
            height=360,
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="#f8f9fa",
            font=dict(family="Arial, sans-serif", color="#333")
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption("Histogram showing the distribution of residue contact degrees in the protein contact network. Degree corresponds to the number of residues within the specified distance threshold.")

    else:
        st.write("No contact data available.")
    if progress is not None:
        try:
            progress.progress(100)
            if progress_label is not None:
                progress_label.text("100% complete — processing complete")
        except Exception:
            pass

    if progress is not None:
        try:
            progress.progress(100)
        except Exception:
            pass

if uploaded_file:
    st.session_state["is_demo"] = False
    progress_bar = mid_col.progress(0)
    progress_label = mid_col.empty()
    progress_label.text("0% complete — waiting to start")
    with st.spinner("Processing uploaded PDB file..."):
        structure = load_structure_from_upload(progress_bar, progress_label)
        run_pcn_app(structure, progress=progress_bar, progress_label=progress_label)

else:
    st.session_state["is_demo"] = (demo_selection != "None")

    demo_active = st.session_state.get("is_demo", False)
    demo_id = st.session_state.get("demo_selection", demo_selection)

    if demo_active and demo_id and demo_id != "None":
        progress_bar = mid_col.progress(0)
        progress_label = mid_col.empty()
        progress_label.text("0% complete — waiting to start")
        with st.spinner(f"Loading demo {demo_id}..."):
            structure = load_demo_structure(demo_id, progress_bar, progress_label)
            run_pcn_app(structure, progress=progress_bar, progress_label=progress_label)
    else:
        st.info("Upload a PDB file or try the demo to begin.")
