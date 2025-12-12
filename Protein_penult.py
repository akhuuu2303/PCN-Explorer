import streamlit as st
from Bio.PDB import PDBParser, PDBList
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import tempfile

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="PCN Explorer", layout="wide")

# Top-right Help hyperlink
st.markdown("""
<div style='text-align: right; font-size: 18px; margin-top: -40px;'>
    <a href="https://github.com/akhuuu2303/PCN-Explorer/blob/main/HELP.md"
       target="_blank"
       style="color:#4A90E2; text-decoration:none; font-weight:bold;">
       Help
    </a>
</div>
""", unsafe_allow_html=True)

st.title("Protein Contact Network (PCN) Explorer")
st.caption("Analyze residue contacts · Visualize networks · Export PCN data")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Upload & Settings")

uploaded_file = st.sidebar.file_uploader("Upload a PDB file", type=["pdb"])
use_demo = st.sidebar.button("Try Demo (1CRN)")

threshold = st.sidebar.number_input(
    "Contact threshold (Å)",
    min_value=2.0,
    max_value=15.0,
    value=5.0,
    step=0.1,
)
# ---------------------------------------------------------
# RESIDUE TYPE MAP
# ---------------------------------------------------------
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
    "hydrophobic": "#FFA500",
    "polar": "#1f77b4",
    "positive": "#2ca02c",
    "negative": "#d62728",
}

default_color = "#8da0cb"  # softer blue for single-colour, works on dark bg

# ---------------------------------------------------------
# PCN COMPUTATION
# ---------------------------------------------------------
def compute_pcn_df(structure, model_id, chain_id, threshold):
    # defensive: model/chain selection
    model = structure[int(model_id) - 1]
    chain = model[chain_id]

    residues = []
    coords = []

    for res in chain.get_residues():
        if "CA" in res:
            try:
                coords.append(res["CA"].get_coord())
                residues.append(res)
            except Exception:
                continue

    if len(coords) == 0:
        # return safe empty placeholders
        return pd.DataFrame(), pd.DataFrame(), [], np.empty((0, 3)), []

    coords = np.vstack(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
    adjacency = (dist_matrix <= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    labels = [f"{res.get_resname().strip()}_{res.get_id()[1]}" for res in residues]

    adj_df = pd.DataFrame(adjacency, index=labels, columns=labels)
    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return adj_df, dist_df, labels, coords, residues

# ---------------------------------------------------------
# ENHANCED PLOTLY 3D VISUALIZER (Option 1)
# ---------------------------------------------------------
def draw_pcn_plot_enhanced(labels, coords, adjacency, dist_matrix, residues):
    """
    Enhanced Plotly visualization with:
      - Degree-based filtering
      - Node glow
      - Node size scaled by degree
      - Edge thickness scaling
      - Hover to show neighbors
      - Residue-type colouring (disabled for demo)
    """

    # ---------------------------------------------
    # SAFETY CHECKS
    # ---------------------------------------------
    if len(labels) == 0 or coords.size == 0:
        st.warning("No Cα residues found for this selection.")
        return

    N = len(labels)
    adj_np = np.array(adjacency)

    # ---------------------------------------------
    # COMPUTE NODE DEGREES (NUMPY 2.0 SAFE)
    # ---------------------------------------------
    degrees = np.sum(adj_np, axis=0)

    deg_min = degrees.min() if len(degrees) > 0 else 0
    deg_ptp = np.ptp(degrees) if np.ptp(degrees) > 0 else 1

    # compute node sizes based on degree
    min_size, max_size = 6, 18
    deg_norm = (degrees - deg_min) / deg_ptp
    node_sizes = min_size + deg_norm * (max_size - min_size)

    # ---------------------------------------------
    # RESIDUE COLOR MODE (demo forces single colour)
    # ---------------------------------------------
    if st.session_state.get("is_demo", False):
        colour_mode = "Single Colour"
    else:
        colour_mode = st.radio(
            "Node Colour Mode",
            ["Single Colour", "Residue Type"],
            horizontal=True,
            label_visibility="collapsed",
        )

    # ---------------------------------------------
    # DEGREE FILTER SELECTOR (disabled for demo)
    # ---------------------------------------------
    if st.session_state.get("is_demo", False):
        degree_filter = "None"
    else:
        max_degree = int(degrees.max()) if N > 0 else 0
        degree_options = ["None"] + list(range(max_degree + 1))

        degree_filter = st.selectbox(
            "Highlight nodes with degree:",
            degree_options,
            index=0
        )

    # ---------------------------------------------
    # NODE COLORS + DEGREE FILTER
    # ---------------------------------------------
    node_colors = []

    for i, res in enumerate(residues):
        name = (res.get_resname() or "").strip().upper()
        if name not in res_type_map:
            name = "UNK"

        base_category = res_type_map[name]
        base_color = residue_type_colors.get(base_category, default_color)

        # apply residue-type mode
        colour = base_color if colour_mode == "Residue Type" else default_color

        # apply degree filter
        if degree_filter != "None":
            target = int(degree_filter)
            if degrees[i] == target:
                node_colors.append(colour)  # highlight
            else:
                node_colors.append("rgba(200,200,200,0.15)")  # faded
        else:
            node_colors.append(colour)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # ---------------------------------------------
    # HOVER TEXT (neighbors)
    # ---------------------------------------------
    neighbor_texts = []
    for i in range(N):
        neighbors = np.where(adj_np[i] == 1)[0]
        if len(neighbors) == 0:
            neighbor_texts.append(f"{labels[i]}<br>No contacts")
        else:
            names = ", ".join(labels[j] for j in neighbors[:8])
            neighbor_texts.append(
                f"{labels[i]}<br>Connected to: {names} ({len(neighbors)} contacts)"
            )

    # ---------------------------------------------
    # EDGES (hide if degree filter excludes them)
    # ---------------------------------------------
    base_edges, hover_edges = [], []

    for i in range(N):
        for j in range(i + 1, N):
            if adj_np[i, j] == 1:

                # hide edges NOT connected to filtered nodes
                if degree_filter != "None":
                    target = int(degree_filter)
                    if not (degrees[i] == target or degrees[j] == target):
                        continue

                d = dist_matrix[i, j]
                rel = max(0.01, (threshold - d) / threshold)
                width = 1 + 4 * rel
                opacity = 0.35 + 0.65 * rel

                x0, y0, z0 = coords[i]
                x1, y1, z1 = coords[j]

                base_edges.append(
                    go.Scatter3d(
                        x=[x0, x1],
                        y=[y0, y1],
                        z=[z0, z1],
                        mode="lines",
                        line=dict(color=f"rgba(0,0,0,{opacity:.2f})", width=width),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

                hover_edges.append(
                    go.Scatter3d(
                        x=[x0, x1],
                        y=[y0, y1],
                        z=[z0, z1],
                        mode="lines",
                        line=dict(color="red", width=width + 6),
                        hoverinfo="text",
                        hovertext=f"{labels[i]} — {labels[j]}<br>{d:.2f} Å",
                        opacity=0.0,
                        showlegend=False,
                    )
                )

    # ---------------------------------------------
    # NODE GLOW LAYER
    # ---------------------------------------------
    glow = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=[s * 1.8 for s in node_sizes],
            color=node_colors,
            opacity=0.18,
            line=dict(width=0),
        ),
        hoverinfo="none",
        showlegend=False,
    )

    # ---------------------------------------------
    # MAIN NODE LAYER
    # ---------------------------------------------
    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="black")
        ),
        text=neighbor_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    # ---------------------------------------------
    # BUILD FIGURE
    # ---------------------------------------------
    fig = go.Figure(data=base_edges + hover_edges + [glow, node_trace])

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        height=750,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------
    # RESIDUE TYPE LEGEND
    # ---------------------------------------------
    if colour_mode == "Residue Type":
        st.markdown("""
        ### Residue Type Legend
        <div style="display:flex; gap:20px; font-size:14px; margin-bottom:10px;">
            <div><span style="color:#FFA500;">■</span> Hydrophobic</div>
            <div><span style="color:#1f77b4;">■</span> Polar</div>
            <div><span style="color:#2ca02c;">■</span> Positive</div>
            <div><span style="color:#d62728;">■</span> Negative</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD STRUCTURES
# ---------------------------------------------------------
def load_structure_from_upload():
    pdb_bytes = uploaded_file.getvalue()
    pdb_text = pdb_bytes.decode("utf-8")
    parser = PDBParser(QUIET=True)
    return parser.get_structure("uploaded", StringIO(pdb_text))


def load_demo_structure():
    pdbl = PDBList()
    temp_dir = tempfile.gettempdir()
    pdb_path = pdbl.retrieve_pdb_file("1CRN", file_format="pdb", pdir=temp_dir)
    parser = PDBParser(QUIET=True)
    return parser.get_structure("demo", pdb_path)


# ---------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------
def run_pcn_app(structure):

    st.success("PDB Loaded Successfully")

    models = list(structure)
    model_ids = list(range(1, len(models) + 1))
    model_choice = st.selectbox("Select NMR Model", model_ids)

    chains = list(models[model_choice - 1].get_chains())
    chain_ids = [c.id for c in chains]
    chain_choice = st.selectbox("Select Chain", chain_ids)

    adj_df, dist_df, labels, coords, residues = compute_pcn_df(
        structure, model_choice, chain_choice, threshold
    )

    # ---------------- METRICS ----------------
    num_nodes = len(labels)
    num_edges = int(np.sum(adj_df.values) // 2) if num_nodes > 0 else 0
    max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
    density = num_edges / max_edges if max_edges > 0 else 0

    st.markdown(
        f"### **Network Summary**  \n"
        f"- **Nodes:** {num_nodes}  \n"
        f"- **Edges:** {num_edges}  \n"
        f"- **Graph Density:** {density:.4f}"
    )

    st.subheader("Distance Matrix (Preview)")
    if not dist_df.empty:
        st.dataframe(dist_df.iloc[:10, :10])
    else:
        st.write("No distance data to show.")

    st.subheader("Adjacency Matrix (Preview)")
    if not adj_df.empty:
        st.dataframe(adj_df.iloc[:10, :10])
    else:
        st.write("No adjacency data to show.")

    # Downloads
    if not dist_df.empty:
        st.download_button("Download Distance Matrix (CSV)", dist_df.to_csv().encode(), "distance.csv")
    if not adj_df.empty:
        st.download_button("Download Adjacency Matrix (CSV)", adj_df.to_csv().encode(), "adjacency.csv")

    # SIF + TXT
    edges = []
    N = len(labels)
    adjacency_np = adj_df.values if not adj_df.empty else np.zeros((0, 0))

    for i in range(N):
        for j in range(i + 1, N):
            if adjacency_np[i][j] == 1:
                edges.append(f"{labels[i]} pp {labels[j]}")

    sif_text = "\n".join(edges)

    if edges:
        st.download_button("Download SIF (Cytoscape)", sif_text, "network.sif")
        st.download_button("Download Edge List (TXT)", sif_text, "edges.txt")

    # Plot
    draw_pcn_plot_enhanced(labels, coords, adjacency_np, dist_df.values if not dist_df.empty else np.zeros((N, N)), residues)


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if uploaded_file:
    st.session_state["is_demo"] = False
    structure = load_structure_from_upload()
    run_pcn_app(structure)

elif use_demo:
    st.session_state["is_demo"] = True
    structure = load_demo_structure()
    run_pcn_app(structure)

else:
    st.info("Upload a PDB file or try the demo to begin.")

