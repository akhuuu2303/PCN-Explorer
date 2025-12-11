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

default_color = "#636EFA"


# ---------------------------------------------------------
# PCN COMPUTATION
# ---------------------------------------------------------
def compute_pcn_df(structure, model_id, chain_id, threshold):
    model = structure[int(model_id) - 1]
    chain = model[chain_id]

    residues = []
    coords = []

    for res in chain.get_residues():
        if "CA" in res:
            residues.append(res)
            coords.append(res["CA"].get_coord())

    coords = np.vstack(coords)

    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
    adjacency = (dist_matrix <= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    labels = [f"{res.get_resname()}_{res.get_id()[1]}" for res in residues]

    adj_df = pd.DataFrame(adjacency, index=labels, columns=labels)
    dist_df = pd.DataFrame(dist_matrix, index=labels, columns=labels)

    return adj_df, dist_df, labels, coords, residues


# ---------------------------------------------------------
# PLOTLY 3D VISUALIZER
# ---------------------------------------------------------
def draw_pcn_plot(labels, coords, adjacency, dist_matrix, residues):

    # ---------------- DISABLE RESIDUE TYPE FOR DEMO ----------------
    if st.session_state.get("is_demo", False):
        colour_mode = "Single Colour"
    else:
        colour_mode = st.radio(
            "Node Colour Mode",
            ["Single Colour", "Residue Type"],
            horizontal=True,
            label_visibility="collapsed",
        )

    # Assign node colours
    node_colors = []
    for res in residues:
        name = res.get_resname().upper().strip()
        if name not in res_type_map:
            name = "UNK"

        category = res_type_map.get(name, "polar")  # hydrophobic/polar/positive/negative

        if colour_mode == "Residue Type":
            node_colors.append(residue_type_colors[category])
        else:
            node_colors.append(default_color)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # ---------- BUILD EDGE TRACES ----------
    base_edges = []
    hover_edges = []
    N = len(labels)

    for i in range(N):
        for j in range(i + 1, N):
            if adjacency[i][j] == 1:
                d = dist_matrix[i][j]
                x0, y0, z0 = coords[i]
                x1, y1, z1 = coords[j]

                base_edges.append(
                    go.Scatter3d(
                        x=[x0, x1],
                        y=[y0, y1],
                        z=[z0, z1],
                        mode="lines",
                        line=dict(color="black", width=2),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

                hover_edges.append(
                    go.Scatter3d(
                        x=[x0, x1],
                        y=[y0, y1],
                        z=[z0, z1],
                        mode="lines+markers",
                        line=dict(color="red", width=6),
                        marker=dict(size=4, color="red", opacity=0),
                        hoverinfo="text",
                        hovertext=f"{labels[i]} — {labels[j]}<br>Distance: {d:.2f} Å",
                        opacity=0.0,
                        hoverlabel=dict(bgcolor="red", font_size=14, font_color="white"),
                    )
                )

    # ---------- NODE TRACE ----------
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=7, color=node_colors, line=dict(width=2, color="white")),
        text=labels,
        hovertemplate="Residue: %{text}<extra></extra>",
    )

    # ---------- FIGURE ----------
    fig = go.Figure(base_edges + hover_edges + [node_trace])
    fig.update_layout(
        height=700,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    if colour_mode == "Residue Type":
        st.markdown("### Residue Type Legend")
        st.markdown("""
        <div style="display:flex; gap:20px; font-size:16px;">
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
# MAIN WEBSITE LOGIC
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

    # ---------------- METRICS (NODES/EDGES/GRAPH DENSITY) ----------------
    num_nodes = len(labels)
    num_edges = int(np.sum(adj_df.values) // 2)
    max_edges = num_nodes * (num_nodes - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0

    st.markdown(
        f"### **Network Summary**  \n"
        f"- **Nodes:** {num_nodes}  \n"
        f"- **Edges:** {num_edges}  \n"
        f"- **Graph Density:** {density:.4f}"
    )

    st.subheader("Distance Matrix (Preview)")
    st.dataframe(dist_df.iloc[:10, :10])

    st.subheader("Adjacency Matrix (Preview)")
    st.dataframe(adj_df.iloc[:10, :10])

    # Downloads
    st.download_button("Download Distance Matrix (CSV)", dist_df.to_csv().encode(), "distance.csv")
    st.download_button("Download Adjacency Matrix (CSV)", adj_df.to_csv().encode(), "adjacency.csv")

    # SIF + TXT
    edges = []
    N = len(labels)
    adjacency_np = adj_df.values

    for i in range(N):
        for j in range(i + 1, N):
            if adjacency_np[i][j] == 1:
                edges.append(f"{labels[i]} pp {labels[j]}")

    sif_text = "\n".join(edges)

    st.download_button("Download SIF (Cytoscape)", sif_text, "network.sif")
    st.download_button("Download Edge List (TXT)", sif_text, "edges.txt")

    # Plot
    draw_pcn_plot(labels, coords, adjacency_np, dist_df.values, residues)


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

