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

# Help link
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
    min_value=1.0,
    max_value=40.0,
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

default_color = "#8da0cb"

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

    if len(coords) == 0:
        return pd.DataFrame(), pd.DataFrame(), [], np.empty((0, 3)), []

    coords = np.vstack(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
    adjacency = (dist_matrix <= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)

    labels = [f"{res.get_resname().strip()}_{res.get_id()[1]}" for res in residues]

    return (
        pd.DataFrame(adjacency, index=labels, columns=labels),
        pd.DataFrame(dist_matrix, index=labels, columns=labels),
        labels,
        coords,
        residues
    )

# ---------------------------------------------------------
# ENHANCED PLOT FUNCTION
# ---------------------------------------------------------
def draw_pcn_plot_enhanced(labels, coords, adjacency, dist_matrix, residues):

    if len(labels) == 0:
        st.warning("No residues available for visualization.")
        return

    N = len(labels)
    adj_np = np.array(adjacency)

    # Degrees
    degrees = np.sum(adj_np, axis=0)
    deg_min = degrees.min() if len(degrees) else 0
    deg_ptp = np.ptp(degrees) if np.ptp(degrees) > 0 else 1
    min_size, max_size = 6, 18
    node_sizes = min_size + ((degrees - deg_min) / deg_ptp) * (max_size - min_size)

    # ---------------------------------------------------------
    # RESIDUE COLOR MODE (legend at top)
    # ---------------------------------------------------------
    if st.session_state.get("is_demo", False):
        colour_mode = "Single Colour"
    else:
        colour_mode = st.radio(
            "Node Colour Mode",
            ["Single Colour", "Residue Type"],
            label_visibility="collapsed",
            horizontal=True
        )

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
    # DEGREE FILTER
    # ---------------------------------------------------------
    if st.session_state.get("is_demo", False):
        degree_filter = "None"
    else:
        max_degree = int(degrees.max()) if N else 0
        degree_filter = st.selectbox(
            "Highlight residues with degree:",
            ["None"] + list(range(max_degree + 1)),
            index=0
        )

    # ---------------------------------------------------------
    # NODE COLORS WITH DEGREE FILTER
    # ---------------------------------------------------------
    node_colors = []
    for i, res in enumerate(residues):

        name = res.get_resname().strip().upper()
        category = res_type_map.get(name, "UNK")
        base_color = residue_type_colors.get(category, default_color)

        # apply residue type mode
        col = base_color if colour_mode == "Residue Type" else default_color

        # apply filter
        if degree_filter != "None":
            if degrees[i] == int(degree_filter):
                node_colors.append(col)
            else:
                node_colors.append("rgba(180,180,180,0.2)")
        else:
            node_colors.append(col)

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # ---------------------------------------------------------
    # HOVER TEXT
    # ---------------------------------------------------------
    neighbor_texts = []
    for i in range(N):
        neighbors = np.where(adj_np[i] == 1)[0]
        if len(neighbors):
            names = ", ".join(labels[j] for j in neighbors[:8])
            neighbor_texts.append(f"{labels[i]}<br>Contacts: {names} ({len(neighbors)} total)")
        else:
            neighbor_texts.append(f"{labels[i]}<br>No contacts")

    # ---------------------------------------------------------
    # EDGES
    # ---------------------------------------------------------
    base_edges = []
    hover_edges = []

    for i in range(N):
        for j in range(i + 1, N):

            if adj_np[i, j] != 1:
                continue

            if degree_filter != "None":
                if (degrees[i] != int(degree_filter)) and (degrees[j] != int(degree_filter)):
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
                    line=dict(color="red", width=width + 5),
                    hoverinfo="text",
                    hovertext=f"{labels[i]} — {labels[j]}<br>{d:.2f} Å",
                    opacity=0.0,
                    showlegend=False
                )
            )

    # ---------------------------------------------------------
    # NODE GLOW + MAIN NODES
    # ---------------------------------------------------------
    glow = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(
            size=[s * 1.8 for s in node_sizes],
            color=node_colors,
            opacity=0.20,
            line=dict(width=0),
        ),
        hoverinfo="none",
        showlegend=False
    )

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="black")),
        text=neighbor_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False
    )

    fig = go.Figure(data=base_edges + hover_edges + [glow, node_trace])

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=750
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# LOAD STRUCTURES
# ---------------------------------------------------------
def load_structure_from_upload():
    text = uploaded_file.read().decode("utf-8")
    return PDBParser(QUIET=True).get_structure("uploaded", StringIO(text))

def load_demo_structure():
    pdbl = PDBList()
    temp = tempfile.gettempdir()
    path = pdbl.retrieve_pdb_file("1CRN", file_format="pdb", pdir=temp)
    return PDBParser(QUIET=True).get_structure("demo", path)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def run_pcn_app(structure):

    st.success("PDB Loaded Successfully")

    model_ids = list(range(1, len(structure) + 1))
    model_choice = st.selectbox("Select NMR Model", model_ids)

    chain_ids = [c.id for c in structure[model_choice - 1].get_chains()]
    chain_choice = st.selectbox("Select Chain", chain_ids)

    adj_df, dist_df, labels, coords, residues = compute_pcn_df(
        structure, model_choice, chain_choice, threshold
    )

    num_nodes = len(labels)
    num_edges = int(np.sum(adj_df.values) // 2) if num_nodes else 0
    max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 1
    density = num_edges / max_edges

    st.markdown(
        f"### **Network Summary**\n"
        f"- **Nodes:** {num_nodes}\n"
        f"- **Edges:** {num_edges}\n"
        f"- **Graph Density:** {density:.4f}"
    )

    # Matrices
    st.subheader("Distance Matrix (Preview)")
    st.dataframe(dist_df.iloc[:10, :10])

    st.subheader("Adjacency Matrix (Preview)")
    st.dataframe(adj_df.iloc[:10, :10])

    # Downloads
    st.download_button("Download Distance Matrix (CSV)", dist_df.to_csv().encode(), "distance.csv")
    st.download_button("Download Adjacency Matrix (CSV)", adj_df.to_csv().encode(), "adjacency.csv")

    # SIF Export
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj_df.values[i][j] == 1:
                edges.append(f"{labels[i]} pp {labels[j]}")

    sif_text = "\n".join(edges)
    st.download_button("Download SIF (Cytoscape)", sif_text, "network.sif")
    # Plot
    draw_pcn_plot_enhanced(
        labels,
        coords,
        adj_df.values,
        dist_df.values,
        residues
    )

    # ---- CONTACT COUNT TABLE ----
    st.subheader("Residue Contact Counts")

    if len(labels) > 0:
        degrees = np.sum(adj_df.values, axis=0)
        contact_df = pd.DataFrame({
            "Residue": labels,
            "Number of Contacts": degrees
        }).sort_values("Number of Contacts", ascending=False)

        st.dataframe(contact_df, use_container_width=True)

    else:
        st.write("No contact data available.")


   


# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if uploaded_file:
    st.session_state["is_demo"] = False
    run_pcn_app(load_structure_from_upload())

elif use_demo:
    st.session_state["is_demo"] = True
    run_pcn_app(load_demo_structure())

else:
    st.info("Upload a PDB file or try the demo to begin.")
