
import streamlit as st
from Bio.PDB import PDBParser, PDBList
import numpy as np
import pandas as pd
import tempfile
from io import StringIO
import traceback
import plotly.graph_objects as go

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="PCN Explorer", layout="wide")

# -------------------------------------------------------
# HELP BUTTON 
# -------------------------------------------------------
if "show_help" not in st.session_state:
    st.session_state.show_help = False

# -------------------------------------------------------
# TOP HEADER WITH HELP BUTTON
# -------------------------------------------------------
top_col1, top_col2 = st.columns([8, 1])

with top_col1:
    st.markdown(
        """
        <h1 style="margin-bottom:-8px;"> Protein Contact Network Explorer</h1>
        <p style="font-size:16px; color:gray;">Analyze residue contacts · Visualize networks · Export PCN data</p>
        """,
        unsafe_allow_html=True
    )

with top_col2:
    if st.button(" Help", use_container_width=True):
        st.session_state.show_help = not st.session_state.show_help

# -------------------------------------------------------
# -------------------------------------------------------
if st.session_state.show_help:
    st.markdown("## Help & Documentation")
    st.markdown("""
    ### What this tool does
    - Computes **Protein Contact Networks (PCNs)**  
    - Uses Cα–Cα distances  
    - Allows 3D visualization  
    - Exports matrices + Cytoscape files  

    ### How to use
    1. Upload a PDB file (or try demo)
    2. Select **NMR Model** (if available)
    3. Select **Chain**
    4. Enter **contact threshold**
    5. View top-left preview of **distance** and **adjacency matrices**
    6. Explore interactive **3D structure graph**
    7. Download:
        - Distance matrix (CSV)
        - Adjacency matrix (CSV)
        - SIF file (Cytoscape)
        - TXT edge list

    ### Contact rule
    Residues are considered in contact if:  
    **Cα–Cα distance ≤ threshold**

    ### Node coloring (optional)
    - Hydrophobic → orange  
    - Polar → blue  
    - Positive → red  
    - Negative → green   

    ---
    """)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload PDB file", type=["pdb"])
    use_demo = st.button("Try Demo (1CRN)")

    st.divider()

    threshold = st.number_input(
        "Contact threshold (Å)",
        min_value=0.1,
        max_value=20.0,
        value=5.0,
        step=0.1
    )

    color_mode = st.selectbox("Node color mode", ["Single color", "By residue property"])

# -------------------------------------------------------
# RESIDUE PROPERTIES
# -------------------------------------------------------
RES_PROP = {
    "ALA":"hydrophobic","VAL":"hydrophobic","ILE":"hydrophobic","LEU":"hydrophobic",
    "MET":"hydrophobic","PHE":"hydrophobic","TRP":"hydrophobic","PRO":"hydrophobic",

    "SER":"polar","THR":"polar","ASN":"polar","GLN":"polar",
    "CYS":"polar","TYR":"polar","GLY":"polar",

    "LYS":"positive","ARG":"positive","HIS":"positive",

    "ASP":"negative","GLU":"negative"
}

COLOR_MAP = {
    "hydrophobic": "orange",
    "polar": "royalblue",
    "positive": "red",
    "negative": "green",
    "unknown": "gray",
    "default": "cyan"
}

def residue_property(res3):
    return RES_PROP.get(res3, "unknown")

# -------------------------------------------------------
# STRUCTURE LOADING
# -------------------------------------------------------
def parse_structure_from_upload(uploaded_file):
    raw = uploaded_file.getvalue()
    try:
        text = raw.decode("utf-8")
        parser = PDBParser(QUIET=True)
        return parser.get_structure("uploaded", StringIO(text))
    except:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        temp.write(raw)
        temp.flush()
        parser = PDBParser(QUIET=True)
        return parser.get_structure("uploaded", temp.name)

def load_demo_structure():
    pdbl = PDBList()
    path = pdbl.retrieve_pdb_file("1CRN", file_format="pdb", pdir=tempfile.gettempdir())
    parser = PDBParser(QUIET=True)
    return parser.get_structure("demo", path)

def extract_chain_data(structure, model_id, chain_id):
    model = structure[model_id]
    chain = model[chain_id]

    coords = []
    residues = []

    for r in chain.get_residues():
        if "CA" in r:
            try:
                coords.append(r["CA"].get_coord())
            except:
                continue
            residues.append(r)

    coords = np.vstack(coords)
    labels = [f"{r.get_resname()}_{r.get_id()[1]}" for r in residues]
    return coords, labels, residues

def compute_matrices(coords, threshold):
    diff = coords[:,None,:] - coords[None,:,:]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    adj = (dist <= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return dist, adj

# -------------------------------------------------------
# 3D VISUALIZATION
# -------------------------------------------------------
def plot_with_edges(coords, labels, dist, threshold, residues, color_mode):

    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    node_colors = []
    categories_used = set()

    for r in residues:
        prop = residue_property(r.get_resname())
        categories_used.add(prop)
        node_colors.append(
            COLOR_MAP[prop] if color_mode == "By residue property" else COLOR_MAP["default"]
        )

    legend_traces = []
    if color_mode == "By residue property":
        for prop in categories_used:
            legend_traces.append(
                go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode="markers",
                    marker=dict(size=8, color=COLOR_MAP[prop]),
                    name=prop
                )
            )

    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=6, color=node_colors),
        text=labels,
        hoverinfo="text",
        name="Residues"
    )

    edges = []
    N = len(coords)
    for i in range(N):
        for j in range(i+1, N):
            if dist[i,j] <= threshold:
                edges.append(
                    go.Scatter3d(
                        x=[x[i], x[j]],
                        y=[y[i], y[j]],
                        z=[z[i], z[j]],
                        mode="lines",
                        line=dict(color="black", width=1),
                        hoverinfo="text",
                        text=f"{labels[i]} - {labels[j]}: {dist[i,j]:.2f} Å",
                        showlegend=False
                    )
                )

    fig = go.Figure(data=legend_traces + [node_trace] + edges)
    fig.update_layout(height=700, legend=dict(orientation="h", y=1.12))
    return fig

# -------------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------------
structure = None
if uploaded_file:
    structure = parse_structure_from_upload(uploaded_file)
elif use_demo:
    structure = load_demo_structure()

if structure is None:
    st.info("Upload a PDB file or use the demo to begin.")
    st.stop()

try:
    # STRUCTURE SELECTION
    st.header("Structure Selection")

    # FIXED: display real model numbering (1–N instead of 0–N)
    models = list(structure)
    model_ids_display = [m.id + 1 for m in models]
    selected_display = st.selectbox("Select NMR Model", model_ids_display)
    model = selected_display - 1  # convert back to BioPython indexing

    chain_ids = [c.id for c in structure[model].get_chains()]
    chain = st.selectbox("Select Chain", chain_ids)

    coords, labels, residues = extract_chain_data(structure, model, chain)
    dist, adj = compute_matrices(coords, threshold)

    st.success(f"Loaded {len(labels)} residues from Model {selected_display}, Chain {chain}")

    # MATRIX PREVIEW
    st.subheader("Matrix Previews (Top-left 10×10)")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Distance Matrix**")
        st.dataframe(pd.DataFrame(dist, index=labels, columns=labels).iloc[:10,:10])

    with col2:
        st.write("**Adjacency Matrix**")
        st.dataframe(pd.DataFrame(adj, index=labels, columns=labels).iloc[:10,:10])

    # 3D NETWORK
    st.markdown("---")
    st.subheader("Interactive 3D PCN Structure")
    fig = plot_with_edges(coords, labels, dist, threshold, residues, color_mode)
    st.plotly_chart(fig, use_container_width=True)

    # FULL MATRICES + DOWNLOADS
    st.markdown("---")
    st.subheader("Full Matrices & Downloads")

    dist_df = pd.DataFrame(dist, index=labels, columns=labels)
    adj_df = pd.DataFrame(adj, index=labels, columns=labels)

    st.write("### Distance Matrix")
    st.dataframe(dist_df)

    st.write("### Adjacency Matrix")
    st.dataframe(adj_df)

    st.download_button("Download Distance CSV", dist_df.to_csv().encode(), "distance.csv")
    st.download_button("Download Adjacency CSV", adj_df.to_csv().encode(), "adjacency.csv")

    sif, txt = [], []
    N = len(labels)

    for i in range(N):
        for j in range(i+1, N):
            if adj[i,j] == 1:
                sif.append(f"{labels[i]} pp {labels[j]}")
                txt.append(f"{labels[i]} {labels[j]}")

    st.download_button("Download SIF (Cytoscape)", "\n".join(sif), "network.sif")
    st.download_button("Download Edge List (TXT)", "\n".join(txt), "edges.txt")

except Exception as e:
    st.error("Error processing structure")
    st.code(traceback.format_exc())
