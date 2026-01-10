# PCN Explorer: Protein Contact Network Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

**PCN Explorer** is a computational tool designed to analyze protein tertiary structures using graph theory. By abstracting amino acid residues as nodes and their spatial interactions as edges, it constructs **Protein Contact Networks (PCNs)** to quantify structural stability, communication bottlenecks, and functional hubs.

This application supports both standard X-ray crystallographic structures (`.pdb`) and multi-model NMR ensembles, providing real-time 3D visualization alongside rigorous topological metrics.

---

## Key Features

### 1. Network Construction & Visualization
* **Geometric Thresholding:** Constructs networks based on Cα–Cα Euclidean distances (default: 7.5 Å).
* **3D Interactive Explorer:** Real-time rendering of nodes and edges with distinct coloring for Hydrophobic, Polar, Positive, and Negative residues.
* **Hub Identification:** Automatic highlighting of high-degree nodes (structural hubs).

### 2. Advanced Toplogical Metrics
* **Wasserman-Faust Closeness Centrality:** Implements the improved formula for closeness centrality to accurately handle disconnected components (common in fragmented PDB chains).
* **Degree–Betweenness Analysis:** A specialized scatter plot that partitions residues into four functional roles based on user-defined percentiles:
    * <span style="color:#d32f2f">**Global Critical:**</span> High Degree / High Betweenness
    * <span style="color:#f57c00">**Bottlenecks:**</span> Low Degree / High Betweenness
    * <span style="color:#1976d2">**Structural Hubs:**</span> High Degree / Low Betweenness
    * <span style="color:#757575">**Peripheral:**</span> Low Degree / Low Betweenness

### 3. Data Export
* **Adjacency & Distance Matrices:** Downloadable CSVs and interactive Heatmaps.
* **Network Files:** Export networks as `.sif` (Cytoscape compatible) or Edge Lists.
* **Statistical Reports:** Comprehensive CSV reports containing degree, clustering coefficients, and centrality scores for every residue.

---

## Installation Procedures:

### Prerequisites
Ensure you have **Python 3.9+** installed.

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/PCN-Explorer.git](https://github.com/YourUsername/PCN-Explorer.git)
cd PCN-Explorer
```
### 2.Set Up The Environment
```bash
# Create a virtual environment
python -m venv venv

# For Windows
.\venv\Scripts\activate

# For Mac/Linux
source venv/bin/activate
```
### 3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
### 4. Run the application 
```bash
streamlit run Protein_penult.py
```
## Methodology

### Network Definition
The Protein Contact Network (PCN) is defined as an undirected graph $G = (V, E)$ where:
* **Nodes ($V$):** Cα atoms of the amino acid residues.
* **Edges ($E$):** Exist between residues $i$ and $j$ if the Euclidean distance $d_{ij} \le$ Threshold (default 7.5 Å).
* **Adjacency Matrix ($A$):** $A_{ij} = 1$ if connected, $0$ otherwise.

### Mathematical Formulations

#### 1. Node Degree (Hub Identification)
The degree $k_i$ of a residue $i$ is the number of other residues it is in contact with. Residues with a degree in the top percentile (e.g., top 10%) are identified as **Hubs**.

$$k_i = \sum_{j} A_{ij}$$

#### 2. Local Clustering Coefficient
Measures the degree to which a residue's neighbors are also connected to each other (local cohesiveness). For a residue $i$ with degree $k_i$:

$$C_i = \frac{2e_i}{k_i(k_i - 1)}$$

Where $e_i$ is the number of actual edges between the neighbors of residue $i$.

#### 3. Closeness Centrality (Normalized)
Represents how close a residue is to all other residues. It is calculated using the **Wasserman and Faust** formula to strictly normalize values between 0 and 1, even for disconnected graphs.

$$C_{close}(i) = \frac{n-1}{\sum_{j \neq i} d(i,j)} \cdot \frac{n-1}{N-1}$$

Where $d(i, j)$ is the shortest path distance, $n$ is the number of reachable nodes, and $N$ is the total nodes.

#### 4. Betweenness Centrality
Quantifies the influence of a residue on the flow of information through the network. It is defined as the fraction of all shortest paths $\sigma_{st}$ between any pair of nodes $(s, t)$ that pass through node $i$.

$$C_{between}(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

### Functional Partitioning
Residues are classified into four functional roles by comparing their metrics against user-defined percentile thresholds (Default: Top 10% Degree, Top 5% Betweenness):

1.  **Global Critical:** High Degree, High Betweenness (Structural & Functional Hubs).
2.  **Structural Hubs:** High Degree, Low Betweenness (Local Stability Centers).
3.  **Bottlenecks:** Low Degree, High Betweenness (Critical Bridges/Connectors).
4.  **Peripheral:** Low Degree, Low Betweenness (Surface/Flexible Regions).
