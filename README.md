# Protein Contact Network (PCN) Explorer

PCN-Explorer is a Streamlit-based application for computing and visualizing residue-level Protein Contact Networks (PCNs) using protein structures in PDB format. The tool supports standard X-ray structures and multi-model NMR ensembles, allowing users to interactively select models, chains, and distance thresholds for contact definition.

---

## Overview

Protein Contact Networks represent amino acid residues as nodes and residue–residue contacts as edges.  
This tool constructs PCNs using Cα–Cα geometric distances and provides both numerical matrix outputs and interactive 3D structural visualizations.

---

## Features

- Upload PDB files or use a built-in example (1CRN)
- Supports multi-model NMR structures with model selection
- Supports multi-chain structures with chain selection
- Computes:
  - Full Cα–Cα distance matrix
  - Binary adjacency matrix (threshold-based contacts)
- Interactive 3D visualization of the residue network with a choice of highlighting specific residues and nodes given number of contacts or degree
- Downloadable outputs:
  - Distance matrix (CSV)
  - Adjacency matrix (CSV)
  - SIF file (Cytoscape-compatible)
  - TXT edge list
- Optional residue-type coloring (hydrophobic, polar, positive, negative)
- A Residue Degree Distribution graph 
  

## Method

1. Cα coordinates are extracted for all residues in the selected chain.
2. Pairwise Euclidean distances are computed to form an N × N distance matrix.
3. The adjacency matrix is defined as:
   
   **contact(i, j) = 1 if Cα–Cα distance ≤ threshold, else 0**

4. An interactive network visualization is generated with residues as nodes and threshold-based contacts as edges.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/PCN-Explorer
cd PCN-Explorer
```

To install the necessary dependencies do:
```bash 
pip install -r requirements.txt
```

## Launch the Application with:

```bash 
streamlit run Protein_penult.py
```
## Requirements
The following dependencies will be installed into your system:

-plotly

-numpy

-pandas

-biopython

-streamlit

-networkx

-Pillow

-scipy

## Authors 
Akhurath Ganapathy (2025)
Sanjana Vijay Krishnan (2025)






