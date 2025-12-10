HELP.md — PCN Explorer User Guide
Protein Contact Network (PCN) Explorer — Help & Instructions

This page provides a quick reference guide for using the PCN Explorer web application, including file upload, parameter selection, visualization options, and available downloads.

## 1. What is a Protein Contact Network?
A Protein Contact Network (PCN) is a graph-based representation of a protein structure where:

Nodes = amino acid residues (represented by their Cα atoms)

Edges = residue pairs whose Cα–Cα distance is ≤ a chosen threshold

PCNs reveal structural connectivity patterns, residue interactions, stability features, and topological properties of proteins.

## 2. Uploading a PDB File

You can upload any valid PDB file using the sidebar:

Click Browse files

Select a .pdb file from your computer

The app will automatically parse:

Chains
Models (if NMR ensemble)
Residue coordinates
Alternatively, click Try Demo (1CRN) to load a sample protein.

## 3. Selecting NMR Model and Chain

If the uploaded PDB contains:

 Multiple NMR models

You will see:
Select NMR Model: 1, 2, 3, ...

Select exactly one model to analyze.

Multiple chains

You will see:
Select Chain: A, B, C, ...

Only one chain can be analyzed per PCN.

## 4. Contact Threshold

The contact threshold determines which residues are considered “in contact.”

Default: 5.0 Å

Residue pairs with Cα–Cα distance ≤ threshold receive a value 1 in the adjacency matrix.

## 5. Matrices Provided
🔹 Distance Matrix (CSV)

An N × N matrix containing pairwise Cα distances.

🔹 Adjacency Matrix (CSV)

Binary contact matrix:

1 → residues are in contact  
0 → no contact  


Both matrices are displayed (top-left preview) and can be downloaded.

## 6. Network Visualization

The interactive Plotly viewer shows:

Residues as nodes

Distance-threshold contacts as edges

Hover labels showing:

Residue name

Residue number

Coordinates

Color modes:

Single color

Residue type categories:

Hydrophobic

Polar

Positive

Negative

Edges turn into thin black lines connecting nodes that are in contact.

## 7. Downloadable forms
✔ Distance Matrix (CSV)

✔ Adjacency Matrix (CSV)

✔ SIF File (for Cytoscape)

Format:

RESIDUE1 pp RESIDUE2

✔ TXT Edge List

A simple list of contacting residue pairs.

These formats allow downstream network analysis using external tools.

## 8. Common Errors & Solutions
“No CA atoms found”

Your PDB file may contain:

Missing residues

Non-standard chains

Non-protein entries

Ensure CA atoms exist.

“Model index out of range”

You may have selected a model number not present in the structure.

Blank visualization

Occurs if:

No residues were parsed

Threshold is too low

The structure contains only heteroatoms

Increase the threshold or check your chain selection.

## 9. Citation / Academic Use

If you use PCN Explorer for a publication or project, please include:

PCN Explorer — Protein Contact Network Generator (2025), by Akhurath Ganapathy.
