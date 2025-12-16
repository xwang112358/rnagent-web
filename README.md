# ROBIN: Repository Of BInders to Nucleic acids

A Flask web application for exploring RNA-targeting small molecules from the ROBIN dataset, featuring UMAP visualization, molecular property analysis, and an AI-powered chatbot for RNA structure insights.

## Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate web
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
AZURE_OPENAI_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
SECRET_KEY=your_flask_secret_key
```

### 3. Run the Server

```bash
python server.py
```

The app will be available at `http://127.0.0.1:5000`

## Codebase Structure

```
├── server.py              # Flask server with all API endpoints
├── chatbox.py             # RCSB PDB data fetching and LLM prompting
├── templates/
│   ├── index.html         # Landing page with target selection
│   └── analyze.html       # Analysis dashboard with visualizations
├── static/                # Static assets (images)
├── robin_clean.csv        # Main dataset (~24k molecules, 5 RNA targets)
├── robin_sequence.csv     # RNA target sequences
├── umap_cache/            # Pre-computed UMAP embeddings (auto-generated)
├── RCSB_data/             # Cached PDB structure data (auto-generated)
└── environment.yml        # Conda environment specification
```

## Features

- **Target Selection**: Choose from 5 RNA targets (TPP, Glutamine_RS, ZTP, SAM_ll, PreQ1)
- **UMAP Visualization**: Interactive chemical space plot with k-means clustering
- **Molecular Properties**: QED and SA score distributions with filtering
- **AI Chatbot**: Context-aware Q&A about RNA structures using RCSB PDB data

