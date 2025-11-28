from flask import Flask, render_template, request, jsonify
import pandas as pd
import requests
import json
import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

app = Flask(__name__)

# Load the dataset once when the server starts
df = pd.read_csv("robin_clean.csv")

# Cache directory for UMAP embeddings
CACHE_DIR = Path("umap_cache")
CACHE_DIR.mkdir(exist_ok=True)

def smiles_to_image_base64(smiles, size=(200, 200)):
    """Convert SMILES to base64-encoded PNG image for hover display."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating image for SMILES {smiles}: {e}")
        return None

def compute_and_cache_umap(target):
    """Compute UMAP embeddings for a target and cache to disk."""
    print(f"Computing UMAP for {target}...")
    
    # Get active molecules
    active_df = df[df[target] == 1]
    
    if len(active_df) < 3:
        return None
    
    smiles_list = active_df["Smile"].tolist()
    names_list = active_df["Name"].tolist()
    
    try:
        import chemplot as cp
        plotter = cp.Plotter.from_smiles(smiles_list)
        umap_df = plotter.umap()
        
        # Generate base64 images for each molecule
        print(f"Generating molecule images for {target}...")
        images_list = [smiles_to_image_base64(smiles) for smiles in smiles_list]
        
        # Extract coordinates as lists
        cache_data = {
            'umap_x': umap_df['UMAP-1'].tolist(),
            'umap_y': umap_df['UMAP-2'].tolist(),
            'names': names_list,
            'smiles': smiles_list,
            'images': images_list
        }
        
        # Save to cache
        cache_file = CACHE_DIR / f"{target}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"✓ Cached UMAP for {target} ({len(names_list)} molecules)")
        return cache_data
        
    except Exception as e:
        print(f"✗ Error computing UMAP for {target}: {e}")
        return None

def load_cached_umap(target):
    """Load UMAP embeddings from cache."""
    cache_file = CACHE_DIR / f"{target}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

# Pre-compute UMAP embeddings for all targets on startup
def initialize_umap_cache():
    """Pre-compute and cache UMAP embeddings for all targets."""
    valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
    print("\n" + "="*50)
    print("Initializing UMAP cache...")
    print("="*50)
    
    for target in valid_targets:
        cache_data = load_cached_umap(target)
        if cache_data is None:
            compute_and_cache_umap(target)
        else:
            print(f"✓ Using cached UMAP for {target} ({len(cache_data['names'])} molecules)")
    
    print("="*50)
    print("UMAP cache initialization complete!\n")

# Initialize cache on startup
# Run this in a separate thread to not block server startup
def init_cache_async():
    import threading
    def run_init():
        import time
        time.sleep(1)  # Give server time to start
        initialize_umap_cache()
    thread = threading.Thread(target=run_init, daemon=True)
    thread.start()

# Only initialize on the main process (not on reloader)
if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    init_cache_async()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    target = request.form["target"]
    
    # PDB ID mapping
    pdb_ids = {
        "TPP": "2GDI",
        "Glutamine_RS": "6QN3",
        "ZTP": "5BTP",
        "SAM_ll": "2QWY",
        "PreQ1": "3FU2"
    }
    
    # Check if target is valid
    valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
    if target not in valid_targets:
        result = f"Invalid target. Please choose from: {', '.join(valid_targets)}"
    else:
        # Count active molecules (where target column equals 1)
        num_hits = (df[target] == 1).sum()
        pdb_id = pdb_ids[target]
        result = f"Number of active molecules (hits) for {target}: {num_hits}"
        
        # Fetch PDB title from RCSB API
        pdb_title = "Title not available"
        try:
            api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                pdb_title = data.get("struct", {}).get("title", "Title not available")
        except Exception as e:
            pdb_title = f"Error fetching title: {str(e)}"
        
        # Get example active molecules (up to 10)
        active_molecules = df[df[target] == 1]["Smile"].head(10).tolist()
        examples = "<br>".join(active_molecules) if active_molecules else "No active molecules found"
    
    return render_template("analyze.html", analysis=result, target=target, examples=examples, 
                         pdb_id=pdb_id, pdb_title=pdb_title)


@app.route("/regenerate_umap/<target>")
def regenerate_umap(target):
    """Manually regenerate UMAP cache for a specific target."""
    valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
    if target not in valid_targets:
        return jsonify({"error": f"Invalid target: {target}"}), 400
    
    cache_data = compute_and_cache_umap(target)
    if cache_data:
        return jsonify({"success": f"UMAP regenerated for {target}", "num_molecules": len(cache_data['names'])})
    else:
        return jsonify({"error": f"Failed to generate UMAP for {target}"}), 500


@app.route("/umap_plot", methods=["POST"])
def umap_plot():
    try:
        target = request.form["target"]
        
        # Validate target
        valid_targets = ["TPP", "Glutamine_RS", "ZTP", "SAM_ll", "PreQ1"]
        if target not in valid_targets:
            return jsonify({"error": f"Invalid target: {target}"}), 400
        
        # Load cached UMAP data
        cache_data = load_cached_umap(target)
        
        if cache_data is None:
            return jsonify({"error": f"No UMAP data available for {target}. Not enough active molecules (need at least 3)."}), 400
        
        # Import Bokeh libraries
        from bokeh.plotting import figure
        from bokeh.models import HoverTool, ColumnDataSource
        from bokeh.embed import components
        
        # Create ColumnDataSource with all data including images
        source = ColumnDataSource(data={
            'x': cache_data['umap_x'],
            'y': cache_data['umap_y'],
            'name': cache_data['names'],
            'smiles': cache_data['smiles'],
            'image': cache_data['images']
        })
        
        # Create figure
        p = figure(
            width=900,
            height=600,
            title=f"UMAP Visualization of Active Molecules for {target} ({len(cache_data['names'])} molecules)",
            x_axis_label="UMAP-1",
            y_axis_label="UMAP-2",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="scale_width"
        )
        
        # Add scatter points
        p.circle(
            'x', 'y',
            source=source,
            size=10,
            color='#2196F3',
            alpha=0.7,
            line_color='white',
            line_width=1
        )
        
        # Configure hover tool with molecule image
        hover = HoverTool(tooltips="""
            <div style="width: 250px;">
                <div style="font-weight: bold; font-size: 14px; margin-bottom: 5px;">
                    @name
                </div>
                <div style="margin-bottom: 5px;">
                    <strong>UMAP-1:</strong> @x{0.00}<br>
                    <strong>UMAP-2:</strong> @y{0.00}
                </div>
                <div style="border: 2px solid #2196F3; padding: 5px; background-color: white;">
                    <img src="@image" width="200" style="display: block; margin: 0 auto;">
                </div>
                <div style="font-size: 10px; color: #666; margin-top: 5px; font-family: monospace;">
                    @smiles
                </div>
            </div>
        """)
        p.add_tools(hover)
        
        # Style the plot
        p.background_fill_color = "#f8f9fa"
        p.border_fill_color = "white"
        p.title.text_font_size = "14pt"
        p.title.align = "center"
        
        # Get components (script and div) for embedding
        script, div = components(p)
        
        # Return as JSON with script and div
        return jsonify({
            'script': script,
            'div': div
        })
        
    except Exception as e:
        return jsonify({"error": f"Error generating UMAP plot: {str(e)}"}), 500


if __name__ == "__main__":
    # Enable debug mode and the reloader so changes are picked up automatically
    # During development you can run with: python server.py
    # If you prefer the flask CLI, set FLASK_APP and FLASK_DEBUG (see README/run notes below)
    app.run(debug=True, use_reloader=True)

    # Optional: to auto-refresh the browser (no manual refresh) install `livereload`:
    #   pip install livereload
    # Then replace the app.run(...) above with the lines below (or uncomment and run):
    # from livereload import Server
    # server = Server(app.wsgi_app)
    # server.watch('templates/')
    # server.watch('static/')
    # server.watch('*.py')
    # server.serve(port=5000, host='127.0.0.1')
