"""
Flask Server for SABER Annotation GUI
Serves static files and handles API endpoints
"""

import os
import json
import logging
import traceback
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import zarr
import numpy as np
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(data_path: str, output_path: str = None):
    """Create Flask application with configuration"""
    
    # Get the directory where this script is located
    base_dir = Path(__file__).parent
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Files in base directory: {list(base_dir.glob('*'))}")
    
    app = Flask(__name__,
                template_folder=str(base_dir),
                static_folder=str(base_dir))
    CORS(app)
    
    # Store configuration
    app.config['DATA_PATH'] = Path(data_path)
    app.config['OUTPUT_PATH'] = Path(output_path) if output_path else None
    
    # Open zarr store
    try:
        app.zarr_root = zarr.open(str(app.config['DATA_PATH']), mode='r')
        app.run_ids = list(app.zarr_root.keys())
        logger.info(f"Loaded {len(app.run_ids)} runs from {data_path}")
    except Exception as e:
        logger.error(f"Failed to open zarr store: {e}")
        app.zarr_root = None
        app.run_ids = []
    
    return app


def extract_mask_values(masks: np.ndarray) -> tuple:
    """Extract mask values from 2D or 3D mask array - vectorized for speed"""
    mask_values = []
    extracted_masks = []
    
    if len(masks.shape) == 2:
        # 2D label map - extract individual masks using vectorized operations
        unique_values = np.unique(masks[masks > 0])
        num_masks = len(unique_values)
        
        if num_masks == 0:
            return [], []
        
        # Vectorized mask extraction
        masks_3d = masks[np.newaxis, :, :] == unique_values[:, np.newaxis, np.newaxis]
        extracted_masks = [masks_3d[i].astype(np.uint8) for i in range(num_masks)]
        mask_values = unique_values.tolist()
        
    elif len(masks.shape) == 3:
        # Stack of masks - extract values
        for i, mask in enumerate(masks):
            unique_vals = np.unique(mask[mask > 0])
            if len(unique_vals) > 0:
                mask_values.append(float(unique_vals[0]))
            else:
                mask_values.append(float(i + 1))
            extracted_masks.append(mask.astype(np.uint8))
    
    return mask_values, extracted_masks


def apply_rotation(image: np.ndarray, masks: np.ndarray) -> tuple:
    """Apply 90-degree rotation to image and masks"""
    if image.ndim == 2:
        # 2D grayscale
        image = np.rot90(image, k=-1)
        if masks.ndim == 2:
            masks = np.rot90(masks, k=-1)
        else:
            masks = np.rot90(masks, k=-1, axes=(1, 2))
    elif image.ndim == 3 and image.shape[0] == 3:
        # 2D RGB
        image = np.rot90(image, k=-1, axes=(1, 2))
        if masks.ndim == 2:
            masks = np.rot90(masks, k=-1)
        else:
            masks = np.rot90(masks, k=-1, axes=(1, 2))
    elif image.ndim == 3:
        # 3D volume
        image = np.rot90(image, k=-1, axes=(1, 2))
        if isinstance(masks, list):
            masks = [np.rot90(m, k=-1, axes=(1, 2)) if m.ndim == 3 
                    else np.rot90(m, k=-1) for m in masks]
        else:
            masks = np.rot90(masks, k=-1, axes=(-2, -1))
    
    return image, masks

def _sanitize_for_log(s: str) -> str:
    # remove common log-breaking chars + a couple of unicode line separators
    return (s or "").replace("\r", "").replace("\n", "").replace("\t", "").replace("\u2028", "").replace("\u2029", "")



def run_server(data_path: str, 
               output_path: str = None,
               host: str = '0.0.0.0', 
               port: int = 8080,
               debug: bool = False):
    """Run the Flask server"""
    
    app = create_app(data_path, output_path)
    base_dir = Path(__file__).parent
    
    @app.route('/')
    def index():
        """Serve the main HTML interface"""
        try:
            return render_template('gui.html')
        except Exception as e:
            logger.error(f"Error serving index: {_sanitize_for_log(str(e))}")
            logger.error(f"Template folder: {app.template_folder}")
            logger.error(f"Template files: {list(Path(app.template_folder).glob('*.html'))}")
            return f"Error: {e}", 500
    
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files (CSS, JS)"""
        try:
            safe_filename = filename.replace('\r\n', '').replace('\n', '').replace('\r', '')
            logger.info(f"Serving static file: {safe_filename} from {base_dir}")
            return send_from_directory(base_dir, filename)
        except Exception as e:
            safe_filename = filename.replace('\r\n', '').replace('\n', '').replace('\r', '')
            logger.error(f"Error serving static file {safe_filename}: {_sanitize_for_log(str(e))}")
            return "File not found", 404
    
    @app.route('/api/runs')
    def get_runs():
        """Get list of available runs"""
        return jsonify({'runs': app.run_ids})
    
    @app.route('/api/runs/<run_id>')
    def get_run_data(run_id):
        """Get data for a specific run"""
        if run_id not in app.run_ids:
            return jsonify({'error': 'Run not found'}), 404
        
        try:
            # Check if rotation requested
            apply_rotate = request.args.get('rotate', 'false').lower() == 'true'
            
            # Read data
            image = app.zarr_root[run_id][0][:]
            try:
                masks = app.zarr_root[run_id]['labels'][0][:]
            except:
                masks = app.zarr_root[run_id]['masks'][:]
            
            # Apply rotation if requested
            if apply_rotate:
                image, masks = apply_rotation(image, masks)
            
            # Handle different image dimensions
            if image.ndim == 2:
                # 2D grayscale - keep as is
                pass
            elif image.ndim == 3 and image.shape[0] == 3:
                # RGB image - convert to grayscale for web display
                image = np.mean(image, axis=0)
            elif image.ndim == 3:
                # 3D volume - take middle slice for 2D web view
                nz = image.shape[0]
                mid_z = nz // 2
                image = image[mid_z]
                if len(masks.shape) == 4:
                    masks = masks[:, mid_z, :, :]
                elif len(masks.shape) == 3 and masks.shape[0] > 1:
                    # Assume first dim is Z
                    masks = masks[mid_z, :, :]
            
            # Extract mask values
            mask_values, extracted_masks = extract_mask_values(masks)
            
            # Prepare response
            response_data = {
                'image': image.tolist(),
                'masks': [m.tolist() for m in extracted_masks],
                'mask_values': mask_values,
                'shape': list(image.shape)
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            sanitized_run_id = run_id.replace('\r', '').replace('\n', '')
            logger.error(f"Error loading run {sanitized_run_id}: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/save', methods=['POST'])
    def save_annotations():
        """Save annotations to JSON file"""
        try:
            data = request.json
            
            # Determine output path
            if app.config['OUTPUT_PATH']:
                output_file = app.config['OUTPUT_PATH'] / 'annotations.json'
            else:
                # Save in current directory if no output path specified
                output_file = Path('annotations.json')
            
            # Create directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the annotations
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved annotations to {output_file}")
            return jsonify({'success': True, 'file': str(output_file)})
            
        except Exception as e:
            logger.error(f"Error saving annotations: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/status')
    def get_status():
        """Get server status"""
        status = {
            'runs_loaded': len(app.run_ids),
            'data_path': str(app.config['DATA_PATH']),
            'output_path': str(app.config['OUTPUT_PATH']) if app.config['OUTPUT_PATH'] else None,
        }
        
        return jsonify(status)
    
    # Run the server
    logger.info(f"Starting server at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)