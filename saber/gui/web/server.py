"""
Complete SAM2-ET Annotation GUI Web Server Implementation
=========================================================
This is the full server.py file with integrated HTML interface
"""

import os
import json
import logging
import base64
import io
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS
import zarr
import numpy as np
from PIL import Image
import traceback

logger = logging.getLogger(__name__)

# Complete HTML template with full functionality
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM2-ET Tomogram Annotation GUI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header h1 {
            font-size: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .help-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
        }

        .container {
            flex: 1;
            display: flex;
            gap: 1rem;
            padding: 1rem;
            max-width: 1600px;
            width: 100%;
            margin: 0 auto;
        }

        .left-panel {
            width: 200px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            max-height: calc(100vh - 120px);
        }

        .panel-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }

        .run-list {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .run-item {
            padding: 0.75rem;
            background: #f7fafc;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
            font-size: 0.9rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .run-item:hover {
            background: #edf2f7;
            transform: translateX(5px);
        }

        .run-item.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .run-item.good {
            border-color: #48bb78;
            background: #f0fff4;
        }

        .run-item.good.active {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        }

        .right-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
        }

        .viewer-container {
            flex: 1;
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
            min-height: 400px;
        }

        .viewer {
            flex: 1;
            position: relative;
            background: #2d3748;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
        }

        .viewer-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            z-index: 10;
        }

        canvas {
            width: 100%;
            height: 100%;
            display: block;
            cursor: crosshair;
            image-rendering: pixelated;
        }

        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
            padding: 1rem;
            background: #f7fafc;
            border-radius: 8px;
        }

        .class-selector {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex: 1;
        }

        .class-dropdown {
            flex: 1;
            padding: 0.5rem;
            border: 2px solid #e2e8f0;
            border-radius: 6px;
            background: white;
            font-size: 0.95rem;
        }

        .save-btn {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
        }

        .status-bar {
            margin-top: 1rem;
            padding: 1rem;
            background: #f7fafc;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
        }

        .loading-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.2rem;
            z-index: 100;
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }

        .modal h2 {
            margin-bottom: 1rem;
            color: #2d3748;
        }

        .modal-close {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 1rem;
        }

        .error-message {
            background: #fed7d7;
            color: #742a2a;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SAM2-ET Tomogram Annotation GUI</h1>
        <button class="help-btn" onclick="showHelp()">Help & Tutorial</button>
    </div>

    <div class="container">
        <div class="left-panel">
            <div class="panel-title">Run IDs (<span id="runCount">0</span>)</div>
            <div class="run-list" id="runList">
                <!-- Run items will be dynamically added here -->
            </div>
        </div>

        <div class="right-panel">
            <div class="viewer-container">
                <div class="viewer">
                    <div class="viewer-label">Unprocessed Masks</div>
                    <canvas id="leftCanvas"></canvas>
                    <div class="loading-overlay" id="leftLoading" style="display:none;">Loading...</div>
                </div>
                <div class="viewer">
                    <div class="viewer-label">Accepted Masks</div>
                    <canvas id="rightCanvas"></canvas>
                    <div class="loading-overlay" id="rightLoading" style="display:none;">Loading...</div>
                </div>
            </div>

            <div class="controls">
                <div class="class-selector">
                    <label for="classDropdown">Class:</label>
                    <select id="classDropdown" class="class-dropdown">
                        <!-- Classes will be dynamically added -->
                    </select>
                </div>
                <button class="save-btn" onclick="saveSegmentation()">Save Segmentation (S)</button>
            </div>

            <div class="status-bar">
                <div>Current Run: <strong id="currentRun">None</strong></div>
                <div>Accepted Masks: <strong id="acceptedCount">0</strong></div>
                <div>Good Runs: <strong id="goodRunCount">0</strong></div>
            </div>

            <div id="errorMessage" class="error-message" style="display:none;"></div>
        </div>
    </div>

    <div class="modal" id="helpModal">
        <div class="modal-content">
            <h2>Welcome to SAM2-ET Annotation GUI</h2>
            <p><strong>Keyboard Shortcuts:</strong></p>
            <ul>
                <li>← / → : Navigate between runs</li>
                <li>↑ / ↓ : Change class selection</li>
                <li>D : Mark run as good</li>
                <li>F : Unmark run</li>
                <li>S : Save annotations</li>
                <li>R : Undo last mask</li>
            </ul>
            <p><strong>Mouse Controls:</strong></p>
            <ul>
                <li>Click on masks in left viewer to accept them</li>
                <li>Accepted masks appear in right viewer</li>
            </ul>
            <button class="modal-close" onclick="hideHelp()">Got it!</button>
        </div>
    </div>

    <script>
        // Application State
        const app = {
            runs: [],
            currentRunIndex: 0,
            currentRunId: null,
            goodRuns: new Set(),
            classes: {{ class_names | tojson }},
            selectedClass: null,
            currentData: null,
            classDict: {},
            acceptedMasks: new Set(),
            maskColors: [
                [31, 119, 180], [255, 127, 14], [44, 160, 44],
                [214, 39, 40], [148, 103, 189], [140, 86, 75],
                [227, 119, 194], [0, 128, 128], [188, 189, 34],
                [23, 190, 207]
            ]
        };

        // Initialize application
        async function init() {
            console.log('Initializing application...');
            
            // Initialize class dictionary
            app.classes.forEach((className, idx) => {
                app.classDict[className] = {
                    value: idx + 1,
                    masks: [],
                    color: app.maskColors[idx % app.maskColors.length]
                };
            });
            app.selectedClass = app.classes[0];

            // Populate class dropdown
            populateClassDropdown();
            
            // Load runs
            await loadRuns();
            
            // Setup event listeners
            setupEventListeners();
        }

        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const data = await response.json();
                
                app.runs = data.runs;
                document.getElementById('runCount').textContent = data.total;
                
                populateRunList();
                
                if (app.runs.length > 0) {
                    await loadRun(0);
                }
            } catch (error) {
                showError('Failed to load runs: ' + error.message);
            }
        }

        function populateRunList() {
            const runList = document.getElementById('runList');
            runList.innerHTML = '';
            
            app.runs.forEach((run, idx) => {
                const item = document.createElement('div');
                item.className = 'run-item';
                item.textContent = run;
                item.onclick = () => loadRun(idx);
                
                if (idx === app.currentRunIndex) {
                    item.classList.add('active');
                }
                
                if (app.goodRuns.has(run)) {
                    item.classList.add('good');
                }
                
                runList.appendChild(item);
            });
        }

        function populateClassDropdown() {
            const dropdown = document.getElementById('classDropdown');
            dropdown.innerHTML = '';
            
            app.classes.forEach(className => {
                const option = document.createElement('option');
                option.value = className;
                option.textContent = className;
                dropdown.appendChild(option);
            });
            
            dropdown.onchange = (e) => {
                app.selectedClass = e.target.value;
            };
        }

        async function loadRun(index) {
            if (index < 0 || index >= app.runs.length) return;
            
            // Show loading
            document.getElementById('leftLoading').style.display = 'flex';
            document.getElementById('rightLoading').style.display = 'flex';
            
            app.currentRunIndex = index;
            app.currentRunId = app.runs[index];
            document.getElementById('currentRun').textContent = app.currentRunId;
            
            // Reset state for new run
            app.acceptedMasks.clear();
            Object.keys(app.classDict).forEach(key => {
                app.classDict[key].masks = [];
            });
            
            try {
                const response = await fetch(`/api/runs/${app.currentRunId}/data`);
                if (!response.ok) throw new Error('Failed to load data');
                
                app.currentData = await response.json();
                
                // Update UI
                populateRunList();
                renderCanvases();
                updateStatus();
                
            } catch (error) {
                showError('Failed to load run data: ' + error.message);
            } finally {
                // Hide loading
                document.getElementById('leftLoading').style.display = 'none';
                document.getElementById('rightLoading').style.display = 'none';
            }
        }

        function renderCanvases() {
            if (!app.currentData) return;
            
            renderCanvas('leftCanvas', true);
            renderCanvas('rightCanvas', false);
        }

        function renderCanvas(canvasId, showUnaccepted) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            
            if (!app.currentData) return;
            
            const width = app.currentData.width;
            const height = app.currentData.height;
            
            canvas.width = width;
            canvas.height = height;
            
            // Clear canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, width, height);
            
            // Draw base image if available
            if (app.currentData.image_data) {
                const img = new Image();
                img.onload = function() {
                    ctx.drawImage(img, 0, 0);
                    
                    // Draw masks on top
                    drawMasks(ctx, showUnaccepted);
                };
                img.src = 'data:image/png;base64,' + app.currentData.image_data;
            } else {
                // Draw masks only
                drawMasks(ctx, showUnaccepted);
            }
        }

        function drawMasks(ctx, showUnaccepted) {
            if (!app.currentData || !app.currentData.masks) return;
            
            app.currentData.masks.forEach((maskData, idx) => {
                let shouldShow = false;
                let color = app.maskColors[idx % app.maskColors.length];
                
                if (showUnaccepted) {
                    // Left canvas: show unaccepted masks
                    shouldShow = !Array.from(app.acceptedMasks).includes(idx) &&
                                !Object.values(app.classDict).some(c => c.masks.includes(idx));
                } else {
                    // Right canvas: show accepted masks
                    Object.entries(app.classDict).forEach(([className, classData]) => {
                        if (classData.masks.includes(idx)) {
                            shouldShow = true;
                            color = classData.color;
                        }
                    });
                }
                
                if (shouldShow && maskData) {
                    ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.4)`;
                    
                    // Draw mask overlay
                    const imageData = ctx.createImageData(app.currentData.width, app.currentData.height);
                    for (let i = 0; i < maskData.length; i++) {
                        if (maskData[i] > 0) {
                            imageData.data[i * 4] = color[0];
                            imageData.data[i * 4 + 1] = color[1];
                            imageData.data[i * 4 + 2] = color[2];
                            imageData.data[i * 4 + 3] = 100;
                        }
                    }
                    ctx.putImageData(imageData, 0, 0);
                }
            });
        }

        function setupEventListeners() {
            // Keyboard navigation
            document.addEventListener('keydown', async (e) => {
                switch(e.key) {
                    case 'ArrowLeft':
                        if (app.currentRunIndex > 0) {
                            await loadRun(app.currentRunIndex - 1);
                        }
                        break;
                    case 'ArrowRight':
                        if (app.currentRunIndex < app.runs.length - 1) {
                            await loadRun(app.currentRunIndex + 1);
                        }
                        break;
                    case 'ArrowUp':
                        const dropdown = document.getElementById('classDropdown');
                        if (dropdown.selectedIndex > 0) {
                            dropdown.selectedIndex--;
                            app.selectedClass = dropdown.value;
                        }
                        break;
                    case 'ArrowDown':
                        const dropdown2 = document.getElementById('classDropdown');
                        if (dropdown2.selectedIndex < dropdown2.options.length - 1) {
                            dropdown2.selectedIndex++;
                            app.selectedClass = dropdown2.value;
                        }
                        break;
                    case 'd':
                    case 'D':
                        app.goodRuns.add(app.currentRunId);
                        populateRunList();
                        updateStatus();
                        break;
                    case 'f':
                    case 'F':
                        app.goodRuns.delete(app.currentRunId);
                        populateRunList();
                        updateStatus();
                        break;
                    case 's':
                    case 'S':
                        await saveSegmentation();
                        break;
                    case 'r':
                    case 'R':
                        undoLastMask();
                        break;
                }
            });

            // Canvas click handling
            document.getElementById('leftCanvas').addEventListener('click', (e) => {
                if (!app.currentData || !app.currentData.masks) return;
                
                const rect = e.target.getBoundingClientRect();
                const x = Math.floor((e.clientX - rect.left) * (app.currentData.width / rect.width));
                const y = Math.floor((e.clientY - rect.top) * (app.currentData.height / rect.height));
                
                // Find which mask was clicked
                const pixelIndex = y * app.currentData.width + x;
                
                for (let i = 0; i < app.currentData.masks.length; i++) {
                    if (app.currentData.masks[i] && app.currentData.masks[i][pixelIndex] > 0 &&
                        !app.acceptedMasks.has(i) &&
                        !Object.values(app.classDict).some(c => c.masks.includes(i))) {
                        
                        // Accept this mask for current class
                        app.classDict[app.selectedClass].masks.push(i);
                        app.acceptedMasks.add(i);
                        renderCanvases();
                        updateStatus();
                        break;
                    }
                }
            });
        }

        function undoLastMask() {
            const currentClassMasks = app.classDict[app.selectedClass].masks;
            if (currentClassMasks.length > 0) {
                const lastMask = currentClassMasks.pop();
                app.acceptedMasks.delete(lastMask);
                renderCanvases();
                updateStatus();
            }
        }

        async function saveSegmentation() {
            try {
                const response = await fetch('/api/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        run_id: app.currentRunId,
                        class_dict: app.classDict,
                        accepted_masks: Array.from(app.acceptedMasks),
                        good_runs: Array.from(app.goodRuns)
                    })
                });
                
                if (!response.ok) throw new Error('Save failed');
                
                const result = await response.json();
                console.log('Saved:', result);
                
                // Show success message
                showError('Segmentation saved successfully!', false);
                
            } catch (error) {
                showError('Failed to save: ' + error.message);
            }
        }

        function updateStatus() {
            document.getElementById('acceptedCount').textContent = 
                Object.values(app.classDict).reduce((sum, c) => sum + c.masks.length, 0);
            document.getElementById('goodRunCount').textContent = app.goodRuns.size;
        }

        function showError(message, isError = true) {
            const errorDiv = document.getElementById('errorMessage');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            errorDiv.style.background = isError ? '#fed7d7' : '#c6f6d5';
            errorDiv.style.color = isError ? '#742a2a' : '#22543d';
            
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function showHelp() {
            document.getElementById('helpModal').classList.add('show');
        }

        function hideHelp() {
            document.getElementById('helpModal').classList.remove('show');
        }

        // Initialize on load
        window.onload = init;
    </script>
</body>
</html>
"""

def create_app(data_path, output_path=None, class_names=None, dask_processor=None):
    """Create Flask application with full functionality."""
    
    app = Flask(__name__)
    CORS(app)
    
    # Store configuration
    app.config['DATA_PATH'] = Path(data_path)
    app.config['OUTPUT_PATH'] = Path(output_path) if output_path else None
    app.config['CLASS_NAMES'] = class_names or ['object']
    app.config['DASK_PROCESSOR'] = dask_processor
    
    # Load Zarr data
    try:
        app.zarr_root = zarr.open(str(app.config['DATA_PATH']), mode='r')
        app.run_ids = list(app.zarr_root.keys())
        logger.info(f"Loaded {len(app.run_ids)} run IDs from Zarr file")
    except Exception as e:
        logger.error(f"Failed to load Zarr data: {e}")
        app.zarr_root = None
        app.run_ids = []
    
    @app.route('/')
    def index():
        """Serve the main HTML interface."""
        return render_template_string(
            HTML_TEMPLATE,
            data_path=str(app.config['DATA_PATH']),
            class_names=app.config['CLASS_NAMES']
        )
    
    @app.route('/api/runs')
    def get_runs():
        """Get list of available run IDs."""
        return jsonify({
            'runs': app.run_ids,
            'total': len(app.run_ids)
        })
    
    @app.route('/api/runs/<run_id>/data')
    def get_run_data(run_id):
        """Get data for a specific run with proper image encoding."""
        if not app.zarr_root or run_id not in app.zarr_root:
            return jsonify({'error': 'Run ID not found'}), 404
        
        try:
            # Load data
            image = app.zarr_root[run_id]['image'][:]
            masks = app.zarr_root[run_id].get('labels', 
                     app.zarr_root[run_id].get('masks', []))[:]
            
            # Handle transposition if needed
            if image.shape[0] < image.shape[1]:
                image = image.T
                masks = np.swapaxes(masks, 1, 2)
            
            height, width = image.shape[:2]
            
            # Normalize and convert image to PNG
            image_normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_normalized, mode='L')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode as base64
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            # Process masks - convert to simple binary arrays
            mask_list = []
            for mask in masks:
                # Flatten mask and convert to list for JSON serialization
                mask_binary = (mask > 0.5).astype(np.uint8).flatten().tolist()
                mask_list.append(mask_binary)
            
            return jsonify({
                'image_data': image_base64,
                'masks': mask_list,
                'width': width,
                'height': height,
                'num_masks': len(masks)
            })
            
        except Exception as e:
            logger.error(f"Error loading run {run_id}: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/save', methods=['POST'])
    def save_annotation():
        """Save annotation data to output Zarr file."""
        if not app.config['OUTPUT_PATH']:
            return jsonify({'message': 'Output path not configured - running in read-only mode'}), 200
        
        try:
            data = request.json
            run_id = data.get('run_id')
            class_dict = data.get('class_dict')
            accepted_masks = data.get('accepted_masks', [])
            good_runs = data.get('good_runs', [])
            
            # Open output zarr
            output_zarr = zarr.open(str(app.config['OUTPUT_PATH']), mode='a')
            
            # Save to group
            group = output_zarr.require_group(run_id)
            
            # Save metadata
            group.attrs['class_dict'] = json.dumps(class_dict)
            group.attrs['accepted_masks'] = accepted_masks
            
            # Save good runs list to root
            output_zarr.attrs['good_run_ids'] = good_runs
            
            # Copy original data if needed
            if run_id in app.zarr_root:
                group['image'] = app.zarr_root[run_id]['image'][:]
                
                # Save accepted masks by class
                original_masks = app.zarr_root[run_id].get('labels',
                                app.zarr_root[run_id].get('masks', []))[:]
                
                # Create labeled masks array
                labeled_masks = []
                for class_name, class_info in class_dict.items():
                    class_mask = np.zeros_like(original_masks[0]) if len(original_masks) > 0 else np.zeros((256, 256))
                    for mask_idx in class_info.get('masks', []):
                        if mask_idx < len(original_masks):
                            class_mask = np.logical_or(class_mask, original_masks[mask_idx] > 0)
                    labeled_masks.append(class_mask.astype(np.uint8))
                
                if labeled_masks:
                    group['labels'] = np.stack(labeled_masks)
            
            return jsonify({
                'status': 'saved',
                'run_id': run_id,
                'accepted_count': len(accepted_masks),
                'good_runs_count': len(good_runs)
            })
            
        except Exception as e:
            logger.error(f"Error saving annotation: {traceback.format_exc()}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/status')
    def get_status():
        """Get server and Dask cluster status."""
        status = {
            'server': 'running',
            'data_path': str(app.config['DATA_PATH']),
            'output_path': str(app.config['OUTPUT_PATH']) if app.config['OUTPUT_PATH'] else None,
            'runs_loaded': len(app.run_ids),
            'classes': app.config['CLASS_NAMES']
        }
        
        if app.config['DASK_PROCESSOR']:
            status['dask'] = app.config['DASK_PROCESSOR'].get_status()
        
        return jsonify(status)
    
    return app


def run_server(data_path, output_path=None, host='0.0.0.0', port=8080,
               dask_scheduler=None, n_workers=4, class_names=None, debug=False):
    """Run the Flask server with optional Dask integration."""
    
    # Initialize Dask processor if needed
    dask_processor = None
    if dask_scheduler or n_workers > 0:
        from .dask_processor import DaskProcessor
        dask_processor = DaskProcessor(
            scheduler=dask_scheduler,
            n_workers=n_workers if not dask_scheduler else 0
        )
        dask_processor.start()
    
    # Create Flask app
    app = create_app(
        data_path=data_path,
        output_path=output_path,
        class_names=class_names,
        dask_processor=dask_processor
    )
    
    try:
        # Run Flask server
        logger.info(f"Starting server at http://{host}:{port}")
        app.run(host=host, port=port, debug=debug)
    finally:
        # Cleanup Dask
        if dask_processor:
            dask_processor.close()