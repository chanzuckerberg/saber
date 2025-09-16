"""
Web Server Implementation for SAM2-ET Annotation GUI
Mirrors the PyQt5 GUI functionality in a web interface
"""

import os
import json
import logging
import base64
import io
import traceback
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file
from flask_cors import CORS
import zarr
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# HTML Template with complete GUI functionality
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM2-ET Annotation GUI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: white;
            padding: 10px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
        }

        .header h1 {
            font-size: 20px;
            color: #333;
        }

        .container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }

        .panel {
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }

        .left-panel {
            width: 200px;
        }

        .run-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .run-item {
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
            margin-bottom: 4px;
            font-size: 14px;
        }

        .run-item:hover {
            background: #f0f0f0;
        }

        .run-item.active {
            background: #2196F3;
            color: white;
        }

        .middle-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .canvas-container {
            flex: 1;
            display: flex;
            gap: 10px;
            padding: 10px;
            background: #fafafa;
            position: relative;
        }

        .canvas-wrapper {
            flex: 1;
            position: relative;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .canvas-label {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 10;
        }

        canvas {
            max-width: 100%;
            max-height: 100%;
            cursor: crosshair;
        }

        .controls {
            padding: 10px;
            background: white;
            border-top: 1px solid #e0e0e0;
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .right-panel {
            width: 250px;
            background: white;
            padding: 15px;
            display: flex;
            flex-direction: column;
        }

        .class-manager {
            flex: 1;
            display: flex;
            flex-direction: column;
        }

        .class-manager h3 {
            font-size: 16px;
            margin-bottom: 10px;
            color: #333;
        }

        .add-class-form {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
        }

        .add-class-form input {
            flex: 1;
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .add-class-form button {
            padding: 6px 12px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .add-class-form button:hover {
            background: #45a049;
        }

        .class-list {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }

        .class-item {
            display: flex;
            align-items: center;
            padding: 8px;
            cursor: pointer;
            border-radius: 4px;
            margin-bottom: 4px;
        }

        .class-item:hover {
            background: #f0f0f0;
        }

        .class-item.active {
            background: #e3f2fd;
        }

        .class-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 10px;
            border: 1px solid #ddd;
        }

        .class-name {
            flex: 1;
            font-size: 14px;
        }

        .remove-class-btn {
            width: 100%;
            padding: 8px;
            margin-top: 10px;
            background: #f44336;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .remove-class-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }

        .remove-class-btn:not(:disabled):hover {
            background: #da190b;
        }

        .button {
            padding: 8px 16px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }

        .button:hover {
            background: #1976D2;
        }

        .button.secondary {
            background: #757575;
        }

        .button.secondary:hover {
            background: #616161;
        }

        .status-bar {
            padding: 8px 20px;
            background: #37474F;
            color: white;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
        }

        .shortcuts-help {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            z-index: 1000;
            display: none;
        }

        .shortcuts-help.show {
            display: block;
        }

        .shortcuts-help h3 {
            margin-bottom: 15px;
        }

        .shortcuts-help table {
            width: 100%;
            font-size: 14px;
        }

        .shortcuts-help td {
            padding: 5px 10px;
        }

        .shortcuts-help td:first-child {
            font-weight: bold;
            text-align: right;
        }

        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 999;
            display: none;
        }

        .overlay.show {
            display: block;
        }

        .loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>SAM2-ET Annotation GUI</h1>
        <button class="button" onclick="toggleHelp()">Help (H)</button>
    </div>

    <div class="container">
        <div class="panel left-panel">
            <div class="run-list" id="runList"></div>
        </div>

        <div class="middle-panel">
            <div class="canvas-container">
                <div class="canvas-wrapper">
                    <div class="canvas-label">Unprocessed Masks</div>
                    <canvas id="leftCanvas"></canvas>
                </div>
                <div class="canvas-wrapper">
                    <div class="canvas-label">Accepted Masks</div>
                    <canvas id="rightCanvas"></canvas>
                </div>
            </div>
            <div class="controls">
                <button class="button secondary" onclick="importAnnotations()">Import JSON</button>
                <button class="button" onclick="exportAnnotations()">Export JSON</button>
                <span style="flex: 1;"></span>
                <span id="maskInfo" style="font-size: 14px; color: #666;"></span>
            </div>
        </div>

        <div class="panel right-panel">
            <div class="class-manager">
                <h3>Segmentation Classes</h3>
                <div class="add-class-form">
                    <input type="text" id="classNameInput" placeholder="Enter class name...">
                    <button onclick="addClass()">Add</button>
                </div>
                <div class="class-list" id="classList"></div>
                <button class="remove-class-btn" id="removeClassBtn" onclick="removeClass()" disabled>
                    Remove Selected Class
                </button>
            </div>
        </div>
    </div>

    <div class="status-bar">
        <span id="statusLeft">Ready</span>
        <span id="statusRight">Use A/D to navigate runs, W/S to switch classes</span>
    </div>

    <div class="overlay" id="overlay" onclick="toggleHelp()"></div>
    <div class="shortcuts-help" id="shortcutsHelp">
        <h3>Keyboard Shortcuts</h3>
        <table>
            <tr><td>A / ←</td><td>Previous run</td></tr>
            <tr><td>D / →</td><td>Next run</td></tr>
            <tr><td>W / ↑</td><td>Previous class</td></tr>
            <tr><td>S / ↓</td><td>Next class</td></tr>
            <tr><td>R</td><td>Remove selected mask</td></tr>
            <tr><td>H</td><td>Toggle help</td></tr>
            <tr><td>Click mask</td><td>Accept (left) / Select (right)</td></tr>
        </table>
        <button class="button" style="margin-top: 15px; width: 100%;" onclick="toggleHelp()">Close</button>
    </div>

    <script>
        // TAB10 color palette
        const TAB10_COLORS = [
            [31, 119, 180],   // blue
            [255, 127, 14],   // orange
            [44, 160, 44],    // green
            [214, 39, 40],    // red
            [148, 103, 189],  // purple
            [140, 86, 75],    // brown
            [227, 119, 194],  // pink
            [0, 128, 128],    // teal
            [188, 189, 34],   // olive
            [23, 190, 207],   // cyan
        ];

        // Global state
        let state = {
            runs: [],
            currentRunIndex: 0,
            currentRunId: null,
            currentData: null,
            classes: {},
            selectedClass: null,
            annotations: {},
            maskValueToIndex: {},
            indexToMaskValue: {},
            usedColorIndices: new Set(),
            highlightedMaskValue: null,
            leftMaskVisibility: {},
            rightMaskVisibility: {},
            lastClickPos: null,
            currentMaskIndex: 0
        };

        // Initialize the application
        async function init() {
            await loadRuns();
            setupEventListeners();
            updateStatus('Application initialized');
        }

        // Load available runs
        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const data = await response.json();
                state.runs = data.runs;
                renderRunList();
                if (state.runs.length > 0) {
                    await selectRun(0);
                }
            } catch (error) {
                console.error('Failed to load runs:', error);
                updateStatus('Failed to load runs', 'error');
            }
        }

        // Render run list
        function renderRunList() {
            const runList = document.getElementById('runList');
            runList.innerHTML = '';
            
            state.runs.forEach((runId, index) => {
                const item = document.createElement('div');
                item.className = 'run-item';
                item.textContent = runId;
                item.onclick = () => selectRun(index);
                if (index === state.currentRunIndex) {
                    item.classList.add('active');
                }
                runList.appendChild(item);
            });
        }

        // Select a run
        async function selectRun(index) {
            if (index < 0 || index >= state.runs.length) return;
            
            state.currentRunIndex = index;
            state.currentRunId = state.runs[index];
            
            // Update UI
            renderRunList();
            updateStatus(`Loading run: ${state.currentRunId}`);
            
            // Load run data
            try {
                const response = await fetch(`/api/runs/${state.currentRunId}`);
                const data = await response.json();
                state.currentData = data;
                
                // Process mask values
                processMaskValues(data);
                
                // Initialize mask visibility
                initializeMaskVisibility();
                
                // Load existing annotations for this run
                loadExistingAnnotations();
                
                // Render canvases
                renderCanvases();
                
                updateStatus(`Loaded run: ${state.currentRunId}`);
            } catch (error) {
                console.error('Failed to load run data:', error);
                updateStatus('Failed to load run data', 'error');
            }
        }

        // Process mask values from data
        function processMaskValues(data) {
            state.maskValueToIndex = {};
            state.indexToMaskValue = {};
            
            if (data.mask_values) {
                data.mask_values.forEach((value, index) => {
                    state.maskValueToIndex[value] = index;
                    state.indexToMaskValue[index] = value;
                });
            }
        }

        // Initialize mask visibility states
        function initializeMaskVisibility() {
            state.leftMaskVisibility = {};
            state.rightMaskVisibility = {};
            
            if (state.currentData && state.currentData.masks) {
                state.currentData.masks.forEach((_, index) => {
                    const maskValue = state.indexToMaskValue[index];
                    state.leftMaskVisibility[maskValue] = true;
                    state.rightMaskVisibility[maskValue] = false;
                });
            }
        }

        // Load existing annotations for current run
        function loadExistingAnnotations() {
            if (!state.annotations[state.currentRunId]) return;
            
            const runAnnotations = state.annotations[state.currentRunId];
            
            // Clear class masks
            Object.keys(state.classes).forEach(className => {
                state.classes[className].masks = [];
            });
            
            // Restore annotations
            Object.entries(runAnnotations).forEach(([maskValueStr, className]) => {
                const maskValue = parseFloat(maskValueStr);
                
                if (state.classes[className] && state.maskValueToIndex[maskValue] !== undefined) {
                    state.classes[className].masks.push(maskValue);
                    state.leftMaskVisibility[maskValue] = false;
                    state.rightMaskVisibility[maskValue] = true;
                }
            });
        }

        // Render both canvases
        function renderCanvases() {
            renderCanvas('leftCanvas', true);
            renderCanvas('rightCanvas', false);
        }

        // Render a single canvas
        function renderCanvas(canvasId, isLeft) {
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            
            if (!state.currentData) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                return;
            }
            
            // Set canvas size
            const width = state.currentData.shape[1];
            const height = state.currentData.shape[0];
            canvas.width = width;
            canvas.height = height;
            
            // Draw base image with normalization
            const imageData = ctx.createImageData(width, height);
            const baseImage = state.currentData.image;
            
            // Find min and max values for normalization
            let minVal = Infinity;
            let maxVal = -Infinity;
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const val = baseImage[y][x];
                    minVal = Math.min(minVal, val);
                    maxVal = Math.max(maxVal, val);
                }
            }
            
            // Normalize to 0-255 range
            const range = maxVal - minVal;
            const scale = range > 0 ? 255 / range : 1;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = (y * width + x) * 4;
                    const normalized = Math.round((baseImage[y][x] - minVal) * scale);
                    imageData.data[idx] = normalized;
                    imageData.data[idx + 1] = normalized;
                    imageData.data[idx + 2] = normalized;
                    imageData.data[idx + 3] = 255;
                }
            }
            ctx.putImageData(imageData, 0, 0);
            
            // Draw masks
            state.currentData.masks.forEach((mask, maskIndex) => {
                const maskValue = state.indexToMaskValue[maskIndex];
                const visibility = isLeft ? state.leftMaskVisibility : state.rightMaskVisibility;
                
                if (!visibility[maskValue]) return;
                
                // Get color for this mask
                let color;
                if (!isLeft) {
                    // Find which class this mask belongs to
                    let className = null;
                    if (state.annotations[state.currentRunId]) {
                        className = state.annotations[state.currentRunId][maskValue.toString()];
                    }
                    if (className && state.classes[className]) {
                        const colorIndex = state.classes[className].colorIndex;
                        color = TAB10_COLORS[colorIndex % TAB10_COLORS.length];
                    } else {
                        color = TAB10_COLORS[maskIndex % TAB10_COLORS.length];
                    }
                } else {
                    color = TAB10_COLORS[maskIndex % TAB10_COLORS.length];
                }
                
                // Draw mask with transparency
                ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0.4)`;
                
                for (let y = 0; y < height; y++) {
                    for (let x = 0; x < width; x++) {
                        if (mask[y][x] > 0.5) {
                            ctx.fillRect(x, y, 1, 1);
                        }
                    }
                }
                
                // Draw boundary if this mask is highlighted
                if (maskValue === state.highlightedMaskValue && visibility[maskValue]) {
                    drawMaskBoundary(ctx, mask, width, height);
                }
            });
        }

        // Draw mask boundary
        function drawMaskBoundary(ctx, mask, width, height) {
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            // Simple boundary detection
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    if (mask[y][x] > 0.5) {
                        // Check if it's a boundary pixel
                        const isBoundary = 
                            (x === 0 || mask[y][x-1] <= 0.5) ||
                            (x === width-1 || mask[y][x+1] <= 0.5) ||
                            (y === 0 || mask[y-1][x] <= 0.5) ||
                            (y === height-1 || mask[y+1][x] <= 0.5);
                        
                        if (isBoundary) {
                            ctx.fillStyle = 'white';
                            ctx.fillRect(x, y, 1, 1);
                        }
                    }
                }
            }
        }

        // Class management functions
        function getNextColorIndex() {
            let index = 0;
            while (state.usedColorIndices.has(index)) {
                index++;
            }
            return index;
        }

        function addClass() {
            const input = document.getElementById('classNameInput');
            const className = input.value.trim();
            
            if (!className) {
                alert('Please enter a class name');
                return;
            }
            
            if (state.classes[className]) {
                alert(`Class '${className}' already exists`);
                return;
            }
            
            const colorIndex = getNextColorIndex();
            state.usedColorIndices.add(colorIndex);
            
            state.classes[className] = {
                value: colorIndex + 1,
                colorIndex: colorIndex,
                masks: []
            };
            
            input.value = '';
            renderClassList();
            selectClass(className);
            updateStatus(`Added class: ${className}`);
        }

        function removeClass() {
            if (!state.selectedClass) return;
            
            if (!confirm(`Remove class '${state.selectedClass}'? This will remove all associated mask assignments.`)) {
                return;
            }
            
            const className = state.selectedClass;
            const colorIndex = state.classes[className].colorIndex;
            
            // Free up color index
            state.usedColorIndices.delete(colorIndex);
            
            // Remove all annotations for this class
            Object.keys(state.annotations).forEach(runId => {
                const runAnnotations = state.annotations[runId];
                Object.keys(runAnnotations).forEach(maskValue => {
                    if (runAnnotations[maskValue] === className) {
                        delete runAnnotations[maskValue];
                        
                        // Move mask back to left panel
                        const maskVal = parseFloat(maskValue);
                        state.leftMaskVisibility[maskVal] = true;
                        state.rightMaskVisibility[maskVal] = false;
                    }
                });
            });
            
            // Remove class
            delete state.classes[className];
            state.selectedClass = null;
            
            renderClassList();
            renderCanvases();
            updateStatus(`Removed class: ${className}`);
        }

        function selectClass(className) {
            state.selectedClass = className;
            renderClassList();
            updateStatus(`Selected class: ${className}`);
        }

        function renderClassList() {
            const classList = document.getElementById('classList');
            classList.innerHTML = '';
            
            Object.keys(state.classes).forEach(className => {
                const classData = state.classes[className];
                const color = TAB10_COLORS[classData.colorIndex % TAB10_COLORS.length];
                
                const item = document.createElement('div');
                item.className = 'class-item';
                if (className === state.selectedClass) {
                    item.classList.add('active');
                }
                
                const colorBox = document.createElement('div');
                colorBox.className = 'class-color';
                colorBox.style.backgroundColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                
                const nameSpan = document.createElement('span');
                nameSpan.className = 'class-name';
                nameSpan.textContent = className;
                
                item.appendChild(colorBox);
                item.appendChild(nameSpan);
                item.onclick = () => selectClass(className);
                
                classList.appendChild(item);
            });
            
            // Update remove button state
            const removeBtn = document.getElementById('removeClassBtn');
            removeBtn.disabled = !state.selectedClass;
        }

        // Canvas interaction
        function setupEventListeners() {
            // Canvas clicks
            document.getElementById('leftCanvas').addEventListener('click', (e) => handleCanvasClick(e, true));
            document.getElementById('rightCanvas').addEventListener('click', (e) => handleCanvasClick(e, false));
            
            // Keyboard shortcuts
            document.addEventListener('keydown', handleKeyPress);
            
            // Class name input enter key
            document.getElementById('classNameInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') addClass();
            });
        }

        function handleCanvasClick(event, isLeft) {
            const canvas = event.target;
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((event.clientX - rect.left) * canvas.width / rect.width);
            const y = Math.floor((event.clientY - rect.top) * canvas.height / rect.height);
            
            if (!state.currentData) return;
            
            // Find masks at this position
            const maskHits = [];
            state.currentData.masks.forEach((mask, index) => {
                const maskValue = state.indexToMaskValue[index];
                const visibility = isLeft ? state.leftMaskVisibility : state.rightMaskVisibility;
                
                if (mask[y] && mask[y][x] > 0.5 && visibility[maskValue]) {
                    maskHits.push(index);
                }
            });
            
            if (maskHits.length === 0) return;
            
            // Handle overlapping masks
            if (!state.lastClickPos || state.lastClickPos.x !== x || state.lastClickPos.y !== y) {
                state.lastClickPos = {x, y};
                state.currentMaskIndex = 0;
            } else {
                state.currentMaskIndex = (state.currentMaskIndex + 1) % maskHits.length;
            }
            
            const hitIndex = maskHits[state.currentMaskIndex];
            const maskValue = state.indexToMaskValue[hitIndex];
            
            if (isLeft) {
                // Accept mask to selected class
                if (!state.selectedClass) {
                    updateStatus('No class selected - please add and select a class first', 'warning');
                    return;
                }
                
                // Move mask to right panel
                state.leftMaskVisibility[maskValue] = false;
                state.rightMaskVisibility[maskValue] = true;
                
                // Add to class
                state.classes[state.selectedClass].masks.push(maskValue);
                
                // Update annotations
                if (!state.annotations[state.currentRunId]) {
                    state.annotations[state.currentRunId] = {};
                }
                state.annotations[state.currentRunId][maskValue.toString()] = state.selectedClass;
                
                // Highlight the newly accepted mask
                state.highlightedMaskValue = maskValue;
                
                renderCanvases();
                updateStatus(`Mask ${maskValue} assigned to ${state.selectedClass}`);
            } else {
                // Toggle selection on right panel
                if (state.highlightedMaskValue === maskValue) {
                    state.highlightedMaskValue = null;
                } else {
                    state.highlightedMaskValue = maskValue;
                }
                renderCanvases();
            }
        }

        function handleKeyPress(event) {
            switch(event.key.toLowerCase()) {
                case 'a':
                case 'arrowleft':
                    navigateRun(-1);
                    break;
                case 'd':
                case 'arrowright':
                    navigateRun(1);
                    break;
                case 'w':
                case 'arrowup':
                    navigateClass(-1);
                    break;
                case 's':
                case 'arrowdown':
                    navigateClass(1);
                    break;
                case 'r':
                    removeHighlightedMask();
                    break;
                case 'h':
                    toggleHelp();
                    break;
            }
        }

        function navigateRun(direction) {
            const newIndex = state.currentRunIndex + direction;
            if (newIndex >= 0 && newIndex < state.runs.length) {
                selectRun(newIndex);
            }
        }

        function navigateClass(direction) {
            const classNames = Object.keys(state.classes);
            if (classNames.length === 0) return;
            
            let currentIndex = classNames.indexOf(state.selectedClass);
            if (currentIndex === -1) currentIndex = 0;
            
            const newIndex = currentIndex + direction;
            if (newIndex >= 0 && newIndex < classNames.length) {
                selectClass(classNames[newIndex]);
            }
        }

        function removeHighlightedMask() {
            if (state.highlightedMaskValue === null) {
                updateStatus('No mask selected to remove', 'warning');
                return;
            }
            
            const maskValue = state.highlightedMaskValue;
            const maskIndex = state.maskValueToIndex[maskValue];
            
            // Check if mask is on right panel
            if (!state.rightMaskVisibility[maskValue]) {
                updateStatus('Selected mask is not on the right panel', 'warning');
                return;
            }
            
            // Find which class this mask belongs to
            let className = null;
            if (state.annotations[state.currentRunId]) {
                className = state.annotations[state.currentRunId][maskValue.toString()];
                if (className) {
                    delete state.annotations[state.currentRunId][maskValue.toString()];
                }
            }
            
            // Remove from class
            if (className && state.classes[className]) {
                const masks = state.classes[className].masks;
                const idx = masks.indexOf(maskValue);
                if (idx > -1) {
                    masks.splice(idx, 1);
                }
            }
            
            // Move back to left panel
            state.leftMaskVisibility[maskValue] = true;
            state.rightMaskVisibility[maskValue] = false;
            state.highlightedMaskValue = null;
            
            renderCanvases();
            updateStatus(`Removed mask ${maskValue} from ${className}`);
        }

        // Import/Export functions
        async function exportAnnotations() {
            const dataStr = JSON.stringify(state.annotations, null, 2);
            const dataBlob = new Blob([dataStr], {type: 'application/json'});
            const url = URL.createObjectURL(dataBlob);
            
            const link = document.createElement('a');
            link.href = url;
            link.download = 'annotations.json';
            link.click();
            
            URL.revokeObjectURL(url);
            updateStatus('Annotations exported');
        }

        async function importAnnotations() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            
            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                
                try {
                    const text = await file.text();
                    const loadedAnnotations = JSON.parse(text);
                    
                    console.log('Loaded annotations:', loadedAnnotations);
                    console.log('Current run:', state.currentRunId);
                    console.log('Available mask values:', Object.keys(state.maskValueToIndex));
                    
                    // Update annotations
                    Object.assign(state.annotations, loadedAnnotations);
                    
                    // Extract all unique classes
                    const allClasses = new Set();
                    let annotationCount = 0;
                    Object.values(state.annotations).forEach(runAnnotations => {
                        Object.values(runAnnotations).forEach(className => {
                            allClasses.add(className);
                            annotationCount++;
                        });
                    });
                    
                    console.log('Found classes:', Array.from(allClasses));
                    console.log('Total annotations:', annotationCount);
                    
                    // Add any missing classes
                    allClasses.forEach(className => {
                        if (!state.classes[className]) {
                            const colorIndex = getNextColorIndex();
                            state.usedColorIndices.add(colorIndex);
                            state.classes[className] = {
                                value: colorIndex + 1,
                                colorIndex: colorIndex,
                                masks: []
                            };
                            console.log(`Created class: ${className} with color index ${colorIndex}`);
                        }
                    });
                    
                    renderClassList();
                    
                    // If we have annotations for the current run, load them
                    if (loadedAnnotations[state.currentRunId]) {
                        console.log('Loading annotations for current run:', loadedAnnotations[state.currentRunId]);
                        loadExistingAnnotations();
                        renderCanvases();
                        updateStatus(`Imported ${annotationCount} annotations for ${Object.keys(loadedAnnotations).length} runs`);
                    } else {
                        // Find first run with annotations and switch to it
                        const firstRunWithAnnotations = Object.keys(loadedAnnotations)[0];
                        if (firstRunWithAnnotations) {
                            const runIndex = state.runs.indexOf(firstRunWithAnnotations);
                            if (runIndex >= 0) {
                                console.log('Switching to run with annotations:', firstRunWithAnnotations);
                                selectRun(runIndex);
                                updateStatus(`Imported annotations and switched to ${firstRunWithAnnotations}`);
                                return;
                            }
                        }
                        renderCanvases();
                        updateStatus(`Imported ${annotationCount} annotations (current run has no annotations)`);
                    }
                } catch (error) {
                    console.error('Failed to import annotations:', error);
                    updateStatus('Failed to import annotations', 'error');
                }
            };
            
            input.click();
        }

        // UI utilities
        function toggleHelp() {
            const overlay = document.getElementById('overlay');
            const help = document.getElementById('shortcutsHelp');
            overlay.classList.toggle('show');
            help.classList.toggle('show');
        }

        function updateStatus(message, type = 'info') {
            const statusLeft = document.getElementById('statusLeft');
            statusLeft.textContent = message;
            statusLeft.style.color = type === 'error' ? '#f44336' : 
                                     type === 'warning' ? '#ff9800' : '#fff';
        }

        // Initialize on load
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""


def create_app(data_path: str, output_path: str = None, class_names: List[str] = None):
    """Create Flask application with configuration"""
    app = Flask(__name__)
    CORS(app)
    
    # Store configuration
    app.config['DATA_PATH'] = Path(data_path)
    app.config['OUTPUT_PATH'] = Path(output_path) if output_path else None
    app.config['CLASS_NAMES'] = class_names or []
    
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
    """Extract mask values from 2D or 3D mask array"""
    mask_values = []
    extracted_masks = []
    
    if len(masks.shape) == 2:
        # 2D label map - extract individual masks
        unique_values = np.unique(masks[masks > 0])
        for val in unique_values:
            mask_values.append(float(val))
            extracted_masks.append((masks == val).astype(np.float32))
    elif len(masks.shape) == 3:
        # Stack of masks - extract values
        for i, mask in enumerate(masks):
            unique_vals = np.unique(mask[mask > 0])
            if len(unique_vals) > 0:
                mask_values.append(float(unique_vals[0]))
            else:
                mask_values.append(float(i + 1))
            extracted_masks.append(mask.astype(np.float32))
    
    return mask_values, extracted_masks


def get_mask_boundary(mask: np.ndarray) -> List[List[int]]:
    """Get boundary points for a mask using OpenCV"""
    try:
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        if not contours:
            return []
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Extract points
        if largest.shape[1] == 1:
            pts = largest.squeeze(axis=1)
        else:
            pts = largest.reshape(-1, 2)
        
        # Subsample if too many points
        if len(pts) > 100:
            step = max(1, len(pts) // 50)
            pts = pts[::step]
        
        return pts.tolist()
    except:
        return []


def run_server(data_path: str, 
               output_path: str = None,
               host: str = '0.0.0.0', 
               port: int = 8080,
               dask_scheduler: str = None,
               n_workers: int = 4,
               class_names: List[str] = None,
               debug: bool = False):
    """Run the Flask server"""
    
    app = create_app(data_path, output_path, class_names)
    
    # Initialize Dask if configured
    if dask_scheduler or n_workers > 0:
        from .dask_processor import DaskProcessor
        app.dask_processor = DaskProcessor(dask_scheduler, n_workers)
        app.dask_processor.start()
    else:
        app.dask_processor = None
    
    @app.route('/')
    def index():
        """Serve the main HTML interface"""
        return render_template_string(HTML_TEMPLATE)
    
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
            # Read data
            image = app.zarr_root[run_id][0][:]
            try:
                masks = app.zarr_root[run_id]['labels'][0][:]
            except:
                masks = app.zarr_root[run_id]['masks'][:]
            
            # Handle 2D/3D cases
            if image.ndim == 2:
                nx, ny = image.shape
                if nx < ny:
                    image = image.T
                    masks = np.swapaxes(masks, -2, -1)
            elif image.ndim == 3 and image.shape[0] == 3:
                # RGB image
                _, nx, ny = image.shape
                if nx < ny:
                    image = np.transpose(image, (0, 2, 1))
                    masks = np.swapaxes(masks, -2, -1)
                # Convert to grayscale for simplicity
                image = np.mean(image, axis=0)
            elif image.ndim == 3:
                # 3D volume - take middle slice for 2D view
                nz, nx, ny = image.shape
                if nx < ny:
                    image = np.swapaxes(image, 1, 2)
                    masks = np.swapaxes(masks, -2, -1)
                # Take middle slice
                mid_z = nz // 2
                image = image[mid_z]
                if len(masks.shape) == 4:
                    masks = masks[:, mid_z, :, :]
            
            # Extract mask values
            mask_values, extracted_masks = extract_mask_values(masks)
            
            # Prepare response
            response_data = {
                'image': image.tolist(),
                'masks': [m.tolist() for m in extracted_masks],
                'mask_values': mask_values,
                'shape': image.shape
            }
            
            # Add boundaries if needed (optional, for optimization)
            # boundaries = [get_mask_boundary(m) for m in extracted_masks]
            # response_data['boundaries'] = boundaries
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error loading run {run_id}: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/save', methods=['POST'])
    def save_annotations():
        """Save annotations"""
        if not app.config['OUTPUT_PATH']:
            return jsonify({'error': 'No output path configured'}), 400
        
        try:
            data = request.json
            output_file = app.config['OUTPUT_PATH'] / 'annotations.json'
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
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
        
        if app.dask_processor:
            status['dask'] = app.dask_processor.get_status()
        
        return jsonify(status)
    
    # Run the server
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    # For testing
    run_server('/path/to/data', port=8080, debug=True)