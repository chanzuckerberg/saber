// SABER Annotation GUI - JavaScript Application

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
    currentMaskIndex: 0,
    applyRotation: false,
    boundaryCache: {},
    dataCache: {},  // Cache loaded run data
    prefetchPromise: null  // Track prefetch operations
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
        const url = `/api/runs/${state.currentRunId}${state.applyRotation ? '?rotate=true' : ''}`;
        const response = await fetch(url);
        const data = await response.json();
        state.currentData = data;
        
        // Process mask values
        processMaskValues(data);
        
        // Initialize mask visibility
        initializeMaskVisibility();
        
        // Clear caches
        state.boundaryCache = {};
        
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

// Get boundary points for mask (with caching)
function getBoundaryPoints(mask, width, height, maskValue) {
    if (state.boundaryCache[maskValue]) {
        return state.boundaryCache[maskValue];
    }
    
    const points = [];
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (mask[y][x] > 0.5) {
                const isBoundary = 
                    (x === 0 || mask[y][x-1] <= 0.5) ||
                    (x === width-1 || mask[y][x+1] <= 0.5) ||
                    (y === 0 || mask[y-1][x] <= 0.5) ||
                    (y === height-1 || mask[y+1][x] <= 0.5);
                
                if (isBoundary) {
                    points.push({x, y});
                }
            }
        }
    }
    
    state.boundaryCache[maskValue] = points;
    return points;
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
            const boundaryPoints = getBoundaryPoints(mask, width, height, maskValue);
            ctx.fillStyle = 'white';
            const thickness = 2; // Increase this value for thicker outline
            boundaryPoints.forEach(point => {
                ctx.fillRect(point.x - Math.floor(thickness/2), 
                           point.y - Math.floor(thickness/2), 
                           thickness, thickness);
            });
        }
    });
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
    
    // Remove all annotations for this class across all runs
    Object.keys(state.annotations).forEach(runId => {
        const runAnnotations = state.annotations[runId];
        const toRemove = [];
        Object.entries(runAnnotations).forEach(([maskValue, cls]) => {
            if (cls === className) {
                toRemove.push(maskValue);
            }
        });
        toRemove.forEach(maskValue => {
            delete runAnnotations[maskValue];
            // Move mask back to left panel if current run
            if (runId === state.currentRunId) {
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

// Toggle rotation
async function toggleRotation() {
    state.applyRotation = !state.applyRotation;
    
    const rotateBtn = document.getElementById('rotateBtn');
    rotateBtn.textContent = state.applyRotation ? 'Rotate: ON' : 'Rotate: OFF';
    rotateBtn.classList.toggle('active', state.applyRotation);
    
    // Clear cache and reload current run
    state.boundaryCache = {};
    
    if (state.currentRunId) {
        await selectRun(state.currentRunIndex);
    }
    
    updateStatus(`Rotation ${state.applyRotation ? 'enabled' : 'disabled'}`);
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
    updateStatus(`Removed mask ${maskValue} from ${className || 'unknown class'}`);
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
                }
            });
            
            renderClassList();
            
            // Reload current run or switch to first annotated run
            if (loadedAnnotations[state.currentRunId]) {
                loadExistingAnnotations();
                renderCanvases();
                updateStatus(`Imported ${annotationCount} annotations`);
            } else {
                const firstRunWithAnnotations = Object.keys(loadedAnnotations)[0];
                if (firstRunWithAnnotations) {
                    const runIndex = state.runs.indexOf(firstRunWithAnnotations);
                    if (runIndex >= 0) {
                        await selectRun(runIndex);
                        updateStatus(`Imported annotations and switched to ${firstRunWithAnnotations}`);
                        return;
                    }
                }
                renderCanvases();
                updateStatus(`Imported ${annotationCount} annotations`);
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