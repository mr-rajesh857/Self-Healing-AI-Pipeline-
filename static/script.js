// Global variables
let currentStep = 0;
let uploadedData = null;
let selectedTargetColumn = null;
let selectedTaskType = null;
let analysisResults = null;
let bestModel = null;

// API base URL
const API_BASE = 'http://localhost:5000';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateStepIndicator();
});

// Event listeners
function initializeEventListeners() {
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Column selection
    document.getElementById('confirmColumns').addEventListener('click', confirmColumnSelection);
    
    // Task type selection
    document.querySelectorAll('.task-option').forEach(option => {
        option.addEventListener('click', selectTaskType);
    });
    document.getElementById('confirmTask').addEventListener('click', confirmTaskType);
    
    // Analysis continuation
    document.getElementById('continueTraining').addEventListener('click', proceedToTraining);
    
    // Feature selection buttons
    document.querySelectorAll('.fs-btn').forEach(btn => {
        btn.addEventListener('click', selectFeatureMethod);
    });
    
    // Code generation
    document.getElementById('generateCode').addEventListener('click', generateCode);
    
    // Modal
    document.getElementById('closeModal').addEventListener('click', closeModal);
    document.getElementById('copyCode').addEventListener('click', copyGeneratedCode);
}

// File handling
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = '#764ba2';
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.style.borderColor = '#667eea';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showUploadProgress();
    
    fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            uploadedData = data;
            displayColumns(data.columns);
            nextStep();
        } else {
            alert('Error uploading file: ' + data.error);
        }
        hideUploadProgress();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error uploading file');
        hideUploadProgress();
    });
}

function showUploadProgress() {
    document.getElementById('uploadProgress').style.display = 'block';
    let progress = 0;
    const interval = setInterval(() => {
        progress += 10;
        document.getElementById('progressFill').style.width = progress + '%';
        if (progress >= 100) {
            clearInterval(interval);
        }
    }, 100);
}

function hideUploadProgress() {
    document.getElementById('uploadProgress').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';
}

// Column selection
function displayColumns(columns) {
    const grid = document.getElementById('columnsGrid');
    grid.innerHTML = '';
    
    columns.forEach(column => {
        const columnDiv = document.createElement('div');
        columnDiv.className = 'column-item';
        columnDiv.textContent = column;
        columnDiv.addEventListener('click', () => selectColumn(columnDiv, column));
        grid.appendChild(columnDiv);
    });
}

function selectColumn(element, column) {
    document.querySelectorAll('.column-item').forEach(item => {
        item.classList.remove('selected');
    });
    element.classList.add('selected');
    selectedTargetColumn = column;
    document.getElementById('confirmColumns').disabled = false;
}

function confirmColumnSelection() {
    if (selectedTargetColumn) {
        nextStep();
    }
}

// Task type selection
function selectTaskType(e) {
    document.querySelectorAll('.task-option').forEach(option => {
        option.classList.remove('selected');
    });
    e.currentTarget.classList.add('selected');
    selectedTaskType = e.currentTarget.dataset.task;
    document.getElementById('confirmTask').disabled = false;
}

function confirmTaskType() {
    if (selectedTaskType) {
        nextStep();
        startAnalysis();
    }
}

// Analysis
function startAnalysis() {
    const data = {
        target_column: selectedTargetColumn,
        task_type: selectedTaskType
    };
    
    fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            analysisResults = data;
            displayAnalysisResults(data);
            document.getElementById('continueTraining').style.display = 'block';
        } else {
            alert('Error in analysis: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error in analysis');
    });
}

function displayAnalysisResults(data) {
    document.getElementById('analysisStatus').style.display = 'none';
    const resultsDiv = document.getElementById('analysisResults');
    resultsDiv.style.display = 'block';
    
    let html = '<h3>Analysis Results</h3>';
    
    // Drift detection
    if (data.drift_results && data.drift_results.length > 0) {
        html += '<div class="metric-card"><h4>Data Drift Detected</h4>';
        data.drift_results.forEach(drift => {
            const severity = drift.psi > 0.5 ? 'high' : 'medium';
            html += `<div class="drift-item ${severity}">
                <span>${drift.column}</span>
                <span>PSI: ${drift.psi.toFixed(4)}</span>
            </div>`;
        });
        html += '</div>';
    } else {
        html += '<div class="metric-card"><h4>No Significant Drift Detected</h4></div>';
    }
    
    // Base model performance
    if (data.base_model_performance) {
        html += '<div class="metric-card"><h4>Base Model Performance</h4>';
        Object.entries(data.base_model_performance).forEach(([key, value]) => {
            if (typeof value === 'number') {
                html += `<p><strong>${key}:</strong> ${value.toFixed(4)}</p>`;
            }
        });
        html += '</div>';
    }
    
    resultsDiv.innerHTML = html;
}

function proceedToTraining() {
    nextStep();
}

// Feature selection and model training
function selectFeatureMethod(e) {
    const method = e.currentTarget.dataset.method;
    
    // Update button state
    document.querySelectorAll('.fs-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    e.currentTarget.classList.add('active');
    
    // Start training with selected method
    trainModels(method);
}

function trainModels(method) {
    const data = {
        target_column: selectedTargetColumn,
        task_type: selectedTaskType,
        feature_method: method
    };
    
    // Show loading
    const resultsDiv = document.getElementById('trainingResults');
    resultsDiv.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i><p>Training models...</p></div>';
    
    fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayTrainingResults(data);
            if (data.best_model) {
                bestModel = data.best_model;
                showResultsStep();
            }
        } else {
            alert('Error in training: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error in training');
    });
}

function displayTrainingResults(data) {
    const resultsDiv = document.getElementById('trainingResults');
    let html = '<h3>Training Results</h3>';
    
    if (data.selected_features) {
        html += `<div class="metric-card">
            <h4>Selected Features (${data.selected_features.length})</h4>
            <p>${data.selected_features.join(', ')}</p>
        </div>`;
    }
    
    if (data.model_performances) {
        html += '<h4>Model Performance Comparison</h4>';
        Object.entries(data.model_performances).forEach(([model, performance]) => {
            html += '<div class="metric-card">';
            html += `<h4>${model}</h4>`;
            Object.entries(performance).forEach(([metric, value]) => {
                if (typeof value === 'number') {
                    html += `<p><strong>${metric}:</strong> ${value.toFixed(4)}</p>`;
                }
            });
            html += '</div>';
        });
    }
    
    resultsDiv.innerHTML = html;
}

function showResultsStep() {
    currentStep = 5;
    updateStepIndicator();
    showStep(currentStep);
    
    if (bestModel) {
        displayBestModel(bestModel);
    }
}

function displayBestModel(model) {
    const infoDiv = document.getElementById('bestModelInfo');
    let html = `
        <div class="metric-card">
            <h3>🏆 Best Model Found</h3>
            <p><strong>Model:</strong> ${model.name}</p>
            <p><strong>Feature Selection:</strong> ${model.feature_method}</p>
            <p><strong>Score:</strong> ${model.score.toFixed(4)}</p>
            <p><strong>Features Used:</strong> ${model.features_count}</p>
        </div>
    `;
    infoDiv.innerHTML = html;
}

// Code generation
function generateCode() {
    if (!bestModel) {
        alert('No best model available');
        return;
    }
    
    fetch(`${API_BASE}/generate-code`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model: bestModel.name,
            feature_method: bestModel.feature_method,
            task_type: selectedTaskType,
            target_column: selectedTargetColumn
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('generatedCode').textContent = data.code;
            document.getElementById('codeModal').style.display = 'block';
        } else {
            alert('Error generating code: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error generating code');
    });
}

function copyGeneratedCode() {
    const code = document.getElementById('generatedCode').textContent;
    navigator.clipboard.writeText(code).then(() => {
        alert('Code copied to clipboard!');
    });
}

function closeModal() {
    document.getElementById('codeModal').style.display = 'none';
}

// Step navigation
function nextStep() {
    if (currentStep < 5) {
        currentStep++;
        updateStepIndicator();
        showStep(currentStep);
    }
}

function updateStepIndicator() {
    document.querySelectorAll('.step').forEach((step, index) => {
        step.classList.remove('active', 'completed');
        if (index < currentStep) {
            step.classList.add('completed');
        } else if (index === currentStep) {
            step.classList.add('active');
        }
    });
}

function showStep(stepIndex) {
    document.querySelectorAll('.step-content').forEach((content, index) => {
        content.classList.remove('active');
        if (index === stepIndex) {
            content.classList.add('active');
        }
    });
}