// Main application JavaScript
let currentECGData = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const generateSampleBtn = document.getElementById('generateSampleBtn');
const manualInputBtn = document.getElementById('manualInputBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const visualizationSection = document.getElementById('visualizationSection');
const resultsSection = document.getElementById('resultsSection');

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
generateSampleBtn.addEventListener('click', generateSampleECG);
analyzeBtn.addEventListener('click', analyzeECG);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// File Processing
function handleFile(file) {
    showNotification(`Loading ${file.name}...`, 'info');

    const reader = new FileReader();

    reader.onload = function (e) {
        try {
            const content = e.target.result;
            let ecgData;

            // Parse based on file type
            if (file.name.endsWith('.json')) {
                const json = JSON.parse(content);
                ecgData = json.signal || json.data || json.ecg_signal || json;
                if (Array.isArray(ecgData) && ecgData.length > 0 && typeof ecgData[0] === 'object') {
                    ecgData = ecgData.map(item => item.value || item.amplitude || item);
                }
            } else if (file.name.endsWith('.csv') || file.name.endsWith('.txt')) {
                // Parse CSV/TXT - assume one value per line or comma-separated
                ecgData = content.split(/[\n,\r]+/)
                    .map(val => parseFloat(val.trim()))
                    .filter(val => !isNaN(val));
            }

            if (ecgData && ecgData.length > 0) {
                currentECGData = ecgData;
                visualizeECG(ecgData);
                showSection(visualizationSection);
                hideSection(resultsSection);
                showNotification(`✓ Loaded ${ecgData.length} samples from ${file.name}`, 'success');
            } else {
                showNotification('Could not parse ECG data from file', 'error');
            }
        } catch (error) {
            showNotification('Error reading file: ' + error.message, 'error');
        }
    };

    reader.readAsText(file);
}

// Generate Sample ECG
function generateSampleECG() {
    showNotification('Generating sample ECG...', 'info');

    // Generate a synthetic ECG-like signal
    const fs = 360; // Sampling frequency
    const duration = 10; // seconds
    const numSamples = fs * duration;

    const ecgData = [];
    for (let i = 0; i < numSamples; i++) {
        const t = i / fs;

        // Simulate ECG waveform with multiple harmonics
        const heartRate = 1.2; // Hz
        let signal = 0;

        // P wave
        signal += 0.2 * Math.sin(2 * Math.PI * heartRate * t);

        // QRS complex (sharp peak)
        signal += 1.0 * Math.sin(2 * Math.PI * heartRate * 3 * t);

        // T wave
        signal += 0.3 * Math.sin(2 * Math.PI * heartRate * 2 * t);

        // Add some noise
        signal += 0.05 * (Math.random() - 0.5);

        ecgData.push(signal);
    }

    currentECGData = ecgData;
    visualizeECG(ecgData);
    showSection(visualizationSection);
    hideSection(resultsSection);
    showNotification('✓ Sample ECG generated successfully!', 'success');
}

// Visualize ECG using Plotly
function visualizeECG(ecgData) {
    const trace = {
        y: ecgData,
        type: 'scatter',
        mode: 'lines',
        line: {
            color: '#1976d2',
            width: 1.5
        },
        name: 'ECG Signal'
    };

    const layout = {
        title: {
            text: 'ECG Signal Visualization',
            font: { color: '#212121', size: 18, family: 'Inter' }
        },
        xaxis: {
            title: 'Sample',
            color: '#757575',
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        yaxis: {
            title: 'Amplitude',
            color: '#757575',
            gridcolor: '#e0e0e0',
            showgrid: true
        },
        paper_bgcolor: '#fafafa',
        plot_bgcolor: '#ffffff',
        font: { color: '#212121', family: 'Inter' },
        margin: { t: 50, r: 30, b: 50, l: 60 },
        hovermode: 'closest'
    };

    const config = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };

    Plotly.newPlot('ecgPlot', [trace], layout, config);
}

// Analyze ECG
async function analyzeECG() {
    if (!currentECGData) {
        showNotification('Please load ECG data first', 'error');
        return;
    }

    // Show loading state
    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                ecg_signal: currentECGData
            })
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();

        if (result.success) {
            displayResults(result);
            showSection(resultsSection);
            showNotification('✓ Analysis complete!', 'success');

            if (result.demo_mode) {
                setTimeout(() => {
                    showNotification('⚠️ Demo mode: Predictions are simulated', 'warning');
                }, 1500);
            }
        } else {
            showNotification('Error: ' + (result.error || 'Unknown error'), 'error');
        }
    } catch (error) {
        showNotification('Error analyzing ECG: ' + error.message, 'error');
    } finally {
        // Reset button state
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display Results
function displayResults(result) {
    // Prediction
    const predictionIcon = document.getElementById('predictionIcon');
    const predictionLabel = document.getElementById('predictionLabel');
    const predictionConfidence = document.getElementById('predictionConfidence');

    const isNormal = result.prediction.toLowerCase().includes('normal');

    predictionIcon.className = 'prediction-icon ' + (isNormal ? 'normal' : 'abnormal');
    predictionIcon.textContent = isNormal ? '✓' : '⚠';
    predictionLabel.textContent = result.prediction;

    const maxProb = Math.max(...Object.values(result.probabilities));
    predictionConfidence.textContent = `Confidence: ${(maxProb * 100).toFixed(1)}%`;

    // Probabilities
    const probabilitiesContainer = document.getElementById('probabilitiesContainer');
    probabilitiesContainer.innerHTML = '';

    for (const [className, probability] of Object.entries(result.probabilities)) {
        const barDiv = document.createElement('div');
        barDiv.className = 'probability-bar';

        barDiv.innerHTML = `
            <div class="probability-label">
                <span>${className}</span>
                <span><strong>${(probability * 100).toFixed(1)}%</strong></span>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: 0%"></div>
            </div>
        `;

        probabilitiesContainer.appendChild(barDiv);

        // Animate bar
        setTimeout(() => {
            const fill = barDiv.querySelector('.probability-fill');
            fill.style.width = `${probability * 100}%`;
        }, 100);
    }

    // Features
    const featuresContainer = document.getElementById('featuresContainer');
    featuresContainer.innerHTML = '';

    for (const [featureName, featureValue] of Object.entries(result.features)) {
        const featureDiv = document.createElement('div');
        featureDiv.className = 'feature-item';

        featureDiv.innerHTML = `
            <div class="feature-label">${featureName}</div>
            <div class="feature-value">${featureValue.toFixed(2)}</div>
        `;

        featuresContainer.appendChild(featureDiv);
    }
}

// Utility Functions
function showSection(section) {
    section.style.display = 'block';
    section.style.animation = 'fadeIn 0.5s ease-out';
}

function hideSection(section) {
    section.style.display = 'none';
}

// Notification System
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existing = document.querySelector('.snackbar');
    if (existing) {
        existing.remove();
    }

    const snackbar = document.createElement('div');
    snackbar.className = 'snackbar';
    snackbar.textContent = message;

    // Color based on type
    if (type === 'success') {
        snackbar.style.background = '#4caf50';
    } else if (type === 'error') {
        snackbar.style.background = '#f44336';
    } else if (type === 'warning') {
        snackbar.style.background = '#ff9800';
    } else {
        snackbar.style.background = '#2196f3';
    }

    document.body.appendChild(snackbar);

    // Auto remove after 3 seconds
    setTimeout(() => {
        snackbar.style.animation = 'slideDown 0.3s ease-out reverse';
        setTimeout(() => snackbar.remove(), 300);
    }, 3000);
}

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();

        if (!health.model_loaded) {
            console.warn('Warning: No model loaded on server');
        }

        if (health.demo_mode) {
            showNotification('Running in DEMO mode - predictions are simulated', 'warning');
        }
    } catch (error) {
        console.error('Could not connect to server:', error);
        showNotification('Warning: Could not connect to server', 'error');
    }
});
