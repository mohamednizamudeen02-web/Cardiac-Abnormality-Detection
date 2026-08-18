/**
 * Cardiac Monitor Pro — Advanced UI Logic v3.0
 * Gemini-quality frontend architecture: modular, accessible, reactive.
 */

'use strict';

// ── State ───────────────────────────────────────────────────────────────────
let socket = null;
let currentECGData = null;
let isMonitoring = false;
let streamInterval = null;
let lastPredictionResult = null;
let currentTheme = localStorage.getItem('cm-theme') || 'dark';

// ── DOM References ───────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const uploadArea = $('uploadArea');
const fileInput = $('fileInput');
const generateSampleBtn = $('generateSampleBtn');
const startStreamBtn = $('startStreamBtn');
const analyzeBtn = $('analyzeBtn');
const ecgSection = $('ecgSection');
const resultsSection = $('resultsSection');
const xaiSection = $('xaiSection');
const featuresSection = $('featuresSection');
const realTimeBtn = $('realTimeBtn');
const realTimeModal = $('realTimeModal');
const patientsModal = $('patientsModal');
const settingsModal = $('settingsModal');
const themeToggle = $('themeToggle');
const themeSwitch = $('themeSwitch');
const toastContainer = $('toastContainer');

// ── Role-Based Access Control ────────────────────────────────────────────────
// Role is server-rendered into body.dataset.role at page load (no extra request)
const USER_ROLE = document.body.dataset.role || 'viewer';
const isViewer = USER_ROLE === 'viewer';

// ── Utility: debounce ────────────────────────────────────────────────────────
function debounce(fn, delay) {
    let timer;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

// ── Utility: section visibility ──────────────────────────────────────────────
function showSection(el) {
    el.style.display = 'block';
    requestAnimationFrame(() => el.classList.add('fade-in'));
}
function hideSection(el) { el.style.display = 'none'; el.classList.remove('fade-in'); }

// ── Toast Notifications ──────────────────────────────────────────────────────
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `<span class="toast-bar"></span><span>${message}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast--out');
        toast.addEventListener('animationend', () => toast.remove());
    }, 3500);
}

// ── WebSocket Init ───────────────────────────────────────────────────────────
function initializeSocket() {
    socket = io();
    // Silently update the status dot on connect/disconnect — no toast spam
    socket.on('connect', () => {
        const dot = document.querySelector('.status-dot');
        if (dot) { dot.style.background = 'var(--success)'; dot.style.boxShadow = '0 0 12px var(--success)'; }
    });
    socket.on('disconnect', () => {
        const dot = document.querySelector('.status-dot');
        if (dot) { dot.style.background = 'var(--danger)'; dot.style.boxShadow = '0 0 12px var(--danger)'; }
    });
    socket.on('prediction_update', updateRealtimeDisplay);
    socket.on('critical_alert', data => showNotification(`🚨 CRITICAL: ${data.message}`, 'error'));
    socket.on('monitoring_started', () => {
        $('monitoringStatus').textContent = 'Active';
        $('monitoringStatus').style.color = 'var(--success)';
    });
}

// ── Drag-and-Drop Handlers ───────────────────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } });
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

// ── File Processing ──────────────────────────────────────────────────────────
function handleFile(file) {
    if (isViewer) { showNotification('🔒 View-only access — file upload is disabled', 'error'); return; }
    showNotification(`Loading ${file.name}…`, 'info');
    const reader = new FileReader();
    reader.onload = e => {
        try {
            const content = e.target.result;
            let ecgData;
            if (file.name.endsWith('.json')) {
                const json = JSON.parse(content);
                ecgData = json.signal || json.data || json.ecg_signal || json;
                if (Array.isArray(ecgData) && ecgData.length && typeof ecgData[0] === 'object') {
                    ecgData = ecgData.map(v => v.value ?? v.amplitude ?? v);
                }
            } else {
                ecgData = content.split(/[\n,\r]+/).map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
            }
            if (ecgData?.length) {
                currentECGData = ecgData;
                visualizeECG(ecgData);
                showSection(ecgSection);
                hideSection(resultsSection);
                hideSection(xaiSection);
                showNotification(`✓ Loaded ${ecgData.length.toLocaleString()} samples`, 'success');
            } else {
                showNotification('Could not parse ECG data from file', 'error');
            }
        } catch (err) {
            showNotification(`Error reading file: ${err.message}`, 'error');
        }
    };
    reader.readAsText(file);
}

// ── Realistic P-QRS-T Sample ECG Generator ──────────────────────────────────
function gaussian(t, mu, sigma, amp) {
    return amp * Math.exp(-0.5 * ((t - mu) / sigma) ** 2);
}

function generateSampleECG() {
    if (isViewer) { showNotification('🔒 View-only access. Contact an admin.', 'error'); return; }
    showNotification('Generating sample ECG…', 'info');
    const fs = 360, duration = 10;
    const n = fs * duration;
    const ecgData = new Array(n);
    const bpm = 72, beatPeriod = 60 / bpm;

    for (let i = 0; i < n; i++) {
        const t = i / fs;
        const tBeat = t % beatPeriod;

        // P-wave, QRS complex, T-wave using Gaussians
        let s = gaussian(tBeat, 0.10, 0.025, 0.15)     // P wave
            + gaussian(tBeat, 0.20, 0.005, -0.18)      // Q notch
            + gaussian(tBeat, 0.22, 0.010, 1.10)       // R peak
            + gaussian(tBeat, 0.24, 0.005, -0.12)      // S notch
            + gaussian(tBeat, 0.36, 0.040, 0.22);      // T wave
        s += (Math.random() - 0.5) * 0.03;              // noise
        ecgData[i] = s;
    }

    currentECGData = ecgData;
    visualizeECG(ecgData);
    showSection(ecgSection);
    hideSection(resultsSection);
    hideSection(xaiSection);
    showNotification('✓ Sample ECG generated (P-QRS-T morphology)', 'success');
}

// ── Plotly Theme Helper ──────────────────────────────────────────────────────
function getPlotTheme() {
    const dark = currentTheme === 'dark';
    return {
        paper_bgcolor: dark ? 'rgba(13,17,39,0.95)' : 'rgba(255,255,255,0.95)',
        plot_bgcolor: dark ? 'rgba(6,9,26,0.95)' : 'rgba(238,242,251,0.95)',
        font: { color: dark ? '#b8c4e0' : '#334070', family: 'Inter, sans-serif', size: 12 },
        xaxis: { gridcolor: dark ? 'rgba(0,212,255,.08)' : 'rgba(0,150,200,.15)', color: dark ? '#6b7db3' : '#6478a8', zerolinecolor: 'rgba(0,212,255,.2)' },
        yaxis: { gridcolor: dark ? 'rgba(0,212,255,.08)' : 'rgba(0,150,200,.15)', color: dark ? '#6b7db3' : '#6478a8', zerolinecolor: 'rgba(0,212,255,.2)' },
    };
}

// ── ECG Visualization ────────────────────────────────────────────────────────
function visualizeECG(ecgData) {
    const theme = getPlotTheme();
    const display = ecgData.length > 3600 ? ecgData.slice(0, 3600) : ecgData;
    const x = Array.from({ length: display.length }, (_, i) => (i / 360).toFixed(3));

    const trace = {
        x, y: display, type: 'scatter', mode: 'lines',
        line: { color: '#00d4ff', width: 1.8, shape: 'spline' },
        name: 'ECG Signal',
        hovertemplate: '%{x}s: %{y:.4f}<extra></extra>',
    };
    const layout = {
        ...theme,
        title: { text: 'ECG Signal', font: { color: '#00d4ff', size: 14, family: 'Outfit, sans-serif', weight: 700 }, x: 0.02 },
        xaxis: { ...theme.xaxis, title: { text: 'Time (s)', font: { size: 11 } } },
        yaxis: { ...theme.yaxis, title: { text: 'Amplitude (mV)', font: { size: 11 } } },
        margin: { t: 44, r: 20, b: 48, l: 58 },
        hovermode: 'x unified',
    };
    Plotly.newPlot('ecgPlot', [trace], layout, { responsive: true, displaylogo: false, displayModeBar: true });
}

// ── Analyze ECG ──────────────────────────────────────────────────────────────
const analyzeECG = debounce(async () => {
    if (isViewer) { showNotification('🔒 View-only access — analysis is disabled. Contact an admin.', 'error'); return; }
    if (!currentECGData) { showNotification('Please load ECG data first', 'error'); return; }

    const btnText = analyzeBtn.querySelector('.btn-text');
    const btnLoader = analyzeBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    analyzeBtn.disabled = true;

    try {
        const patientId = $('selectedPatient')?.value || '';
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ecg_signal: currentECGData, patient_id: patientId }),
        });
        if (!res.ok) { const e = await res.json(); throw new Error(e.error || 'Prediction failed'); }
        const result = await res.json();
        if (result.success) {
            displayResults(result);
            displayXAI(result);
            showSection(resultsSection);
            showSection(featuresSection);
            updateLiveStats(result);
            // Refresh history after new analysis
            if (patientId) loadPatientHistory(patientId);
            if (result.demo_mode) showNotification('ℹ️ Demo mode — no trained model loaded', 'warning');
            else showNotification('✓ Analysis complete!', 'success');
        } else {
            showNotification(`Error: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (err) {
        showNotification(`Error: ${err.message}`, 'error');
    } finally {
        btnText.style.display = 'inline-flex';
        btnLoader.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}, 300);

// ── Display Results ──────────────────────────────────────────────────────────
function displayResults(result) {
    lastPredictionResult = result;

    const prediction = result.prediction || '';
    const isNormal = prediction.toLowerCase().includes('normal');
    const isCritical = prediction.toLowerCase().includes('mi') ||
        prediction.toLowerCase().includes('infarct') ||
        prediction.toLowerCase().includes('ischemia') ||
        prediction.toLowerCase().includes('block') ||
        prediction.toLowerCase().includes('vf') ||
        prediction.toLowerCase().includes('vt ');
    const isSerious = !isNormal && (result.risk_level === 'High' || isCritical);

    $('predictionIcon').className = `prediction-icon ${isNormal ? 'normal' : 'abnormal'}`;
    $('predictionIcon').textContent = isNormal ? '✓' : '⚠';

    const labelEl = $('predictionLabel');
    labelEl.textContent = prediction;
    labelEl.style.color = isNormal ? '#26de81' : (isSerious ? '#ff4757' : '#ffa500');
    if (isSerious) {
        labelEl.style.textShadow = '0 0 16px rgba(255,71,87,.35)';
    } else {
        labelEl.style.textShadow = '';
    }

    $('predictionConfidence').textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

    const badge = $('riskBadge');
    badge.textContent = result.risk_level;
    badge.className = `risk-badge ${result.risk_level.toLowerCase()}`;
    if (isSerious) {
        badge.style.animation = 'pulse 1.2s ease-in-out infinite';
    } else {
        badge.style.animation = '';
    }

    // Enhanced colour-coded probability bars
    const probClasses = {
        'Normal Sinus': 'normal',
        'Arrhythmia': 'arrhy',
        'MI / Ischemia': 'mi',
        'Other Abnormality': 'other',
    };
    const container = $('probabilitiesContainer');
    container.innerHTML = '';
    const entries = Object.entries(result.probabilities);
    for (const [cls, prob] of entries) {
        const key = Object.keys(probClasses).find(k => cls.toLowerCase().includes(k.split(' ')[0].toLowerCase())) || 'other';
        const colorClass = probClasses[key] || 'other';
        const isCls = cls.toLowerCase().includes('mi') || cls.toLowerCase().includes('infarct') || cls.toLowerCase().includes('ischemia');
        const isHighProb = prob >= 0.4;
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.setAttribute('role', 'listitem');
        // Show red warning badge for serious conditions with high probability
        const warnBadge = (isCls && isHighProb)
            ? `<span style="margin-left:.4rem;font-size:.6rem;background:#ff475722;color:#ff4757;border:1px solid #ff475766;border-radius:99px;padding:.1rem .45rem;font-weight:800;">⚠ CRITICAL</span>`
            : '';
        row.innerHTML = `
            <span class="prob-label" title="${cls}" style="${(isCls && isHighProb) ? 'color:#ff4757;font-weight:700;' : ''}">${cls}${warnBadge}</span>
            <div class="prob-track"><div class="prob-fill prob-fill--${colorClass}" style="width:0%"></div></div>
            <span class="prob-pct" style="${(isCls && isHighProb) ? 'color:#ff4757;' : ''}"> ${(prob * 100).toFixed(1)}%</span>`;
        container.appendChild(row);
        setTimeout(() => { row.querySelector('.prob-fill').style.width = `${prob * 100}%`; }, 80);
    }

    // Risk needle gauge
    const riskPct = result.risk_level === 'High' ? 85 : result.risk_level === 'Medium' ? 50 : 15;
    const needle = $('riskNeedle');
    if (needle) needle.style.left = `${riskPct}%`;

    // Uncertainty bar
    $('uncertaintyFill').style.width = `${result.uncertainty * 100}%`;
    $('uncertaintyText').textContent = `Uncertainty: ${(result.uncertainty * 100).toFixed(1)}%`;

    // Feature cards
    const fc = $('featuresContainer');
    fc.innerHTML = '';
    for (const [name, val] of Object.entries(result.features)) {
        const d = document.createElement('div');
        d.className = 'feature-item';
        d.setAttribute('role', 'listitem');
        d.innerHTML = `<div class="feature-label">${name}</div><div class="feature-value">${typeof val === 'number' ? val.toFixed(2) : val}</div>`;
        fc.appendChild(d);
    }

    // Serious condition banner
    const existingBanner = $('severityBanner');
    if (existingBanner) existingBanner.remove();
    if (isSerious) {
        const banner = document.createElement('div');
        banner.id = 'severityBanner';
        banner.style.cssText = 'background:rgba(255,71,87,.12);border:1.5px solid rgba(255,71,87,.5);border-radius:12px;padding:.7rem 1rem;margin-top:.75rem;display:flex;align-items:center;gap:.6rem;animation:fadeIn .4s;';
        banner.innerHTML = `<span style="font-size:1.1rem;">🚨</span><div><div style="font-size:.8rem;font-weight:800;color:#ff4757;margin-bottom:.15rem;">CRITICAL FINDING DETECTED</div><div style="font-size:.74rem;color:#b8c4e0;">${prediction} — immediate cardiology review recommended. Do not dismiss these findings.</div></div>`;
        // Insert after prediction-main div
        const predMain = document.querySelector('.prediction-main');
        if (predMain && predMain.parentNode) predMain.parentNode.insertBefore(banner, predMain.nextSibling);
    }
}

// ── Patient Selector ──────────────────────────────────────────────────────────
let _allPatients = [];

async function initPatientSelector() {
    try {
        const res = await fetch('/api/patients');
        const data = await res.json();
        _allPatients = data.patients || [];
        const sel = $('selectedPatient');
        sel.innerHTML = '<option value="">— No patient selected —</option>' +
            _allPatients.map(p => `<option value="${p.patient_id}">${p.name} (${p.patient_id})</option>`).join('');
    } catch { /* silently ignore */ }
}

function updatePatientChip(pid) {
    const p = _allPatients.find(x => x.patient_id === pid);
    const chip = $('patientInfoChip');
    const histBtn = $('viewHistoryBtn');
    if (!p) { chip.style.display = 'none'; histBtn.style.display = 'none'; return; }
    $('chipName').textContent = p.name;
    $('chipMeta').textContent = `${p.age ?? '?'}y · ${p.gender ?? '?'} · ${p.patient_id}`;
    chip.style.display = 'flex';
    histBtn.style.display = 'inline-flex';
}

$('selectedPatient')?.addEventListener('change', e => {
    const pid = e.target.value;
    updatePatientChip(pid);
    if (pid) loadPatientHistory(pid);
    else {
        $('historySection').style.display = 'none';
    }
});

$('viewHistoryBtn')?.addEventListener('click', () => {
    const pid = $('selectedPatient')?.value;
    if (pid) loadPatientHistory(pid);
});

// ── Patient History Timeline ──────────────────────────────────────────────────
async function loadPatientHistory(patientId) {
    const section = $('historySection');
    const tag = $('historyPatientTag');
    const timeline = $('historyTimeline');
    const trendEl = $('trendPlot');

    section.style.display = 'block';
    const p = _allPatients.find(x => x.patient_id === patientId);
    tag.textContent = p ? p.name : patientId;
    timeline.innerHTML = '<li style="color:var(--txt-3);font-size:.8rem;padding:.5rem 0 .5rem 1rem">Loading…</li>';

    try {
        const res = await fetch(`/api/analyses/${patientId}`);
        const data = await res.json();
        const analyses = data.analyses || [];

        if (!analyses.length) {
            timeline.innerHTML = '<li class="history-empty">No analyses recorded for this patient yet.<br>Run an ECG analysis above to begin.</li>';
            trendEl.innerHTML = '';
            return;
        }

        // Render timeline
        timeline.innerHTML = analyses.slice(0, 10).map(a => {
            const d = new Date(a.analysis_timestamp);
            const dateStr = d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const riskCls = (a.risk_level || 'Low').toLowerCase();
            return `<li class="timeline-item">
                <div class="tl-date">${dateStr}</div>
                <div class="tl-pred">${a.prediction || '—'}</div>
                <div class="tl-meta">
                    Conf: ${a.confidence ? (a.confidence * 100).toFixed(1) + '%' : '—'} &nbsp;·&nbsp;
                    <span class="risk-pill risk-pill--${riskCls}">${a.risk_level || '—'}</span>
                </div>
            </li>`;
        }).join('');

        // Risk trend sparkline
        const riskMap = { Low: 1, Medium: 2, High: 3 };
        const reversed = [...analyses].reverse();
        const xs = reversed.map((_, i) => i + 1);
        const ys = reversed.map(a => riskMap[a.risk_level] || 1);
        const labels = reversed.map(a => new Date(a.analysis_timestamp).toLocaleDateString());
        const colors = reversed.map(a => a.risk_level === 'High' ? '#ff4757' : a.risk_level === 'Medium' ? '#ffa500' : '#26de81');

        const theme = getPlotTheme();
        Plotly.newPlot(trendEl, [{
            x: xs, y: ys, text: labels,
            mode: 'lines+markers',
            line: { color: '#00d4ff', width: 2, shape: 'spline' },
            marker: { color: colors, size: 8, line: { width: 1.5, color: '#fff' } },
            hovertemplate: '%{text}: %{y}<extra></extra>',
            fill: 'tozeroy', fillcolor: 'rgba(0,212,255,0.07)',
        }], {
            ...theme,
            yaxis: { ...theme.yaxis, range: [0.5, 3.5], tickvals: [1, 2, 3], ticktext: ['Low', 'Med', 'High'], gridcolor: 'rgba(255,255,255,.05)' },
            xaxis: { ...theme.xaxis, title: 'Analysis #', showgrid: false },
            margin: { l: 48, r: 12, t: 8, b: 32 },
            height: 140,
        }, { displayModeBar: false, responsive: true });

    } catch (err) {
        timeline.innerHTML = `<li class="history-empty">Error loading history: ${err.message}</li>`;
    }
}

// ── Display XAI ──────────────────────────────────────────────────────────────
function displayXAI(result) {
    const theme = getPlotTheme();

    // Grad-CAM
    if (result.gradcam_heatmap) {
        showSection(xaiSection);
        const trace = {
            y: result.gradcam_heatmap, type: 'scatter', mode: 'lines',
            fill: 'tozeroy',
            line: { color: '#00ffc8', width: 2, shape: 'spline' },
            fillcolor: 'rgba(0,255,200,0.12)',
            name: 'Importance',
        };
        const layout = {
            ...theme,
            xaxis: { ...theme.xaxis, title: { text: 'Sample', font: { size: 10 } } },
            yaxis: { ...theme.yaxis, title: { text: 'Importance', font: { size: 10 } } },
            margin: { t: 12, r: 12, b: 36, l: 52 },
            height: 190,
        };
        Plotly.newPlot('gradcamPlot', [trace], layout, { responsive: true, displaylogo: false, displayModeBar: false });
    }

    // SHAP values
    if (result.shap_values) {
        const sc = $('shapContainer');
        sc.innerHTML = '';
        const entries = Object.entries(result.shap_values).sort((a, b) => b[1] - a[1]);
        for (const [feature, value] of entries) {
            const item = document.createElement('div');
            item.className = 'shap-item';
            item.setAttribute('role', 'listitem');
            item.innerHTML = `
                <div class="shap-label" title="${feature}">${feature}</div>
                <div class="shap-bar"><div class="shap-fill" style="width:0%"></div></div>
                <div class="shap-value">${(value * 100).toFixed(0)}%</div>`;
            sc.appendChild(item);
            setTimeout(() => { item.querySelector('.shap-fill').style.width = `${Math.abs(value) * 100}%`; }, 80);
        }
    }

    // Recommendations — mark urgent items red
    if (result.recommendations) {
        const list = $('recommendationsList');
        const urgentKeywords = ['immediate', 'emergency', 'urgent', 'critical', 'call', '911', 'cardiology', 'hospital', 'seek'];
        list.innerHTML = result.recommendations
            .map(r => {
                const isUrgent = urgentKeywords.some(kw => r.toLowerCase().includes(kw));
                const style = isUrgent
                    ? 'color:#ff4757;font-weight:700;background:rgba(255,71,87,.08);border-left:3px solid #ff4757;padding:.25rem .5rem;border-radius:0 6px 6px 0;margin-bottom:.3rem;'
                    : '';
                const prefix = isUrgent ? '🚨 ' : '';
                return `<li style="${style}">${prefix}${r}</li>`;
            })
            .join('');
    }
}

// ── Live Stats ───────────────────────────────────────────────────────────────
function updateLiveStats(result) {
    $('liveHeartRate').textContent = `${(result.features['Heart Rate'] ?? 0).toFixed(0)} bpm`;
    $('liveHRV').textContent = `${(result.features['SDNN (HRV)'] ?? 0).toFixed(0)} ms`;
    $('liveStatus').textContent = result.prediction;
    $('liveConfidence').textContent = `${(result.confidence * 100).toFixed(0)}%`;
}

function updateRealtimeDisplay(data) {
    $('liveHeartRate').textContent = `${data.heart_rate.toFixed(0)} bpm`;
    $('liveStatus').textContent = data.prediction;
    $('liveConfidence').textContent = `${(data.confidence * 100).toFixed(0)}%`;
}

// ── Real-time Stream ─────────────────────────────────────────────────────────
function startRealtimeStream() {
    if (isViewer) { showNotification('🔒 View-only access — live monitoring is disabled.', 'error'); return; }
    if (!currentECGData?.length) { showNotification('⚠️ Please load or generate ECG data first', 'warning'); return; }
    if (!socket) initializeSocket();
    realTimeModal.style.display = 'flex';
}

function startMonitoring() {
    if (isViewer) { showNotification('🔒 View-only access — monitoring is disabled.', 'error'); return; }
    if (!currentECGData?.length) { showNotification('⚠️ Please load or generate ECG data first', 'warning'); return; }
    if (!socket) initializeSocket();
    const patientId = $('patientIdInput').value || 'PATIENT_001';
    socket.emit('start_monitoring', { patient_id: patientId });
    isMonitoring = true;
    $('startMonitoringBtn').disabled = true;
    $('stopMonitoringBtn').disabled = false;
    showNotification('✓ Real-time monitoring started!', 'success');

    let idx = 0;
    streamInterval = setInterval(() => {
        if (!isMonitoring) return;
        const chunk = currentECGData.slice(idx, idx + 100);
        if (chunk.length) { socket.emit('stream_ecg', { ecg_data: chunk, patient_id: patientId }); idx += 100; }
        else idx = 0;
    }, 500);
}

function stopMonitoring() {
    isMonitoring = false;
    clearInterval(streamInterval); streamInterval = null;
    $('startMonitoringBtn').disabled = false;
    $('stopMonitoringBtn').disabled = true;
    $('monitoringStatus').textContent = 'Stopped';
    $('monitoringStatus').style.color = 'var(--danger)';
    showNotification('Monitoring stopped', 'info');
}

// ── PDF Report ───────────────────────────────────────────────────────────────
async function generatePDFReport() {
    if (isViewer) { showNotification('🔒 View-only access — report generation is disabled.', 'error'); return; }
    if (!lastPredictionResult) { showNotification('⚠️ Analyze an ECG first before generating a report', 'warning'); return; }
    showNotification('📄 Generating PDF report…', 'info');
    const patientId = $('selectedPatient')?.value || 'PATIENT_001';
    try {
        const res = await fetch('/api/generate-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: patientId,
                prediction: lastPredictionResult.prediction,
                confidence: lastPredictionResult.confidence,
                uncertainty: lastPredictionResult.uncertainty,
                probabilities: lastPredictionResult.probabilities,
                features: lastPredictionResult.features,
                recommendations: lastPredictionResult.recommendations,
                risk_level: lastPredictionResult.risk_level,
            }),
        });
        if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = Object.assign(document.createElement('a'), { href: url, download: `cardiac_report_${patientId}_${Date.now()}.pdf` });
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showNotification('✅ Report downloaded!', 'success');
        } else {
            const err = await res.json().catch(() => ({}));
            showNotification(`❌ ${err.error || 'Failed to generate report'}`, 'error');
        }
    } catch (err) {
        showNotification(`❌ ${err.message}`, 'error');
    }
}

// ── Medical Report Analyzer ──────────────────────────────────────────────────
const reportUploadArea = $('reportUploadArea');
const reportFileInput = $('reportFileInput');
const browseReportBtn = $('browseReportBtn');
const reportFileName = $('reportFileName');
const analyzeReportBtn = $('analyzeReportBtn');
const analyzeReportBtnText = $('analyzeReportBtnText');
const analyzeReportLoader = $('analyzeReportLoader');
const reportResponseArea = $('reportResponseArea');
const reportMarkdownContent = $('reportMarkdownContent');

let currentReportFile = null;

if (reportUploadArea) {
    reportUploadArea.addEventListener('click', (e) => {
        if (e.target !== browseReportBtn) reportFileInput.click();
    });
    
    browseReportBtn.addEventListener('click', () => reportFileInput.click());
    
    reportUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (isViewer) return;
        reportUploadArea.style.borderColor = 'var(--primary)';
        reportUploadArea.style.background = 'rgba(0, 212, 255, 0.1)';
    });
    
    reportUploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        reportUploadArea.style.borderColor = 'rgba(0,212,255,.2)';
        reportUploadArea.style.background = 'rgba(0,0,0,.15)';
    });
    
    reportUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        if (isViewer) return;
        reportUploadArea.style.borderColor = 'rgba(0,212,255,.2)';
        reportUploadArea.style.background = 'rgba(0,0,0,.15)';
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleReportFile(e.dataTransfer.files[0]);
        }
    });
    
    reportFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleReportFile(e.target.files[0]);
        }
    });
    
    analyzeReportBtn.addEventListener('click', performReportAnalysis);
}

function handleReportFile(file) {
    if (isViewer) {
        showNotification('🔒 View-only access.', 'error');
        return;
    }
    const validTypes = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showNotification('⚠️ Unsupported file type. Please use PDF, JPG, or PNG.', 'warning');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        showNotification('⚠️ File exceeds 10MB limit.', 'error');
        return;
    }
    
    currentReportFile = file;
    reportFileName.textContent = file.name;
    reportFileName.style.display = 'block';
    analyzeReportBtn.disabled = false;
    
    // Hide previous response map if user uploads a new file
    reportResponseArea.style.display = 'none';
    reportMarkdownContent.innerHTML = '';
}

async function performReportAnalysis() {
    if (!currentReportFile) return;
    if (isViewer) {
        showNotification('🔒 View-only access — report analysis disabled.', 'error');
        return;
    }
    
    analyzeReportBtn.disabled = true;
    analyzeReportBtnText.textContent = 'Analyzing...';
    analyzeReportLoader.style.display = 'inline-block';
    
    const formData = new FormData();
    formData.append('file', currentReportFile);
    
    try {
        const res = await fetch('/api/analyze-report', {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        
        if (res.ok && data.success) {
            reportResponseArea.style.display = 'block';
            
            // Parse with marked.js if available
            if (typeof marked !== 'undefined') {
                reportMarkdownContent.innerHTML = marked.parse(data.analysis);
            } else {
                reportMarkdownContent.innerHTML = data.analysis.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<b>$1</b>');
            }
            
            showNotification('✅ Report analysis complete!', 'success');
        } else {
            showNotification(`❌ ${data.error || 'Failed to analyze report'}`, 'error');
        }
    } catch (err) {
        showNotification(`❌ Error analyzing report`, 'error');
    } finally {
        analyzeReportBtnText.textContent = 'Analyze with Gemini';
        analyzeReportLoader.style.display = 'none';
        // Only re-enable if there is NO error and we want them to be able to click it again, but usually users upload a new file.
        // If we want to allow re-analysis, we un-disable it.
        analyzeReportBtn.disabled = false;
    }
}

// ── Patients Modal ───────────────────────────────────────────────────────────
async function showPatientsModal() {
    patientsModal.style.display = 'flex';
    await loadPatients();
}

async function loadPatients(query = '') {
    try {
        const res = await fetch('/api/patients');
        const data = await res.json();
        const tbody = $('patientsTableBody');
        const patients = (data.patients || []).filter(p =>
            !query || p.name.toLowerCase().includes(query.toLowerCase()) || p.patient_id.includes(query)
        );
        if (!patients.length) { tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No patients found.</td></tr>'; return; }
        tbody.innerHTML = patients.map(p => `
            <tr>
                <td><code style="color:var(--primary);font-family:'JetBrains Mono',monospace;font-size:.78rem">${p.patient_id}</code></td>
                <td style="font-weight:600">${p.name}</td>
                <td>${p.age ?? '—'}</td>
                <td>${p.gender ?? '—'}</td>
                <td style="font-size:.78rem;color:var(--txt-3)">${p.contact_info ?? '—'}</td>
                <td style="font-size:.78rem;color:var(--txt-3);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${p.medical_history || ''}">${p.medical_history || '—'}</td>
                <td><span class="risk-badge ${p.total_analyses > 0 ? 'low' : ''}" style="font-size:.7rem">${p.total_analyses ?? 0}</span></td>
            </tr>`).join('');
    } catch (err) {
        $('patientsTableBody').innerHTML = `<tr><td colspan="6" class="table-empty">Error loading patients: ${err.message}</td></tr>`;
    }
}

// Patient search with debounce
$('patientSearch').addEventListener('input', debounce(e => loadPatients(e.target.value), 300));

// ── Settings Modal ───────────────────────────────────────────────────────────
function showSettingsModal() {
    themeSwitch.checked = currentTheme === 'dark';
    settingsModal.style.display = 'flex';
}

themeSwitch.addEventListener('change', () => {
    const newTheme = themeSwitch.checked ? 'dark' : 'light';
    setTheme(newTheme);
    showNotification(`Switched to ${newTheme} mode`, 'info');
});

// ── Theme ────────────────────────────────────────────────────────────────────
function setTheme(theme) {
    currentTheme = theme;
    document.body.classList.toggle('light-theme', theme === 'light');
    localStorage.setItem('cm-theme', theme);

    // Update nav toggle icons
    themeToggle.querySelector('.icon-moon').style.display = theme === 'dark' ? '' : 'none';
    themeToggle.querySelector('.icon-sun').style.display = theme === 'light' ? '' : 'none';
    if (themeSwitch) themeSwitch.checked = theme === 'dark';

    // Re-theme any existing Plotly charts
    const plotTheme = getPlotTheme();
    ['ecgPlot', 'gradcamPlot', 'realtimeECGPlot'].forEach(id => {
        const el = $(id);
        if (el && el.data) Plotly.relayout(id, plotTheme);
    });
}

themeToggle.addEventListener('click', () => {
    const next = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    showNotification(`Switched to ${next} mode`, 'info');
});

// ── Zoom buttons ─────────────────────────────────────────────────────────────
$('zoomInBtn').addEventListener('click', () => {
    const el = $('ecgPlot');
    if (!el?.data) return;
    Plotly.relayout('ecgPlot', { 'xaxis.autorange': false, 'xaxis.range': [0, el.layout.xaxis.range ? el.layout.xaxis.range[1] * 0.6 : 5] });
});
$('zoomOutBtn').addEventListener('click', () => {
    if ($('ecgPlot')?.data) Plotly.relayout('ecgPlot', { 'xaxis.autorange': true });
});
$('downloadECGBtn').addEventListener('click', () => {
    if ($('ecgPlot')?.data) Plotly.downloadImage('ecgPlot', { format: 'png', filename: 'ecg_signal', width: 1200, height: 500 });
});

// ── Modal open / close helpers ───────────────────────────────────────────────
function openModal(modal) { modal.style.display = 'flex'; }
function closeModal(modal) { modal.style.display = 'none'; }

realTimeBtn.addEventListener('click', startRealtimeStream);
$('closeRealTimeModal').addEventListener('click', () => closeModal(realTimeModal));
$('patientsBtn').addEventListener('click', showPatientsModal);
$('closePatientsModal').addEventListener('click', () => closeModal(patientsModal));
$('settingsBtn').addEventListener('click', showSettingsModal);
$('closeSettingsModal').addEventListener('click', () => closeModal(settingsModal));
$('generateReportBtn').addEventListener('click', generatePDFReport);
$('viewExplanationBtn').addEventListener('click', () => {
    xaiSection.style.display === 'none' ? showSection(xaiSection) : hideSection(xaiSection);
});
analyzeBtn.addEventListener('click', analyzeECG);
generateSampleBtn.addEventListener('click', generateSampleECG);
startStreamBtn.addEventListener('click', startRealtimeStream);
$('startMonitoringBtn').addEventListener('click', startMonitoring);
$('stopMonitoringBtn').addEventListener('click', stopMonitoring);

// Close modals on backdrop click
[realTimeModal, patientsModal, settingsModal].forEach(modal => {
    modal.addEventListener('click', e => { if (e.target === modal) closeModal(modal); });
});

// ── Keyboard Shortcuts ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        [realTimeModal, patientsModal, settingsModal].forEach(m => { if (m.style.display !== 'none') closeModal(m); });
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') { e.preventDefault(); themeToggle.click(); }
    if ((e.ctrlKey || e.metaKey) && e.key === 'g') { e.preventDefault(); generateSampleBtn.click(); }
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); if (currentECGData) analyzeBtn.click(); }
});

// ── Particles ────────────────────────────────────────────────────────────────
function createParticles() {
    const container = $('particlesContainer');
    if (!container) return;
    for (let i = 0; i < 28; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        const size = 2 + Math.random() * 3;
        Object.assign(p.style, {
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            width: `${size}px`,
            height: `${size}px`,
            animationDelay: `${Math.random() * 18}s`,
            animationDuration: `${16 + Math.random() * 10}s`,
        });
        container.appendChild(p);
    }
}

// ── Intersection Observer for card animations ────────────────────────────────
function observeCards() {
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('fade-in'); });
    }, { threshold: 0.1 });
    document.querySelectorAll('.card').forEach(c => obs.observe(c));
}

// ── Apply Viewer Restrictions on load ────────────────────────────────────────
function applyViewerRestrictions() {
    if (!isViewer) return;

    // Buttons to disable with a lock tooltip
    const writeElements = [
        generateSampleBtn, startStreamBtn, analyzeBtn, realTimeBtn,
        $('generateReportBtn'), $('startMonitoringBtn'),
        $('analyzeReportBtn')
    ];
    writeElements.forEach(el => {
        if (!el) return;
        el.disabled = true;
        el.title = '🔒 View-only access — contact an admin';
        el.style.opacity = '0.45';
        el.style.cursor = 'not-allowed';
    });

    // Upload area
    if (uploadArea) {
        uploadArea.style.opacity = '0.45';
        uploadArea.style.cursor = 'not-allowed';
        uploadArea.title = '🔒 View-only access';
        uploadArea.setAttribute('tabindex', '-1');
    }
    const reportUploadAreaEl = $('reportUploadArea');
    if (reportUploadAreaEl) {
        reportUploadAreaEl.style.opacity = '0.45';
        reportUploadAreaEl.style.cursor = 'not-allowed';
        reportUploadAreaEl.title = '🔒 View-only access';
        reportUploadAreaEl.setAttribute('tabindex', '-1');
    }
}

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
    setTheme(currentTheme);
    createParticles();
    observeCards();
    applyViewerRestrictions();
    await initPatientSelector();
    // No startup toast — status is shown via the nav status dot
});
