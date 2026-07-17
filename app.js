import { Client, handle_file } from "https://esm.sh/@gradio/client";

// State
let selectedFile = null;
let targetFile = null;
let selectedSpeakerCount = 'Auto-detect';

// DOM Elements
const views = {
    upload: document.getElementById('upload-view'),
    processing: document.getElementById('processing-view'),
    results: document.getElementById('results-view')
};

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const filenameDisplay = document.getElementById('filename-display');
const btnRemove = document.getElementById('btn-remove');

const targetDropZone = document.getElementById('target-drop-zone');
const targetFileInput = document.getElementById('target-file-input');
const targetFileInfo = document.getElementById('target-file-info');
const targetFilenameDisplay = document.getElementById('target-filename-display');
const btnRemoveTarget = document.getElementById('btn-remove-target');
const btnSeparate = document.getElementById('btn-separate');
const speakerBtns = document.querySelectorAll('.pill-btn');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Setup File Upload Listeners
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileSelection(e.target.files[0], 'main');
        }
    });

    // Remove file
    btnRemove.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.classList.remove('visible');
        dropZone.style.display = 'block';
        btnSeparate.disabled = true;
    });

    // Target File Listeners
    targetDropZone.addEventListener('click', () => targetFileInput.click());
    targetDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        targetDropZone.style.backgroundColor = 'rgba(6, 182, 212, 0.1)';
    });
    targetDropZone.addEventListener('dragleave', () => targetDropZone.style.backgroundColor = 'transparent');
    targetDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        targetDropZone.style.backgroundColor = 'transparent';
        if (e.dataTransfer.files.length) handleFileSelection(e.dataTransfer.files[0], 'target');
    });
    targetFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileSelection(e.target.files[0], 'target');
    });
    btnRemoveTarget.addEventListener('click', () => {
        targetFile = null;
        targetFileInput.value = '';
        targetFileInfo.style.display = 'none';
        targetDropZone.style.display = 'block';
    });

    // Speaker Selection
    speakerBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            speakerBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            let val = btn.dataset.val;
            selectedSpeakerCount = val === 'auto' ? 'Auto-detect' : val;
        });
    });

    // Separate Button
    btnSeparate.addEventListener('click', startSeparation);

    // Reset Button
    document.getElementById('btn-reset').addEventListener('click', resetApp);
});

// Helper: Show View
function showView(viewName) {
    Object.values(views).forEach(v => {
        v.classList.remove('active');
        setTimeout(() => { if(!v.classList.contains('active')) v.style.display = 'none'; }, 400);
    });
    
    setTimeout(() => {
        views[viewName].style.display = 'block';
        // force reflow
        void views[viewName].offsetWidth;
        views[viewName].classList.add('active');
    }, 400);
}

// Helper: Toast
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function handleFileSelection(file, type = 'main') {
    // Validate type (basic)
    if (!file.type.startsWith('audio/') && !file.name.match(/\.(wav|mp3|flac|ogg|m4a|wma|aac)$/i)) {
        showToast('Please select a valid audio file.', 'error');
        return;
    }
    
    // Validate size (approx limit for 5 min depends on bitrate, cap at 50MB)
    if (file.size > 50 * 1024 * 1024) {
        showToast('File is too large (max 50MB).', 'error');
        return;
    }

    if (type === 'main') {
        selectedFile = file;
        filenameDisplay.textContent = file.name;
        dropZone.style.display = 'none';
        fileInfo.classList.add('visible');
        btnSeparate.disabled = false;
    } else {
        targetFile = file;
        targetFilenameDisplay.textContent = file.name;
        targetDropZone.style.display = 'none';
        targetFileInfo.style.display = 'flex';
    }
}

async function startSeparation() {
    if (!selectedFile) return;

    showView('processing');
    const titleEl = document.getElementById('status-title');
    const descEl = document.getElementById('status-desc');

    titleEl.textContent = 'Connecting...';
    descEl.textContent = 'Connecting to Hugging Face ZeroGPU';

    try {
        const client = await Client.connect("shauryasuyal/HearUsOut_AIMS", { 
    hf_token: "hf token here" 
});
        
        titleEl.textContent = 'Separating Voices...';
        descEl.textContent = 'AI is analyzing and separating the audio mixture (this may take a few moments on ZeroGPU)';

        // Connect explicitly to the /predict named endpoint we defined in gr.Blocks
        const result = await client.predict("/predict", [
            handle_file(selectedFile), 
            selectedSpeakerCount,
            targetFile ? handle_file(targetFile) : null
        ]);

        const data = result.data;
        // Output from Gradio is [msg, spk1, spk2, spk3, spk4, spk5]
        const statusMsg = data[0];
        
        if (statusMsg && statusMsg.startsWith("ERROR")) {
            throw new Error(statusMsg);
        }

        // Filter out null/undefined audio files
        const audioFiles = [];
        for (let i = 1; i <= 5; i++) {
            if (data[i] && data[i].url) {
                audioFiles.push({
                    index: i,
                    url: data[i].url,
                    filename: `speaker_${i}.wav`
                });
            }
        }

        document.getElementById('res-badge').textContent = `✨ ${statusMsg}`;
        document.getElementById('res-model').textContent = `Engine: SepFormer ZeroGPU`;
        
        document.getElementById('btn-download-all').style.display = 'none'; // ZIP download not easily supported client-side without JS zip libraries

        renderSpeakers(audioFiles);
        showView('results');
        showToast('Separation complete!', 'success');

    } catch (err) {
        console.error(err);
        showToast(err.message, 'error');
        resetApp();
    }
}

function renderSpeakers(speakers) {
    const grid = document.getElementById('speakers-grid');
    grid.innerHTML = '';
    
    speakers.forEach((spk, i) => {
        // Speaker colors map 1->5, loop after
        const colorIdx = ((spk.index - 1) % 5) + 1; 
        
        const card = document.createElement('div');
        card.className = 'speaker-card';
        card.style.animation = `slideIn 0.5s ease forwards ${i * 0.1}s`;
        card.style.opacity = '0';
        
        // Setup card HTML structure
        card.innerHTML = `
            <div class="speaker-top-bar spk-${colorIdx}"></div>
            <div class="speaker-content">
                <div class="speaker-header">
                    <div class="speaker-title">
                        <div class="speaker-avatar spk-${colorIdx}">${spk.index}</div>
                        <span>Speaker ${spk.index}</span>
                    </div>
                </div>
                <canvas id="canvas-${spk.index}" class="waveform-canvas"></canvas>
                <audio controls src="${spk.url}" crossorigin="anonymous"></audio>
                <a href="${spk.url}" download="${spk.filename}" class="download-btn">Download Track</a>
            </div>
        `;
        
        grid.appendChild(card);
        
        // Draw mock waveform for visuals since real decoding in browser takes time
        drawMockWaveform(`canvas-${spk.index}`, colorIdx);
    });
}

function drawMockWaveform(canvasId, colorIdx) {
    const canvas = document.getElementById(canvasId);
    if(!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Get CSS color variable value
    const tempDiv = document.createElement('div');
    tempDiv.className = `spk-${colorIdx}`;
    document.body.appendChild(tempDiv);
    const color = window.getComputedStyle(tempDiv).color;
    document.body.removeChild(tempDiv);
    
    ctx.fillStyle = color;
    
    const bars = 60;
    const barWidth = 3;
    const gap = (canvas.width - (bars * barWidth)) / bars;
    
    for (let i = 0; i < bars; i++) {
        // Generate an elegant, smooth sinusoidal-like envelope
        let envelope = Math.sin(Math.PI * (i / bars));
        let noise = 0.3 + (Math.random() * 0.7);
        let height = (canvas.height * 0.6) * envelope * noise;
        
        // Ensure a minimum height
        height = Math.max(height, 4);
        
        const y = (canvas.height - height) / 2;
        const x = i * (barWidth + gap);
        
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, height, barWidth / 2);
        ctx.fill();
    }
}

function resetApp() {
    selectedFile = null;
    targetFile = null;
    
    fileInput.value = '';
    targetFileInput.value = '';
    fileInfo.classList.remove('visible');
    targetFileInfo.style.display = 'none';
    dropZone.style.display = 'block';
    targetDropZone.style.display = 'block';
    btnSeparate.disabled = true;
    
    showView('upload');
}
