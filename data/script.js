const STALE_TIMEOUT_MS = 2000; // 2 seconds threshold to mark as stale
const POLL_INTERVAL_HZ = 500;  // 2Hz = 500ms
let pollingInterval = null;
let currentDir = null;
let deviceStates = {}; // Stores temp_val, rssi, last_update_ts per device

// Elements
const dirListEl = document.getElementById('dir-list');
const deviceGridEl = document.getElementById('device-grid');
const selectedInfoEl = document.getElementById('selected-info');

// Parse simple CSV string into an array of objects
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    if (lines.length <= 1) return []; // Only header
    
    // header: ts,device,seq,temp_val,temp_scale,hum_val,hum_scale,press_val,press_scale,rssi
    const headers = lines[0].trim().split(',');
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        const values = line.split(',');
        const row = {};
        for(let j = 0; j < headers.length; j++) {
            row[headers[j]] = values[j];
        }
        data.push(row);
    }
    return data;
}

// Render the UI state based on deviceStates
function renderGrid() {
    deviceGridEl.innerHTML = '';
    const now = Date.now();
    let count = 0;
    
    for (const [deviceId, state] of Object.entries(deviceStates)) {
        count++;
        const timeSinceUpdate = now - state.last_update_ts;
        const isStale = timeSinceUpdate > STALE_TIMEOUT_MS;
        
        const card = document.createElement('div');
        card.className = `card ${isStale ? 'stale' : ''}`;
        
        card.innerHTML = `
            <div class="device-id">
                <span>ID: ${deviceId}</span>
                <div class="status-dot" title="${isStale ? 'Stale' : 'Active'}"></div>
            </div>
            
            <div class="metric">
                <span class="metric-label">Timestamp</span>
                <span class="metric-value temp-value" style="font-size: 1rem;">${state.ts}</span>
            </div>
            
            <div class="metric" style="border-bottom: none; margin-bottom: 0; padding-bottom: 0;">
                <span class="metric-label">RSSI</span>
                <span class="metric-value rssi-value">${state.rssi} dBm</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); text-align: right; margin-top: 0.5rem">
                ${isStale ? `Stale (${(timeSinceUpdate / 1000).toFixed(1)}s ago)` : 'Live updates'}
            </div>
        `;
        
        deviceGridEl.appendChild(card);
    }

    if (count === 0) {
        deviceGridEl.innerHTML = `<div class="loading">No devices found in this file yet.</div>`;
    }
}

// Fetches and parses the CSV periodically
async function pollCSV() {
    if (!currentDir) return;
    
    try {
        const response = await fetch(`./${currentDir}rx.csv`);
        if (!response.ok) {
            console.warn(`rx.csv not found in ${currentDir}`);
            return;
        }
        
        const csvText = await response.text();
        const rows = parseCSV(csvText);
        
        const now = Date.now();
        const latestByDevice = {};
        
        // Find the absolute latest row for each device in the entire CSV
        rows.forEach(row => {
            if (row.device) {
                latestByDevice[row.device] = row;
            }
        });
        
        // Assess updates
        for (const [deviceId, row] of Object.entries(latestByDevice)) {
            if (!deviceStates[deviceId]) {
                // Initialize device
                deviceStates[deviceId] = {
                    last_update_ts: now,
                    rssi: row.rssi || '-∞',
                    ts: row.ts
                };
            } else {
                const state = deviceStates[deviceId];
                
                // Always update the RSSI and ts representation in the state so it renders correctly
                state.rssi = row.rssi || '-∞';
                
                // Determine if we actually received a new reading based on ts
                if (state.ts !== row.ts) {
                    state.ts = row.ts;
                    state.last_update_ts = now;
                }
            }
        }
        
        renderGrid();
        
    } catch(e) {
        console.error("Error polling CSV:", e);
    }
}

// Selects a directory and starts the polling mechanism
function selectDirectory(dir) {
    if (currentDir === dir) return;
    
    currentDir = dir;
    deviceStates = {}; // Reset state for new directory
    selectedInfoEl.innerHTML = `Monitoring: <strong>${dir}rx.csv</strong>`;
    
    // Update sidebar UI
    const items = dirListEl.querySelectorAll('.dir-item');
    items.forEach(el => {
        if (el.dataset.dir === dir) el.classList.add('active');
        else el.classList.remove('active');
    });

    // Reset grid
    deviceGridEl.innerHTML = `<div class="loading">Waiting for data...</div>`;

    // Reset polling
    if (pollingInterval) clearInterval(pollingInterval);
    pollCSV(); // Immediate fetch
    pollingInterval = setInterval(pollCSV, POLL_INTERVAL_HZ);
}

// Initialize directory listing
async function initSidebar() {
    try {
        const res = await fetch('./');
        const text = await res.text();
        
        // Parse directory links (usually http or ftp servers like python -m http.server)
        const regex = /<a[^>]*href=["']?([^"'>]+\/)["']?[^>]*>/ig;
        let match;
        const dirs = [];
        
        while ((match = regex.exec(text)) !== null) {
            let dir = match[1];
            if (dir.startsWith('./')) dir = dir.substring(2);
            // Example dir: 20260304_221540/
            if (dir.match(/^20\d{6}_\d{6}\/$/) || dir.match(/^20\d/)) {
                dirs.push(dir);
            }
        }
        
        // Ensure unique and sorted descending
        const uniqueDirs = [...new Set(dirs)].sort().reverse();
        dirListEl.innerHTML = '';
        
        if (uniqueDirs.length === 0) {
            dirListEl.innerHTML = `<li class="loading">No valid timestamped directories found.</li>`;
            return;
        }

        uniqueDirs.forEach(dir => {
            const li = document.createElement('li');
            li.className = 'dir-item';
            li.textContent = dir;
            li.dataset.dir = dir;
            li.onclick = () => selectDirectory(dir);
            dirListEl.appendChild(li);
        });
        
    } catch(err) {
        console.error("Fetch dir failed", err);
        dirListEl.innerHTML = `<li class="loading">Could not load directories. Serve via HTTP.</li>`;
    }
}

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
});
