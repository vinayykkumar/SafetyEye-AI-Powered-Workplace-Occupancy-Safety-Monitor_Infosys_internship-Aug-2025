// Global variables
let currentViolationData = null;
let currentPage = 1;
const itemsPerPage = 10;
let currentViolations = [];
let statsUpdateInterval;
let violationsUpdateInterval;

// Initialize dashboard when page loads
// Initialize dashboard only if dashboard elements exist
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing SafetyEye Dashboard...');

    const isDashboard = document.getElementById('video-upload');
    const isLogsPage = document.getElementById('apply-filter');
    const isSettingsPage = document.getElementById('email-settings-form');

    if (isDashboard || isLogsPage || isSettingsPage) {
        setupEventListeners();
        console.log('✅ Event listeners attached for this page.');

        if (isDashboard) {
            initializeDashboard();
            startAutoUpdates();
        }
    } else {
        console.log('🟡 No dashboard, logs, or settings elements found on this page.');
    }
});



// Initialize dashboard components
function initializeDashboard() {
    console.log('🚀 Initializing SafetyEye Dashboard...');
    
    // Update stats immediately
    updateStats();
    updateViolations();
    
    // Initialize charts
    initializeCharts();

    // Set current date in filters (only if they exist)
    const today = new Date().toISOString().split('T')[0];
    const startInput = document.getElementById('start-date');
    const endInput = document.getElementById('end-date');
    if (startInput && endInput) {
        startInput.value = today;
        endInput.value = today;
    }

}

// Setup all event listeners
function setupEventListeners() {
    // Dashboard controls
    document.getElementById('start-webcam').addEventListener('click', startWebcam);
    document.getElementById('stop-processing').addEventListener('click', stopProcessing);
    document.getElementById('upload-trigger').addEventListener('click', triggerVideoUpload);
    document.getElementById('video-upload').addEventListener('change', handleVideoUpload);
    
    // Modal controls
    setupModalControls();
    
    // Logs page controls
    console.log("🔍 Checking for log page buttons...");

    if (document.getElementById('apply-filter')) {
        console.log("✅ Found log page buttons, attaching listeners...");

        document.getElementById('apply-filter').addEventListener('click', applyDateFilter);
        document.getElementById('reset-filter').addEventListener('click', resetFilters);
        document.getElementById('download-csv').addEventListener('click', downloadCSV);
        document.getElementById('refresh-table').addEventListener('click', refreshTable);
        document.getElementById('prev-page').addEventListener('click', previousPage);
        document.getElementById('next-page').addEventListener('click', nextPage);
        document.getElementById('type-filter').addEventListener('change', applyTypeFilter);
    }
    
    // Settings page controls
    if (document.getElementById('email-settings-form')) {
        document.getElementById('email-settings-form').addEventListener('submit', saveEmailSettings);
        document.getElementById('test-email').addEventListener('click', openTestEmailModal);
        document.getElementById('confirm-test-email').addEventListener('click', sendTestEmail);
        document.getElementById('save-system-settings').addEventListener('click', saveSystemSettings);
        document.getElementById('clear-violations').addEventListener('click', confirmClearViolations);
        document.getElementById('reset-system').addEventListener('click', confirmResetSystem);
        document.getElementById('confirm-action').addEventListener('click', handleConfirmAction);
    }
    
    // Email buttons
    const emailButtons = document.querySelectorAll('#send-email-btn, #log-send-email');
    emailButtons.forEach(btn => {
        if (btn) {
            btn.addEventListener('click', sendViolationEmail);
        }
    });
}

// Setup modal controls
function setupModalControls() {
    // Close modals when clicking X
    document.querySelectorAll('.close-modal').forEach(closeBtn => {
        closeBtn.addEventListener('click', function() {
            this.closest('.modal').style.display = 'none';
        });
    });
    
    // Close modals when clicking outside
    window.addEventListener('click', function(event) {
        document.querySelectorAll('.modal').forEach(modal => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
    
    // Escape key to close modals
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            document.querySelectorAll('.modal').forEach(modal => {
                modal.style.display = 'none';
            });
        }
    });
}

// Start auto-updates
function startAutoUpdates() {
    // Update stats every 5 seconds
    statsUpdateInterval = setInterval(updateStats, 5000);
    
    // Update violations every 3 seconds
    violationsUpdateInterval = setInterval(updateViolations, 3000);
}

// Update dashboard statistics
async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        // Update stat displays
        document.getElementById('violation-count').textContent = stats.violation_count_today;
        document.getElementById('compliance-rate').textContent = stats.compliance_rate + '%';
        document.getElementById('current-occupancy').textContent = stats.current_occupancy;
        
        // Update detection mode
        const modeElement = document.getElementById('mode-text');
        if (modeElement) {
            modeElement.textContent = stats.detection_mode;
        }
        
        // Update processing status
        updateProcessingStatus(stats.processing_status);
        
        // Update charts
        updateCharts(stats);
        
    } catch (error) {
        console.error('Error updating stats:', error);
    }
}

// Update processing status display
function updateProcessingStatus(status) {
    const statusElement = document.getElementById('processing-status');
    const stopButton = document.getElementById('stop-processing');
    const videoFeed = document.getElementById('video-feed');
    const noFeed = document.getElementById('no-feed');
    
    if (status === 'active') {
        statusElement.textContent = 'LIVE';
        statusElement.className = 'status-indicator status-online';
        stopButton.disabled = false;
        if (videoFeed) videoFeed.style.display = 'block';
        if (noFeed) noFeed.style.display = 'none';
    } else {
        statusElement.textContent = 'OFFLINE';
        statusElement.className = 'status-indicator status-offline';
        stopButton.disabled = true;
        if (noFeed) noFeed.style.display = 'flex';
        if (videoFeed) videoFeed.style.display = 'none';
    }
}

// Update violations list
async function updateViolations() {
    try {
        const response = await fetch('/api/violations');
        const violations = await response.json();
        
        updateViolationsDisplay(violations);
        
    } catch (error) {
        console.error('Error updating violations:', error);
    }
}

// Update violations display
function updateViolationsDisplay(violations) {
    const violationsList = document.getElementById('violations-list');
    const countBadge = document.getElementById('recent-violations-count');
    
    if (!violationsList) return;
    
    // Update count badge
    if (countBadge) {
        countBadge.textContent = violations.length;
    }
    
    // Clear current list
    violationsList.innerHTML = '';
    
    if (violations.length === 0) {
        violationsList.innerHTML = `
            <div class="no-violations">
                <i class="fas fa-check-circle"></i>
                <h4>No Violations Detected</h4>
                <p>All safety protocols are being followed</p>
            </div>
        `;
        return;
    }
    
    // Add violation cards
    violations.forEach(violation => {
        const violationCard = createViolationCard(violation);
        violationsList.appendChild(violationCard);
    });
}

// Create violation card element
function createViolationCard(violation) {
    const card = document.createElement('div');
    card.className = 'violation-card';
    
    const badgeClass = violation.violation_type.includes('HELMET') ? 'badge-danger' : 'badge-warning';
    const icon = violation.violation_type.includes('HELMET') ? 'fa-hard-hat' : 'fa-vest';
    
    card.innerHTML = `
        <div class="violation-header">
            <span class="violation-badge ${badgeClass}">
                <i class="fas ${icon}"></i> ${violation.violation_type}
            </span>
            <span class="violation-time">${formatTime(violation.timestamp)}</span>
        </div>
        <div class="violation-details">
            <div class="detail-item">
                <span class="detail-label">Person ID:</span>
                <span class="detail-value">${violation.person_id}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Frame:</span>
                <span class="detail-value">${violation.frame}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Helmet Conf:</span>
                <span class="detail-value">${violation.helmet_conf.toFixed(2)}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Vest Conf:</span>
                <span class="detail-value">${violation.vest_conf.toFixed(2)}</span>
            </div>
        </div>
        <div class="violation-actions">
            <button class="btn btn-secondary btn-sm view-details" data-violation='${JSON.stringify(violation)}'>
                <i class="fas fa-eye"></i> View Details
            </button>
        </div>
    `;
    
    // Add click event for view details
    card.querySelector('.view-details').addEventListener('click', function() {
        const violationData = JSON.parse(this.getAttribute('data-violation'));
        showViolationDetails(violationData);
    });
    
    return card;
}

// Show violation details in modal
function showViolationDetails(violation) {
    currentViolationData = violation;
    
    // Update modal content
    document.getElementById('detail-type').textContent = violation.violation_type;
    document.getElementById('detail-timestamp').textContent = violation.timestamp;
    document.getElementById('detail-person-id').textContent = violation.person_id;
    document.getElementById('detail-frame').textContent = violation.frame;
    document.getElementById('detail-helmet-conf').textContent = violation.helmet_conf.toFixed(3);
    document.getElementById('detail-vest-conf').textContent = violation.vest_conf.toFixed(3);
    
    // Update image
    const imageElement = document.getElementById('detail-image');
    if (violation.image_path && violation.image_path.trim() !== '') {
        imageElement.src = violation.image_path;
        imageElement.style.display = 'block';
    } else {
        imageElement.style.display = 'none';
    }
    
    // Show modal
    document.getElementById('violation-modal').style.display = 'block';
}

// Start webcam processing
async function startWebcam() {
    try {
        const response = await fetch('/api/start_webcam');
        const result = await response.json();
        
        if (result.status === 'started') {
            showNotification('Webcam started successfully', 'success');
            updateProcessingStatus('active');
        } else {
            showNotification('Processing already running', 'warning');
        }
    } catch (error) {
        console.error('Error starting webcam:', error);
        showNotification('Failed to start webcam', 'error');
    }
}

// Stop video processing
async function stopProcessing() {
    try {
        const response = await fetch('/api/stop_processing');
        const result = await response.json();
        
        if (result.status === 'stopped') {
            showNotification('Processing stopped', 'success');
            updateProcessingStatus('inactive');
        }
    } catch (error) {
        console.error('Error stopping processing:', error);
        showNotification('Failed to stop processing', 'error');
    }
}

// Trigger video upload
function triggerVideoUpload() {
    document.getElementById('video-upload').click();
}

// Handle video upload
async function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('video', file);
    
    try {
        const response = await fetch('/api/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.status === 'started') {
            showNotification('Video upload and processing started', 'success');
            updateProcessingStatus('active');
        } else {
            showNotification('Processing already running', 'warning');
        }
    } catch (error) {
        console.error('Error uploading video:', error);
        showNotification('Failed to upload video', 'error');
    }
    
    // Reset file input
    event.target.value = '';
}

// Send violation email
async function sendViolationEmail() {
    if (!currentViolationData) {
        showNotification('No violation data available', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/violations/send-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(currentViolationData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Email sent successfully', 'success');
            // Close modal
            document.querySelector('.modal').style.display = 'none';
        } else {
            showNotification('Failed to send email', 'error');
        }
    } catch (error) {
        console.error('Error sending email:', error);
        showNotification('Error sending email', 'error');
    }
}

// Initialize charts
function initializeCharts() {
    // Violations by type chart
    const violationsCtx = document.getElementById('violationsChart');
    if (violationsCtx) {
        window.violationsChart = new Chart(violationsCtx, {
            type: 'doughnut',
            data: {
                labels: ['No Helmet', 'No Vest', 'Other'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ['#f56565', '#ed8936', '#a0aec0'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    // Compliance trend chart
    const complianceCtx = document.getElementById('complianceChart');
    if (complianceCtx) {
        window.complianceChart = new Chart(complianceCtx, {
            type: 'line',
            data: {
                labels: ['9:00', '10:00', '11:00', '12:00', '13:00', '14:00'],
                datasets: [{
                    label: 'Compliance Rate',
                    data: [95, 92, 88, 85, 90, 93],
                    borderColor: '#48bb78',
                    backgroundColor: 'rgba(72, 187, 120, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 80,
                        max: 100
                    }
                }
            }
        });
    }
}

// Update charts with new data
function updateCharts(stats) {
    if (window.violationsChart) {
        window.violationsChart.data.datasets[0].data = [
            stats.violations_by_type?.no_helmet || 0,
            stats.violations_by_type?.no_vest || 0,
            stats.violations_by_type?.other || 0
        ];
        window.violationsChart.update();
    }
}

// Logs Page Functions
async function applyDateFilter() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;

    if (!startDate || !endDate) {
        showNotification('Please select both start and end dates', 'warning');
        return;
    }

    try {
        console.log("📡 Fetching violations from:", `/api/violations/date-range?start=${startDate}&end=${endDate}`);
        const response = await fetch(`/api/violations/date-range?start=${startDate}&end=${endDate}`);
        const violations = await response.json();
        console.log("✅ Received violations:", violations);

        currentViolations = violations;
        currentPage = 1;

        if (violations.length === 0) {
            showNotification('No violations found for this range', 'warning');
        } else {
            showNotification(`Loaded ${violations.length} violations`, 'success');
        }

        updateViolationsTable();
        updateSummaryStats(violations);

        // Enable/disable CSV button
        document.getElementById('download-csv').disabled = violations.length === 0;

    } catch (error) {
        console.error('❌ Error applying date filter:', error);
        showNotification('Error loading violations', 'error');
    }
}



function resetFilters() {
    document.getElementById('start-date').value = '';
    document.getElementById('end-date').value = '';
    document.getElementById('type-filter').value = 'all';
    
    currentViolations = [];
    currentPage = 1;
    updateViolationsTable();
    updateSummaryStats([]);
    
    document.getElementById('download-csv').disabled = true;
    showNotification('Filters reset — showing all violations', 'info');
}

function updateViolationsTable() {
    const tableBody = document.getElementById('violations-table-body');
    if (!tableBody) return;

    console.log("🧾 Updating table with violations:", currentViolations);

    tableBody.innerHTML = '';

    if (!currentViolations || currentViolations.length === 0) {
        tableBody.innerHTML = `
            <tr>
                <td colspan="8" class="no-data">No violations found for selected date range</td>
            </tr>
        `;
        updatePagination();
        return;
    }

    // Filter by type if selected
    const typeFilter = document.getElementById('type-filter').value;
    let filteredViolations = currentViolations;

    if (typeFilter === 'no_hardhat') {
        filteredViolations = currentViolations.filter(v => v.violation_type.toLowerCase().includes('no_helmet'));
    } else if (typeFilter === 'no_vest') {
        filteredViolations = currentViolations.filter(v => v.violation_type.toLowerCase().includes('no_vest'));
    } else if (typeFilter === 'both') {
        filteredViolations = currentViolations.filter(v =>
            v.violation_type.toLowerCase().includes('no_helmet') &&
            v.violation_type.toLowerCase().includes('no_vest')
        );
    }

    // Pagination
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const pageViolations = filteredViolations.slice(startIndex, endIndex);

    // Build table rows
    pageViolations.forEach(violation => {
        const row = document.createElement('tr');
        const badgeClass = violation.violation_type.includes('helmet') ? 'badge-danger' : 'badge-warning';
        row.innerHTML = `
            <td>${violation.id}</td>
            <td>${violation.timestamp}</td>
            <td><span class="violation-badge ${badgeClass}">${violation.violation_type}</span></td>
            <td>${violation.person_id}</td>
            <td>${violation.frame}</td>
            <td>${violation.helmet_conf.toFixed(2)}</td>
            <td>${violation.vest_conf.toFixed(2)}</td>
            <td>
                <button class="btn btn-secondary btn-sm view-log-details" data-violation='${JSON.stringify(violation)}'>
                    <i class="fas fa-eye"></i> Details
                </button>
            </td>
        `;
        row.querySelector('.view-log-details').addEventListener('click', function () {
            const vData = JSON.parse(this.getAttribute('data-violation'));
            showLogDetails(vData);
        });
        tableBody.appendChild(row);
    });

    updatePagination(filteredViolations.length);
}


function updatePagination(totalItems = 0) {
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    const currentPageElement = document.getElementById('current-page');
    const totalPagesElement = document.getElementById('total-pages');
    const prevButton = document.getElementById('prev-page');
    const nextButton = document.getElementById('next-page');
    
    if (currentPageElement) currentPageElement.textContent = currentPage;
    if (totalPagesElement) totalPagesElement.textContent = totalPages;
    if (prevButton) prevButton.disabled = currentPage === 1;
    if (nextButton) nextButton.disabled = currentPage === totalPages || totalPages === 0;
}

function previousPage() {
    if (currentPage > 1) {
        currentPage--;
        updateViolationsTable();
    }
}

function nextPage() {
    const totalPages = Math.ceil(currentViolations.length / itemsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        updateViolationsTable();
    }
}

function applyTypeFilter() {
    updateViolationsTable();
}

function updateSummaryStats(violations) {
    const totalViolations = violations.length;
    const noHelmetCount = violations.filter(v => v.violation_type.includes('HELMET')).length;
    const noVestCount = violations.filter(v => v.violation_type.includes('VEST')).length;
    
    document.getElementById('total-violations').textContent = totalViolations;
    document.getElementById('no-helmet-count').textContent = noHelmetCount;
    document.getElementById('no-vest-count').textContent = noVestCount;
}

function showLogDetails(violation) {
    currentViolationData = violation;
    
    // Update modal content
    document.getElementById('log-id').textContent = violation.id;
    document.getElementById('log-timestamp').textContent = violation.timestamp;
    document.getElementById('log-type').textContent = violation.violation_type;
    document.getElementById('log-person-id').textContent = violation.person_id;
    document.getElementById('log-frame').textContent = violation.frame;
    document.getElementById('log-helmet-conf').textContent = violation.helmet_conf.toFixed(3);
    document.getElementById('log-vest-conf').textContent = violation.vest_conf.toFixed(3);
    document.getElementById('log-image-path').textContent = violation.image_path || 'Not available';
    
    // Update image
    const imageElement = document.getElementById('log-image');
    const noImageElement = document.getElementById('no-image');
    
    if (violation.image_path && violation.image_path.trim() !== '') {
        imageElement.src = violation.image_path;
        imageElement.style.display = 'block';
        noImageElement.style.display = 'none';
    } else {
        imageElement.style.display = 'none';
        noImageElement.style.display = 'block';
    }
    
    // Show modal
    document.getElementById('log-modal').style.display = 'block';
}

async function downloadCSV() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    if (!startDate || !endDate) {
        showNotification('Please select date range first', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/violations/download-csv?start=${startDate}&end=${endDate}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `violations_${startDate}_to_${endDate}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showNotification('CSV download started', 'success');
        } else {
            showNotification('Failed to download CSV', 'error');
        }
    } catch (error) {
        console.error('Error downloading CSV:', error);
        showNotification('Error downloading CSV', 'error');
    }
}

function refreshTable() {
    applyDateFilter();
}

// Settings Page Functions
async function saveEmailSettings(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const settings = Object.fromEntries(formData);
    
    try {
        const response = await fetch('/api/settings/email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(settings)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Email settings saved successfully', 'success');
        } else {
            showNotification('Failed to save email settings', 'error');
        }
    } catch (error) {
        console.error('Error saving email settings:', error);
        showNotification('Error saving email settings', 'error');
    }
}

function openTestEmailModal() {
    document.getElementById('test-email-modal').style.display = 'block';
}

async function sendTestEmail() {
    const recipient = document.getElementById('test-recipient').value;
    
    if (!recipient) {
        showNotification('Please enter recipient email', 'warning');
        return;
    }
    
    try {
        // Create test violation data
        const testViolation = {
            violation_type: 'TEST_VIOLATION',
            timestamp: new Date().toISOString(),
            person_id: 'TEST',
            frame: '0',
            helmet_conf: 0.0,
            vest_conf: 0.0
        };
        
        const response = await fetch('/api/violations/send-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(testViolation)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Test email sent successfully', 'success');
            document.getElementById('test-email-modal').style.display = 'none';
        } else {
            showNotification('Failed to send test email', 'error');
        }
    } catch (error) {
        console.error('Error sending test email:', error);
        showNotification('Error sending test email', 'error');
    }
}

async function saveSystemSettings() {
    const settings = {
        report_frequency: document.getElementById('report-frequency').value,
        violation_threshold: document.getElementById('violation-threshold').value,
        confidence_threshold: document.getElementById('confidence-threshold').value
    };
    
    // In a real implementation, you would send this to the backend
    showNotification('System settings saved successfully', 'success');
}

function confirmClearViolations() {
    showConfirmationModal(
        'Clear All Violations',
        'Are you sure you want to clear all violation records? This action cannot be undone.',
        'clearViolations'
    );
}

function confirmResetSystem() {
    showConfirmationModal(
        'Reset System',
        'Are you sure you want to reset the system? This will clear all data and restore default settings.',
        'resetSystem'
    );
}

function showConfirmationModal(title, message, action) {
    document.getElementById('confirmation-title').textContent = title;
    document.getElementById('confirmation-message').textContent = message;
    document.getElementById('confirm-action').setAttribute('data-action', action);
    document.getElementById('confirmation-modal').style.display = 'block';
}

function handleConfirmAction() {
    const action = this.getAttribute('data-action');
    
    switch (action) {
        case 'clearViolations':
            clearViolations();
            break;
        case 'resetSystem':
            resetSystem();
            break;
    }
    
    document.getElementById('confirmation-modal').style.display = 'none';
}

async function clearViolations() {
    // In a real implementation, you would call a backend API
    showNotification('Violations cleared successfully', 'success');
}

async function resetSystem() {
    // In a real implementation, you would call a backend API
    showNotification('System reset successfully', 'success');
}

// Utility Functions
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString() + ' ' + date.toLocaleDateString();
}

function updateSystemUptime() {
    const uptimeElement = document.getElementById('system-uptime');
    if (uptimeElement) {
        // This would normally come from the backend
        const startTime = new Date();
        const now = new Date();
        const diff = now - startTime;
        
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        
        uptimeElement.textContent = `${days}d ${hours}h ${minutes}m`;
    }
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close">&times;</button>
    `;
    
    // Add styles if not already added
    if (!document.querySelector('#notification-styles')) {
        const styles = document.createElement('style');
        styles.id = 'notification-styles';
        styles.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 15px 20px;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                border-left: 4px solid #667eea;
                z-index: 10000;
                display: flex;
                align-items: center;
                gap: 12px;
                max-width: 400px;
                animation: slideInRight 0.3s ease;
            }
            
            .notification-success {
                border-left-color: #48bb78;
            }
            
            .notification-error {
                border-left-color: #f56565;
            }
            
            .notification-warning {
                border-left-color: #ed8936;
            }
            
            .notification-content {
                display: flex;
                align-items: center;
                gap: 8px;
                flex: 1;
            }
            
            .notification-close {
                background: none;
                border: none;
                font-size: 1.2rem;
                cursor: pointer;
                color: #a0aec0;
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(styles);
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
    
    // Close on click
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.style.animation = 'slideInRight 0.3s ease reverse';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    });
}

function getNotificationIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (statsUpdateInterval) clearInterval(statsUpdateInterval);
    if (violationsUpdateInterval) clearInterval(violationsUpdateInterval);
});

console.log('✅ SafetyEye Dashboard JavaScript loaded successfully!');