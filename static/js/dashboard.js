// Dashboard Configuration
const PERIODS = ['90', '120', '180', '365'];

// Initialize dashboard with data from window.dashboardData
function initializeDashboard() {
    // Check if dashboard data is properly initialized
    if (!window.dashboardData || !window.dashboardData.locations) {
        console.error('Dashboard data not properly initialized');
        return;
    }

    try {
        // Render initial HTML structure
        renderDashboardStructure();

        // Initialize components
        const gauge = initializeGauge();
        gauge.set(window.dashboardData.periods['90'].mood || 0);

        const trendChart = initializeTrendChart(
            PERIODS.map(period => window.dashboardData.periods[period].mood || 0)
        );

        // Initialize Word Clouds
        initializeWordClouds();

        // Store references for later updates
        window.dashboardState = {
            gauge,
            trendChart
        };
    } catch (error) {
        console.error('Error initializing dashboard:', error);
    }
}

function renderDashboardStructure() {
    const root = document.getElementById('dashboard-root');
    if (!root) {
        console.error('Dashboard root element not found');
        return;
    }

    root.innerHTML = `
        <div class="container mt-5">
            <h1 class="mb-4">Review Sentiment Dashboard</h1>

            <!-- Location Filter -->
            <div class="row mb-4">
                <div class="col-md-4">
                    <label class="filter-label" for="location-filter">Select Property:</label>
                    <select id="location-filter" class="form-select">
                        <option value="">All Properties</option>
                        ${(window.dashboardData.locations || []).map(location => 
                            `<option value="${location}">${location}</option>`
                        ).join('')}
                    </select>
                </div>
            </div>

            <!-- Dashboard Content -->
            <div id="dashboard-content">
                <!-- Gauge and Trend -->
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Current Sentiment (90 Days)</h5>
                                <div class="gauge-container">
                                    <canvas id="gauge-90"></canvas>
                                </div>
                                <div class="sentiment-score">${(window.dashboardData.periods['90'].mood || 0).toFixed(1)}%</div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Sentiment Trend</h5>
                                <canvas id="trendLine"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Word Clouds -->
                <div id="wordcloud-container"></div>

                <!-- AI Insights -->
                <div id="summaries-container" class="row mb-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">AI-Generated Insights</h5>
                                <div class="summaries-content">
                                    <div class="accordion" id="summariesAccordion">
                                        ${PERIODS.map(period => `
                                            <div class="accordion-item">
                                                <h2 class="accordion-header">
                                                    <button class="accordion-button collapsed" type="button" 
                                                            data-bs-toggle="collapse" 
                                                            data-bs-target="#summary${period}">
                                                        Last ${period} Days Analysis
                                                    </button>
                                                </h2>
                                                <div id="summary${period}" class="accordion-collapse collapse">
                                                    <div class="accordion-body" id="summaryContent${period}">
                                                        <div class="summary-placeholder">No analysis available for this period.</div>
                                                    </div>
                                                </div>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Top Words Tables -->
                <div class="row mt-4">
                    ${PERIODS.map(period => renderWordTable(period)).join('')}
                </div>
            </div>
        </div>
    `;

    // Add event listener for location filter
    const locationFilter = document.getElementById('location-filter');
    if (locationFilter) {
        locationFilter.addEventListener('change', handleLocationChange);
    }
}
function renderWordTable(period) {
    const periodData = window.dashboardData.periods[period] || {};
    const title = period === '365' ? 'Year' : period + ' Days';

    return `
        <div class="col-md-3">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Top Words (${title})</h5>
                    <div class="table-responsive">
                        <table class="table table-striped" id="topWordsTable_${period}">
                            <thead>
                                <tr>
                                    <th>Word</th>
                                    <th>Count</th>
                                    <th>Sentiment</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${(periodData.words || []).map(([word, count]) => {
                                    const sentiment = (periodData.sentiments || {})[word] || 0;
                                    const sentimentClass = getSentimentClass(sentiment);
                                    return `
                                        <tr>
                                            <td class="clickable" onclick="analyzeWord('${word}')">${word}</td>
                                            <td>${count}</td>
                                            <td>
                                                <span class="sentiment-indicator ${sentimentClass}"></span>
                                                ${sentiment.toFixed(2)}
                                            </td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function initializeGauge() {
    const gaugeEl = document.getElementById('gauge-90');
    const gauge = new Gauge(gaugeEl);
    gauge.setOptions({
        angle: -0.2,
        lineWidth: 0.2,
        radiusScale: 1,
        pointer: {
            length: 0.6,
            strokeWidth: 0.035,
            color: '#000000'
        },
        staticLabels: {
            font: "12px sans-serif",
            labels: [20, 40, 60, 80, 100],
            color: "#000000",
            fractionDigits: 0
        },
        staticZones: [
            { strokeStyle: "#FF4D4D", min: 0, max: 20 },
            { strokeStyle: "#FFA500", min: 20, max: 40 },
            { strokeStyle: "#FFD700", min: 40, max: 60 },
            { strokeStyle: "#9ACD32", min: 60, max: 80 },
            { strokeStyle: "#32CD32", min: 80, max: 100 }
        ],
        limitMax: false,
        limitMin: false,
        highDpiSupport: true
    });
    gauge.maxValue = 100;
    gauge.setMinValue(0);
    gauge.animationSpeed = 32;
    return gauge;
}

function initializeTrendChart(data) {
    const ctx = document.getElementById('trendLine').getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: PERIODS.map(p => p + ' Days'),
            datasets: [{
                label: 'Sentiment Score',
                data: data,
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                tension: 0.4,
                fill: true,
                pointBackgroundColor: '#4CAF50',
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: context => `Sentiment: ${context.raw.toFixed(1)}%`
                    }
                }
            }
        }
    });
}

function initializeWordClouds() {
    const container = document.getElementById('wordcloud-container');
    const template = period => `
        <div class="col-md-${period === '365' ? '8 offset-md-2' : '4'}">
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">Last ${period} Days</h5>
                    <div class="wordcloud-container ${period === '365' ? 'large' : ''}" onclick="promptWord('${period}')">
                        <img id="wordcloud_${period}" 
                             src="/static/${window.dashboardData.periods[period].wordcloud}?t=${window.dashboardData.timestamp}"
                             class="wordcloud-image" 
                             alt="${period} day wordcloud">
                    </div>
                </div>
            </div>
        </div>
    `;

    const regularClouds = document.createElement('div');
    regularClouds.className = 'row mb-4';
    regularClouds.innerHTML = PERIODS.slice(0, 3).map(template).join('');

    const yearCloud = document.createElement('div');
    yearCloud.className = 'row mb-4';
    yearCloud.innerHTML = template('365');

    container.appendChild(regularClouds);
    container.appendChild(yearCloud);
}

async function handleLocationChange() {
    const location = this.value;
    document.querySelector('h1').textContent = `Review Sentiment Dashboard ${location ? '- ' + location : ''}`;

    try {
        const response = await fetch('/filter?location=' + encodeURIComponent(location));
        const data = await response.json();
        window.dashboardData = data;
        updateDashboard(data);
    } catch (err) {
        console.error('Error fetching filter data:', err);
    }
}

function updateDashboard(data) {
    const { gauge, trendChart } = window.dashboardState;

    gauge.set(data.mood_90 || 0);
    document.querySelector('.sentiment-score').textContent = `${(data.mood_90 || 0).toFixed(1)}%`;

    trendChart.data.datasets[0].data = PERIODS.map(period => data['mood_' + period] || 0);
    trendChart.update();

    PERIODS.forEach(period => {
        const img = document.getElementById(`wordcloud_${period}`);
        const tbody = document.querySelector(`#topWordsTable_${period} tbody`);
        const summaryContent = document.getElementById(`summaryContent${period}`);

        if (img && data[`wc_${period}`]) {
            img.src = `/static/${data['wc_' + period]}?t=${Date.now()}`;
        }

        if (tbody && data[`top_words_${period}`]) {
            tbody.innerHTML = data[`top_words_${period}`].map(([word, count]) => {
                const sentiment = (data[`word_sentiments_${period}`] || {})[word] || 0;
                const sentimentClass = getSentimentClass(sentiment);
                return `
                    <tr>
                        <td class="clickable" onclick="analyzeWord('${word}')">${word}</td>
                        <td>${count}</td>
                        <td>
                            <span class="sentiment-indicator ${sentimentClass}"></span>
                            ${sentiment.toFixed(2)}
                        </td>
                    </tr>
                `;
            }).join('');
        }

        if (summaryContent && data[`summary_${period}`]) {
            const summary = data[`summary_${period}`];
            const actualPeriod = summary.actual_period !== parseInt(period)
                ? `<div class="period-note">Showing analysis for ${summary.actual_period} days</div>`
                : '';

            summaryContent.innerHTML = `
                ${actualPeriod}
                <div class="summary-section">
                    <div class="overview-section">
                        <h6>Overall Sentiment</h6>
                        <p class="summary-text">${summary.text || 'No overview available.'}</p>
                    </div>
                    
                    ${summary.trouble_points && summary.trouble_points.length > 0 ? `
                        <div class="trouble-points mt-4">
                            <h6>Areas of Concern</h6>
                            <ul class="concern-list">
                                ${summary.trouble_points.map(point => 
                                    `<li>${point}</li>`
                                ).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    
                    ${summary.suggestions && summary.suggestions.length > 0 ? `
                        <div class="suggestions mt-4">
                            <h6>Recommendations</h6>
                            <ul class="suggestion-list">
                                ${summary.suggestions.map(sugg => 
                                    `<li>${sugg}</li>`
                                ).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }
    });
}

function getSentimentClass(sentiment) {
    if (sentiment > 0) return 'sentiment-positive';
    if (sentiment < 0) return 'sentiment-negative';
    return 'sentiment-neutral';
}

const wordModal = new bootstrap.Modal(document.getElementById('wordAnalysisModal'));

async function analyzeWord(word) {
    try {
        const response = await fetch(`/word_analysis/${encodeURIComponent(word)}`);
        const data = await response.json();
        displayWordAnalysis(data);
        wordModal.show();
    } catch (err) {
        console.error('Error analyzing word:', err);
    }
}

function displayWordAnalysis(data) {
    document.querySelector('.word-title').textContent = `Analysis of "${data.word}"`;
    document.querySelector('.stats-grid').innerHTML = `
        <div class="stat-box">
            <span class="stat-label">Total Appearances</span>
            <span class="stat-value">${data.total_appearances}</span>
        </div>
        <div class="stat-box">
            <span class="stat-label">Positive Contexts</span>
            <span class="stat-value">${data.positive_contexts}</span>
        </div>
        <div class="stat-box">
            <span class="stat-label">Negative Contexts</span>
            <span class="stat-value">${data.negative_contexts}</span>
        </div>
    `;

    document.querySelector('.context-list').innerHTML = data.contexts.map(context => `
        <div class="context-card ${getSentimentClass(context.sentiment)}">
            <div class="context-text">${context.comment}</div>
            <div class="context-metadata">
                Rating: ${context.rating} | Date: ${context.date}
            </div>
        </div>
    `).join('');
}

function promptWord(period) {
    const word = prompt('Enter word to analyze from word cloud:');
    if (word) analyzeWord(word);
}

window.analyzeWord = analyzeWord;
window.promptWord = promptWord;

document.addEventListener('DOMContentLoaded', initializeDashboard);