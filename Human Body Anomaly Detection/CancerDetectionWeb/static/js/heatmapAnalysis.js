/**
 * Heatmap Analysis Module
 * Handles the visualization and analysis of medical scan heatmaps
 */

// Initialize the heatmap visualization
function initHeatmapAnalysis() {
    // Get DOM elements
    const scanSelector = document.getElementById('scan-selector');
    const heatmapPlaceholder = document.getElementById('heatmap-placeholder');
    const heatmapLoading = document.getElementById('heatmap-loading');
    const heatmapDisplay = document.getElementById('heatmap-display');
    const visualizationTabs = document.getElementById('visualization-tabs');
    const visualizationContent = document.getElementById('visualization-content');
    
    // Make sure all required elements exist
    if (!scanSelector || !heatmapPlaceholder || !heatmapLoading || !heatmapDisplay) {
        console.error('Required heatmap elements not found in the DOM');
        return;
    }
    
    // Load available scans
    loadAvailableScans();
    
    // Set up event handlers
    scanSelector.addEventListener('change', handleScanSelection);
    
    // Setup visualization tabs if they exist
    if (visualizationTabs) {
        const tabs = visualizationTabs.querySelectorAll('.visualization-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                tabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to clicked tab
                this.classList.add('active');
                
                // Switch visualization content
                const tabName = this.getAttribute('data-tab');
                switchVisualization(tabName);
            });
        });
    }
    
    // Function to load available scans
    function loadAvailableScans() {
        scanSelector.disabled = true;
        
        fetch('/api/patient/scan_history')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error: ${response.status} - ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                scanSelector.innerHTML = '<option value="">Select a scan to visualize</option>';
                
                if (data.history && data.history.length > 0) {
                    data.history.forEach(scan => {
                        const option = document.createElement('option');
                        option.value = scan.id;
                        
                        // Format date for display
                        const scanDate = new Date(scan.date);
                        const formattedDate = scanDate.toLocaleDateString();
                        
                        option.textContent = `${scan.disease.charAt(0).toUpperCase() + scan.disease.slice(1)} (${scan.result}) - ${formattedDate}`;
                        scanSelector.appendChild(option);
                    });
                    
                    // Once scans are loaded, enable the selector
                    scanSelector.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error fetching scan history for selector:', error);
                // Show error message in selector
                scanSelector.innerHTML = '<option value="">Error loading scans</option>';
                
                // Update the placeholder with error message
                heatmapPlaceholder.innerHTML = `
                    <i class="fas fa-exclamation-triangle text-4xl mb-4 text-red-400"></i>
                    <p>Error loading scan history. Please try refreshing the page.</p>
                    <p class="text-sm text-blue-200 mt-2">${error.message}</p>
                `;
                
                scanSelector.disabled = false;
            });
    }
    
    // Handle scan selection
    function handleScanSelection() {
        const scanId = scanSelector.value;
        
        // Handle empty selection
        if (!scanId) {
            heatmapPlaceholder.classList.remove('hidden');
            heatmapLoading.classList.add('hidden');
            heatmapDisplay.classList.add('hidden');
            if (visualizationTabs) {
                visualizationTabs.classList.add('hidden');
            }
            return;
        }
        
        // Show loading state
        heatmapPlaceholder.classList.add('hidden');
        heatmapLoading.classList.remove('hidden');
        heatmapDisplay.classList.add('hidden');
        
        // Store the currently selected scan ID for the report generation
        const currentlyViewedScanIdElement = document.getElementById('currently-viewed-scan-id');
        if (currentlyViewedScanIdElement) {
            currentlyViewedScanIdElement.value = scanId;
        }
        
        // Fetch heatmap visualization data
        fetchHeatmapData(scanId);
    }
    
    // Function to fetch heatmap data with retry capability
    function fetchHeatmapData(scanId, retryCount = 0) {
        const maxRetries = 2;
        
        // First clear any existing error message
        const heatmapContainer = document.getElementById('heatmap-container');
        const existingError = document.getElementById('heatmap-error');
        if (existingError) {
            try {
                existingError.remove();
            } catch (e) {
                console.warn('Error removing existing error:', e);
            }
        }
        
        fetch(`/api/patient/heatmap/${scanId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error: ${response.status} - ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                // Process the heatmap data
                processHeatmapData(data);
            })
            .catch(error => {
                console.error('Error fetching heatmap data:', error);
                
                // Try to retry if not exceeded max retries
                if (retryCount < maxRetries) {
                    console.log(`Retrying fetch (${retryCount + 1}/${maxRetries})...`);
                    setTimeout(() => {
                        fetchHeatmapData(scanId, retryCount + 1);
                    }, 1000); // Wait 1 second before retry
                } else {
                    // Show error message if all retries failed
                    heatmapLoading.classList.add('hidden');
                    heatmapPlaceholder.classList.remove('hidden');
                    
                    // Create error message element
                    const errorElement = document.createElement('div');
                    errorElement.id = 'heatmap-error';
                    errorElement.className = 'text-center py-4';
                    errorElement.innerHTML = `
                        <i class="fas fa-exclamation-triangle text-4xl mb-4 text-red-400"></i>
                        <p>Error loading visualization data. Please try again.</p>
                        <p class="text-sm text-blue-200 mt-2">${error.message}</p>
                        <button id="retry-button" class="mt-4 py-2 px-4 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm">
                            <i class="fas fa-sync-alt mr-1"></i> Retry
                        </button>
                    `;
                    
                    // Safely add the error message
                    try {
                        heatmapPlaceholder.innerHTML = '';
                        heatmapPlaceholder.appendChild(errorElement);
                        
                        // Add event listener to retry button
                        const retryButton = document.getElementById('retry-button');
                        if (retryButton) {
                            retryButton.addEventListener('click', () => {
                                heatmapPlaceholder.classList.add('hidden');
                                heatmapLoading.classList.remove('hidden');
                                fetchHeatmapData(scanId, 0); // Reset retry count
                            });
                        }
                    } catch (e) {
                        console.warn('Error adding error message:', e);
                        // Fallback to simpler error handling
                        heatmapPlaceholder.innerHTML = `
                            <i class="fas fa-exclamation-triangle text-4xl mb-4 text-red-400"></i>
                            <p>Error loading visualization data. Please try again.</p>
                        `;
                    }
                }
            });
    }
    
    // Process heatmap data and update UI
    function processHeatmapData(data) {
        // Hide loading state
        heatmapLoading.classList.add('hidden');
        heatmapDisplay.classList.remove('hidden');
        if (visualizationTabs) {
            visualizationTabs.classList.remove('hidden');
        }
        
        // Set heatmap title
        const heatmapTitle = document.getElementById('heatmap-title');
        if (heatmapTitle) {
            heatmapTitle.textContent = `${data.disease.charAt(0).toUpperCase() + data.disease.slice(1)} Scan Analysis`;
        }
        
        // Set confidence indicator
        const confidenceElement = document.getElementById('confidence-indicator');
        if (confidenceElement) {
            const confidenceValue = data.confidence || 0;
            
            if (data.result === 'Positive' || data.result === 'Tuberculosis') {
                confidenceElement.className = 'px-3 py-1 text-xs rounded-full bg-red-500/20 text-red-200';
                confidenceElement.textContent = `${data.result} (${confidenceValue}% confidence)`;
            } else {
                confidenceElement.className = 'px-3 py-1 text-xs rounded-full bg-green-500/20 text-green-200';
                confidenceElement.textContent = `${data.result} (${confidenceValue}% confidence)`;
            }
        }
        
        // Force reload of both original and heatmap images to ensure fresh content
        const timestamp = new Date().getTime();
        
        // Set images
        const originalImage = document.getElementById('original-scan-image');
        const heatmapImage = document.getElementById('heatmap-image');
        
        if (originalImage && heatmapImage) {
            // Clear any previous error messages
            const errorElement = document.getElementById('heatmap-error');
            if (errorElement) {
                errorElement.remove();
            }
            
            if (!data.filename) {
                // If no scan image available, show placeholder
                originalImage.src = `https://via.placeholder.com/400x400.png?text=${data.disease.toUpperCase()}+Scan`;
                heatmapImage.src = `https://via.placeholder.com/400x400.png?text=Heatmap+Not+Available`;
            } else {
                // Use the separate original and heatmap images
                console.log("Original file:", data.original_filename || data.filename);
                console.log("Heatmap file:", data.heatmap_filename || `heatmap_${data.filename}`);
                
                // Add timestamp to bypass browser cache
                const timestamp = new Date().getTime();
                
                // For original image
                const originalFilename = data.original_filename ? data.original_filename : data.filename;
                const originalSrc = `/uploads/${originalFilename}?t=${timestamp}`;
                
                // For heatmap image
                const heatmapFilename = data.heatmap_filename ? data.heatmap_filename : `heatmap_${data.filename}`;
                const heatmapSrc = `/uploads/${heatmapFilename}?t=${timestamp}`;
                
                console.log("Loading original from:", originalSrc);
                console.log("Loading heatmap from:", heatmapSrc);
                
                // Set the image sources with cache-busting
                originalImage.src = originalSrc;
                heatmapImage.src = heatmapSrc;
                
                // Add title for clarification
                originalImage.title = "Original scan image";
                heatmapImage.title = "Heatmap overlay highlighting regions of interest";
                
                // Add loading indicator and better error handling
                const originalContainer = originalImage.parentElement;
                const heatmapContainer = heatmapImage.parentElement;
                
                if (originalContainer && heatmapContainer) {
                    // Clear previous loading indicators if any
                    const existingLoadingIndicators = document.querySelectorAll('.loading-indicator');
                    existingLoadingIndicators.forEach(indicator => {
                        try {
                            indicator.remove();
                        } catch (e) {
                            console.warn('Error removing loading indicator:', e);
                        }
                    });
                    
                    // Add loading indicator before images are loaded
                    const addLoadingIndicator = (container) => {
                        const loadingIndicator = document.createElement('div');
                        loadingIndicator.className = 'loading-indicator absolute inset-0 flex items-center justify-center bg-black/30';
                        loadingIndicator.innerHTML = '<i class="fas fa-spinner fa-spin text-2xl text-white"></i>';
                        container.style.position = 'relative';
                        
                        // Safely append the loading indicator
                        try {
                            container.appendChild(loadingIndicator);
                            return loadingIndicator;
                        } catch (e) {
                            console.warn('Error adding loading indicator:', e);
                            return null;
                        }
                    };
                    
                    const originalLoadingIndicator = addLoadingIndicator(originalContainer);
                    const heatmapLoadingIndicator = addLoadingIndicator(heatmapContainer);
                    
                    // Handle original image loading
                    originalImage.onload = () => {
                        if (originalLoadingIndicator && originalContainer.contains(originalLoadingIndicator)) {
                            try {
                                originalLoadingIndicator.remove();
                            } catch (e) {
                                console.warn('Error removing original loading indicator:', e);
                            }
                        }
                    };
                    
                    // Handle heatmap image loading
                    heatmapImage.onload = () => {
                        if (heatmapLoadingIndicator && heatmapContainer.contains(heatmapLoadingIndicator)) {
                            try {
                                heatmapLoadingIndicator.remove();
                            } catch (e) {
                                console.warn('Error removing heatmap loading indicator:', e);
                            }
                        }
                    };
                }
                
                // Add error handling for images
                originalImage.onerror = () => {
                    console.error('Failed to load original scan image');
                    originalImage.src = 'https://via.placeholder.com/400x400.png?text=Original+Image+Not+Available';
                    if (originalContainer) {
                        const indicator = originalContainer.querySelector('.loading-indicator');
                        if (indicator) {
                            try {
                                indicator.remove();
                            } catch (e) {
                                console.warn('Error removing loading indicator on error:', e);
                            }
                        }
                    }
                };
                
                heatmapImage.onerror = () => {
                    console.error('Failed to load heatmap image, attempting to regenerate');
                    
                    // Try to force regenerate the heatmap by calling the API again
                    const scanId = scanSelector.value;
                    const retryTimestamp = new Date().getTime();
                    
                    // Forcefully regenerate the heatmap
                    fetch(`/api/patient/heatmap/${scanId}?force=true&t=${retryTimestamp}`)
                        .then(response => response.json())
                        .then(newData => {
                            if (newData && !newData.error && newData.heatmap_filename) {
                                console.log("Regenerated heatmap, loading new image");
                                const newTimestamp = new Date().getTime();
                                heatmapImage.src = `/uploads/${newData.heatmap_filename}?t=${newTimestamp}`;
                            } else {
                                // If regeneration still fails, show placeholder
                                console.error("Regeneration failed, showing placeholder");
                                heatmapImage.src = `https://via.placeholder.com/400x400.png?text=Heatmap+Not+Available`;
                                
                                if (heatmapContainer) {
                                    const indicator = heatmapContainer.querySelector('.loading-indicator');
                                    if (indicator) {
                                        try {
                                            indicator.remove();
                                        } catch (e) {
                                            console.warn('Error removing loading indicator on error:', e);
                                        }
                                    }
                                }
                            }
                        })
                        .catch(error => {
                            console.error("Error regenerating heatmap:", error);
                            heatmapImage.src = `https://via.placeholder.com/400x400.png?text=Heatmap+Not+Available`;
                            
                            if (heatmapContainer) {
                                const indicator = heatmapContainer.querySelector('.loading-indicator');
                                if (indicator) {
                                    try {
                                        indicator.remove();
                                    } catch (e) {
                                        console.warn('Error removing loading indicator on error:', e);
                                    }
                                }
                            }
                        });
                };
            }
            
            // Add explanation of heatmap colors
            const heatmapContainer = document.querySelector('.heatmap-image');
            if (heatmapContainer) {
                // Check if legend already exists
                if (!document.getElementById('heatmap-legend')) {
                    const legendHTML = `
                        <div id="heatmap-legend" class="mt-2 flex items-center justify-center text-xs">
                            <div class="flex items-center">
                                <span class="w-3 h-3 inline-block rounded-sm bg-blue-400 mr-1"></span>
                                <span class="mr-2">Normal</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-3 h-3 inline-block rounded-sm bg-yellow-400 mr-1"></span>
                                <span class="mr-2">Low probability</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-3 h-3 inline-block rounded-sm bg-orange-500 mr-1"></span>
                                <span class="mr-2">Medium probability</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-3 h-3 inline-block rounded-sm bg-red-500 mr-1"></span>
                                <span>High probability</span>
                            </div>
                        </div>
                    `;
                    
                    // Add the legend to heatmap container
                    const legendContainer = document.createElement('div');
                    legendContainer.innerHTML = legendHTML;
                    
                    try {
                        heatmapContainer.appendChild(legendContainer);
                    } catch (e) {
                        console.warn('Error adding legend:', e);
                    }
                }
            }
        }
        
        // Set analysis breakdown
        const analysisElement = document.getElementById('heatmap-analysis');
        if (analysisElement) {
            analysisElement.innerHTML = '';
            
            // Add analysis points
            if (data.analysis && data.analysis.length > 0) {
                data.analysis.forEach(point => {
                    const li = document.createElement('li');
                    li.textContent = point;
                    analysisElement.appendChild(li);
                });
            } else if (data.features) {
                // Use features if analysis not available
                for (const [key, feature] of Object.entries(data.features)) {
                    const li = document.createElement('li');
                    li.textContent = `${key.replace('_', ' ')}: ${feature.interpretation}`;
                    analysisElement.appendChild(li);
                }
            } else {
                const li = document.createElement('li');
                li.textContent = 'No detailed analysis available for this scan.';
                analysisElement.appendChild(li);
            }
        }
        
        // Add additional heatmap explanation if needed
        updateHeatmapExplanation(data.disease, data.result);
    }
    
    // Add more detailed explanation based on disease type and result
    function updateHeatmapExplanation(disease, result) {
        const explanationContainer = document.getElementById('heatmap-container');
        if (!explanationContainer) return;
        
        // Remove existing explanation if any
        const existingExplanation = document.getElementById('heatmap-explanation');
        if (existingExplanation) {
            existingExplanation.remove();
        }
        
        // Create new explanation
        const explanationDiv = document.createElement('div');
        explanationDiv.id = 'heatmap-explanation';
        explanationDiv.className = 'mt-4 p-3 bg-blue-900/20 rounded-lg text-sm';
        
        let explanationText = "";
        if (disease === 'fracture') {
            if (result === 'Positive') {
                explanationText = `
                    <h5 class="font-medium mb-2">About Fracture Heatmap:</h5>
                    <p>The heatmap highlights potential fracture lines and areas of concern. 
                    Red and orange areas indicate high probability regions where bone discontinuities 
                    may exist. This AI analysis should be confirmed by clinical evaluation.</p>
                `;
            } else {
                explanationText = `
                    <h5 class="font-medium mb-2">About Normal Bone Heatmap:</h5>
                    <p>The blue areas in this heatmap indicate normal bone structure with no 
                    significant abnormalities detected. The AI system has analyzed the bone density 
                    and edge patterns and found them consistent with normal bone structure.</p>
                `;
            }
        } else if (disease === 'tb') {
            if (result === 'Positive' || result === 'Tuberculosis') {
                explanationText = `
                    <h5 class="font-medium mb-2">About TB Heatmap:</h5>
                    <p>The heatmap highlights areas in the lungs that may contain tuberculosis infiltrates. 
                    Red and orange areas represent regions with abnormal patterns consistent with TB. 
                    These findings should be confirmed by a pulmonologist.</p>
                `;
            } else {
                explanationText = `
                    <h5 class="font-medium mb-2">About Normal Lung Heatmap:</h5>
                    <p>The blue coloring indicates normal lung fields without significant abnormalities. 
                    The AI system has analyzed the lung texture, density, and patterns and found them 
                    consistent with normal lung tissue without signs of tuberculosis.</p>
                `;
            }
        }
        
        explanationDiv.innerHTML = explanationText;
        
        // Add the explanation to the container
        const vizContent = document.getElementById('visualization-content');
        if (vizContent && explanationContainer.contains(vizContent)) {
            explanationContainer.insertBefore(explanationDiv, vizContent);
        } else {
            explanationContainer.appendChild(explanationDiv);
        }
    }
    
    // Switch visualization content based on tab
    function switchVisualization(tabName) {
        if (!visualizationContent) return;
        
        const scanId = scanSelector.value;
        if (!scanId) return;
        
        // Show loading state
        visualizationContent.innerHTML = `
            <div class="text-center py-4">
                <div class="inline-block mb-3">
                    <i class="fas fa-spinner fa-spin text-4xl text-blue-400"></i>
                </div>
                <p>Loading ${tabName} visualization...</p>
            </div>
        `;
        visualizationContent.classList.remove('hidden');
        
        // Fetch visualization data with retry capability
        fetchVisualizationData(scanId, tabName);
    }
    
    // Function to fetch visualization data with retry
    function fetchVisualizationData(scanId, tabName, retryCount = 0) {
        const maxRetries = 2;
        
        fetch(`/api/patient/visualization/${scanId}/${tabName}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error: ${response.status} - ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                // Update visualization content
                const imageUrl = data.image_url || `https://via.placeholder.com/400x400.png?text=${tabName.charAt(0).toUpperCase() + tabName.slice(1)}+Analysis`;
                const description = data.description || 'No description available for this visualization type.';
                
                visualizationContent.innerHTML = `
                    <div class="p-3">
                        <img src="${imageUrl}" alt="${tabName} visualization" class="w-full rounded-lg shadow-lg object-contain max-h-[250px]" onerror="this.src='https://via.placeholder.com/400x400.png?text=Visualization+Not+Available'" />
                        <div class="mt-4">
                            <h5 class="font-medium mb-2">${tabName.charAt(0).toUpperCase() + tabName.slice(1)} Analysis:</h5>
                            <p>${description}</p>
                        </div>
                    </div>
                `;
            })
            .catch(error => {
                console.error(`Error loading ${tabName} visualization:`, error);
                
                // Try to retry if not exceeded max retries
                if (retryCount < maxRetries) {
                    console.log(`Retrying visualization fetch (${retryCount + 1}/${maxRetries})...`);
                    setTimeout(() => {
                        fetchVisualizationData(scanId, tabName, retryCount + 1);
                    }, 1000);
                } else {
                    // Show error message if all retries failed
                    visualizationContent.innerHTML = `
                        <div class="text-center py-4">
                            <i class="fas fa-exclamation-triangle text-4xl mb-4 text-red-400"></i>
                            <p>Error loading visualization. Please try again.</p>
                            <p class="text-sm text-blue-200 mt-2">${error.message}</p>
                            <button class="retry-viz-btn mt-4 py-2 px-4 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm">
                                <i class="fas fa-sync-alt mr-1"></i> Retry
                            </button>
                        </div>
                    `;
                    
                    // Add event listener to retry button
                    const retryButton = visualizationContent.querySelector('.retry-viz-btn');
                    if (retryButton) {
                        retryButton.addEventListener('click', () => {
                            switchVisualization(tabName);
                        });
                    }
                }
            });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initHeatmapAnalysis); 