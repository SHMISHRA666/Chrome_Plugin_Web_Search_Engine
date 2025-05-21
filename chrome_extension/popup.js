document.addEventListener('DOMContentLoaded', function() {
    const queryInput = document.getElementById('query');
    const searchButton = document.getElementById('search');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const answerDiv = document.getElementById('answer');

    const API_BASE_URL = 'http://localhost:5000';

    async function processCurrentPage() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab || !tab.url) {
                throw new Error('No active tab found');
            }

            // Skip chrome:// and other restricted URLs
            if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
                throw new Error('Cannot access restricted pages');
            }

            // Get page content
            let content;
            try {
                const [{result}] = await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    function: () => document.body.innerText
                });
                content = result;
            } catch (scriptError) {
                console.error('Script execution error:', scriptError);
                throw new Error('Cannot access page contents. Please ensure the extension has proper permissions.');
            }

            // Process the page
            const response = await fetch(`${API_BASE_URL}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    url: tab.url,
                    title: tab.title,
                    content: content
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || 'Failed to process page');
            }

            const data = await response.json();
            if (data.status !== 'success') {
                throw new Error(data.error || 'Failed to process page');
            }

        } catch (error) {
            console.error('Error processing page:', error);
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
            throw error; // Re-throw to be handled by the caller
        }
    }

    async function handleSearch() {
        const query = queryInput.value.trim();
        if (!query) return;

        // Show loading state
        loadingDiv.style.display = 'block';
        errorDiv.style.display = 'none';
        answerDiv.style.display = 'none';
        resultsDiv.innerHTML = '';

        try {
            // Send query to backend using the /process endpoint
            const response = await fetch(`${API_BASE_URL}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            if (!response.ok) {
                throw new Error('Search failed');
            }

            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Search failed');
            }

            // Display answer if available
            if (data.answer) {
                answerDiv.textContent = data.answer;
                answerDiv.style.display = 'block';
            }

            // Display only the most relevant search result if available
            if (data.search_results && data.search_results.results && data.search_results.results.length > 0) {
                const bestResult = data.search_results.results[0]; // Get the most relevant result
                resultsDiv.innerHTML = `
                    <div class="result-item">
                        <div class="result-title">
                            ${bestResult.title}
                            <button class="view-source" data-url="${bestResult.url}">View Source</button>
                        </div>
                        <div class="result-content">
                            ${highlightText(bestResult.content, bestResult.highlight_start, bestResult.highlight_end)}
                        </div>
                    </div>
                `;

                // Add click handler for the View Source button
                const viewSourceBtn = resultsDiv.querySelector('.view-source');
                if (viewSourceBtn) {
                    viewSourceBtn.addEventListener('click', function() {
                        chrome.tabs.create({ url: this.dataset.url });
                    });
                }
            }

        } catch (error) {
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
        } finally {
            loadingDiv.style.display = 'none';
        }
    }

    function highlightText(text, start, end) {
        if (start === -1 || end === -1) return text;
        return text.substring(0, start) +
               `<span class="highlight">${text.substring(start, end)}</span>` +
               text.substring(end);
  }
  
  // Event listeners
    searchButton.addEventListener('click', handleSearch);
    queryInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
            handleSearch();
    }
  });

    // Remove this line to avoid processing the page on popup open
    // processCurrentPage();
}); 