// Initialize WebSearchAgent
let webSearchAgent = null;

// Add processing cache and debounce timer
const processedUrls = new Set();
let processingTimer = null;

// Connect to backend
async function connectToBackend() {
    try {
        const response = await fetch('http://localhost:5000/connect');
        const data = await response.json();
        if (data.status === 'connected') {
            console.log('Connected to backend');
            return true;
        }
    } catch (error) {
        console.error('Failed to connect to backend:', error);
    }
    return false;
}

// Process current tab
async function processCurrentTab() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab || !tab.url) {
            console.log('No active tab or URL found');
            return;
        }

        // Skip if URL was recently processed
        if (processedUrls.has(tab.url)) {
            console.log('URL recently processed, skipping:', tab.url);
            return;
        }

        // Skip private/confidential sites
        const privateDomains = [
            'gmail.com', 'google.com/mail',
            'whatsapp.com', 'web.whatsapp.com',
            'facebook.com/messages',
            'linkedin.com/messaging',
            'outlook.com', 'office.com',
            'slack.com',
            'teams.microsoft.com'
        ];
        
        if (privateDomains.some(domain => tab.url.includes(domain))) {
            console.log('Skipping private site:', tab.url);
            return;
        }

        // Process URL
        const response = await fetch('http://localhost:5000/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: tab.url })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            console.log('Processed URL:', tab.url);
            // Add to processed URLs cache
            processedUrls.add(tab.url);
            // Remove from cache after 5 minutes
            setTimeout(() => processedUrls.delete(tab.url), 5 * 60 * 1000);
        }
    } catch (error) {
        console.error('Failed to process tab:', error);
    }
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'search') {
        fetch('http://localhost:5000/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: request.query })
        })
        .then(response => response.json())
        .then(data => sendResponse(data))
        .catch(error => {
            console.error('Search failed:', error);
            sendResponse({ error: 'Search failed' });
        });
        return true;
    }
    
    if (request.action === 'getStats') {
        fetch('http://localhost:5000/stats')
        .then(response => response.json())
        .then(data => sendResponse(data))
        .catch(error => {
            console.error('Failed to get stats:', error);
            sendResponse({ error: 'Failed to get stats' });
        });
        return true;
    }
});

// Listen for tab updates with debounce
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        // Clear existing timer
        if (processingTimer) {
            clearTimeout(processingTimer);
        }
        // Set new timer
        processingTimer = setTimeout(() => {
            processPage(tab);
        }, 1000); // 1 second debounce
    }
});

async function processPage(tab) {
    try {
        if (!tab || !tab.url) {
            console.log('No tab or URL found');
            return;
        }

        // Skip if URL was recently processed
        if (processedUrls.has(tab.url)) {
            console.log('URL recently processed, skipping:', tab.url);
            return;
        }

        // Skip chrome:// and other restricted URLs
        if (tab.url.startsWith('chrome://') || tab.url.startsWith('chrome-extension://')) {
            console.log('Skipping restricted page:', tab.url);
            return;
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
            return;
        }

        // Send to backend
        const response = await fetch('http://localhost:5000/process', {
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

        console.log('Successfully processed page:', tab.url);
        // Add to processed URLs cache
        processedUrls.add(tab.url);
        // Remove from cache after 5 minutes
        setTimeout(() => processedUrls.delete(tab.url), 5 * 60 * 1000);

    } catch (error) {
        console.error('Error processing page:', error);
    }
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'processCurrentPage') {
    chrome.tabs.query({ active: true, currentWindow: true }, async ([tab]) => {
      if (tab) {
        await processPage(tab);
        sendResponse({ success: true });
      }
    });
    return true; // Keep the message channel open for async response
  }
});

// Initialize
connectToBackend().then(connected => {
    if (connected) {
        processCurrentTab();
    }
}); 