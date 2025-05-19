// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'highlight') {
    highlightText(request.start, request.end);
  }
});

// Highlight text on page
function highlightText(start, end) {
  // Remove existing highlights
  const existingHighlights = document.querySelectorAll('.search-highlight');
  existingHighlights.forEach(el => {
    const parent = el.parentNode;
    parent.replaceChild(document.createTextNode(el.textContent), el);
    parent.normalize();
  });
  
  // Find text node containing the target position
  function findTextNode(node, targetPos) {
    if (node.nodeType === Node.TEXT_NODE) {
      if (node.length >= targetPos) {
        return node;
      }
      targetPos -= node.length;
    }
    
    for (let child of node.childNodes) {
      const result = findTextNode(child, targetPos);
      if (result) return result;
    }
    return null;
  }
  
  // Find start and end nodes
  const startNode = findTextNode(document.body, start);
  const endNode = findTextNode(document.body, end);
  
  if (startNode && endNode) {
    const range = document.createRange();
    range.setStart(startNode, start);
    range.setEnd(endNode, end);
    
    // Create highlight
    const span = document.createElement('span');
    span.className = 'search-highlight';
    span.style.backgroundColor = 'yellow';
    range.surroundContents(span);
    
    // Scroll to highlight
    span.scrollIntoView({
      behavior: 'smooth',
      block: 'center'
    });
  }
}

// Check if URL includes highlight parameters
if (window.location.href.includes('highlight.js')) {
  const urlParams = new URLSearchParams(window.location.search);
  const start = parseInt(urlParams.get('start'));
  const end = parseInt(urlParams.get('end'));
  
  if (!isNaN(start) && !isNaN(end)) {
    highlightText(start, end);
  }
} 