/**
 * AskHC - Frontend Application
 * Simple chat interface for RAG queries
 */

// ===========================================
// Configuration
// ===========================================
const API_BASE_URL = '/api';

// ===========================================
// DOM Elements
// ===========================================
const startupScreen = document.getElementById('startup-screen');
const startupStatus = document.getElementById('startup-status');
const mainApp = document.getElementById('main-app');
const messagesContainer = document.getElementById('messages');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');

// ===========================================
// Startup - Wait for Server
// ===========================================

/**
 * Check if server is ready and show main app
 */
async function waitForServer() {
    const maxAttempts = 60;  // Try for 60 seconds
    let attempts = 0;

    while (attempts < maxAttempts) {
        try {
            const response = await fetch('/health');
            if (response.ok) {
                // Server is ready!
                startupStatus.textContent = 'Ready!';
                setTimeout(() => {
                    startupScreen.style.display = 'none';
                    mainApp.style.display = 'flex';
                    questionInput.focus();
                }, 500);
                return;
            }
        } catch (e) {
            // Server not ready yet
        }

        attempts++;
        await new Promise(r => setTimeout(r, 1000));
    }

    // Timeout
    startupStatus.textContent = 'Server not responding. Please refresh.';
}

// ===========================================
// Chat Functions
// ===========================================

/**
 * Send a question to the API
 */
async function askQuestion(question) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                include_sources: false,
            }),
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        return {
            answer: 'Sorry, there was an error processing your request.',
            status: 'error',
        };
    }
}

/**
 * Add a message to the chat
 */
function addMessage(content, role, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    let html = `<div class="message-content"><p>${escapeHtml(content)}</p>`;

    if (sources && sources.length > 0) {
        html += `
            <div class="sources">
                <div class="sources-title">Sources (${sources.length}):</div>
                ${sources.map((s, i) => `
                    <div class="source-item">[${i + 1}] ${escapeHtml(s.content)}</div>
                `).join('')}
            </div>
        `;
    }

    html += '</div>';
    messageDiv.innerHTML = html;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===========================================
// Event Handlers
// ===========================================

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const question = questionInput.value.trim();
    if (!question) return;

    // Disable input
    questionInput.disabled = true;
    sendBtn.disabled = true;

    // Add user message
    addMessage(question, 'user');
    questionInput.value = '';

    // Add thinking indicator
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message assistant';
    thinkingDiv.innerHTML = '<div class="message-content"><p>Thinking...</p></div>';
    messagesContainer.appendChild(thinkingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Get response
    const response = await askQuestion(question);

    // Remove thinking indicator
    thinkingDiv.remove();

    // Add assistant message (sources disabled)
    addMessage(response.answer, 'assistant', null);

    // Re-enable input
    questionInput.disabled = false;
    sendBtn.disabled = false;
    questionInput.focus();
});

// ===========================================
// Initialize
// ===========================================
document.addEventListener('DOMContentLoaded', () => {
    waitForServer();
});
