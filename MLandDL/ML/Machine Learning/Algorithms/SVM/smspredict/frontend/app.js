const form = document.getElementById('predict-form');
const status = document.getElementById('status');
const messageInput = document.getElementById('message');

function setStatus(text, type = 'info') {
  status.className = `status ${type}`;
  status.textContent = text;
  status.classList.remove('hidden');
}

function updateResult(message, label) {
  const level = /spam/i.test(label) ? 'spam' : 'safe';
  status.className = `status ${level}`;
  status.innerHTML = `
    <strong>Prediction:</strong> ${label}
    <p class="result-detail">${message}</p>
  `;
  status.classList.remove('hidden');
}

function extractLabel(responseMessage) {
  const match = responseMessage.match(/ is (.+)$/i);
  return match ? match[1] : responseMessage;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();

  if (!text) {
    setStatus('Please enter a message before submitting.', 'error');
    return;
  }

  setStatus('Analyzing message...', 'info');

  try {
    const response = await fetch('/api/predict/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.message || 'Unable to classify the message.');
    }

    const label = extractLabel(payload.message || 'No result returned.');
    updateResult(payload.message, label);
  } catch (error) {
    setStatus(error.message || 'Unexpected error.', 'error');
  }
});

setStatus('Ready to classify your SMS message.', 'info');
