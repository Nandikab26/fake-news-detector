async function classifyNews() {
    const newsText = document.getElementById('newsText').value;
    if (!newsText.trim()) {
        alert('Please enter some news text to classify.');
        return;
    }

    const loadingDiv = document.getElementById('loading');
    const resultArea = document.getElementById('result-area');
    const predictionBadge = document.getElementById('prediction-badge');

    loadingDiv.style.display = 'block';
    resultArea.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: newsText })
        });

        const data = await response.json();

        if (data.error) {
            alert(`An error occurred: ${data.error}`);
        } else {
            predictionBadge.textContent = data.prediction;
            if (data.prediction === 'RELIABLE') {
                predictionBadge.className = 'badge bg-reliable';
            } else {
                predictionBadge.className = 'badge bg-fake';
            }
            resultArea.style.display = 'block';
        }

    } catch (error) {
        console.error('Error:', error);
        alert('An unexpected error occurred. Please check the console.');
    } finally {
        loadingDiv.style.display = 'none';
    }
}