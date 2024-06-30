# Backend

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Store reported messages in-memory (replace with a database in production)
reported_messages = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report', methods=['POST'])
def report():
    data = request.get_json()
    if 'message' in data:
        reported_messages.append(data['message'])
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Invalid request'}), 400

@app.route('/get_reports')
def get_reports():
    return jsonify({'reported_messages': reported_messages})

if __name__ == '__main__':
    app.run(debug=True)


# Frontend
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smishing Detection Feedback</title>
</head>
<body>
    <h1>Smishing Detection Feedback</h1>
    <form id="feedbackForm">
        <label for="message">Report Suspicious Message:</label>
        <textarea id="message" name="message" rows="4" cols="50" required></textarea>
        <br>
        <button type="button" onclick="submitFeedback()">Submit</button>
    </form>
    <div id="reportedMessages"></div>

    <script>
        function submitFeedback() {
            const message = document.getElementById('message').value;

            fetch('/report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Thank you for your feedback!');
                    document.getElementById('message').value = '';
                    loadReportedMessages();
                } else {
                    alert('Error submitting feedback. Please try again.');
                }
            });
        }

        function loadReportedMessages() {
            fetch('/get_reports')
            .then(response => response.json())
            .then(data => {
                const reportedMessagesDiv = document.getElementById('reportedMessages');
                reportedMessagesDiv.innerHTML = '<h2>Reported Messages:</h2>';
                
                if (data.reported_messages.length > 0) {
                    data.reported_messages.forEach(message => {
                        reportedMessagesDiv.innerHTML += `<p>${message}</p>`;
                    });
                } else {
                    reportedMessagesDiv.innerHTML += '<p>No reported messages yet.</p>';
                }
            });
        }

        // Load reported messages on page load
        document.addEventListener('DOMContentLoaded', loadReportedMessages);
    </script>
</body>
</html>
