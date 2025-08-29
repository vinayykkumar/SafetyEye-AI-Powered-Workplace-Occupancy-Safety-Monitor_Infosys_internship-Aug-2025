import requests

# Alerts: Send notifications for missing gear
def send_alert(message, webhook_url):
    payload = {'text': message}
    requests.post(webhook_url, json=payload)

# Example usage:
# send_alert('PPE violation detected!', 'https://your-webhook-url')
