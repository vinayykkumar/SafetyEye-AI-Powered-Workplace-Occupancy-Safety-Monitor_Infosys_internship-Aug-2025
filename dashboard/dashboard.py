from flask import Flask, render_template

# Dashboard: Live view + compliance stats
app = Flask(__name__)

@app.route('/')
def home():
    return 'PPE Compliance Dashboard Coming Soon!'

if __name__ == '__main__':
    app.run(debug=True)
