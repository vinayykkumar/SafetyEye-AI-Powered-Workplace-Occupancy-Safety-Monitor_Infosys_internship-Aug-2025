# Use stable Python slim image (Debian Bookworm for compatibility)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (updated for Bookworm/Debian 12)
# Install system dependencies (updated for Bookworm/Debian 12)
# Install system dependencies (updated for Bookworm/Debian 12)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    wget \
    # Install Tkinter and its runtime dependencies
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (adjust if your structure differs)
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check (optional: pings Streamlit)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
CMD ["streamlit", "run", "dashboard/dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]