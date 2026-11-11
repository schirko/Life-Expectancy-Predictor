# Life Expectancy Predictor - container build for the VPS (Coolify).
# Streamlit Community Cloud ignores this file; it only matters where Docker is used.

# Python 3.12: the app's docs recommend 3.10-3.12 because the shap package can
# lag behind brand-new Python releases.
FROM python:3.12-slim

# Do not write .pyc files; show log lines immediately in the Coolify log view.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the packages first, in their own layer. Docker reuses this layer when
# only the app's code changes, so most redeploys skip the slow install step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app: code, config, data, and the trained model in model/.
COPY . .

# Streamlit listens on 8501. Enter the same number as the port in Coolify.
EXPOSE 8501

# Coolify uses this to tell whether the app is up. Streamlit answers "ok" here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)" || exit 1

# 0.0.0.0 lets Coolify's proxy reach the app from outside the container.
CMD ["streamlit", "run", "Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
