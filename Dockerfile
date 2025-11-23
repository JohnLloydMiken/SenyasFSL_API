# ==============================
#  BASE IMAGE
# ==============================
FROM python:3.10-slim

# ==============================
#  INSTALL SYSTEM DEPENDENCIES
# ==============================
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libhdf5-dev \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# ==============================
#  WORKDIR
# ==============================
WORKDIR /app

# ==============================
#  COPY PROJECT FILES
# ==============================
COPY . /app

# ==============================
#  UPGRADE PIP
# ==============================
RUN pip install --no-cache-dir --upgrade pip

# ==============================
#  INSTALL PYTHON DEPENDENCIES
# ==============================
RUN pip install --no-cache-dir fastapi uvicorn numpy pydantic tensorflow-cpu

# ==============================
#  EXPOSE PORT (Railway uses PORT)
# ==============================
EXPOSE 8000

# ==============================
#  START FASTAPI
# ==============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
