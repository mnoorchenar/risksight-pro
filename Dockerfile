FROM python:3.11-slim

WORKDIR /app

# HuggingFace Spaces needs a non-root user
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]
```

**Your project folder should look like this:**
```
risksight-pro/
├── app.py
├── requirements.txt
└── Dockerfile