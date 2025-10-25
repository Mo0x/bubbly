FROM python:3.12-slim

WORKDIR /app
COPY bubble_watch.py .

RUN pip install --no-cache-dir pandas numpy matplotlib plyer

# Create a cron job
RUN apt-get update && apt-get install -y cron

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["bash", "/app/run.sh"]