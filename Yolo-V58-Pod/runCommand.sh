docker build -t yolopod .
docker stop yolopod
docker rm yolopod
docker run -d --gpus 0 -p 8501:8501 --name yolopod yolopod
docker logs -f --since=1m yolopod
