FROM python:3.11-slim

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas matplotlib scipy

RUN useradd -m -u 1000 sandbox
COPY executor.py /executor.py

USER 1000
WORKDIR /code

CMD ["python", "-u", "/executor.py"]
