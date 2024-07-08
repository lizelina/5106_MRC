FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install -i https://nexus.4pd.io/repository/pypi-all/simple/ -U pip
RUN pip config set global.index-url https://nexus.4pd.io/repository/pypi-all/simple/
# 设置工作目录
WORKDIR /workspace/MRC

# 复制requirements.txt并安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制项目代码到工作目录
COPY . .

# 暴露端口80
EXPOSE 80

# 设置容器启动时执行的命令
CMD ["python", "digest_mrc.py"]