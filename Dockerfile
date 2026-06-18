# =========================================================================
# BYOS (Bring Your Own Container) SageMaker 推理镜像 - XGBoost
# =========================================================================
# SageMaker 推理容器约定:
#   - 容器以 `docker run <image> serve` 启动
#   - 必须监听 8080 端口
#   - 必须实现 GET  /ping        (健康检查, 返回 200)
#   - 必须实现 POST /invocations (推理请求)
#   - 模型文件会被解压到 /opt/ml/model
# =========================================================================
FROM python:3.11-slim

LABEL maintainer="payermax"
LABEL description="BYOS SageMaker inference image for XGBoost model"

# 避免 python 写 pyc / 缓冲 stdout, 方便查看 CloudWatch 日志
ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/program:${PATH}"

# 安装 nginx (反向代理) 及构建依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 拷贝推理服务代码到 /opt/program
COPY code/ /opt/program/
WORKDIR /opt/program

# serve 脚本需要可执行权限
RUN chmod +x /opt/program/serve

# SageMaker 通过 `serve` 命令启动容器
ENTRYPOINT ["serve"]
