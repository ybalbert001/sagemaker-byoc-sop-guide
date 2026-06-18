# BYOS SageMaker 推理镜像 — XGBoost

一个自定义（BYOC / Bring Your Own Container）的 Amazon SageMaker 推理镜像，用于部署 XGBoost 模型。镜像遵循 SageMaker 推理容器约定，可直接用于创建实时推理端点（Real-time Endpoint）、批量转换（Batch Transform）或多模型端点。

## 目录结构

```
.
├── Dockerfile            # 镜像定义 (python:3.11-slim + nginx + gunicorn)
├── requirements.txt      # Python 依赖 (xgboost / flask / gunicorn ...)
├── build_and_push.sh     # 构建并推送镜像到 ECR
├── test_local.sh         # 本地端到端测试
├── deploy.py             # 用镜像 + S3 模型创建 SageMaker 端点
└── code/
    ├── serve             # 容器入口: 启动 nginx + gunicorn
    ├── nginx.conf        # nginx 反向代理 (对外 8080)
    ├── wsgi.py           # gunicorn WSGI 入口
    └── predictor.py      # Flask 应用: /ping 与 /invocations
```

## SageMaker 容器约定

| 要求 | 实现 |
|------|------|
| 以 `serve` 启动 | `Dockerfile` 的 `ENTRYPOINT ["serve"]` |
| 监听 8080 | `nginx.conf` 中 `listen 8080` |
| `GET /ping` 健康检查 | `predictor.py` 的 `ping()`，能加载模型返回 200 |
| `POST /invocations` 推理 | `predictor.py` 的 `invocations()` |
| 模型挂载到 `/opt/ml/model` | SageMaker 自动从 `ModelDataUrl` 解压 |

## 模型文件约定

SageMaker 会把 `ModelDataUrl` 指向的 `model.tar.gz` 解压到容器内的 `/opt/ml/model`。
本镜像会在该目录按以下顺序查找模型：

`xgboost-model` → `model.json` → `model.ubj` → `model.bst` → `model.model` → 通配 `*.json/*.ubj/*.bst/*.model`

题目假设模型文件位于 `s3://payermax-bucket/artifact`。请确保把 XGBoost 模型打包成 tar.gz，**模型文件位于 tar 包根目录**：

```bash
# 假设本地有训练好的模型文件 xgboost-model
tar -czvf model.tar.gz xgboost-model
aws s3 cp model.tar.gz s3://payermax-bucket/artifact/model.tar.gz
```

> 注意：`ModelDataUrl` 必须指向具体的 `.tar.gz` 对象（如 `s3://payermax-bucket/artifact/model.tar.gz`），而不是前缀目录。

## 使用步骤

### 1. 本地测试（推荐先做）

```bash
chmod +x test_local.sh
./test_local.sh
```

脚本会训练一个示例模型、构建镜像、启动容器并调用 `/ping`、`/invocations`。

### 2. 构建并推送到 ECR

```bash
chmod +x build_and_push.sh
./build_and_push.sh xgboost-byos-inference us-east-1
```

输出形如：
`<account>.dkr.ecr.us-east-1.amazonaws.com/xgboost-byos-inference:latest`

### 3. 部署端点

```bash
export IMAGE_URI=<account>.dkr.ecr.us-east-1.amazonaws.com/xgboost-byos-inference:latest
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<SageMakerExecutionRole>
export MODEL_DATA_URL=s3://payermax-bucket/artifact/model.tar.gz
python3 deploy.py
```

### 4. 调用端点

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name xgboost-byos-endpoint \
  --content-type text/csv \
  --accept application/json \
  --body $'0.2,0.9,0.1,0.5' \
  /dev/stdout
```

## 输入 / 输出格式

请求（`Content-Type`）：
- `text/csv`：无表头，每行一个样本，逗号分隔特征
- `application/json`：`{"instances": [[...], [...]]}` 或直接 `[[...], [...]]`

响应（由 `Accept` 决定）：
- `application/json`（默认）：`{"predictions": [...]}`
- `text/csv`：每行一个预测值

## 可调环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_SERVER_WORKERS` | CPU 核数 | gunicorn worker 数 |
| `MODEL_SERVER_TIMEOUT` | 60 | 请求超时（秒） |
| `MODEL_DIR` | `/opt/ml/model` | 模型目录 |

## 安全提示

SageMaker 实时端点默认通过 IAM（SigV4 签名）进行访问控制，端点本身不对公网直接暴露未认证接口。请确保：
- SageMaker 执行角色仅授予访问 `s3://payermax-bucket/artifact` 的最小权限；
- 调用方使用 IAM 凭证签名调用 `sagemaker-runtime:InvokeEndpoint`。
