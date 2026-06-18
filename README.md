# SageMaker BYOC 部署 SOP — XGBoost

## 流程概览

```
训练模型 → 打包上传 S3 → 构建推理镜像 → 推送 ECR → 创建端点 → 调用推理
```

## Step 1: 训练模型

```bash
python3 train.py --output-dir ./model_output
```

## Step 2: 打包模型并上传 S3

```bash
cd model_output
tar -czvf model.tar.gz model.json
aws s3 cp model.tar.gz s3://<your-bucket>/artifact/model.tar.gz
```

> 模型文件必须位于 tar 包根目录。

## Step 3: 构建推理镜像并推送 ECR

```bash
chmod +x build_and_push.sh
./build_and_push.sh <image-name> <region>
# 例: ./build_and_push.sh xgboost-byos-inference us-east-1
```

## Step 4: 部署 SageMaker 端点

```bash
export IMAGE_URI=<account>.dkr.ecr.<region>.amazonaws.com/<image-name>:latest
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<SageMakerExecutionRole>
export MODEL_DATA_URL=s3://<your-bucket>/artifact/model.tar.gz
python3 deploy.py
```

## Step 5: 调用端点

```bash
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name xgboost-byos-endpoint \
  --content-type text/csv \
  --accept application/json \
  --body $'20.0,3,15,18.0,50.0,2,1,14,0,0.1,1,500,0,1,0.8' \
  /dev/stdout
```

支持的输入格式:
- `text/csv` — 无表头，逗号分隔
- `application/json` — `{"instances": [[...], [...]]}`

## 目录结构

```
├── train.py              # 模型训练脚本 (含 mock 数据)
├── build_and_push.sh     # 构建并推送镜像到 ECR
├── deploy.py             # 创建 SageMaker 端点
├── Dockerfile            # 推理容器定义
├── requirements.txt      # Python 依赖
└── code/
    ├── serve             # 容器入口 (nginx + gunicorn)
    ├── nginx.conf        # 反向代理 (端口 8080)
    ├── wsgi.py           # WSGI 入口
    └── predictor.py      # Flask 推理服务 (/ping, /invocations)
```
