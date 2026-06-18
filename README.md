# SageMaker BYOC 部署 SOP — XGBoost

## 流程概览

```
训练模型 → 打包上传 S3 → 构建推理镜像 → 推送 ECR → 创建端点 → 调用推理
```

## 前置条件

安装训练依赖：

```bash
pip install xgboost
```

SageMaker 执行角色需具备以下权限：

- **S3**: 读写模型文件所在 bucket
- **ECR**: `ecr:CreateRepository`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`, `ecr:PutImage`, `ecr:BatchCheckLayerAvailability`
- **SageMaker**: `sagemaker:CreateModel`, `sagemaker:CreateEndpointConfig`, `sagemaker:CreateEndpoint`, `sagemaker:InvokeEndpoint`

## Step 0: 设置环境变量

```bash
export AWS_REGION=us-east-1
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export IMAGE_NAME=xgboost-byos-inference
export BUCKET=<your-bucket>
export SAGEMAKER_ROLE_ARN=arn:aws:iam::${ACCOUNT}:role/<SageMakerExecutionRole>
```

## Step 1: 训练模型

```bash
python3 train.py --output-dir ./model_output
```

## Step 2: 打包模型并上传 S3

```bash
cd model_output
tar -czvf model.tar.gz model.json
aws s3 cp model.tar.gz s3://${BUCKET}/artifact/model.tar.gz
cd ..
```

> 模型文件必须位于 tar 包根目录。

## Step 3: 构建推理镜像并推送 ECR

```bash
chmod +x build_and_push.sh
./build_and_push.sh ${IMAGE_NAME} ${AWS_REGION}
```

## Step 4: 部署 SageMaker 端点

```bash
export IMAGE_URI=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}:latest
export MODEL_DATA_URL=s3://${BUCKET}/artifact/model.tar.gz
python3 deploy.py
```

## Step 5: 调用端点

```python
import json
import boto3

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

response = runtime.invoke_endpoint(
    EndpointName="xgboost-byos-endpoint",
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps({
        "instances": [
            [20.0, 3, 15, 18.0, 50.0, 2, 1, 14, 0, 0.1, 1, 500, 0, 1, 0.8],
            [300.0, 1, 2, 25.0, 300.0, 7, 3, 3, 1, 0.8, 0, 10, 5, 8, 5.2],
        ]
    }),
)

result = json.loads(response["Body"].read().decode("utf-8"))
print(result)
# {"predictions": [0.38, 0.81]}  (示例值, 因训练数据随机生成每次结果不同)
```

## Step 6: 清理资源

端点会持续产生费用，不再使用时及时删除：

```bash
aws sagemaker delete-endpoint --endpoint-name xgboost-byos-endpoint --region ${AWS_REGION}
```

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
