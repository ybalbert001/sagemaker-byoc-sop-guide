#!/usr/bin/env python3
# =========================================================================
# deploy.py - 用构建好的镜像 + S3 模型创建 SageMaker 实时推理端点
# =========================================================================
# 前置条件:
#   - 已运行 build_and_push.sh 将镜像推送到 ECR
#   - 模型已打包为 model.tar.gz 上传到 S3 (模型文件 model.json 位于 tar 包根目录)
#   - 具备一个可被 SageMaker 假设的执行角色 (SAGEMAKER_ROLE_ARN)
#
# 用法:
#   export IMAGE_URI=<account>.dkr.ecr.<region>.amazonaws.com/xgboost-byos-inference:latest
#   export MODEL_DATA_URL=s3://<bucket>/artifact/model.tar.gz
#   export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<SageMakerExecutionRole>
#   python3 deploy.py
# =========================================================================
import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
IMAGE_URI = os.environ["IMAGE_URI"]
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
MODEL_DATA = os.environ["MODEL_DATA_URL"]
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")

sm = boto3.client("sagemaker", region_name=REGION)
suffix = time.strftime("%Y%m%d-%H%M%S")
model_name = f"xgboost-byos-{suffix}"
config_name = f"xgboost-byos-cfg-{suffix}"
endpoint_name = os.environ.get("ENDPOINT_NAME", "xgboost-byos-endpoint")

print(f"Creating model {model_name} ...")
sm.create_model(
    ModelName=model_name,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": MODEL_DATA,
        "Environment": {"MODEL_SERVER_WORKERS": "2", "MODEL_SERVER_TIMEOUT": "60"},
    },
    ExecutionRoleArn=ROLE_ARN,
)

print(f"Creating endpoint config {config_name} ...")
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": INSTANCE_TYPE,
            "InitialInstanceCount": 1,
        }
    ],
)

# 若端点已存在则更新, 否则创建
existing = sm.list_endpoints(NameContains=endpoint_name)["Endpoints"]
if any(e["EndpointName"] == endpoint_name for e in existing):
    print(f"Updating endpoint {endpoint_name} ...")
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
else:
    print(f"Creating endpoint {endpoint_name} ...")
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)

print("等待端点 InService ...")
waiter = sm.get_waiter("endpoint_in_service")
waiter.wait(EndpointName=endpoint_name)
print(f"端点就绪: {endpoint_name}")
