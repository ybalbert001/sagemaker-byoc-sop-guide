#!/usr/bin/env bash
# =========================================================================
# build_and_push.sh - 构建镜像并推送到 Amazon ECR
# =========================================================================
# 用法:
#   ./build_and_push.sh <image-name> [region]
# 示例:
#   ./build_and_push.sh xgboost-byos-inference us-east-1
# =========================================================================
set -euo pipefail

IMAGE_NAME="${1:-xgboost-byos-inference}"
REGION="${2:-${AWS_REGION:-us-east-1}}"
TAG="${TAG:-latest}"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"
FULLNAME="${ECR_URI}:${TAG}"

echo ">> Account : ${ACCOUNT_ID}"
echo ">> Region  : ${REGION}"
echo ">> Image   : ${FULLNAME}"

# 1. 创建 ECR 仓库(若不存在)
aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${REGION}" >/dev/null 2>&1 || \
    aws ecr create-repository --repository-name "${IMAGE_NAME}" --region "${REGION}" >/dev/null

# 2. 登录 ECR
aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 3. 构建镜像 (linux/amd64 — SageMaker 推理实例为 x86_64)
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${TAG}" .

# 4. 打标签并推送
docker tag "${IMAGE_NAME}:${TAG}" "${FULLNAME}"
docker push "${FULLNAME}"

echo ""
echo ">> 推送完成: ${FULLNAME}"
echo ">> 在创建 SageMaker Model 时使用该镜像 URI。"
