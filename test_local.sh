#!/usr/bin/env bash
# =========================================================================
# test_local.sh - 本地验证推理镜像
# =========================================================================
# 步骤:
#   1. 训练一个示例 XGBoost 模型并保存到 ./model_local/xgboost-model
#   2. 构建镜像
#   3. 以 `serve` 启动容器, 将 ./model_local 挂载到 /opt/ml/model
#   4. 调用 /ping 与 /invocations 验证
# =========================================================================
set -euo pipefail

IMAGE_NAME="xgboost-byos-inference"
CONTAINER_NAME="xgb-byos-test"
MODEL_DIR="$(pwd)/model_local"

echo ">> [1/4] 生成示例模型..."
mkdir -p "${MODEL_DIR}"
python3 - <<'PY'
import numpy as np, xgboost as xgb, os
rng = np.random.RandomState(0)
X = rng.rand(200, 4)
y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
dtrain = xgb.DMatrix(X, label=y)
booster = xgb.train({"objective": "binary:logistic", "max_depth": 3}, dtrain, num_boost_round=10)
out = os.path.join("model_local", "xgboost-model")
booster.save_model(out)
print("saved", out)
PY

echo ">> [2/4] 构建镜像..."
docker build --platform linux/amd64 -t "${IMAGE_NAME}:latest" .

echo ">> [3/4] 启动容器..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER_NAME}" \
    -p 8080:8080 \
    -v "${MODEL_DIR}:/opt/ml/model:ro" \
    "${IMAGE_NAME}:latest" serve
sleep 5

echo ">> [4/4] 测试端点..."
echo "-- /ping --"
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8080/ping

echo "-- /invocations (CSV) --"
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: text/csv" \
    -H "Accept: application/json" \
    --data-binary $'0.2,0.9,0.1,0.5\n0.8,0.7,0.3,0.4'
echo ""

echo "-- /invocations (JSON) --"
curl -s -X POST http://localhost:8080/invocations \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"instances": [[0.2, 0.9, 0.1, 0.5], [0.8, 0.7, 0.3, 0.4]]}'
echo ""

echo ">> 清理容器..."
docker rm -f "${CONTAINER_NAME}" >/dev/null
echo ">> 本地测试完成。"
