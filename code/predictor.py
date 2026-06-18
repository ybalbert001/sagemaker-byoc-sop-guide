# =========================================================================
# predictor.py - XGBoost 推理 Flask 应用
# =========================================================================
# 实现 SageMaker 推理容器要求的两个端点:
#   GET  /ping        健康检查
#   POST /invocations 推理
#
# 模型加载约定:
#   SageMaker 会把 model.tar.gz (来自 s3://payermax-bucket/artifact)
#   解压到 /opt/ml/model。本文件会在该目录下查找 XGBoost 模型文件。
#   支持的模型文件名(按优先级): xgboost-model, model.json, model.ubj,
#   model.bst, *.json/*.ubj/*.bst/*.model 等。
# =========================================================================
import io
import json
import os
import glob

import flask
import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")

# 候选模型文件名(SageMaker 内置 XGBoost 默认导出名为 xgboost-model)
_MODEL_CANDIDATES = [
    "xgboost-model",
    "model.json",
    "model.ubj",
    "model.bst",
    "model.model",
]


class ModelHandler:
    """单例方式懒加载 XGBoost 模型, 避免每个 worker 重复加载。"""

    model = None

    @classmethod
    def _locate_model_file(cls):
        # 1. 优先匹配已知文件名
        for name in _MODEL_CANDIDATES:
            path = os.path.join(MODEL_DIR, name)
            if os.path.isfile(path):
                return path
        # 2. 回退: 按扩展名通配
        for pattern in ("*.json", "*.ubj", "*.bst", "*.model"):
            matches = sorted(glob.glob(os.path.join(MODEL_DIR, pattern)))
            if matches:
                return matches[0]
        return None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            model_path = cls._locate_model_file()
            if model_path is None:
                raise FileNotFoundError(
                    f"在 {MODEL_DIR} 下未找到 XGBoost 模型文件。"
                    f"已检查文件名: {_MODEL_CANDIDATES} 及 *.json/*.ubj/*.bst/*.model"
                )
            booster = xgb.Booster()
            booster.load_model(model_path)
            cls.model = booster
            print(f"Loaded XGBoost model from {model_path}", flush=True)
        return cls.model


# Flask 应用
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """健康检查: 能成功加载模型则返回 200, 否则 404。"""
    healthy = True
    try:
        ModelHandler.get_model()
    except Exception as e:  # noqa: BLE001
        print(f"Health check failed: {e}", flush=True)
        healthy = False
    status = 200 if healthy else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


def _get_feature_names():
    """从已加载的模型中获取 feature names (可能为空)。"""
    try:
        model = ModelHandler.get_model()
        names = model.feature_names
        return names if names else None
    except Exception:
        return None


def _parse_input(data, content_type):
    """将请求体解析为 xgboost.DMatrix。

    支持:
      - text/csv          : 无表头 CSV, 每行一个样本
      - application/json  : {"instances": [[...], [...]]} 或 [[...], [...]]
    """
    content_type = (content_type or "").lower()
    feature_names = _get_feature_names()

    if "text/csv" in content_type:
        df = pd.read_csv(io.StringIO(data.decode("utf-8")), header=None)
        if feature_names and len(df.columns) == len(feature_names):
            df.columns = feature_names
        return xgb.DMatrix(df, feature_names=feature_names)

    if "application/json" in content_type or "application/jsonlines" in content_type:
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict):
            instances = payload.get("instances", payload.get("inputs"))
        else:
            instances = payload
        arr = np.asarray(instances, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return xgb.DMatrix(arr, feature_names=feature_names)

    raise ValueError(f"不支持的 Content-Type: {content_type}. 请使用 text/csv 或 application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    """推理端点。根据 Accept 头返回 CSV 或 JSON 预测结果。"""
    content_type = flask.request.content_type
    accept = flask.request.headers.get("Accept", "application/json")

    try:
        dmatrix = _parse_input(flask.request.get_data(), content_type)
    except Exception as e:  # noqa: BLE001
        return flask.Response(
            response=json.dumps({"error": f"输入解析失败: {e}"}),
            status=400,
            mimetype="application/json",
        )

    try:
        model = ModelHandler.get_model()
        preds = model.predict(dmatrix)
    except Exception as e:  # noqa: BLE001
        return flask.Response(
            response=json.dumps({"error": f"推理失败: {e}"}),
            status=500,
            mimetype="application/json",
        )

    preds_list = np.asarray(preds).tolist()

    if accept and "text/csv" in accept.lower():
        out = io.StringIO()
        pd.DataFrame(preds_list).to_csv(out, header=False, index=False)
        return flask.Response(response=out.getvalue(), status=200, mimetype="text/csv")

    return flask.Response(
        response=json.dumps({"predictions": preds_list}),
        status=200,
        mimetype="application/json",
    )
