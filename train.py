"""
train.py - XGBoost 模型训练示例脚本

使用 mock 数据和特征定义训练一个二分类 XGBoost 模型，
输出的模型文件可直接用于 predictor.py 推理服务。

用法:
    python train.py [--output-dir ./model_output]
"""
import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# =========================================================================
# 特征定义 (Mock)
# =========================================================================
FEATURE_DEFINITIONS = [
    {"name": "transaction_amount", "dtype": "float", "description": "交易金额(美元)"},
    {"name": "transaction_count_7d", "dtype": "int", "description": "近7天交易次数"},
    {"name": "transaction_count_30d", "dtype": "int", "description": "近30天交易次数"},
    {"name": "avg_amount_7d", "dtype": "float", "description": "近7天平均交易金额"},
    {"name": "max_amount_30d", "dtype": "float", "description": "近30天最大单笔金额"},
    {"name": "merchant_category", "dtype": "int", "description": "商户类别编码(0-9)"},
    {"name": "payment_method", "dtype": "int", "description": "支付方式编码(0-4)"},
    {"name": "hour_of_day", "dtype": "int", "description": "交易发生小时(0-23)"},
    {"name": "is_weekend", "dtype": "int", "description": "是否周末(0/1)"},
    {"name": "device_risk_score", "dtype": "float", "description": "设备风险评分(0-1)"},
    {"name": "ip_country_match", "dtype": "int", "description": "IP与注册国家是否匹配(0/1)"},
    {"name": "account_age_days", "dtype": "int", "description": "账户年龄(天)"},
    {"name": "failed_attempts_24h", "dtype": "int", "description": "24小时内失败尝试次数"},
    {"name": "velocity_1h", "dtype": "int", "description": "1小时内交易频率"},
    {"name": "amount_to_avg_ratio", "dtype": "float", "description": "当前金额与历史均值的比值"},
]

FEATURE_NAMES = [f["name"] for f in FEATURE_DEFINITIONS]
LABEL_COL = "is_fraud"


def generate_mock_data(n_samples=10000, seed=42):
    """生成 mock 训练数据，模拟支付欺诈检测场景。"""
    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        "transaction_amount": rng.exponential(scale=50, size=n_samples),
        "transaction_count_7d": rng.poisson(lam=5, size=n_samples),
        "transaction_count_30d": rng.poisson(lam=20, size=n_samples),
        "avg_amount_7d": rng.exponential(scale=40, size=n_samples),
        "max_amount_30d": rng.exponential(scale=100, size=n_samples),
        "merchant_category": rng.integers(0, 10, size=n_samples),
        "payment_method": rng.integers(0, 5, size=n_samples),
        "hour_of_day": rng.integers(0, 24, size=n_samples),
        "is_weekend": rng.integers(0, 2, size=n_samples),
        "device_risk_score": rng.beta(2, 5, size=n_samples),
        "ip_country_match": rng.integers(0, 2, size=n_samples),
        "account_age_days": rng.integers(1, 1000, size=n_samples),
        "failed_attempts_24h": rng.poisson(lam=0.5, size=n_samples),
        "velocity_1h": rng.poisson(lam=2, size=n_samples),
        "amount_to_avg_ratio": rng.exponential(scale=1.0, size=n_samples),
    })

    # 基于特征组合生成带有一定规律的标签
    fraud_score = (
        0.3 * (data["transaction_amount"] > 150).astype(float)
        + 0.2 * (data["device_risk_score"] > 0.5).astype(float)
        + 0.15 * (data["ip_country_match"] == 0).astype(float)
        + 0.15 * (data["failed_attempts_24h"] >= 2).astype(float)
        + 0.1 * (data["velocity_1h"] >= 5).astype(float)
        + 0.1 * (data["amount_to_avg_ratio"] > 3).astype(float)
    )
    fraud_prob = 1 / (1 + np.exp(-5 * (fraud_score - 0.4)))
    data[LABEL_COL] = rng.binomial(1, fraud_prob)

    print(f"生成 {n_samples} 条样本, 欺诈比例: {data[LABEL_COL].mean():.2%}")
    return data


def train(output_dir, n_samples=10000):
    """训练 XGBoost 模型并保存到 output_dir。"""
    data = generate_mock_data(n_samples=n_samples)

    X = data[FEATURE_NAMES]
    y = data[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_NAMES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "seed": 42,
    }

    print("开始训练...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    # 评估
    y_pred_prob = booster.predict(dval)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\n=== 验证集评估 ===")
    print(classification_report(y_val, y_pred, target_names=["正常", "欺诈"]))
    print(f"AUC: {roc_auc_score(y_val, y_pred_prob):.4f}")

    # 特征重要性
    importance = booster.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n=== 特征重要性 (Top 10) ===")
    for feat, score in sorted_imp[:10]:
        print(f"  {feat:30s} {score:.4f}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.json")
    booster.save_model(model_path)
    print(f"\n模型已保存: {model_path}")

    return booster


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 XGBoost 欺诈检测模型")
    parser.add_argument(
        "--output-dir", default="./model_output", help="模型输出目录"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000, help="mock 样本数量"
    )
    args = parser.parse_args()

    train(output_dir=args.output_dir, n_samples=args.n_samples)
