# shap_analysis.py（稳健版：兼容多种 shap 输出形态，生成 app 需要的 PNG）
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')
import json
import os
from pathlib import Path

import sys
import io
# 强制 stdout 使用 UTF-8（解决 Windows 子进程编码问题）
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass

# 从 utils 导入共享函数
from utils import apply_woe_map, ensure_dir

# 设置中文字体（若系统无该字体可根据环境调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("===== SHAP 可解释性分析（稳健版） =====")

# 确保目录存在
ensure_dir("shap_images")

# 必需文件检查（现在都在 model/ 目录下）
required_files = [
    'model/config.json', 
    'model/base_models_tuned.pkl', 
    'model/scaler_lr.pkl', 
    'model/woe_maps.pkl', 
    'model/lr_features.pkl', 
    'model/tree_features.pkl'
]
for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"缺少必需文件：{f}，请先运行训练流程生成这些文件")

with open('model/config.json', 'r') as f:
    CONFIG = json.load(f)

with open('model/base_models_tuned.pkl', 'rb') as f:
    base_models = pickle.load(f)
with open('model/scaler_lr.pkl', 'rb') as f:
    scaler_lr = pickle.load(f)
with open('model/woe_maps.pkl', 'rb') as f:
    woe_maps = pickle.load(f)
with open('model/lr_features.pkl', 'rb') as f:
    lr_features = pickle.load(f)
with open('model/tree_features.pkl', 'rb') as f:
    tree_features = pickle.load(f)
# ---- 加载 selected_features（IV筛选后的原始特征名） ----
try:
    with open('model/selected_features.pkl', 'rb') as f:
        selected_features = pickle.load(f)
except FileNotFoundError:
    selected_features = CONFIG.get('feature_cols', [])   # 降级方案

# ---- 读取 use_woe_for_lr ----
use_woe_for_lr = CONFIG.get('use_woe_for_lr', True)   # 默认 True（兼容旧模型）
DATA_PATH = CONFIG.get('data_path', 'nigerian_sme_loans_full.parquet')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH) if str(DATA_PATH).endswith('.parquet') else pd.read_csv(DATA_PATH)
SAMPLE_N = min(5000, len(df))
if SAMPLE_N <= 0:
    raise ValueError("数据样本为空，无法进行 SHAP 分析")
df_sample = df.sample(n=SAMPLE_N, random_state=42).copy()
print(f"抽样 {SAMPLE_N} 行用于 SHAP 分析")

# ---- 预处理（与训练逻辑保持一致） ----
if 'application_date' in df_sample.columns:
    df_sample['application_date'] = pd.to_datetime(df_sample['application_date'], errors='coerce')
if 'disbursement_date' in df_sample.columns:
    df_sample['disbursement_date'] = pd.to_datetime(df_sample['disbursement_date'], errors='coerce')
if 'application_date' in df_sample.columns and 'disbursement_date' in df_sample.columns:
    df_sample['days_to_disburse'] = (df_sample['disbursement_date'] - df_sample['application_date']).dt.days
if 'application_date' in df_sample.columns:
    df_sample['application_month'] = df_sample['application_date'].dt.month
    df_sample['application_quarter'] = df_sample['application_date'].dt.quarter

drop_cols = ['loan_id', 'business_id', 'application_date', 'disbursement_date', 'default_90d', 'default_180d']
for c in drop_cols:
    if c in df_sample.columns:
        df_sample.drop(columns=[c], inplace=True, errors='ignore')

# 填充与缩尾（使用 utils 中的清洗函数，但这里直接写也可以）
num_cols = df_sample.select_dtypes(include=['float64', 'int64']).columns.tolist()
for col in num_cols:
    df_sample[col].fillna(df_sample[col].median(), inplace=True)
    lower = df_sample[col].quantile(0.005)
    upper = df_sample[col].quantile(0.995)
    df_sample[col] = df_sample[col].clip(lower, upper)
cat_cols = df_sample.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if df_sample[col].isnull().any():
        df_sample[col].fillna(df_sample[col].mode().iloc[0] if not df_sample[col].mode().empty else "Other", inplace=True)

# ---- 构建 LR 特征（与训练时的 use_woe_for_lr 保持一致） ----
if use_woe_for_lr:
    # WOE 编码模式
    X_lr = pd.DataFrame()
    for col in lr_features:          # lr_features 是原始特征名
        if col in df_sample.columns:
            if col in woe_maps:
                X_lr[col] = apply_woe_map(df_sample[col], woe_maps[col])
            else:
                X_lr[col] = pd.to_numeric(df_sample[col], errors='coerce').fillna(0)
        else:
            X_lr[col] = 0.0
else:
    # 独热编码模式（与训练时一致）
    # 1) 确保 selected_features 中的列都存在，缺失列补默认值
    df_clean = df_sample.copy()
    for col in selected_features:
        if col not in df_clean.columns:
            if col in df_clean.select_dtypes(include=['object']).columns or col in woe_maps:
                df_clean[col] = 'Other'   # 分类默认
            else:
                df_clean[col] = 0          # 数值默认
    # 2) 对 selected_features 进行独热编码
    X_lr_raw = pd.get_dummies(df_clean[selected_features], drop_first=True)
    # 3) 对齐训练时的列（lr_features 是独热列名）
    for col in lr_features:
        if col not in X_lr_raw.columns:
            X_lr_raw[col] = 0
    X_lr = X_lr_raw[lr_features]   # 按顺序选择

# 标准化
X_lr_scaled = scaler_lr.transform(X_lr)

# ---- 构建树特征（独热编码并对齐） ----
X_tree = pd.get_dummies(df_sample, drop_first=True)
for col in tree_features:
    if col not in X_tree.columns:
        X_tree[col] = 0
X_tree = X_tree[tree_features]
X_tree_vals = X_tree.values

def save_plot(fig, fname):
    path = Path('shap_images') / fname
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print("已保存:", path)

# ---- XGBoost 分析 ----
if 'XGBoost' in base_models:
    try:
        print("分析 XGBoost 模型...")
        xgb_model = base_models['XGBoost']
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_tree, check_additivity=False)
        if isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[1] if len(shap_values) >= 2 else shap_values[0]
        else:
            shap_vals_to_plot = shap_values

        # Summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_to_plot, X_tree, feature_names=tree_features, show=False, max_display=15)
        plt.title('XGBoost SHAP Summary')
        plt.tight_layout()
        save_plot(plt.gcf(), 'shap_xgb_summary.png')
        plt.close()
        print("生成 shap_xgb_summary.png")

        # Bar
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals_to_plot, X_tree, feature_names=tree_features, plot_type='bar', show=False, max_display=15)
        plt.title('XGBoost 特征重要性')
        plt.tight_layout()
        save_plot(plt.gcf(), 'shap_xgb_bar.png')
        plt.close()
        print("生成 shap_xgb_bar.png")

        # Waterfall（高风险样本）
        probs = xgb_model.predict_proba(X_tree_vals)[:, 1]
        high_risk_idx = int(np.argmax(probs))
        try:
            if isinstance(shap_values, list) and len(shap_values) >= 2:
                single_shap = shap_values[1][high_risk_idx]
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list,tuple)) else explainer.expected_value
            else:
                single_shap = shap_vals_to_plot[high_risk_idx]
                base_val = explainer.expected_value
            exp = shap.Explanation(values=single_shap,
                                   base_values=base_val,
                                   data=X_tree.iloc[high_risk_idx].values,
                                   feature_names=tree_features)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=10, show=False)
            plt.title(f'XGBoost 高风险样本解释 (概率={probs[high_risk_idx]:.2%})')
            plt.tight_layout()
            save_plot(plt.gcf(), 'shap_xgb_waterfall.png')
            plt.close()
            print("生成 shap_xgb_waterfall.png")
        except Exception as e:
            print("生成 XGBoost 瀑布图失败:", e)
    except Exception as e:
        print("XGBoost SHAP 分析出错：", e)

# ---- Stacking 分析 ----
if 'Stacking' in base_models:
    try:
        print("分析 Stacking 元模型...")
        meta_model = base_models['Stacking']
        probs_list = []
        meta_feature_names = []
        if 'LogisticRegression' in base_models:
            lr_probs = base_models['LogisticRegression'].predict_proba(X_lr_scaled)[:, 1].reshape(-1, 1)
            probs_list.append(lr_probs); meta_feature_names.append('LR_Prob')
        if 'XGBoost' in base_models:
            xgb_probs = base_models['XGBoost'].predict_proba(X_tree_vals)[:, 1].reshape(-1, 1)
            probs_list.append(xgb_probs); meta_feature_names.append('XGB_Prob')
        if 'LightGBM' in base_models:
            lgb_probs = base_models['LightGBM'].predict_proba(X_tree_vals)[:, 1].reshape(-1, 1)
            probs_list.append(lgb_probs); meta_feature_names.append('LGB_Prob')

        if len(probs_list) < 1:
            print("警告：未找到基模型概率，跳过 Stacking SHAP 分析")
        else:
            X_meta = np.hstack(probs_list)
            predict_fn = lambda x: meta_model.predict_proba(x)[:, 1]
            background = X_meta[:min(100, X_meta.shape[0])]
            explainer_meta = shap.KernelExplainer(predict_fn, background, link="logit")
            shap_vals_meta = explainer_meta.shap_values(X_meta, nsamples=100)
            if isinstance(shap_vals_meta, list):
                shap_vals_pos = np.array(shap_vals_meta[1]) if len(shap_vals_meta) >= 2 else np.array(shap_vals_meta[0])
            else:
                shap_vals_pos = np.array(shap_vals_meta)

            if shap_vals_pos.ndim == 2 and shap_vals_pos.shape[1] == X_meta.shape[1]:
                plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_vals_pos, pd.DataFrame(X_meta, columns=meta_feature_names), feature_names=meta_feature_names, show=False)
                plt.title('Stacking 元模型 SHAP Summary')
                plt.tight_layout()
                save_plot(plt.gcf(), 'shap_stacking_summary.png')
                plt.close()
                print("生成 shap_stacking_summary.png")

                high_idx = int(np.argmax(X_meta.sum(axis=1))) if X_meta.size>0 else 0
                try:
                    base_val = explainer_meta.expected_value
                except:
                    base_val = None
                try:
                    exp_meta = shap.Explanation(values=shap_vals_pos[high_idx],
                                               base_values=base_val,
                                               data=X_meta[high_idx],
                                               feature_names=meta_feature_names)
                    plt.figure(figsize=(8, 4))
                    shap.plots.waterfall(exp_meta, max_display=3, show=False)
                    plt.title('Stacking 高风险样本解释')
                    plt.tight_layout()
                    save_plot(plt.gcf(), 'shap_stacking_waterfall.png')
                    plt.close()
                    print("生成 shap_stacking_waterfall.png")
                except Exception as e:
                    print("生成 Stacking 瀑布图失败:", e)
            else:
                if shap_vals_pos.ndim >= 2:
                    feat_imp = np.abs(shap_vals_pos).mean(axis=0)
                else:
                    feat_imp = np.abs(shap_vals_pos)
                plt.figure(figsize=(6, 4))
                plt.bar(meta_feature_names, feat_imp, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(meta_feature_names)])
                plt.title('Stacking 元模型特征重要性（平均 |SHAP|）')
                plt.ylabel('平均 |SHAP|')
                plt.tight_layout()
                save_plot(plt.gcf(), 'shap_stacking_bar.png')
                plt.close()
                print("生成 shap_stacking_bar.png (退化方案)")
    except Exception as e:
        print("Stacking SHAP 分析出错：", e)

print("SHAP 分析结束。请检查生成的 PNG 文件：shap_xgb_summary.png, shap_xgb_bar.png, shap_xgb_waterfall.png, shap_stacking_summary.png / shap_stacking_bar.png, shap_stacking_waterfall.png（视可用模型而定）")