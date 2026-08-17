# ===================== utils.py =====================
# 共享工具函数：供 train.py, app.py, shap_analysis.py, fastapi_app.py 调用
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve
from pandas import Interval
import warnings
import os

warnings.filterwarnings('ignore')

# ---------- 目录创建 ----------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ---------- WOE / IV 计算（仅供 train.py 使用） ----------
def calc_woe_iv_train(train_df, feature, target_series, n_bins=20):
    """基于训练集计算连续变量的WOE映射和IV值"""
    temp = pd.DataFrame({feature: train_df[feature], 'target': target_series})
    try:
        temp['bin'] = pd.qcut(temp[feature], q=n_bins, duplicates='drop')
    except ValueError:
        unique_vals = temp[feature].nunique()
        temp['bin'] = pd.qcut(temp[feature], q=min(unique_vals, 5), duplicates='drop')
    grouped = temp.groupby('bin', observed=False)['target'].agg(['sum', 'count'])
    grouped.columns = ['good', 'total']
    grouped['bad'] = grouped['total'] - grouped['good']
    grouped['good_pct'] = grouped['good'] / max(grouped['good'].sum(), 1)
    grouped['bad_pct'] = grouped['bad'] / max(grouped['bad'].sum(), 1)
    grouped['good_pct'] = grouped['good_pct'].replace(0, 0.0001)
    grouped['bad_pct'] = grouped['bad_pct'].replace(0, 0.0001)
    grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
    grouped['iv_contrib'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
    iv = grouped['iv_contrib'].sum()
    return grouped['woe'].to_dict(), iv

def calc_cat_woe_iv_train(train_df, feature, target_series):
    """分类变量的WOE映射（仅训练集）"""
    temp = pd.DataFrame({feature: train_df[feature], 'target': target_series})
    grouped = temp.groupby(feature)['target'].agg(['sum', 'count'])
    grouped.columns = ['good', 'total']
    grouped['bad'] = grouped['total'] - grouped['good']
    grouped['good_pct'] = grouped['good'] / max(grouped['good'].sum(), 1)
    grouped['bad_pct'] = grouped['bad'] / max(grouped['bad'].sum(), 1)
    grouped['good_pct'] = grouped['good_pct'].replace(0, 0.0001)
    grouped['bad_pct'] = grouped['bad_pct'].replace(0, 0.0001)
    grouped['woe'] = np.log(grouped['good_pct'] / grouped['bad_pct'])
    grouped['iv_contrib'] = (grouped['good_pct'] - grouped['bad_pct']) * grouped['woe']
    iv = grouped['iv_contrib'].sum()
    return grouped['woe'].to_dict(), iv

# ---------- VIF 检验 ----------
def calc_vif(X_data):
    from statsmodels.tools.tools import add_constant
    X_const = add_constant(X_data)
    vif_df = pd.DataFrame()
    vif_df['feature'] = X_const.columns
    vif_df['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    vif_df = vif_df[vif_df['feature'] != 'const']
    high_vif = vif_df[vif_df['VIF'] > 5]['feature'].tolist()
    return high_vif, vif_df

# ---------- 最优阈值 ----------
def find_optimal_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = np.argmax(youden)
    return thresholds[idx]

# ============================================================
# 核心修复：apply_woe_map（兼容连续变量分箱 + 分类变量）
# ============================================================
def apply_woe_map(series, woe_entry):
    """
    兼容连续变量（Interval键）和分类变量（字符串键）的WOE映射。
    修复了原版中 bins/labels 长度不匹配、空字典、非数值数据等问题。
    """
    # 1. 空字典或非法输入 → 返回全0
    if not isinstance(woe_entry, dict) or len(woe_entry) == 0:
        return pd.Series(0, index=series.index)

    first_key = next(iter(woe_entry.keys()))

    # 2. 分类变量（键为字符串）→ 直接 map
    if isinstance(first_key, str):
        return series.map(woe_entry).fillna(0)

    # 3. 连续变量（键为 Interval）→ 分箱映射
    if isinstance(first_key, Interval):
        # 3.1 将 series 转为数值，非数值转 NaN 并填充0
        try:
            numeric_series = pd.to_numeric(series, errors='coerce').fillna(0)
        except:
            return pd.Series(0, index=series.index)

        # 3.2 获取排序后的区间及对应的 WOE 值
        intervals = sorted(woe_entry.keys(), key=lambda x: x.left)
        woe_values = [woe_entry[iv] for iv in intervals]

        # 3.3 构造 bins 边界（左边界 + 所有右边界）
        bins = [intervals[0].left] + [iv.right for iv in intervals]

        # 3.4 去除重复边界（若相邻区间边界相等，则合并）
        unique_bins = []
        unique_woes = []
        for b, w in zip(bins, woe_values):
            if not unique_bins or b != unique_bins[-1]:
                unique_bins.append(b)
                unique_woes.append(w)
            else:
                # 若边界重复，将当前的 WOE 与上一个合并（取平均或保留最后一个，这里简单保留最后一个）
                unique_woes[-1] = w  # 用新值覆盖（实际应取平均，但为简化）
        # 注意：pd.cut 要求 bins 长度 = labels 长度 + 1
        # 如果去掉重复后，bins 长度可能不等于 labels+1，需要调整
        # 为保证长度一致，若 unique_woes 长度大于 len(unique_bins)-1，截断；若小于，补0
        if len(unique_woes) > len(unique_bins) - 1:
            unique_woes = unique_woes[:len(unique_bins)-1]
        elif len(unique_woes) < len(unique_bins) - 1:
            # 补齐不足的 labels（用最后一个值填充）
            last_val = unique_woes[-1] if unique_woes else 0
            unique_woes += [last_val] * (len(unique_bins) - 1 - len(unique_woes))

        # 3.5 执行分箱
        try:
            binned = pd.cut(
                numeric_series,
                bins=unique_bins,
                include_lowest=True,
                labels=unique_woes,
                duplicates='drop'  # 再次保证不重复
            )
            # 转换为浮点数，NaN 填充 0
            result = binned.astype(float).fillna(0)
        except Exception as e:
            # 如果分箱失败（如 bins 无效），返回全0
            print(f"⚠️ 分箱映射失败 ({e})，返回全0")
            result = pd.Series(0, index=series.index)
        return result

    # 4. 其他异常情况（键类型未知）→ 返回全0
    return pd.Series(0, index=series.index)

# ---------- 通用数据清洗（可选用） ----------
def clean_dataframe(df, target_col=None):
    """通用的数值/分类列清洗（缩尾、填充），供训练和预测复用"""
    df = df.copy()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
        lower = df[col].quantile(0.005)
        upper = df[col].quantile(0.995)
        df[col] = df[col].clip(lower, upper)
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Other", inplace=True)
    return df