# ===================== train.py =====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')
import json
import os
# 从 utils 导入共享函数
from utils import (
    ensure_dir,
    calc_woe_iv_train,
    calc_cat_woe_iv_train,
    calc_vif,
    find_optimal_threshold,
    apply_woe_map,
    clean_dataframe
)
import sys
import io
# 强制 stdout 使用 UTF-8（解决 Windows 子进程编码问题）
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass

# ===================== 默认配置 =====================
DEFAULT_CONFIG = {
    "iv_threshold": 0.02,
    "vif_threshold": 5,
    "use_woe_for_lr": True,
    "use_iv_filter": True,
    "use_vif_filter": True,
    "sample_frac": 0.2,
    "test_size": 0.3,
    "random_state": 42,
    "cv_folds": 3,
    "n_iter_random": 8,
    "model_selection": ["lr", "xgb", "lgb"],
}

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("===== 自适应风控模型训练 =====")
    
    # ---------- 1. 加载配置 ----------
    config_path = 'temp_config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        print("✅ 加载用户配置（来自 temp_config.json）")
        CONFIG = DEFAULT_CONFIG.copy()
        CONFIG.update(user_config)
        CONFIG.setdefault('sample_frac', 1.0)
        CONFIG.setdefault('test_size', 0.3)
        CONFIG.setdefault('n_iter_random', 8)
        CONFIG.setdefault('model_selection', ["lr", "xgb", "lgb"])
        CONFIG.setdefault('use_iv_filter', True)
        CONFIG.setdefault('use_vif_filter', True)
    else:
        print("⚠️ 未找到 temp_config.json，使用默认配置（可能训练全量数据）")
        CONFIG = DEFAULT_CONFIG.copy()
        CONFIG['data_path'] = 'nigerian_sme_loans_full.parquet'
    
    print("当前配置:", json.dumps(CONFIG, indent=2))
    
    # ---------- 2. 读取数据 ----------
    data_path = CONFIG.get('data_path')
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    print(f"正在读取数据: {data_path}")
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    print(f"原始数据形状: {df.shape}")
    
    # 数据量截断抽样（强制上限20万行）
    MAX_ROWS = 200000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=CONFIG.get("random_state", 42))
        print(f"⚠️ 数据量超过 {MAX_ROWS}，随机抽样至 {len(df)} 条")
    else:
        print(f"✅ 数据量 {len(df)}，未达上限，全量使用")

    # ---------- 3. 提取目标列和特征列 ----------
    target_col = CONFIG.get('target_col')
    if target_col is None or target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不在数据中，请检查配置")
    
    feature_cols = CONFIG.get('feature_cols', [])
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target_col]
        print("未指定特征列，将使用所有非目标列")
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"警告: 以下特征列在数据中不存在，将被忽略: {missing}")
            feature_cols = [c for c in feature_cols if c in df.columns]
        if len(feature_cols) == 0:
            raise ValueError("没有可用的特征列，请检查配置")
    
    print(f"目标列: {target_col}")
    print(f"特征列数: {len(feature_cols)}")
    
    df = df[feature_cols + [target_col]].copy()
    df[target_col] = df[target_col].astype(int)
    
    # ---------- 4. 数据清洗 ----------
    df = clean_dataframe(df, target_col=target_col)
    
    y = df[target_col]
    print(f"全局违约占比: {y.mean():.4%}")
    
    # ---------- 5. 划分训练/测试 ----------
    X = df.drop(target_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=y
    )
    print(f"训练集样本: {len(y_train):,}, 违约率: {y_train.mean():.4%}")
    
    # ---------- 6. 特征工程：IV筛选 + WOE编码（可选） ----------
    continuous_cols = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    woe_maps = {}
    iv_dict = {}
    selected_features = []
    
    if CONFIG.get("use_iv_filter", True):
        print("\n--- 开始IV值筛选 ---")
        for col in continuous_cols:
            woe_map, iv = calc_woe_iv_train(X_train, col, y_train, n_bins=20)
            iv_dict[col] = iv
            if iv > CONFIG["iv_threshold"]:
                selected_features.append(col)
                woe_maps[col] = woe_map
        for col in categorical_cols:
            woe_map, iv = calc_cat_woe_iv_train(X_train, col, y_train)
            iv_dict[col] = iv
            if iv > CONFIG["iv_threshold"]:
                selected_features.append(col)
                woe_maps[col] = woe_map
        print(f"IV > {CONFIG['iv_threshold']} 保留特征数: {len(selected_features)}")
        sorted_iv = sorted(iv_dict.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 IV:", sorted_iv[:10])
    else:
        selected_features = continuous_cols + categorical_cols
        if CONFIG["use_woe_for_lr"]:
            for col in continuous_cols:
                woe_map, _ = calc_woe_iv_train(X_train, col, y_train, n_bins=20)
                woe_maps[col] = woe_map
            for col in categorical_cols:
                woe_map, _ = calc_cat_woe_iv_train(X_train, col, y_train)
                woe_maps[col] = woe_map
    
    # ---------- 保存分类变量的所有类别（用于前端下拉菜单） ----------
    print("\n--- 保存分类变量选项列表 ---")
    categorical_cols_all = X_train.select_dtypes(include=['object']).columns.tolist()
    categorical_options = {}
    for col in categorical_cols_all:
        unique_vals = X_train[col].dropna().unique().tolist()
        if "Other" not in unique_vals:
            unique_vals.append("Other")
        categorical_options[col] = unique_vals
        print(f"  {col}: {len(unique_vals)} 个选项")
    
    ensure_dir("model")
    with open('model/categorical_options.json', 'w', encoding='utf-8') as f:
        json.dump(categorical_options, f, ensure_ascii=False, indent=2)
    print("分类选项已保存到 model/categorical_options.json")

    # ---------- 7. 构建LR特征（WOE编码 + 标准化） ----------
    if CONFIG["use_woe_for_lr"]:
        X_train_lr_woe = pd.DataFrame()
        X_test_lr_woe = pd.DataFrame()
        for col in selected_features:
            if col in woe_maps:
                X_train_lr_woe[col] = X_train[col].map(woe_maps[col]).fillna(0)
                X_test_lr_woe[col] = X_test[col].map(woe_maps[col]).fillna(0)
            else:
                X_train_lr_woe[col] = X_train[col]
                X_test_lr_woe[col] = X_test[col]
        if CONFIG.get("use_vif_filter", True) and X_train_lr_woe.shape[1] > 1:
            print("\n--- 开始VIF多重共线性检验（迭代剔除） ---")
            features_keep = X_train_lr_woe.columns.tolist()
            while True:
                high_vif, vif_report = calc_vif(X_train_lr_woe[features_keep])
                if not high_vif:
                    break
                max_vif_feature = high_vif[0]
                features_keep.remove(max_vif_feature)
                print(f"  剔除 {max_vif_feature} (VIF={vif_report[vif_report['feature']==max_vif_feature]['VIF'].values[0]:.2f})")
                if len(features_keep) == 1:
                    break
            X_train_lr_woe = X_train_lr_woe[features_keep]
            X_test_lr_woe = X_test_lr_woe[features_keep]
            print(f"VIF剔除后保留特征数: {len(features_keep)}")
        
        scaler_lr = StandardScaler()
        X_train_lr_scaled = scaler_lr.fit_transform(X_train_lr_woe)
        X_test_lr_scaled = scaler_lr.transform(X_test_lr_woe)
        lr_feature_names = X_train_lr_woe.columns.tolist()
    else:
        X_train_lr_raw = pd.get_dummies(X_train[selected_features], drop_first=True)
        X_test_lr_raw = pd.get_dummies(X_test[selected_features], drop_first=True)
        X_train_lr_raw, X_test_lr_raw = X_train_lr_raw.align(X_test_lr_raw, join='left', axis=1, fill_value=0)
        if CONFIG.get("use_vif_filter", True) and X_train_lr_raw.shape[1] > 1:
            print("\n--- 开始VIF多重共线性检验（迭代剔除） ---")
            features_keep = X_train_lr_raw.columns.tolist()
            while True:
                high_vif, vif_report = calc_vif(X_train_lr_raw[features_keep])
                if not high_vif:
                    break
                max_vif_feature = high_vif[0]
                features_keep.remove(max_vif_feature)
                print(f"  剔除 {max_vif_feature} (VIF={vif_report[vif_report['feature']==max_vif_feature]['VIF'].values[0]:.2f})")
                if len(features_keep) == 1:
                    break
            X_train_lr_raw = X_train_lr_raw[features_keep]
            X_test_lr_raw = X_test_lr_raw[features_keep]
        scaler_lr = StandardScaler()
        X_train_lr_scaled = scaler_lr.fit_transform(X_train_lr_raw)
        X_test_lr_scaled = scaler_lr.transform(X_test_lr_raw)
        lr_feature_names = X_train_lr_raw.columns.tolist()
    
    # ---------- 8. 树模型特征（独热编码） ----------
    print(f"树模型将使用 IV 筛选后的 {len(selected_features)} 个特征进行独热编码")
    X_train_tree = pd.get_dummies(X_train[selected_features], drop_first=True)
    X_test_tree = pd.get_dummies(X_test[selected_features], drop_first=True)
    X_train_tree, X_test_tree = X_train_tree.align(X_test_tree, join='left', axis=1, fill_value=0)
    tree_feature_names = X_train_tree.columns.tolist()
    X_train_tree_raw = X_train_tree.values
    X_test_tree_raw = X_test_tree.values
    print(f"树模型特征数: {len(tree_feature_names)}")
    
    # ---------- 9. 训练模型 ----------
    results = {}
    best_models = {}
    
    # 9.1 逻辑回归
    if "lr" in CONFIG["model_selection"]:
        print("\n--- 训练逻辑回归 ---")
        lr_param_dist = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
        lr_base = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=CONFIG["random_state"])
        rs = RandomizedSearchCV(lr_base, lr_param_dist, n_iter=CONFIG["n_iter_random"], cv=CONFIG["cv_folds"],
                                scoring='roc_auc', random_state=CONFIG["random_state"], n_jobs=2, verbose=1)
        rs.fit(X_train_lr_scaled, y_train)
        best_lr = rs.best_estimator_
        best_models['LogisticRegression'] = best_lr
        y_prob = best_lr.predict_proba(X_test_lr_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks = max(tpr - fpr)
        y_pred = best_lr.predict(X_test_lr_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results['LogisticRegression'] = {'AUC': auc, 'KS': ks, 'Acc': acc, 'Prec': prec, 'Recall': rec, 'F1': f1}
        print(f"  AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
    
    # 9.2 XGBoost
    if "xgb" in CONFIG["model_selection"]:
        print("\n--- 训练 XGBoost ---")
        xgb_param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.9, 1.0]
        }
        scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
        xgb_base = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=CONFIG["random_state"],
                                 use_label_encoder=False, eval_metric='logloss')
        rs = RandomizedSearchCV(xgb_base, xgb_param_dist, n_iter=CONFIG["n_iter_random"], cv=CONFIG["cv_folds"],
                                scoring='roc_auc', random_state=CONFIG["random_state"], n_jobs=2, verbose=1)
        rs.fit(X_train_tree_raw, y_train)
        best_xgb = rs.best_estimator_
        best_models['XGBoost'] = best_xgb
        y_prob = best_xgb.predict_proba(X_test_tree_raw)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks = max(tpr - fpr)
        y_pred = best_xgb.predict(X_test_tree_raw)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results['XGBoost'] = {'AUC': auc, 'KS': ks, 'Acc': acc, 'Prec': prec, 'Recall': rec, 'F1': f1}
        print(f"  AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
    
    # 9.3 LightGBM
    if "lgb" in CONFIG["model_selection"]:
        print("\n--- 训练 LightGBM ---")
        lgb_param_dist = {
            'n_estimators': [100, 200, 300],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.7, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.9, 1.0]
        }
        lgb_base = LGBMClassifier(class_weight='balanced', random_state=CONFIG["random_state"], verbose=-1)
        rs = RandomizedSearchCV(lgb_base, lgb_param_dist, n_iter=CONFIG["n_iter_random"], cv=CONFIG["cv_folds"],
                                scoring='roc_auc', random_state=CONFIG["random_state"], n_jobs=2, verbose=1)
        rs.fit(X_train_tree_raw, y_train)
        best_lgb = rs.best_estimator_
        best_models['LightGBM'] = best_lgb
        y_prob = best_lgb.predict_proba(X_test_tree_raw)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks = max(tpr - fpr)
        y_pred = best_lgb.predict(X_test_tree_raw)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results['LightGBM'] = {'AUC': auc, 'KS': ks, 'Acc': acc, 'Prec': prec, 'Recall': rec, 'F1': f1}
        print(f"  AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
    
    # ---------- 10. Stacking 融合 ----------
    if len(best_models) >= 2:
        print("\n--- 训练 Stacking 融合模型 ---")
        meta_train_list = []
        meta_test_list = []
        for name, model in best_models.items():
            if name == 'LogisticRegression':
                X_train_meta = X_train_lr_scaled
                X_test_meta = X_test_lr_scaled
            else:
                X_train_meta = X_train_tree_raw
                X_test_meta = X_test_tree_raw
            prob_train = model.predict_proba(X_train_meta)[:, 1].reshape(-1, 1)
            prob_test = model.predict_proba(X_test_meta)[:, 1].reshape(-1, 1)
            meta_train_list.append(prob_train)
            meta_test_list.append(prob_test)
        X_meta_train = np.hstack(meta_train_list)
        X_meta_test = np.hstack(meta_test_list)
        
        meta_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=CONFIG["random_state"])
        meta_lr.fit(X_meta_train, y_train)
        best_models['Stacking'] = meta_lr
        y_prob = meta_lr.predict_proba(X_meta_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks = max(tpr - fpr)
        y_pred = meta_lr.predict(X_meta_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results['Stacking'] = {'AUC': auc, 'KS': ks, 'Acc': acc, 'Prec': prec, 'Recall': rec, 'F1': f1}
        print(f"  Stacking AUC: {auc:.4f}, KS: {ks:.4f}, F1: {f1:.4f}")
        optimal_threshold = find_optimal_threshold(y_test, y_prob)
        print(f"最优阈值 (Youden): {optimal_threshold:.4f}")
    else:
        model = list(best_models.values())[0]
        if 'LogisticRegression' in best_models:
            prob = model.predict_proba(X_test_lr_scaled)[:, 1]
        else:
            prob = model.predict_proba(X_test_tree_raw)[:, 1]
        optimal_threshold = find_optimal_threshold(y_test, prob)
    
    # ---------- 11. 保存所有产物到 model/ 目录 ----------
    print("\n--- 保存模型和配置 ---")
    ensure_dir("model")
    
    with open('model/base_models_tuned.pkl', 'wb') as f:
        pickle.dump(best_models, f)
    
    with open('model/scaler_lr.pkl', 'wb') as f:
        pickle.dump(scaler_lr, f)
    with open('model/woe_maps.pkl', 'wb') as f:
        pickle.dump(woe_maps, f)
    with open('model/lr_features.pkl', 'wb') as f:
        pickle.dump(lr_feature_names, f)
    with open('model/tree_features.pkl', 'wb') as f:
        pickle.dump(tree_feature_names, f)
    with open('model/best_threshold.pkl', 'wb') as f:
        pickle.dump(optimal_threshold, f)
# ---- 保存 selected_features（IV筛选后的原始特征名） ----
    with open('model/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)   

    with open('model/config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model/model_performance.csv')
    print("所有文件保存完成！")

    # ========== 新增：保存训练集和测试集预测结果（用于监控/PSI） ==========
    print("\n--- 保存训练集和测试集预测结果用于监控 ---")
    # 复制原始特征（未编码）
    X_test_original = X_test.copy()
    test_df = X_test_original.copy()
    test_df['y_true'] = y_test.values

    X_train_original = X_train.copy()
    train_df = X_train_original.copy()
    train_df['y_true'] = y_train.values

    # 定义函数：为数据集添加各模型预测概率
    def add_predictions(df, X_lr_scaled, X_tree_raw, models):
        for name, model in models.items():
            if name == 'LogisticRegression':
                prob = model.predict_proba(X_lr_scaled)[:, 1]
            elif name in ['XGBoost', 'LightGBM']:
                prob = model.predict_proba(X_tree_raw)[:, 1]
            elif name == 'Stacking':
                # 构建 meta 特征
                meta_list = []
                for sub_name in models:
                    if sub_name == 'LogisticRegression':
                        sub_prob = models[sub_name].predict_proba(X_lr_scaled)[:, 1].reshape(-1, 1)
                    elif sub_name in ['XGBoost', 'LightGBM']:
                        sub_prob = models[sub_name].predict_proba(X_tree_raw)[:, 1].reshape(-1, 1)
                    else:
                        continue
                    meta_list.append(sub_prob)
                if meta_list:
                    meta_arr = np.hstack(meta_list)
                    prob = model.predict_proba(meta_arr)[:, 1]
                else:
                    continue
            else:
                continue
            df[f'prob_{name}'] = prob
        return df

    # 为测试集添加预测
    test_df = add_predictions(test_df, X_test_lr_scaled, X_test_tree_raw, best_models)
    test_df.to_parquet('model/test_predictions.parquet', index=False)

    # 为训练集添加预测（用于 PSI 基准）
    train_df = add_predictions(train_df, X_train_lr_scaled, X_train_tree_raw, best_models)
    train_df.to_parquet('model/train_predictions.parquet', index=False)

    print("训练集和测试集预测结果已保存至 model/ 目录")
    # ========== 新增结束 ==========

    print("\n===== 训练结束 =====")