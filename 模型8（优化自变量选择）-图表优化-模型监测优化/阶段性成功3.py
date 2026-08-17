# ===================== app.py（修正Header和Sidebar不透明度，完美透出背景图） =====================
import streamlit as st
# ---- 必须最先设置页面配置 ----
st.set_page_config(page_title="⚡ 自适应小微企业风控系统", layout="wide")

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
import json
import traceback
from datetime import datetime
import base64
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
# 从 utils 导入共享函数
from utils import apply_woe_map, ensure_dir
from pandas import Interval  # 用于判断WOE映射类型
import subprocess
import sys
warnings.filterwarnings('ignore')

# ===================== 确保必要目录存在 =====================

print("当前工作目录:", os.getcwd())
print("脚本所在目录:", os.path.dirname(os.path.abspath(__file__)))


ensure_dir("model")
ensure_dir("shap_images")
ensure_dir("predictions_history")


# ===================== 【安全读取】背景图并转为 Base64 字符串 =====================
def get_background_base64(img_name="可视化.png"):
    with open(img_name, "rb") as f:
        img_byte_data = f.read()
    return base64.b64encode(img_byte_data).decode()
bg_base64_str = get_background_base64("可视化.png")
bg_css_url = f"url('data:image/png;base64,{bg_base64_str}')"

# ===================== 1. 注入修复级科技感 CSS（使用拼接 + 修正顶部与侧边栏遮挡） =====================
# ===================== 1. 主 CSS（背景 + 全局样式） =====================
css_part_1 = """
<style>
    /* ---- 1. 全局超暗夜背景 ---- */
    .stApp, [data-testid="stAppViewContainer"] {
        background: #0a1229 !important;
        background-image: 
"""

css_part_2 = """
         !important;
        background-size: cover !important;         
        background-position: center 0% !important;
        background-repeat: no-repeat !important;
        color: #ffffff !important;
    }

    /* ---- 2. 全局文字色强制 ---- */
    h1, h2, h3, h4, h5, h6, span, p, label, div, strong, b {
        color: #e6f1ff !important;
    }
    h1, h2, h3 {
        color: #00e5ff !important;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.4) !important;
        border-bottom: 1px solid rgba(0, 229, 255, 0.2) !important;
        padding-bottom: 10px !important;
    }

    /* ---- 3. 折叠框 Expander 美化 ---- */
    .streamlit-expanderHeader {
        background: rgba(0, 30, 60, 0.8) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        color: #00e5ff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        backdrop-filter: blur(4px) !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(0, 229, 255, 0.15) !important;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2) !important;
    }
    .streamlit-expanderContent {
        background: rgba(8, 15, 36, 0.95) !important;
        border: 1px solid rgba(0, 229, 255, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        color: #ffffff !important;
        backdrop-filter: blur(4px) !important;
    }

    /* ---- 4. 下拉框 统一蓝色边框+青色箭头+深色弹窗 ---- */
    div[data-baseweb="select"] svg,
    div[data-testid="stNumberInput"] svg {
        fill: #00e5ff !important;
        color: #00e5ff !important;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiselect div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] {
        background: rgba(16, 30, 60, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid #00e5ff !important;
        border-radius: 4px !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] ul {
        background: #0a1229 !important;
        border: 1px solid #00e5ff !important;
        border-radius: 4px !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.8) !important;
        padding: 4px 0 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"],
    [data-baseweb="popover"] li[role="option"],
    [data-baseweb="popover"] [role="option"] {
        background: #0a1229 !important;
        color: #e6f1ff !important;
        padding: 8px 16px !important;
        transition: background 0.2s;
        font-size: 15px !important;
        list-style: none !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"]:hover,
    [data-baseweb="popover"] li[role="option"]:hover {
        background: rgba(0, 229, 255, 0.2) !important;
        color: #ffffff !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li[role="option"][aria-selected="true"],
    [data-baseweb="popover"] li[role="option"][aria-selected="true"] {
        background: rgba(0, 229, 255, 0.15) !important;
        color: #00e5ff !important;
    }
    div[data-baseweb="input"] input {
        color: #ffffff !important;
    }

    /* ---- 5. Number Input 数字输入框 哑光蓝边框 ---- */
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: rgba(16, 30, 60, 0.8) !important;
        border: 1px solid #00e5ff !important;
        border-radius: 4px !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    [data-testid="stNumberInput"] input[data-testid="stNumberInputField"] {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
        color: #e6f1ff !important;
    }
    [data-testid="stNumberInput"] button {
        background: rgba(0, 40, 80, 0.8) !important;
        border: 1px solid #00e5ff !important;
        color: #00e5ff !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background: rgba(0, 229, 255, 0.2) !important;
    }

    /* ---- 6. 文件上传框 FileUploader ---- */
    [data-testid="stFileUploader"] section {
        background: rgba(8, 20, 48, 0.6) !important;
        border: 1px dashed rgba(0, 229, 255, 0.4) !important;
        color: #e6f1ff !important;
        border-radius: 6px !important;
        backdrop-filter: blur(4px) !important;
    }
    [data-testid="stFileUploader"] section span {
        color: #e6f1ff !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] section button {
        background: #0a1229 !important;
        color: #e6f1ff !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 4px !important;
        padding: 6px 16px !important;
        transition: none !important;
    }
    [data-testid="stFileUploader"] button:hover,
    [data-testid="stFileUploader"] section button:hover {
        background: #0a1229 !important;
        color: #e6f1ff !important;
        border-color: rgba(0, 229, 255, 0.6) !important;
    }

    /* ---- 7. 警告/提示弹窗 Alert ---- */
    .stAlert {
        background: rgba(0, 40, 80, 0.6) !important;
        border-left: 4px solid #00e5ff !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.1) !important;
    }
    .stAlert > div { color: #ffffff !important; }

    /* ---- 8. Metric指标卡片 带扫描动画（通用背景不带 !important） ---- */
    div[data-testid="metric-container"] {
        background: rgba(8, 20, 48, 0.7);   /* 去掉 !important，允许后面覆盖 */
        position: relative !important;
        border: 1px solid transparent !important;
        border-radius: 6px !important;
        padding: 15px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6) !important;
        background-clip: padding-box !important;
        overflow: hidden !important;
    }
    div[data-testid="metric-container"]::before {
        content: '' !important;
        position: absolute !important;
        inset: -2px !important;
        z-index: -1 !important;
        background: linear-gradient(135deg, #00e5ff, transparent 30%, transparent 70%, #0055ff) !important;
        border-radius: 8px !important;
        padding: 2px !important;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0) !important;
        -webkit-mask-composite: xor !important;
        mask-composite: exclude !important;
    }
    div[data-testid="metric-container"]::after {
        content: '' !important;
        position: absolute !important;
        top: 0; left: -100%; width: 50%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.1), transparent) !important;
        transform: skewX(-25deg) !important;
        animation: scan 4s infinite linear !important;
        pointer-events: none !important;
    }
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 200%; }
    }
    div[data-testid="stMetricValue"] {
        color: #00e5ff !important;
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.5) !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #88c0ff !important;
    }

    /* ---- 专门覆盖高风险（第2列）和低风险（第4列）背景颜色 ---- */
    div[data-testid="column"]:nth-child(2) div[data-testid="metric-container"] {
        background: rgba(255, 0, 0, 0.25) !important;
        border-color: #ff0000 !important;
    }
    div[data-testid="column"]:nth-child(2) div[data-testid="stMetricValue"] {
        color: #ff5555 !important;
    }
    div[data-testid="column"]:nth-child(4) div[data-testid="metric-container"] {
        background: rgba(0, 255, 0, 0.25) !important;
        border-color: #00ff00 !important;
    }
    div[data-testid="column"]:nth-child(4) div[data-testid="stMetricValue"] {
        color: #55ff55 !important;
    }

    /* ---- 9. 侧边栏 Sidebar ---- */
    [data-testid="stSidebar"] {
        background: rgba(6, 14, 30, 0.2) !important;
        backdrop-filter: blur(8px) !important;
        border-right: 1px solid rgba(0, 229, 255, 0.2) !important;
        top: 60px !important;
        height: calc(100vh - 50px) !important;
        border-top: 1px solid rgba(0, 229, 255, 0.2) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #c8e0ff !important;
    }

    /* ---- 10. DataFrame 数据表格 ---- */
    .stDataFrame {
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 6px !important;
        background: rgba(8, 20, 48, 0.5) !important;
    }
    .stDataFrame th {
        background: rgba(0, 229, 255, 0.1) !important;
        color: #00e5ff !important;
        border-bottom: 1px solid #00e5ff !important;
    }
    .stDataFrame td {
        color: #ffffff !important;
    }

    /* ---- 11. 按钮 Button 美化 ---- */
    .stButton button {
        background: linear-gradient(135deg, #0077b6, #00e5ff) !important;
        color: #0b1026 !important;
        border: none !important;
        font-weight: bold !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.3) !important;
    }
    .stButton button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 25px rgba(0, 229, 255, 0.6) !important;
    }
    button[data-testid="stBaseButton-secondary"] {
        background: #0a1229 !important;
        color: #e6f1ff !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 4px !important;
        padding: 6px 16px !important;
        transition: none !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background: #0a1229 !important;
        color: #e6f1ff !important;
        border-color: rgba(0, 229, 255, 0.6) !important;
        box-shadow: 0 0 10px rgba(0, 229, 255, 0.2) !important;
    }
    button[data-testid="stBaseButton-secondary"]:active,
    button[data-testid="stBaseButton-secondary"]:focus {
        background: #0a1229 !important;
        color: #e6f1ff !important;
        border-color: rgba(0, 229, 255, 0.3) !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* ---- 12. 折叠面板全局深色兜底 ---- */
    [data-testid="stExpander"] {
        background: transparent !important;
    }
    [data-testid="stExpander"] details {
        background: rgba(8, 20, 48, 0.9) !important;
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
        border-radius: 6px !important;
    }
    [data-testid="stExpander"] summary {
        background: rgba(0, 30, 60, 0.8) !important;
        border: none !important;
        border-radius: 6px 6px 0 0 !important;
        color: #00e5ff !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: rgba(0, 229, 255, 0.15) !important;
    }
    [data-testid="stExpanderDetails"] {
        background: rgba(8, 20, 48, 0.9) !important;
        border-top: 1px solid rgba(0, 229, 255, 0.1) !important;
        padding: 10px !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* ---- 13. 隐藏原生菜单和底部，保留 header ---- */
    #MainMenu, footer {
        visibility: hidden !important;
    }

    /* ---- 14. 顶部栏修正 ---- */
    header[data-testid="stHeader"] {
        background: transparent !important;
        background-image: none !important;
        border-bottom: 1px solid rgba(0, 229, 255, 0.2) !important;
        box-shadow: none !important;
        visibility: visible !important;
        display: flex !important;
    }
    header[data-testid="stHeader"] * {
        background-color: transparent !important;
        background-image: none !important;
    }

    /* ---- 15. 侧边栏展开按钮图标青色 ---- */
    header[data-testid="stHeader"] button[data-testid="stExpandSidebarButton"] svg {
        fill: #00e5ff !important;
        color: #00e5ff !important;
    }
    button[data-testid="stExpandSidebarButton"] {
        background: transparent !important;
        border: none !important;
    }
    button[data-testid="stExpandSidebarButton"]:hover {
        background: rgba(0, 229, 255, 0.1) !important;
    }
</style>

<!-- JS兜底：打开下拉瞬间强制覆盖弹窗内联白色背景 -->
<script>
function fixDropdownDark() {
    const lists = document.querySelectorAll('ul[data-testid="stSelectboxVirtualDropdown"], [data-baseweb="popover"] [role="listbox"]');
    lists.forEach(el=>{
        el.style.background = "#0a1229";
        el.style.border = "1px solid #00e5ff";
        const items = el.querySelectorAll('[role="option"]');
        items.forEach(item=>{
            item.style.background = "#0a1229";
            item.style.color = "#e6f1ff";
        })
    })
}
document.addEventListener('click', ()=>setTimeout(fixDropdownDark, 40));
window.addEventListener('load', fixDropdownDark);
new MutationObserver(()=>setTimeout(fixDropdownDark,30)).observe(document.body,{childList:true,subtree:true})
</script>
"""

# 注入主 CSS（注意使用拼接方式）
st.markdown(css_part_1 + bg_css_url + css_part_2, unsafe_allow_html=True)

# ===================== 2. 追加 header 背景图固定 =====================
st.markdown(
    f"""
    <style>
        header[data-testid="stHeader"] {{
            background-image: {bg_css_url} !important;
            background-size: cover !important;
            background-position: center 0% !important;
            background-attachment: fixed !important;
            background-color: rgba(6, 14, 30, 0.4) !important;
            backdrop-filter: blur(2px) !important;
            border-bottom: 1px solid rgba(0, 229, 255, 0.3) !important;
            box-shadow: 0 2px 15px rgba(0,0,0,0.4) !important;
        }}
        header[data-testid="stHeader"] * {{
            background-color: transparent !important;
            background-image: none !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== 3. 侧边栏导航字体立体感 =====================
st.markdown("""
<style>
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        text-shadow: 0 1px 0 #001122, 0 2px 0 #001122, 0 3px 3px rgba(0,0,0,0.8) !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        letter-spacing: 0.5px !important;
        padding: 8px 12px !important;
        border-radius: 4px !important;
        transition: all 0.3s ease !important;
        color: #88c0ff !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(0, 229, 255, 0.1) !important;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.5) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] {
        color: #00e5ff !important;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.8), 0 0 40px rgba(0, 229, 255, 0.4) !important;
        background: rgba(0, 229, 255, 0.08) !important;
        border-left: 3px solid #00e5ff !important;
        box-shadow: inset 0 0 20px rgba(0, 229, 255, 0.1) !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 4. 修复 st.json / st.code / st.Alert 等组件 =====================
st.markdown("""
<style>
    div[data-testid="stJson"] {
        background: rgba(8, 20, 48, 0.95) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 6px !important;
        padding: 12px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    }
    div[data-testid="stJson"] .react-json-view {
        background: transparent !important;
        color: #e6f1ff !important;
    }
    div[data-testid="stJson"] .react-json-view .object-key { color: #00e5ff !important; }
    div[data-testid="stJson"] .react-json-view .string-value,
    div[data-testid="stJson"] .react-json-view .number-value,
    div[data-testid="stJson"] .react-json-view .boolean-value { color: #e6f1ff !important; }
    div[data-testid="stJson"] .react-json-view .icon-container { fill: #00e5ff !important; }
    div[data-testid="stJson"] * { background-color: transparent !important; }

    div[data-testid="stCode"] {
        background: #0a1229 !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-radius: 6px !important;
        padding: 12px !important;
    }
    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code {
        background: #0a1229 !important;
        color: #00e5ff !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        white-space: pre-wrap !important;
    }
    div[data-testid="stCode"] ::-webkit-scrollbar { background: #0a1229; }
    div[data-testid="stCode"] ::-webkit-scrollbar-thumb { background: #00e5ff; border-radius: 4px; }

    div[data-testid="stAlert"] {
        background: rgba(0, 40, 80, 0.85) !important;
        color: #e6f1ff !important;
        border: 1px solid rgba(0, 229, 255, 0.4) !important;
        border-radius: 6px !important;
        backdrop-filter: blur(4px) !important;
    }
    div[data-testid="stAlert"] > div,
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span { color: #e6f1ff !important; }

    .stMarkdown code {
        background: rgba(0, 40, 80, 0.6) !important;
        color: #00e5ff !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid rgba(0, 229, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 5. 列可见性菜单（白底黑字） =====================
st.markdown("""
<style>
    [data-testid="stDataFrameColumnVisibilityMenu"],
    [data-testid="stDataFrameColumnVisibilityMenu"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    [data-testid="stDataFrameColumnVisibilityMenu"] * {
        color: #000000 !important;
    }
    [data-testid="stDataFrameColumnVisibilityMenu"] div:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    [data-testid="stDataFrameColumnVisibilityMenu"] [aria-selected="true"] {
        background-color: #e0e0e0 !important;
        color: #000000 !important;
    }
    [data-testid="stDataFrameColumnVisibilityMenu"] ::-webkit-scrollbar-thumb {
        background: #888 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 6. 表格内部搜索栏、工具栏、全屏工具栏图标 =====================
st.markdown("""
<style>
    .gdg-search-status label,
    .gdg-search-status span,
    .gdg-search-bar label,
    .gdg-search-bar span {
        color: #000000 !important;
    }
    [data-testid="stDataFrame"] .stSelectbox label,
    [data-testid="stDataFrame"] .stMultiSelect label,
    [data-testid="stDataFrame"] label {
        color: #000000 !important;
    }
    .gdg-search-bar input {
        color: #000000 !important;
        background: #ffffff !important;
    }
    .gdg-table-toolbar span,
    .gdg-table-toolbar div {
        color: #000000 !important;
    }
    .gdg-table-toolbar svg {
        fill: #000000 !important;
        color: #000000 !important;
    }

    /* 表格工具栏图标（普通模式） */
    .stElementToolbar svg {
        fill: #333333 !important;
        color: #333333 !important;
    }
    /* 全屏模式下的表格工具栏图标 */
    [data-testid="stFullScreenFrame"] .stElementToolbar svg {
        fill: #000000 !important;
        color: #000000 !important;
    }
    .stElementToolbar button:hover svg,
    [data-testid="stFullScreenFrame"] .stElementToolbar button:hover svg {
        fill: #666666 !important;
        color: #666666 !important;
    }
    .stElementToolbar button {
        background: transparent !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ===================== 7. 通用 Tooltip（白底黑字） =====================
st.markdown("""
<style>
    [data-baseweb="tooltip"],
    [role="tooltip"],
    .stTooltipContent {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
        border-radius: 4px !important;
        padding: 4px 8px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
        font-size: 14px !important;
    }
    [data-baseweb="tooltip"] *,
    [role="tooltip"] *,
    .stTooltipContent * {
        color: #000000 !important;
        background: transparent !important;
    }
    /* 全屏模式下的 Tooltip */
    [data-testid="stFullScreenFrame"] [data-baseweb="tooltip"],
    [data-testid="stFullScreenFrame"] [role="tooltip"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* 帮助问号图标改为青色 */
button[aria-label^="Help for"] svg {
    stroke: #00e5ff !important;
}
</style>
""", unsafe_allow_html=True)


# ===================== 3. 定义赛博朋克图表配置 =====================
TECH_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#00e5ff', 'family': 'Segoe UI, sans-serif'},
    'xaxis': {
        'showgrid': True, 'gridcolor': 'rgba(0, 229, 255, 0.15)',
        'zeroline': False, 'showline': True, 'linecolor': '#00e5ff',
        'tickfont': {'color': '#88c0ff'}
    },
    'yaxis': {
        'showgrid': True, 'gridcolor': 'rgba(0, 229, 255, 0.15)',
        'zeroline': False, 'showline': True, 'linecolor': '#00e5ff',
        'tickfont': {'color': '#88c0ff'}
    },
    'legend': {
        'font': {'color': '#88c0ff'},
        'bgcolor': 'rgba(8, 20, 48, 0.8)',
        'bordercolor': 'rgba(0, 229, 255, 0.2)'
    }
}

# ===================== 初始化 & 函数定义 =====================
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_data_source' not in st.session_state:
    st.session_state.prediction_data_source = "默认测试集"
if 'train_config_saved' not in st.session_state:
    st.session_state.train_config_saved = False

HISTORY_DIR = "predictions_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_prediction_to_file(df, source_name):
    try:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype(str)
            elif pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
            else:
                df_clean[col] = df_clean[col].astype(str)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"pred_{timestamp}"
        parquet_path = os.path.join(HISTORY_DIR, f"{base_name}.parquet")
        meta_path = os.path.join(HISTORY_DIR, f"{base_name}.json")
        
        df_clean.to_parquet(parquet_path, index=False, engine='pyarrow')
        
        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source_name,
            "rows": len(df_clean),
            "columns": df_clean.columns.tolist(),
            "file": parquet_path
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta
    except Exception as e:
        print(traceback.format_exc())
        st.error(f"❌ 保存失败：{str(e)}\n\n请查看终端输出的详细信息。")
        return None

def list_history_records():
    records = []
    for fname in os.listdir(HISTORY_DIR):
        if fname.endswith('.json'):
            json_path = os.path.join(HISTORY_DIR, fname)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                parquet_path = meta.get('file')
                if parquet_path and os.path.exists(parquet_path):
                    records.append(meta)
            except:
                continue
    records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return records

def delete_record(meta):
    parquet_path = meta.get('file')
    if parquet_path and os.path.exists(parquet_path):
        os.remove(parquet_path)
    json_path = parquet_path.replace('.parquet', '.json') if parquet_path else None
    if json_path and os.path.exists(json_path):
        os.remove(json_path)

@st.cache_resource
def load_models():
    try:
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
        with open('model/best_threshold.pkl', 'rb') as f:
            best_threshold = pickle.load(f)
        with open('model/config.json', 'r') as f:
            config = json.load(f)
        # ---- 新增：加载 selected_features ----
        try:
            with open('model/selected_features.pkl', 'rb') as f:
                selected_features = pickle.load(f)
        except FileNotFoundError:
            selected_features = config.get('feature_cols', [])  # 降级方案
        return base_models, scaler_lr, woe_maps, lr_features, tree_features, best_threshold, config, selected_features
    except FileNotFoundError:
        return None, None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"加载模型时发生错误: {str(e)}")
        return None, None, None, None, None, None, None, None

base_models, scaler_lr, woe_maps, lr_features, tree_features, best_threshold, config, selected_features = load_models()
# ---- 读取 use_woe_for_lr 标志 ----
# 防御：如果config为None，设为空字典
if config is None:
    config = {}
use_woe_for_lr = config.get('use_woe_for_lr', True)   # 默认True（兼容旧模型）
if selected_features is None:
    selected_features = config.get('feature_cols', [])



model_loaded = base_models is not None
if not model_loaded:
    st.sidebar.warning("⚠️ 未找到模型文件，请先训练模型")

@st.cache_data
def load_categorical_options():
    try:
        with open('model/categorical_options.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}
categorical_options = load_categorical_options()

# ---------- WOE映射辅助函数 ----------


def preprocess_single(raw_data):
    df = pd.DataFrame([raw_data])
    
    # ---- 构建LR特征 ----
    if use_woe_for_lr:
        # WOE 编码模式（与训练时 use_woe_for_lr=True 一致）
        X_lr = pd.DataFrame()
        for col in lr_features:          # lr_features 是原始特征名
            if col in woe_maps and col in df.columns:
                X_lr[col] = apply_woe_map(df[col], woe_maps[col])
            elif col in df.columns:
                try:
                    X_lr[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                except:
                    X_lr[col] = 0
            else:
                X_lr[col] = 0
    else:
        # 独热编码模式（与训练时 use_woe_for_lr=False 一致）
        # 1) 对缺失的列填充默认值（数值0，分类 'Other'）
        for col in selected_features:
            if col not in df.columns:
                if col in categorical_options:
                    df[col] = 'Other'
                else:
                    df[col] = 0
        # 2) 对 selected_features 做独热编码（与训练时完全一致）
        X_lr_raw = pd.get_dummies(df[selected_features], drop_first=True)
        # 3) 确保列与训练时一致（lr_features 是独热列名）
        for col in lr_features:
            if col not in X_lr_raw.columns:
                X_lr_raw[col] = 0
        X_lr = X_lr_raw[lr_features]   # 按顺序选择

    # 标准化
    if scaler_lr is not None:
        X_lr_scaled = scaler_lr.transform(X_lr)
    else:
        X_lr_scaled = np.zeros((len(X_lr), max(1, len(X_lr.columns))))
    
    # ---- 构建树特征（始终使用独热编码，与训练一致） ----
    X_tree_raw_df = pd.get_dummies(df[selected_features], drop_first=True)
    for col in tree_features:
        if col not in X_tree_raw_df.columns:
            X_tree_raw_df[col] = 0
    X_tree = X_tree_raw_df[tree_features]
    X_tree_raw = X_tree.values
    
    return X_lr_scaled, X_tree_raw

def predict_single(raw_data):
    if not model_loaded:
        return {'LR': 0.5, 'XGB': 0.5, 'LGB': 0.5, 'Stacking': 0.5, 'pred': 0}
    X_lr_scaled, X_tree_raw = preprocess_single(raw_data)
    lr_prob = base_models['LogisticRegression'].predict_proba(X_lr_scaled)[0, 1]
    xgb_prob = base_models['XGBoost'].predict_proba(X_tree_raw)[0, 1]
    lgb_prob = base_models['LightGBM'].predict_proba(X_tree_raw)[0, 1]
    X_meta = np.array([[lr_prob, xgb_prob, lgb_prob]])
    stacking_prob = base_models['Stacking'].predict_proba(X_meta)[0, 1]
    pred = int(stacking_prob >= best_threshold)
    return {'LR': lr_prob, 'XGB': xgb_prob, 'LGB': lgb_prob, 'Stacking': stacking_prob, 'pred': pred}

def get_risk_level(prob):
    if prob < 0.3: return "低风险", "🟢"
    elif prob < 0.5: return "中低风险", "🟡"
    elif prob < 0.7: return "中高风险", "🟠"
    else: return "高风险", "🔴"

def run_script_with_logging(script_name, status, log_placeholder):
    """
    执行指定的 Python 脚本，实时捕获输出并显示在 status 和 log_placeholder 中。
    返回 (success, log_lines)
    """
    if not os.path.exists(script_name):
        status.update(label=f"❌ 脚本 {script_name} 不存在", state="error")
        return False, []

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            env=env,
            cwd=os.getcwd()
        )
    except Exception as e:
        status.update(label=f"❌ 启动进程失败: {e}", state="error")
        return False, []

    log_lines = []
    for line in process.stdout:
        log_lines.append(line)
        # 实时显示最后30行，便于跟踪进度
        display_lines = log_lines[-30:]
        log_placeholder.code(''.join(display_lines), language='bash')
        status.write(line.strip())

    process.wait()
    if process.returncode == 0:
        return True, log_lines
    else:
        status.update(label=f"❌ 脚本退出码 {process.returncode}", state="error")
        return False, log_lines

def reload_models():
    """清除缓存并重新加载模型，更新全局变量"""
    global base_models, scaler_lr, woe_maps, lr_features, tree_features, best_threshold, config, selected_features, model_loaded
    st.cache_resource.clear()
    base_models, scaler_lr, woe_maps, lr_features, tree_features, best_threshold, config, selected_features = load_models()
    model_loaded = base_models is not None

# ===================== 侧边栏 =====================
st.sidebar.title("⚙️ 系统控制台")
if model_loaded:
    st.sidebar.markdown("**自适应参数调整**")
    iv_threshold = st.sidebar.slider("IV筛选阈值", 0.0, 0.1, float(config.get("iv_threshold", 0.02)), 0.005)
    vif_threshold = st.sidebar.slider("VIF剔除阈值", 3, 10, int(config.get("vif_threshold", 5)), 1)
    use_woe = st.sidebar.checkbox("LR使用WOE编码", value=config.get("use_woe_for_lr", True))
    st.sidebar.info(f"⚡ 当前配置：IV>{iv_threshold}, VIF>{vif_threshold}, WOE={use_woe}")
else:
    st.sidebar.info("请先训练模型以调整参数")

page = st.sidebar.radio(
    "导航",
    ["📊 单体评估", "📈 全域大屏", "📉 模型监控", "🚀 模型训练", "📁 数据管理", "🧠 可解释性分析"]
)

# ===================== 页面1：单体评估 =====================
if page == "📊 单体评估":
    st.title("📊 单体风险评估")
    if not model_loaded:
        st.warning("⚠️ 模型未加载，请先训练模型")
        st.stop()
    
    with st.expander("📋 当前模型配置与特征列表", expanded=False):
        if config:
            col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
            with col_cfg1:
                st.metric("IV 筛选阈值", f"{config.get('iv_threshold', 'N/A')}")
            with col_cfg2:
                st.metric("VIF 剔除阈值", f"{config.get('vif_threshold', 'N/A')}")
            with col_cfg3:
                woe_val = config.get('use_woe_for_lr', 'N/A')
                st.metric("LR 使用 WOE 编码", "✅ 是" if woe_val else "❌ 否")
            st.caption(f"🎯 目标列：{config.get('target_col', '未知')}")
            feature_cols = config.get('feature_cols', [])
            st.caption(f"📌 模型使用的特征（共 {len(feature_cols)} 个）：")
            if feature_cols:
                for f in feature_cols:
                    if f in categorical_options:
                        st.caption(f"  • {f} (分类)")
                    else:
                        st.caption(f"  • {f} (数值)")
            else:
                st.caption("（无特征信息）")
        else:
            st.info("未找到配置文件")
    
    st.markdown("---")
    st.markdown("输入企业信息，一键评估违约风险")
    
    feature_cols = config.get('feature_cols', [])
    if not feature_cols:
        st.warning("未找到特征列表，使用默认表单")
        col1, col2 = st.columns(2)
        with col1:
            business_sector = st.selectbox("行业", ['agriculture','manufacturing','services','retail_trade','technology'])
            business_state = st.selectbox("州", ['Lagos','Abuja','Kano','Rivers','Kaduna'])
            lender = st.selectbox("银行", ['Zenith Bank','Access Bank','First Bank','GTBank','UBA'])
            years_in_business = st.number_input("经营年限", 0, 50, 3)
            num_employees = st.number_input("员工数", 1, 500, 10)
        with col2:
            annual_revenue = st.number_input("年营收", 100000, 100000000, 5000000)
            principal = st.number_input("贷款本金", 100000, 50000000, 1000000)
            interest_rate = st.slider("年利率", 0.05, 0.35, 0.22, 0.01)
            tenor = st.selectbox("期限(月)", [3,6,12,18,24,36])
            collateral = st.number_input("抵押物价值", 0, 50000000, 1500000)
            credit_score = st.number_input("信用评分", 300, 850, 650)
            days_to_disburse = st.number_input("放款天数", 0, 90, 15)
        if st.button("预测"):
            raw = {
                'business_sector': business_sector,
                'business_state': business_state,
                'lender': lender,
                'years_in_business': years_in_business,
                'num_employees': num_employees,
                'annual_revenue_ngn': annual_revenue,
                'principal_ngn': principal,
                'interest_rate_annual': interest_rate,
                'tenor_months': tenor,
                'collateral_value_ngn': collateral,
                'credit_score': credit_score,
                'days_to_disburse': days_to_disburse,
                'application_date': '2024-01-15',
                'disbursement_date': '2024-02-01'
            }
            res = predict_single(raw)
            risk_level, icon = get_risk_level(res['Stacking'])
            st.metric("违约概率", f"{res['Stacking']:.2%}")
            st.metric("风险等级", f"{icon} {risk_level}")
            st.metric("预测结果", "违约" if res['pred'] else "正常")
            fig = px.bar(x=['LR','XGB','LGB','Stacking'], 
                         y=[res['LR'],res['XGB'],res['LGB'],res['Stacking']],
                         color=[res['LR'],res['XGB'],res['LGB'],res['Stacking']],
                         color_continuous_scale='RdYlGn_r', range_color=[0,1])
            fig.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
    else:
        # ---- 有特征列表时，按类型分组 ----
        num_features = []
        cat_features = []
        date_features = []
        for col_name in feature_cols:
            if col_name in ['application_date', 'disbursement_date']:
                date_features.append(col_name)
            elif col_name in categorical_options:
                cat_features.append(col_name)
            else:
                num_features.append(col_name)
        
        raw_input = {}
        
        # 数字输入框
        if num_features:
            st.subheader("📊 数值特征")
            col1, col2 = st.columns(2)
            for i, col_name in enumerate(num_features):
                with col1 if i % 2 == 0 else col2:
                    default_val = 0.0
                    if 'revenue' in col_name.lower() or 'principal' in col_name.lower() or 'collateral' in col_name.lower():
                        default_val = 1000000.0
                    elif 'years' in col_name.lower():
                        default_val = 3.0
                    elif 'employees' in col_name.lower():
                        default_val = 10.0
                    elif 'score' in col_name.lower():
                        default_val = 650.0
                    elif 'rate' in col_name.lower():
                        default_val = 0.22
                    elif 'tenor' in col_name.lower():
                        default_val = 12.0
                    value = st.number_input(f"{col_name}", value=default_val, key=f"num_{col_name}")
                    raw_input[col_name] = value
        
        # 分类下拉框
        if cat_features:
            st.subheader("📋 分类特征")
            col1, col2 = st.columns(2)
            for i, col_name in enumerate(cat_features):
                with col1 if i % 2 == 0 else col2:
                    if col_name in categorical_options:
                        options = categorical_options[col_name].copy()
                    else:
                        options = list(woe_maps[col_name].keys()).copy() if col_name in woe_maps else []
                    if "Other" not in options:
                        options.append("Other")
                    value = st.selectbox(f"{col_name}", options, index=0, key=f"cat_{col_name}")
                    raw_input[col_name] = value
        
        # 日期输入框
        if date_features:
            st.subheader("📅 日期特征")
            col1, col2 = st.columns(2)
            for i, col_name in enumerate(date_features):
                with col1 if i % 2 == 0 else col2:
                    default_date = pd.to_datetime('2024-01-15') if 'application' in col_name else pd.to_datetime('2024-02-01')
                    value = st.date_input(f"{col_name}", value=default_date, key=f"date_{col_name}")
                    raw_input[col_name] = str(value)
        
        # 预测按钮
        if st.button("预测"):
            raw = raw_input.copy()
            res = predict_single(raw)
            risk_level, icon = get_risk_level(res['Stacking'])
            st.metric("违约概率", f"{res['Stacking']:.2%}")
            st.metric("风险等级", f"{icon} {risk_level}")
            st.metric("预测结果", "违约" if res['pred'] else "正常")
            fig = px.bar(x=['LR','XGB','LGB','Stacking'], 
                         y=[res['LR'],res['XGB'],res['LGB'],res['Stacking']],
                         color=[res['LR'],res['XGB'],res['LGB'],res['Stacking']],
                         color_continuous_scale='RdYlGn_r', range_color=[0,1])
            fig.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

# ===================== 页面2：全域大屏 =====================
elif page == "📈 全域大屏":
    st.title("📈 全域风控大屏")
    
    def render_dashboard(df, data_source):
        if df.empty:
            st.warning("数据为空，无法展示")
            return
        
        # ---- 确保有 risk 和 风险等级列 ----
        if 'risk' not in df.columns:
            if '违约概率' in df.columns:
                df['risk'] = df['违约概率']
            else:
                st.error("数据中缺少风险概率列，无法进行可视化")
                return
        if '风险等级' not in df.columns:
            df['风险等级'] = df['risk'].apply(lambda x: get_risk_level(x)[0])
        
        # =========================================================
        # 【核心改动】基于“预测结果数据集”动态提取特征列
        # 不再强依赖 config['feature_cols']，而是看 df 里有什么
        # =========================================================
        # 1. 定义需要排除的列（预测结果列、ID列、日期衍生列等）
        exclude_cols = ['risk', '风险等级', '违约概率', '预测结果', 'pred',
                        'loan_id', 'business_id', 'application_date', 'disbursement_date',
                        'application_month', 'application_quarter', 'days_to_disburse']
        # 2. 获取所有可用列（排除上述列 + 纯数值型的ID列）
        all_cols = [c for c in df.columns if c not in exclude_cols]
        # 3. 进一步过滤：如果某一列全是 NaN 或只有唯一值，则忽略（对绘图无意义）
        available_features = []
        for col in all_cols:
            if df[col].nunique(dropna=False) <= 1:
                continue
            available_features.append(col)
        
        # ---- 先初始化分类和数值特征列表（必须在打印之前） ----
        cat_features = []
        num_features = []
        
        
        if not available_features:
            st.warning("预测结果数据中不包含任何有意义的特征列，无法生成图表")
            return
        
        # ---- 按数据类型自动识别分类特征和数值特征 ----
        for feat in available_features:
            # 如果列是 object / string 类型，或者取值数量少于 12 个（大概率是分类），视为分类
            if df[feat].dtype == 'object':
                cat_features.append(feat)
            elif df[feat].dtype in ['int64', 'float64']:
                # 数值类型中，如果唯一值很少（比如 < 10），也可以视为分类（但这里保守处理，留给用户选择）
                if df[feat].nunique() < 10:
                    # 放入分类，让用户也可以用来分组
                    cat_features.append(feat)
                else:
                    num_features.append(feat)
            else:
                # 其他类型（如日期）先忽略
                pass
        
        # 如果分类特征太多，给一个提示（避免下拉框过长）
        if len(cat_features) > 20:
            st.info(f"ℹ️ 检测到 {len(cat_features)} 个分类特征，下拉框可能较长，建议从业务角度选择最重要的维度。")
        
        # ---- 1. 核心指标卡（6 个） ----
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        total = len(df)
        high_risk = len(df[df['风险等级'] == '高风险'])
        medium_risk = len(df[df['风险等级'] == '中高风险'])  # 中高风险作为代表
        low_risk = len(df[df['风险等级'] == '低风险'])
        avg_risk = df['risk'].mean()
        threshold = best_threshold if best_threshold is not None else 0.5
        reject_count = (df['risk'] > threshold).sum()
        
        with col1:
            st.metric("申报总户数", total)
        with col2:
            st.metric("高风险户数", high_risk, delta=f"{high_risk/total:.1%}" if total>0 else "")
        with col3:
            st.metric("中风险户数", medium_risk, delta=f"{medium_risk/total:.1%}" if total>0 else "")
        with col4:
            st.metric("低风险户数", low_risk, delta=f"{low_risk/total:.1%}" if total>0 else "")
        with col5:
            st.metric("平均违约概率", f"{avg_risk:.2%}")
        
        st.divider()
        
        # ---- 2. 环形图 + 直方图 ----
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("风险等级占比")
            risk_counts = df['风险等级'].value_counts()
            all_levels = ['低风险', '中低风险', '中高风险', '高风险']
            for level in all_levels:
                if level not in risk_counts.index:
                    risk_counts[level] = 0
            risk_counts = risk_counts[all_levels]
            fig_pie = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'低风险':'#4CAF50','中低风险':'#FFEB3B',
                                    '中高风险':'#FF9800','高风险':'#F44336'},
                hole=0.4
            )
            fig_pie.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("申报客户风险评分分布")
            fig_hist = px.histogram(
                df, x='risk', nbins=30,
                color_discrete_sequence=['#00e5ff'],
                labels={'risk': '违约概率', 'count': '企业数量'}
            )
            if best_threshold:
                fig_hist.add_vline(x=best_threshold, line_dash="dash", line_color="#FF4500", annotation_text="拒绝阈值")
            fig_hist.add_vline(x=0.3, line_dash="dot", line_color="#4CAF50", annotation_text="低风险")
            fig_hist.add_vline(x=0.7, line_dash="dot", line_color="#F44336", annotation_text="高风险")
            fig_hist.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20),
                                yaxis_title="企业数量", xaxis_title="违约概率")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        
        # ---- 3. 分类特征平均风险对比 ----
        if cat_features:
            st.subheader("各分类维度平均违约概率对比")
            st.caption("💡 选择分类维度以对比平均风险")
            selected_cat = st.selectbox("选择分类维度", options=cat_features, key="dashboard_cat_select")
            if selected_cat:
                # 如果类别太多，提示可能显示杂乱
                if df[selected_cat].nunique() > 15:
                    st.info(f"ℹ️ 该特征有 {df[selected_cat].nunique()} 个不同取值，柱状图可能较拥挤，建议选择取值较少的分类特征。")
                cat_risk = df.groupby(selected_cat)['risk'].mean().sort_values(ascending=False).reset_index()
                fig_cat = px.bar(
                    cat_risk, x=selected_cat, y='risk',
                    color='risk', color_continuous_scale='RdYlGn_r',
                    labels={'risk': '平均违约概率', selected_cat: '类别'}
                )
                fig_cat.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("当前预测结果数据中无分类特征，跳过此图表")
        
        st.divider()
        
        
        # ---- 5. 散点气泡图（用户选择 X、Y、气泡大小） ----
        st.subheader("多维风险散点气泡图")
        if len(num_features) >= 2:
            # 自动推荐（基于常见关键词，但用户可自由更改）
            default_x = None
            default_y = None
            default_size = None
            for col in num_features:
                if 'revenue' in col.lower() or 'income' in col.lower() or 'turnover' in col.lower():
                    default_x = col
                    break
            if default_x is None:
                default_x = num_features[0]
            for col in num_features:
                if 'principal' in col.lower() or 'loan' in col.lower() or 'amount' in col.lower():
                    default_size = col
                    break
            if default_size is None:
                default_size = num_features[1] if len(num_features) > 1 else num_features[0]
            for col in num_features:
                if 'asset' in col.lower() or 'debt' in col.lower() or 'liability' in col.lower() or 'leverage' in col.lower():
                    default_y = col
                    break
            if default_y is None:
                candidates = [c for c in num_features if c != default_x]
                default_y = candidates[0] if candidates else default_x
            
            col_x = st.selectbox(
                "选择 X 轴（建议选择营收、资产总额等反映企业规模的指标）",
                options=num_features,
                index=num_features.index(default_x) if default_x in num_features else 0,
                key="scatter_x",
                help="建议选择营收、资产总额等反映企业规模的指标"
            )
            col_y = st.selectbox(
                "选择 Y 轴（建议选择资产负债率、贷款/营收比等反映偿债风险的指标）",
                options=num_features,
                index=num_features.index(default_y) if default_y in num_features else min(1, len(num_features)-1),
                key="scatter_y",
                help="建议选择资产负债率、贷款/营收比等反映偿债风险的指标"
            )
            col_size = st.selectbox(
                "选择气泡大小（建议选择贷款金额、授信额度等关注度指标）",
                options=num_features,
                index=num_features.index(default_size) if default_size in num_features else min(1, len(num_features)-1),
                key="scatter_size",
                help="建议选择贷款金额、授信额度等关注度指标"
            )
            
            # 验证：X 和 Y 不能相同
            if col_x == col_y:
                st.warning("⚠️ X 轴和 Y 轴不能相同，请重新选择，否则无法绘制有意义的散点图。")
            else:
                plot_df = df.copy()
                fig_scatter = px.scatter(
                    plot_df,
                    x=col_x,
                    y=col_y,
                    size=col_size,
                    color='风险等级',
                    color_discrete_map={'低风险':'#4CAF50','中低风险':'#FFEB3B',
                                        '中高风险':'#FF9800','高风险':'#F44316'},
                    hover_name=plot_df.index,
                    labels={col_x: col_x, col_y: col_y, col_size: col_size},
                    size_max=30
                )
                fig_scatter.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("当前预测结果数据中数值特征少于2个，无法绘制散点图")
        
        st.divider()
        
        # ---- 6. 箱线图（用户选择数值特征与风险等级的关系） ----
        if num_features:
            st.subheader("关键数值特征与风险等级的关系")
            st.caption("💡 选择多个数值特征，箱线图将展示不同风险等级下这些特征的分布差异。")
            selected_nums = st.multiselect(
                "选择要展示的数值特征",
                options=num_features,
                default=num_features[:3] if len(num_features)>=3 else num_features,
                key="dashboard_num_select",
                help="选择数值特征，查看其在不同风险等级下的分布差异"
            )
            if selected_nums:
                df_melt = df.melt(id_vars=['风险等级'], value_vars=selected_nums,
                                var_name='特征', value_name='值')
                fig_box = px.box(df_melt, x='特征', y='值', color='风险等级',
                            color_discrete_map={'低风险':'#4CAF50','中低风险':'#FFEB3B',
                                                '中高风险':'#FF9800','高风险':'#F44336'})
                fig_box.update_layout(**TECH_THEME, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_box, use_container_width=True)
        
        st.divider()
        
        # ---- 7. 高风险企业列表 ----
        st.subheader("🚨 高风险企业全量列表")
        high_risk_df = df[df['风险等级'] == '高风险'].copy()
        if len(high_risk_df) > 0:
            high_risk_df = high_risk_df.sort_values('risk', ascending=False)
            # 展示所有可用特征（不含排除列）
            display_cols = [c for c in available_features if c in high_risk_df.columns] + ['risk', '风险等级']
            display_df = high_risk_df[display_cols].copy()
            if 'risk' in display_df.columns:
                display_df['违约概率'] = display_df['risk'].apply(lambda x: f"{x:.2%}")
                display_df.drop(columns=['risk'], inplace=True)
            st.dataframe(display_df, use_container_width=True, height=500)
            st.caption(f"共 {len(high_risk_df)} 家高风险企业")
        else:
            st.success("🎉 暂无高风险企业！")
        
        # ---- 8. 数据概况 ----
        with st.expander("📊 预测结果数据概况"):
            st.write("**数值特征描述统计**")
            if num_features:
                st.dataframe(df[num_features].describe(), use_container_width=True)
            else:
                st.info("无数值特征")
            st.write("**分类特征取值分布（前5项）**")
            for col in cat_features[:3]:
                st.write(f"**{col}**")
                st.write(df[col].value_counts().head())
        
        st.caption(f"数据来源：{data_source} | 共 {len(df)} 条记录")
    
    data_options = [("current", "📌 当前预测结果")]
    history_records = list_history_records()
    for idx, rec in enumerate(history_records):
        label = f"📜 {rec.get('timestamp', '')} - {rec.get('source', '')} ({rec.get('rows', 0)}条)"
        data_options.append((f"history_{idx}", label))
    
    selected_key = st.selectbox(
        "选择数据源",
        options=[k for k, _ in data_options],
        format_func=lambda k: dict(data_options)[k],
        index=0
    )
    
    df_display = None
    data_source = ""
    if selected_key == "current":
        if st.session_state.prediction_result is not None and len(st.session_state.prediction_result) > 0:
            df_display = st.session_state.prediction_result.copy()
            data_source = st.session_state.prediction_data_source
        else:
            try:
                df_test = pd.read_parquet('test_data.parquet')
                df_display = df_test.sample(n=min(5000, len(df_test)), random_state=42)
                data_source = "测试集抽样（5000条）"
            except:
                try:
                    df_raw = pd.read_parquet('nigerian_sme_loans_full.parquet')
                    df_display = df_raw.sample(n=min(1000, len(df_raw)), random_state=42)
                    data_source = "原始数据抽样（1000条）"
                except:
                    st.error("❌ 未找到任何数据文件，请检查数据路径")
                    st.stop()
    else:
        idx = int(selected_key.split("_")[1])
        meta = history_records[idx]
        parquet_path = meta.get('file')
        if parquet_path and os.path.exists(parquet_path):
            try:
                df_display = pd.read_parquet(parquet_path)
                data_source = f"历史记录: {meta.get('timestamp', '')} ({meta.get('source', '')})"
            except Exception as e:
                st.error(f"加载历史数据失败: {e}")
                st.stop()
        else:
            st.error("历史数据文件不存在，请检查")
            st.stop()
    
    if df_display is not None:
        if 'risk' not in df_display.columns and '违约概率' in df_display.columns:
            df_display['risk'] = df_display['违约概率']
        elif 'risk' not in df_display.columns:
            with st.spinner("正在对数据预测风险..."):
                risks = []
                for _, row in df_display.iterrows():
                    try:
                        raw = row.to_dict()
                        if 'application_date' not in raw:
                            raw['application_date'] = '2024-01-15'
                        if 'disbursement_date' not in raw:
                            raw['disbursement_date'] = '2024-02-01'
                        result = predict_single(raw)
                        risks.append(result['Stacking'])
                    except:
                        risks.append(0.5)
                df_display['risk'] = risks
        
        render_dashboard(df_display, data_source)
        
        if selected_key == "current":
            if st.button("🗑️ 清除预测结果，恢复默认", use_container_width=True):
                st.session_state.prediction_result = None
                st.session_state.prediction_data_source = "默认测试集"
                st.rerun()

# ===================== 页面3：模型监控 =====================
elif page == "📉 模型监控":
    st.title("📉 模型监控 - 全面评估")

    if not model_loaded:
        st.warning("⚠️ 模型未加载，请先训练模型")
        st.stop()

    # ========== 1. 原有部分：性能表格 + 最佳阈值 ==========
    try:
        perf = pd.read_csv('model/model_performance.csv', index_col=0)
        st.dataframe(perf)
    except:
        st.warning("未找到性能文件 model/model_performance.csv，请先训练模型")
        perf = None

    if best_threshold is not None:
        st.metric("🎯 最佳阈值 (Youden)", f"{best_threshold:.4f}")
    else:
        st.warning("未找到最佳阈值")

    # ========== 2. 动态计算 PSI（替换硬编码） ==========
    train_path = 'model/train_predictions.parquet'
    test_path = 'model/test_predictions.parquet'

    def calculate_psi(expected_probs, actual_probs, bins=10):
        """计算 PSI：expected=训练集, actual=测试集"""
        bin_edges = np.linspace(0, 1, bins + 1)
        expected_counts, _ = np.histogram(expected_probs, bins=bin_edges, density=False)
        actual_counts, _ = np.histogram(actual_probs, bins=bin_edges, density=False)

        expected_pct = expected_counts / len(expected_probs)
        actual_pct = actual_counts / len(actual_probs)

        # 防止除零
        expected_pct = np.clip(expected_pct, 1e-6, 1)
        actual_pct = np.clip(actual_pct, 1e-6, 1)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return psi

    if os.path.exists(train_path) and os.path.exists(test_path):
        try:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)

            # 优先用 Stacking 概率，若不存在则用第一个可用概率列
            train_prob_col = 'prob_Stacking' if 'prob_Stacking' in train_df.columns else [c for c in train_df.columns if c.startswith('prob_')][0]
            test_prob_col = 'prob_Stacking' if 'prob_Stacking' in test_df.columns else [c for c in test_df.columns if c.startswith('prob_')][0]

            train_probs = train_df[train_prob_col].values
            test_probs = test_df[test_prob_col].values

            psi_value = calculate_psi(train_probs, test_probs, bins=10)

            if psi_value < 0.1:
                psi_status = "🟢 稳定"
            elif psi_value < 0.2:
                psi_status = "🟡 轻微偏移"
            else:
                psi_status = "🔴 显著偏移 (建议重训)"

            st.metric("📊 PSI (群体稳定性指数)", f"{psi_value:.4f}", delta=psi_status)
        except Exception as e:
            st.warning(f"计算 PSI 失败: {e}，请检查预测文件是否完整")
    else:
        st.warning("⚠️ 未找到训练集或测试集预测文件，PSI 无法计算。请重新运行 train.py 以生成。")

    st.divider()

    # ========== 3. 新增诊断图表（需要 test_predictions.parquet） ==========
    st.subheader("📈 深入诊断图表")

    if not os.path.exists(test_path):
        st.warning("⚠️ 未找到测试集预测结果文件，请先运行 train.py 并保存 test_predictions.parquet")
        st.stop()

    try:
        df_test = pd.read_parquet(test_path)
    except Exception as e:
        st.error(f"读取测试集文件失败: {e}")
        st.stop()

    # 提取真实标签和 Stacking 概率（优先）
    y_true = df_test['y_true'].values
    if 'prob_Stacking' in df_test.columns:
        y_prob = df_test['prob_Stacking'].values
        model_name = 'Stacking'
    else:
        prob_cols = [c for c in df_test.columns if c.startswith('prob_')]
        if not prob_cols:
            st.error("测试集中没有预测概率列")
            st.stop()
        y_prob = df_test[prob_cols[0]].values
        model_name = prob_cols[0].replace('prob_', '')

    best_thr = best_threshold if best_threshold is not None else 0.5

    # ---------- 3.1 ROC 曲线 ----------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})',
                                 line=dict(color='#00e5ff', width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='随机猜测',
                                 line=dict(color='gray', dash='dash')))
    fig_roc.update_layout(title='ROC 曲线', xaxis_title='假正率 (FPR)', yaxis_title='真正率 (TPR)',title_font=dict(color='#00e5ff'),
                          **TECH_THEME, width=500, height=400, margin=dict(l=40, r=20, t=40, b=40))
    fig_roc.add_annotation(x=0.7, y=0.2, text=f'AUC = {roc_auc:.3f}',
                           showarrow=False, font=dict(color='#88c0ff', size=14))

    # ---------- 3.2 PR 曲线 ----------
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = auc(recall, precision)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AP={ap:.3f})',
                                line=dict(color='#ff7f0e', width=2)))
    fig_pr.add_trace(go.Scatter(x=[0,1], y=[y_true.mean()] * 2, mode='lines', name='随机基线 (正例率)',
                                line=dict(color='gray', dash='dash')))
    fig_pr.update_layout(title='PR 曲线 (精确率-召回率)', xaxis_title='召回率', yaxis_title='精确率',title_font=dict(color='#00e5ff'),
                         **TECH_THEME, width=500, height=400, margin=dict(l=40, r=20, t=40, b=40))
    fig_pr.add_annotation(x=0.7, y=0.8, text=f'AP = {ap:.3f}',
                          showarrow=False, font=dict(color='#88c0ff', size=14))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_roc, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pr, use_container_width=True)

    st.divider()

    # ---------- 3.3 KS 曲线 ----------
# ---------- 3.3 KS 曲线（基于 roc_curve，标准方法） ----------
    fpr_ks, tpr_ks, thresholds_ks = roc_curve(y_true, y_prob)
    ks_values = tpr_ks - fpr_ks
    ks_max = ks_values.max()
    ks_max_idx = np.argmax(ks_values)
    # 阈值索引可能比 fpr 短 1
    if ks_max_idx < len(thresholds_ks):
        ks_threshold = thresholds_ks[ks_max_idx]
    else:
        ks_threshold = 0.5

    fig_ks = go.Figure()
    # 绘制 TPR 和 FPR（确保与 thresholds 长度一致）
    tpr_plot = tpr_ks[:-1] if len(tpr_ks) > len(thresholds_ks) else tpr_ks
    fpr_plot = fpr_ks[:-1] if len(fpr_ks) > len(thresholds_ks) else fpr_ks

    fig_ks.add_trace(go.Scatter(
        x=thresholds_ks, y=tpr_plot,
        mode='lines',
        name='真正率 (TPR)',
        line=dict(color='#e41a1c', width=2)
    ))
    fig_ks.add_trace(go.Scatter(
        x=thresholds_ks, y=fpr_plot,
        mode='lines',
        name='假正率 (FPR)',
        line=dict(color='#377eb8', width=2)
    ))

    # 标注最大 KS 位置
    fig_ks.add_vline(
        x=ks_threshold,
        line_dash="dash",
        line_color="#00e5ff",
        annotation_text=f"KS={ks_max:.3f}",
        annotation_position="top"
    )
    # 标记点
    tpr_at_ks = tpr_ks[ks_max_idx] if ks_max_idx < len(tpr_ks) else tpr_ks[-1]
    fpr_at_ks = fpr_ks[ks_max_idx] if ks_max_idx < len(fpr_ks) else fpr_ks[-1]
    fig_ks.add_trace(go.Scatter(
        x=[ks_threshold], y=[tpr_at_ks],
        mode='markers',
        marker=dict(color='#00e5ff', size=12),
        name='TPR (KS点)',
        showlegend=False
    ))
    fig_ks.add_trace(go.Scatter(
        x=[ks_threshold], y=[fpr_at_ks],
        mode='markers',
        marker=dict(color='#00e5ff', size=12),
        name='FPR (KS点)',
        showlegend=False
    ))

    fig_ks.update_layout(
        title=f'KS 曲线（最大 KS = {ks_max:.3f}，对应阈值 = {ks_threshold:.3f}）',
        xaxis_title='违约概率阈值',
        yaxis_title='累计占比',
        title_font=dict(color='#00e5ff'),
        **TECH_THEME,
        width=600, height=400,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    # fig_ks.update_xaxes(tickformat='.2f', autorange="reversed")

    # ---------- 3.4 混淆矩阵 ----------
    y_pred_thr = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred_thr)
    cm_percent = cm.astype('float') / cm.sum() * 100

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_percent,
        x=['预测正常', '预测违约'],
        y=['真实正常', '真实违约'],
        text=[[f'{cm[0,0]} ({cm_percent[0,0]:.1f}%)', f'{cm[0,1]} ({cm_percent[0,1]:.1f}%)'],
              [f'{cm[1,0]} ({cm_percent[1,0]:.1f}%)', f'{cm[1,1]} ({cm_percent[1,1]:.1f}%)']],
        texttemplate='%{text}',
        textfont={"size": 12, "color": "#000000"},
        colorscale='Blues',
        zmin=0,
        zmax=100,
        showscale=False
    ))
    fig_cm.update_layout(title=f'混淆矩阵 (阈值={best_thr:.3f})', xaxis_title='预测标签', yaxis_title='真实标签',title_font=dict(color='#00e5ff'),
                         **TECH_THEME, width=400, height=350, margin=dict(l=40, r=20, t=40, b=40))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_ks, use_container_width=True)
    with col2:
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ---------- 3.5 风险评分分段违约率 ----------
    scores = y_prob * 100
    bins = np.arange(0, 101, 10)
    labels = [f'{i}-{i+9}' for i in range(0, 100, 10)]
    df_score = pd.DataFrame({'score': scores, 'true': y_true})
    df_score['score_bin'] = pd.cut(df_score['score'], bins=bins, labels=labels, right=False)
    bin_stats = df_score.groupby('score_bin', observed=False).agg(
        total=('true', 'count'),
        bad=('true', 'sum')
    ).reset_index()
    bin_stats['bad_rate'] = bin_stats['bad'] / bin_stats['total'] * 100

    fig_bar = px.bar(
        bin_stats,
        x='score_bin',
        y='bad_rate',
        text='bad_rate',
        color='bad_rate',
        color_continuous_scale='RdYlGn_r',
        labels={'score_bin': '风险评分段', 'bad_rate': '违约率 (%)'},
        title='各风险评分段的违约率 (单调性验证)'
    )
    fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_bar.update_layout(**TECH_THEME, xaxis_tickangle=45, width=700, height=400,title_font=dict(color='#00e5ff'),
                          margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ========== 4. 详细解读 ==========
    st.info(
        "📌 **解读指南**\n"
        "- **PSI < 0.1** 表示模型在训练集和测试集上的概率分布高度一致，泛化能力好。\n"
        "- **ROC 曲线**：AUC 越接近 1 越好，但样本不平衡时需结合 PR 曲线。\n"
        "- **PR 曲线**：特别适用于不平衡数据，AP 值高表示模型在正例（违约）上表现好。\n"
        "- **KS 曲线**：KS > 0.3 表示模型有较好的区分好坏客户的能力。\n"
        "- **混淆矩阵**：重点关注 **漏警**（真实违约却被判正常）和 **误警**（真实正常却被判违约）的比例。\n"
        "- **风险分段图**：理想情况下，违约率应随分数单调递增，验证模型的排序能力。"
    )

# ===================== 页面4：模型训练 =====================
elif page == "🚀 模型训练":
    st.title("🚀 自适应模型训练引擎")
    st.markdown("**第一步：上传训练数据集 → 第二步：选择变量 → 第三步：配置参数 → 第四步：保存配置并手动运行训练**")
    st.divider()
    
    st.subheader("📂 第一步：上传训练数据集")
    uploaded_train_file = st.file_uploader(
        "上传训练数据（CSV 格式）",
        type=['csv'],
        key="train_uploader"
    )
    
    if uploaded_train_file is None:
        st.warning("⚠️ 请先上传 CSV 训练数据集")
        st.markdown("**📌 数据格式要求：**\n- 文件格式：CSV\n- 至少包含：1个目标列（违约标签）+ 1个以上特征列\n- 目标列建议为 0/1 格式（0=正常，1=违约）")
        st.stop()
    
    try:
        df_train_raw = pd.read_csv(uploaded_train_file)
        st.success(f"✅ 数据加载成功！共 {len(df_train_raw)} 行，{len(df_train_raw.columns)} 列")
        with st.expander("📋 数据预览（前5行）"):
            st.dataframe(df_train_raw.head())
        all_columns = df_train_raw.columns.tolist()
        st.caption(f"📌 数据列名：{', '.join(all_columns)}")
        st.divider()
        
        st.subheader("🎯 第二步：选择因变量（目标列）")
        target_col = st.selectbox("选择目标列（违约标签）", options=all_columns, index=0 if all_columns else None)
        if target_col:
            target_values = df_train_raw[target_col].unique()
            st.caption(f"目标列取值: {target_values[:10]}{'...' if len(target_values)>10 else ''}")
            if len(target_values) != 2:
                st.warning(f"⚠️ 目标列有 {len(target_values)} 个不同值，建议检查是否为二分类问题（0/1）")
        st.divider()
        
        st.subheader("📊 第三步：选择自变量（特征列）")
        st.markdown("将自变量归入以下 5 大板块，模型将根据您的选择进行训练")
        feature_groups = {
            "🏢 企业基本画像": {"description": "企业的静态属性，描述企业「是谁」", "features": []},
            "💰 财务能力": {"description": "反映企业的经营规模和还款能力", "features": []},
            "📋 信贷申请要素": {"description": "本次贷款的申请方案", "features": []},
            "🛡️ 风险缓释与征信": {"description": "企业的信用状况和担保情况", "features": []},
        }
        all_available_cols = [c for c in all_columns if c != target_col]

        # 初始化每个板块的 session_state（用于持久化选中状态）
        num_groups = len(feature_groups)
        for idx in range(num_groups):
            key = f"plate_{idx}_selected"
            if key not in st.session_state:
                st.session_state[key] = []

        # 收集所有板块当前选中的特征（从 session_state 读取）
        current_selected = {}
        for idx, (group_name, _) in enumerate(feature_groups.items()):
            key = f"plate_{idx}_selected"
            current_selected[group_name] = st.session_state[key]

        # 计算全局已选集合（所有板块选中的并集）
        all_selected = set()
        for vals in current_selected.values():
            all_selected.update(vals)

        # 逐板块显示，动态生成 options（排除其他板块已选，但保留自己已选）
        for idx, (group_name, group_info) in enumerate(feature_groups.items()):
            with st.expander(f"{group_name}", expanded=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**📌 说明**")
                    st.caption(group_info["description"])
                with col2:
                    # 当前板块已选的值
                    current_plate_selected = set(current_selected[group_name])
                    # 其他板块已选 = 全局已选 - 当前板块已选
                    other_selected = all_selected - current_plate_selected
                    # 可用选项 = 全部可用列 - 其他板块已选
                    options = [c for c in all_available_cols if c not in other_selected]
                    selected = st.multiselect(
                        f"选择 {group_name} 的自变量",
                        options=options,
                        default=list(current_plate_selected),
                        key=f"plate_{idx}_selected"
                    )
                    feature_groups[group_name]["features"] = selected
                    # 注意：session_state 已自动更新，无需手动赋值
                    # 将当前板块已选的特征加入“已选集合”，供后续板块排除
        
        all_selected_features = []
        for group_name, group_info in feature_groups.items():
            all_selected_features.extend(group_info["features"])
        
        st.divider()
        st.subheader("📋 变量选择汇总")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("因变量", target_col if target_col else "未选择")
        with col2:
            st.metric("自变量总数", len(all_selected_features))
        
        if all_selected_features:
            summary_data = []
            for group_name, group_info in feature_groups.items():
                if group_info["features"]:
                    summary_data.append({
                        "板块": group_name,
                        "特征数": len(group_info["features"]),
                        "特征列表": ", ".join(group_info["features"])
                    })
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        else:
            st.warning("⚠️ 请至少选择一个自变量！")
        st.divider()
        
        st.subheader("⚙️ 第四步：训练参数配置")
        col1, col2, col3 = st.columns(3)
        with col1:
            iv_th = st.slider("IV 筛选阈值", 0.0, 0.1, 0.02, 0.005, key="train_iv")
        with col2:
            vif_th = st.slider("VIF 剔除阈值", 3, 10, 5, 1, key="train_vif")
        with col3:
            use_woe = st.checkbox("LR 使用 WOE 编码", value=True, key="train_woe")
        
        st.subheader("🚀 第五步：保存配置并手动运行训练")
        can_save = (target_col is not None and len(all_selected_features) > 0)
        if not can_save:
            if not target_col:
                st.error("❌ 请选择因变量（目标列）")
            if len(all_selected_features) == 0:
                st.error("❌ 请至少选择一个自变量")
        else:
            with st.expander("📋 训练配置摘要"):
                st.json({
                    "数据文件": uploaded_train_file.name,
                    "样本数": len(df_train_raw),
                    "因变量": target_col,
                    "自变量总数": len(all_selected_features),
                    "板块分布": {g: len(f["features"]) for g, f in feature_groups.items() if f["features"]},
                    "IV阈值": iv_th,
                    "VIF阈值": vif_th,
                    "WOE编码": use_woe
                })
            
            if st.button("💾 保存配置并准备训练", use_container_width=True):
                temp_data_path = os.path.join(os.getcwd(), 'temp_train_data.parquet')
                df_train_raw[all_selected_features + [target_col]].to_parquet(temp_data_path, index=False)
                config_data = {
                    'data_path': temp_data_path,
                    'target_col': target_col,
                    'feature_cols': all_selected_features,
                    'iv_threshold': iv_th,
                    'vif_threshold': vif_th,
                    'use_woe_for_lr': use_woe,
                    'sample_frac': 1.0,
                    'test_size': 0.3,
                    'n_iter_random': 8,
                    'model_selection': ["lr", "xgb", "lgb"]
                }
                with open('temp_config.json', 'w') as f:
                    json.dump(config_data, f, indent=2)
                st.session_state.train_config_saved = True
                st.success("✅ 配置已保存！")
            
            if st.session_state.train_config_saved:
                st.divider()
                st.success("✅ 配置已保存，你可以开始训练模型了。")

                # ---- 训练按钮 ----
                col_train, col_shap = st.columns(2)
                with col_train:
                    start_train = st.button("🚀 开始训练 (train.py)", use_container_width=True)
                with col_shap:
                    start_shap = st.button("🧠 生成 SHAP 分析 (shap_analysis.py)", use_container_width=True)

                # ---- 用于显示训练日志的占位符 ----
                log_placeholder = st.empty()

                # ---- 训练状态（防止重复点击） ----
                if 'train_running' not in st.session_state:
                    st.session_state.train_running = False
                if 'shap_running' not in st.session_state:
                    st.session_state.shap_running = False

                # ---- 执行 train.py ----
                if start_train and not st.session_state.train_running:
                    st.session_state.train_running = True
                    with st.status("⏳ 训练进行中...", expanded=True) as status:
                        success, log_lines = run_script_with_logging("train.py", status, log_placeholder)
                        if success:
                            status.update(label="✅ 训练完成！", state="complete")
                            st.success("🎉 模型训练成功！模型已自动重新加载。")
                            # 显示完整日志（可滚动）
                            with st.expander("📋 完整训练日志（点击展开）", expanded=True):
                                st.code(''.join(log_lines), language='bash')
                            # 重新加载模型（不刷新页面）
                            reload_models()
                        else:
                            status.update(label="❌ 训练失败", state="error")
                            st.error("训练异常，请查看上方日志。")
                            # 即使失败也显示完整日志
                            if log_lines:
                                with st.expander("📋 完整训练日志（点击展开）", expanded=True):
                                    st.code(''.join(log_lines), language='bash')
                    st.session_state.train_running = False

                # ---- 执行 shap_analysis.py ----
                if start_shap and not st.session_state.shap_running:
                    st.session_state.shap_running = True
                    if not os.path.exists('model/base_models_tuned.pkl'):
                        st.warning("⚠️ 请先训练模型，再生成 SHAP 分析！")
                        st.session_state.shap_running = False
                    else:
                        with st.status("⏳ SHAP 分析进行中...", expanded=True) as status:
                            success, log_lines = run_script_with_logging("shap_analysis.py", status, log_placeholder)
                            if success:
                                status.update(label="✅ SHAP 分析完成！", state="complete")
                                st.success("🎉 SHAP 图表生成成功！")
                                with st.expander("📋 完整 SHAP 日志（点击展开）", expanded=True):
                                    st.code(''.join(log_lines), language='bash')
                            else:
                                status.update(label="❌ SHAP 分析失败", state="error")
                                st.error("SHAP 分析异常，请查看上方日志。")
                                if log_lines:
                                    with st.expander("📋 完整 SHAP 日志（点击展开）", expanded=True):
                                        st.code(''.join(log_lines), language='bash')
                    st.session_state.shap_running = False

                # ---- 重置配置按钮 ----
                if st.button("🔄 重置配置（重新开始）", use_container_width=True):
                    st.session_state.train_config_saved = False
                    st.session_state.train_running = False
                    st.session_state.shap_running = False
                    st.rerun()
    except Exception as e:
        st.error(f"❌ 数据处理失败: {str(e)}")

# ===================== 页面5：数据管理 =====================
elif page == "📁 数据管理":
    st.title("📁 数据管理与批量预测")
    st.markdown("上传企业数据文件（CSV），系统将自动进行批量风险预测，结果同步到全域大屏")
    
    uploaded_file = st.file_uploader("上传 CSV 文件", type=['csv'])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ 数据加载成功！共 {len(df_upload)} 行，{len(df_upload.columns)} 列")
            with st.expander("📋 数据预览（前5行）"):
                st.dataframe(df_upload.head())
            
            if st.button("🔮 批量预测", use_container_width=True):
                if not model_loaded:
                    st.error("❌ 模型未加载，请先训练模型！")
                else:
                    # ========== 动态获取模型所需的原始特征列 ==========
                    all_features = config.get('feature_cols', []) if config else []
                    derived_cols = ['application_month', 'application_quarter', 'days_to_disburse']
                    raw_numeric = [f for f in all_features if f not in derived_cols]
                    
                    if categorical_options:
                        raw_categorical = list(categorical_options.keys())
                    else:
                        raw_categorical = [col for col in all_features if col in woe_maps]
                    
                    required_cols = list(set(raw_numeric + raw_categorical))
                    
                    if not required_cols:
                        st.warning("⚠️ 无法从模型配置中读取特征列，使用默认列名列表（兼容模式）")
                        required_cols = [
                            'business_sector', 'business_state', 'lender',
                            'years_in_business', 'num_employees',
                            'annual_revenue_ngn', 'principal_ngn',
                            'interest_rate_annual', 'tenor_months',
                            'collateral_value_ngn', 'credit_score'
                        ]
                    
                    missing_cols = [c for c in required_cols if c not in df_upload.columns]
                    if missing_cols:
                        st.warning(f"⚠️ 数据缺少以下列：{missing_cols}，系统将自动填充默认值（数值填0，分类填'Other'），预测结果可能受影响。")
                        for col in missing_cols:
                            if col in categorical_options:
                                default_val = 'Other'
                            else:
                                default_val = 0
                            df_upload[col] = default_val
                    
                    with st.spinner(f"正在对 {len(df_upload)} 条数据进行预测..."):
                        df_predict = df_upload.copy()
                        predictions = []
                        for idx, row in df_predict.iterrows():
                            try:
                                raw = row.to_dict()
                                if 'application_date' not in raw:
                                    raw['application_date'] = '2024-01-15'
                                if 'disbursement_date' not in raw:
                                    raw['disbursement_date'] = '2024-02-01'
                                result = predict_single(raw)
                                predictions.append({
                                    '违约概率': result['Stacking'],
                                    '风险等级': get_risk_level(result['Stacking'])[0],
                                    '预测结果': '违约' if result['pred'] else '正常'
                                })
                            except Exception as e:
                                predictions.append({
                                    '违约概率': np.nan,
                                    '风险等级': '预测失败',
                                    '预测结果': f'错误: {str(e)[:20]}'
                                })
                        df_result = pd.concat([df_predict, pd.DataFrame(predictions)], axis=1)
                        for col in df_result.columns:
                            if df_result[col].dtype == 'object':
                                df_result[col] = df_result[col].astype(str)
                            elif pd.api.types.is_datetime64_any_dtype(df_result[col]):
                                df_result[col] = df_result[col].astype(str)
                            elif pd.api.types.is_numeric_dtype(df_result[col]):
                                df_result[col] = df_result[col].where(pd.notna(df_result[col]), None)
                            else:
                                df_result[col] = df_result[col].astype(str)
                        
                        st.session_state.prediction_result = df_result.copy()
                        st.session_state.prediction_data_source = "批量上传预测"
                        st.session_state.last_df_result = df_result.copy()
                        st.session_state.last_source_name = uploaded_file.name
                        
                        st.success("✅ 预测完成！结果已同步到「全域风控大屏」，请切换页面查看。")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总样本数", len(df_result))
                        with col2:
                            high_risk = len(df_result[df_result['风险等级'] == '高风险'])
                            st.metric("高风险企业", high_risk)
                        with col3:
                            avg_risk = df_result['违约概率'].mean()
                            st.metric("平均违约概率", f"{avg_risk:.2%}" if not pd.isna(avg_risk) else "N/A")
                        
                        with st.expander("📊 预测结果详情", expanded=True):
                            st.dataframe(df_result, use_container_width=True)
                        
                        csv = df_result.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载预测结果 (CSV)",
                            data=csv,
                            file_name="predictions_result.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"❌ 处理失败: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("👆 请上传 CSV 文件开始批量预测")
    
    st.divider()
    if 'last_df_result' in st.session_state and st.session_state.last_df_result is not None:
        st.subheader("💾 当前可保存的预测结果")
        st.write(f"数据来源：{st.session_state.get('last_source_name', '未知')}，共 {len(st.session_state.last_df_result)} 条记录")
        if st.button("💾 保存这次预测结果到历史记录", use_container_width=True):
            df_to_save = st.session_state.last_df_result
            source = st.session_state.get('last_source_name', '未知')
            meta = save_prediction_to_file(df_to_save, source)
            if meta is not None:
                st.success("✅ 已成功保存到历史记录！")
                st.rerun()
            else:
                st.error("❌ 保存失败，请查看终端错误信息。")
    else:
        st.info("暂无预测结果可保存，请先进行批量预测。")
    
    st.divider()
    st.subheader("📜 历史预测记录（已保存）")
    history_records = list_history_records()
    if not history_records:
        st.info("暂无已保存的历史预测记录。")
    else:
        history_df = pd.DataFrame([
            {"时间": r.get('timestamp', ''), "数据来源": r.get('source', ''), "记录数": r.get('rows', 0), "列数": len(r.get('columns', []))} 
            for r in history_records
        ])
        st.dataframe(history_df, use_container_width=True)
        selected_idx = st.selectbox(
            "选择一条记录查看详情",
            options=range(len(history_records)),
            format_func=lambda i: f"{history_records[i].get('timestamp', '')} - {history_records[i].get('source', '')} ({history_records[i].get('rows', 0)}条)"
        )
        if selected_idx is not None:
            meta = history_records[selected_idx]
            parquet_path = meta.get('file')
            if parquet_path and os.path.exists(parquet_path):
                try:
                    df_detail = pd.read_parquet(parquet_path)
                    with st.expander("📊 数据详情", expanded=True):
                        st.dataframe(df_detail, use_container_width=True)
                        csv = df_detail.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载为 CSV",
                            data=csv,
                            file_name=f"prediction_{meta.get('timestamp', '').replace(':', '-').replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"加载记录失败: {e}")
            else:
                st.warning("数据文件不存在，可能已被删除")
            if st.button("🗑️ 删除这条记录", use_container_width=True):
                try:
                    delete_record(meta)
                    st.success("已删除该记录")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
        if st.button("⚠️ 清除所有历史记录", use_container_width=True):
            try:
                for r in history_records:
                    delete_record(r)
                st.success("已清除所有历史记录")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

# ===================== 页面6：可解释性分析 =====================
elif page == "🧠 可解释性分析":
    st.title("🧠 模型可解释性分析 (SHAP)")
    st.markdown("以下图表展示了模型做出决策的依据，帮助你理解哪些特征对风险预测影响最大")

    shap_images = {
        "XGBoost 全局特征重要性 (Summary)": "shap_images/shap_xgb_summary.png",
        "XGBoost 特征重要性排序 (Bar)": "shap_images/shap_xgb_bar.png",
        "XGBoost 高风险样本拆解 (Waterfall)": "shap_images/shap_xgb_waterfall.png",
        "Stacking 元模型特征重要性": "shap_images/shap_stacking_bar.png",
        "Stacking 元模型 SHAP Summary": "shap_images/shap_stacking_summary.png",
        "Stacking 高风险样本解释": "shap_images/shap_stacking_waterfall.png"
    }
    available_images = {name: path for name, path in shap_images.items() if os.path.exists(path)}
    if not available_images:
        st.warning("⚠️ 未找到 SHAP 图片文件，请先运行 `python shap_analysis.py` 生成可解释性图表。")
    else:
        for name, path in available_images.items():
            with st.expander(f"📊 {name}", expanded=False):
                try:
                    st.image(path, width='stretch')
                except Exception as e:
                    st.error(f"图片加载失败: {str(e)}")
        st.info(
            "💡 **解读指南**：\n"
            "- **XGBoost Summary Plot**：红色表示特征值高，蓝色表示低；横轴为 SHAP 值（正值增加违约风险）\n"
            "- **XGBoost Bar Plot**：特征重要性排序，越高影响越大\n"
            "- **XGBoost Waterfall Plot**：单个高风险样本的决策拆解（红色增加风险，蓝色降低风险）\n"
            "- **Stacking 图表**：元模型（逻辑回归）对基模型预测概率的 SHAP 解释，反映各基模型对最终预测的贡献\n"
            "- 若某些图表未显示，表示该模型或步骤未成功生成图片，请检查 `shap_analysis.py` 运行日志。"
        )

# ===================== 底部信息 =====================
st.sidebar.divider()
if st.sidebar.button("🔄 刷新模型（重新加载）", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("""
**系统信息**
- 模型版本: v4.0（自适应变量选择）
- 基模型: LR + XGB + LGB
- 融合: Stacking (元学习器: LR)
- 自适应特征筛选
""")
