import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="COPD 抑郁风险预测器",
    page_icon="🫁",
    layout="wide"
)

# ---------- 使用缓存加载模型 ----------
@st.cache_resource
def load_model():
    # 加载 joblib 保存的 sklearn 包装器模型
    model = joblib.load('xgb_model.pkl')
    return model

model = load_model()
booster = model.get_booster()  # 获取底层 Booster 对象，用于跨版本兼容预测

# ---------- 特征定义 ----------
# 模型训练时使用的特征名称（必须严格一致）
MODEL_FEATURE_NAMES = [
    "Age", "BMI", "Height", "Memory_score", "Functional_dependency",
    "Executive_function_score", "Life_satisfaction", "Self_rated_health"
]

# 界面显示用的特征名称（保持用户友好）
DISPLAY_FEATURE_NAMES = [
    "Age", "BMI", "Height", "Memory score", "Functional dependency",
    "Executive function score", "Life satisfaction", "Self rated health"
]

# ---------- Session State 初始化（必须放在最前面） ----------
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.predicted_class = None
    st.session_state.probabilities = None
    st.session_state.advice = ""
    st.session_state.shap_fig = None

# ---------- 侧边栏：用户输入区 ----------
st.sidebar.title("📋 输入特征")
st.sidebar.markdown("请填写以下信息：")

age = st.sidebar.number_input("Age", min_value=45, max_value=85, value=60, step=1)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1, format="%.1f")
height = st.sidebar.number_input("Height (m)", min_value=0.0, max_value=3.0, value=1.75, step=0.01, format="%.2f")
memory_score = st.sidebar.number_input("Memory score (z-score)", min_value=-5.0, max_value=5.0, value=-0.20, step=0.01, format="%.2f")
functional_dependency = st.sidebar.selectbox(
    "Functional dependency",
    options=[0, 1, 2, 3],
    format_func=lambda x: {0: "Independent (0)", 1: "Low dependency (1)", 2: "Moderate dependency (2)", 3: "High dependency (3)"}[x]
)
executive_function_score = st.sidebar.number_input("Executive function score (z-score)", min_value=-5.0, max_value=5.0, value=-0.20, step=0.01, format="%.2f")
life_satisfaction = st.sidebar.selectbox(
    "Life satisfaction",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {1: "Very poor (1)", 2: "Poor (2)", 3: "Average (3)", 4: "Good (4)", 5: "Excellent (5)"}[x]
)
self_rated_health = st.sidebar.selectbox(
    "Self rated health",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {1: "Very poor (1)", 2: "Poor (2)", 3: "Average (3)", 4: "Good (4)", 5: "Excellent (5)"}[x]
)

# ---------- 主页面 ----------
st.title("🫁 COPD Depression Risk Predictor")
st.markdown("该工具基于 XGBoost 模型，用于预测 COPD 患者的抑郁风险。请在左侧输入特征后点击下方按钮。")

# 预测按钮
predict_clicked = st.button("🔍 开始预测", type="primary", use_container_width=True)

if predict_clicked:
    # --- 1. 按模型训练顺序组装特征 ---
    feature_values = [
        age,
        bmi,
        height,
        memory_score,
        functional_dependency,
        executive_function_score,
        life_satisfaction,
        self_rated_health
    ]
    features = np.array([feature_values])
    
    # --- 2. 使用 Booster 进行预测（绕过 sklearn 兼容问题）---
    dtest = xgb.DMatrix(features, feature_names=MODEL_FEATURE_NAMES)
    raw_pred = booster.predict(dtest, output_margin=True)[0]
    proba_high = 1.0 / (1.0 + np.exp(-raw_pred))
    proba_low = 1.0 - proba_high
    predicted_proba = np.array([proba_low, proba_high])
    predicted_class = 1 if proba_high >= 0.5 else 0
    
    # --- 3. 生成建议 ---
    proba_percent = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据模型预测，您属于**高风险**人群（抑郁风险）。\n"
            f"预测概率为 **{proba_percent:.1f}%**。\n\n"
            "建议您及时咨询专业医疗人员，进行进一步评估和干预。"
        )
    else:
        advice = (
            f"根据模型预测，您属于**低风险**人群。\n"
            f"预测概率为 **{proba_percent:.1f}%**。\n\n"
            "请继续保持健康的生活方式，定期复查。"
        )
    
    # --- 4. 计算 SHAP 值（用于解释预测）---
    try:
        # 修复旧版模型可能缺少 feature_types 属性的问题
        if not hasattr(model, 'feature_types'):
            model.feature_types = None
        
        # 创建解释器（使用 sklearn 包装器以便于获取 expected_value）
        explainer = shap.TreeExplainer(model)
        # 注意：传入的 DataFrame 列名需与模型内部名称一致
        shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=MODEL_FEATURE_NAMES))
        
        # 处理二分类模型的 expected_value 和 shap_values 格式
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            expected_value = explainer.expected_value[1]  # 取正类的基准值
            shap_vals = shap_values[0][:, 1] if shap_values[0].ndim > 1 else shap_values[0]
        else:
            expected_value = explainer.expected_value
            shap_vals = shap_values[0]
        
        # 生成 SHAP force plot（返回 matplotlib 图形）
        plt.figure(figsize=(14, 4))
        shap.plots.force(
            expected_value,
            shap_vals,
            feature_names=MODEL_FEATURE_NAMES,  # 使用模型内部名称
            matplotlib=True,
            show=False
        )
        fig = plt.gcf()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        st.session_state.shap_fig = fig
    except Exception as e:
        st.warning(f"SHAP 可视化生成失败，但预测结果仍有效。错误信息：{e}")
        st.session_state.shap_fig = None
    
    # --- 5. 保存结果到 session_state ---
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.probabilities = predicted_proba
    st.session_state.advice = advice

# ---------- 展示预测结果（如果已预测） ----------
if st.session_state.get("prediction_made", False):
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📊 预测结果")
        class_label = "高风险" if st.session_state.get("predicted_class") == 1 else "低风险"
        st.metric("风险类别", class_label)
        
        proba = st.session_state.get("probabilities")
        if proba is not None:
            proba_df = pd.DataFrame({
                "类别": ["低风险 (0)", "高风险 (1)"],
                "概率": proba
            })
            st.dataframe(proba_df, use_container_width=True)
            st.progress(float(proba[1]), text=f"高风险概率: {proba[1]*100:.1f}%")
        
        st.subheader("💡 建议")
        st.info(st.session_state.get("advice", ""))
    
    with col2:
        st.subheader("🔍 特征重要性解释 (SHAP)")
        fig = st.session_state.get("shap_fig")
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.caption("上图展示了每个特征对本次预测的贡献方向（红色：推高风险，蓝色：降低风险）")
        else:
            st.info("SHAP 可视化暂时无法显示。")
else:
    st.info("👈 请在左侧填写特征信息，然后点击「开始预测」按钮。")