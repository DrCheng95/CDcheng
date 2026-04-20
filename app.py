import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="COPD 抑郁风险预测器",
    page_icon="🫁",
    layout="wide"
)

# ---------- 使用缓存加载模型，避免重复加载 ----------
@st.cache_resource
def load_model():
    model = joblib.load('xgb_model.pkl')
    return model

model = load_model()

# 特征名称与顺序
feature_names = [
    "Age", "BMI", "Height", "Memory score", "Functional dependency", "Executive function score", "Life satisfaction", "Self rated health"]

# ---------- 侧边栏：用户输入区 ----------
st.sidebar.title("📋 输入特征")
st.sidebar.markdown("请填写以下信息：")

age = st.sidebar.number_input("Age", min_value=45, max_value=85, value=60, step=1)
height = st.sidebar.number_input("Height (m)", min_value=0.0, max_value=3.0, value=1.75, step=0.01, format="%.2f")
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1, format="%.1f")

self_rated_health = st.sidebar.selectbox(
    "Self rated health",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {1: "Very poor (1)", 2: "Poor (2)", 3: "Average (3)", 4: "Good (4)", 5: "Excellent (5)"}[x]
)

life_satisfaction = st.sidebar.selectbox(
    "Life satisfaction",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {1: "Very poor (1)", 2: "Poor (2)", 3: "Average (3)", 4: "Good (4)", 5: "Excellent (5)"}[x]
)

functional_dependency = st.sidebar.selectbox(
    "Functional dependency",
    options=[0, 1, 2, 3],
    format_func=lambda x: {0: "Independent (0)", 1: "Low dependency (1)", 2: "Moderate dependency (2)", 3: "High dependency (3)"}[x]
)

memory_score = st.sidebar.number_input("Memory score (z-score)", min_value=-5.0, max_value=5.0, value=-0.20, step=0.01, format="%.2f")
executive_function_score = st.sidebar.number_input("Executive function score (z-score)", min_value=-5.0, max_value=5.0, value=-0.20, step=0.01, format="%.2f")

# 将特征组合成数组
feature_values = [age, bmi, height, memory_score, functional_dependency, executive_function_score, life_satisfaction, self_rated_health]

# ---------- 主页面：标题与预测触发 ----------
st.title("🫁 COPD Depression Risk Predictor")
st.markdown("该工具基于 XGBoost 模型，用于预测 COPD 患者的抑郁风险。请在左侧输入特征后点击下方按钮。")

# 预测按钮
predict_clicked = st.button("🔍 开始预测", type="primary", use_container_width=True)

# 初始化 session_state 用于保存预测结果
if predict_clicked:
    import xgboost as xgb
    
    # 界面输入值，按模型期望顺序重组
    feature_values_ordered = [
        age, bmi, height, memory_score, functional_dependency,
        executive_function_score, life_satisfaction, self_rated_health
    ]
    features = np.array([feature_values_ordered])
    
    # 模型内部特征名称（训练时使用的）
    model_feature_names = [
        "Age", "BMI", "Height", "Memory_score", "Functional_dependency",
        "Executive_function_score", "Life_satisfaction", "Self_rated_health"
    ]
    
    booster = model.get_booster()
    dtest = xgb.DMatrix(features, feature_names=model_feature_names)
    raw_pred = booster.predict(dtest, output_margin=True)[0]
    proba_high = 1.0 / (1.0 + np.exp(-raw_pred))
    proba_low = 1.0 - proba_high
    predicted_proba = np.array([proba_low, proba_high])
    predicted_class = 1 if proba_high >= 0.5 else 0
    
    # 生成建议文本
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
    
    # 计算 SHAP 值（用于解释预测）
    try:
        # 获取底层 Booster 对象，避免 scikit-learn wrapper 的兼容性问题
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
        else:
            booster = model  # 如果模型本身就是 Booster

        # 创建 explainer 直接使用 Booster
        explainer = shap.TreeExplainer(booster)
        
        # 转换输入数据为 DataFrame 以确保特征名称正确传递
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        shap_values = explainer.shap_values(input_df)
        
        # 处理 shap_values 形状：对于二分类 XGBoost，shap_values 可能是 (n_samples, n_features, n_classes) 的列表或数组
        if isinstance(shap_values, list):
            # 多分类情况，但这里是二分类，通常 shap_values 是单个数组
            shap_vals = shap_values[0]  # 取第一个类
        elif shap_values.ndim == 3:
            # 形状 (n_samples, n_features, n_classes)
            shap_vals = shap_values[:, :, 1]  # 取正类（高风险）的 SHAP 值
        else:
            shap_vals = shap_values  # 已经是二维

        # 如果 shap_vals 是二维但只有一行，取第一行
        if shap_vals.ndim == 2:
            shap_vals = shap_vals[0]
            
        # 获取期望值
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            # 二分类通常有两个 expected_value，取正类的
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        
        # 绘制 SHAP force plot
        plt.figure(figsize=(14, 4))
        shap.plots.force(
            expected_value,
            shap_vals,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        fig = plt.gcf()
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)
        st.session_state.shap_fig = fig
    except Exception as e:
        st.warning(f"SHAP 可视化生成失败，但预测结果仍有效。错误信息：{e}")
        st.session_state.shap_fig = None
    
    # 保存到 session_state
    st.session_state.prediction_made = True
    st.session_state.predicted_class = predicted_class
    st.session_state.probabilities = predicted_proba
    st.session_state.advice = advice

# ---------- 展示预测结果（如果已预测） ----------
if st.session_state.get("prediction_made", False):
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📊 预测结果")
        class_label = "高风险" if st.session_state.predicted_class == 1 else "低风险"
        st.metric("风险类别", class_label)
        
        # 显示概率
        proba = st.session_state.probabilities
        proba_df = pd.DataFrame({
            "类别": ["低风险 (0)", "高风险 (1)"],
            "概率": proba
        })
        st.dataframe(proba_df, use_container_width=True)
        
        # 进度条展示高风险概率
        st.progress(float(proba[1]), text=f"高风险概率: {proba[1]*100:.1f}%")
        
        st.subheader("💡 建议")
        st.info(st.session_state.advice)
    
    with col2:
        st.subheader("🔍 特征重要性解释 (SHAP)")
        if st.session_state.shap_fig is not None:
            st.pyplot(st.session_state.shap_fig, use_container_width=True)
            st.caption("上图展示了每个特征对本次预测的贡献方向（红色：推高风险，蓝色：降低风险）")
        else:
            st.info("SHAP 可视化暂时无法显示。")
else:
    st.info("👈 请在左侧填写特征信息，然后点击「开始预测」按钮。")