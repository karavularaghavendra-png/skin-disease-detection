# 🩺 AI Skin Disease Detection

import os
import logging

# Centralised TF warning suppression (must be before TF imports)
import logger as _logger_init  # noqa: F401 — sets TF env vars on import

import json
import tempfile
import pathlib

import streamlit as st
import numpy as np
from PIL import Image

from predict import predict_single_image, load_model_cached

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Disease Detection Using Deep Learning",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL — lazy: only loaded when user clicks Analyze
# ─────────────────────────────────────────────────────────────
try:
    model, class_names = load_model_cached()
except FileNotFoundError:
    model, class_names = None, []
    st.warning(
        "⚠️ Trained model not found. "
        "Please run `python train_model.py --dataset dataset/skin_dataset` first."
    )

# ─────────────────────────────────────────────────────────────
# IMPORTS — single source of truth
# ─────────────────────────────────────────────────────────────
from utils.disease_info import get_disease_info, get_severity, get_disclaimer
from utils.medication_map import MEDICATION_MAP
from utils.ood_detector import is_skin_image
from utils.image_utils import check_image_quality
from utils.model_comparison import get_comparison_table_rows, get_recommendation, MODEL_COMPARISON

from explainability import generate_gradcam_overlay

# PIL image bomb protection
Image.MAX_IMAGE_PIXELS = 50_000_000

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Navy + Teal Medical Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ══════════════════════════════════════════════════════════
       🩺 DARK NAVY + TEAL — Premium Medical AI Theme
       ══════════════════════════════════════════════════════════ */

    /* ── Google Fonts: Inter ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Keyframe Animations ── */
    @keyframes subtleFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 201, 167, 0.12); }
        50% { box-shadow: 0 0 35px rgba(0, 201, 167, 0.22); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes borderPulse {
        0%, 100% { border-color: rgba(0, 201, 167, 0.25); }
        50% { border-color: rgba(0, 201, 167, 0.5); }
    }

    /* ── Apply Inter globally ── */
    html, body, [class*="css"], .stApp, .stMarkdown,
    .stTextInput input, .stSelectbox select, .stButton > button,
    h1, h2, h3, h4, h5, h6, p, span, div, label, li, a, td, th,
    code, pre, blockquote, summary, details {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Hide Streamlit defaults (rainbow bar, footer, menu) ── */
    #MainMenu {visibility: hidden !important;}
    header[data-testid="stHeader"] {display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    .viewerBadge_container__r5tak {display: none !important;}
    .styles_viewerBadge__CvC9N {display: none !important;}

    /* ── Main background — deep navy gradient (full page) ── */
    html, body {
        background: #0A0F1E !important;
    }
    .stApp {
        background: linear-gradient(160deg, #0A0F1E 0%, #0D1B2A 30%, #101c30 60%, #0A0F1E 100%) !important;
        min-height: 100vh;
    }
    /* Subtle animated radial overlay for depth */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 20% 20%, rgba(0, 201, 167, 0.03) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 80%, rgba(0, 180, 216, 0.02) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar — dark slate with teal left accent ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0f1520 100%) !important;
        border-right: none !important;
        border-left: 3px solid #00C9A7 !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3) !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00C9A7 !important;
        text-shadow: 0 0 14px rgba(0, 201, 167, 0.3);
        font-weight: 700 !important;
        letter-spacing: -0.2px !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stCaption {
        color: #94A3B8 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: #E0FFF8 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(0, 201, 167, 0.12) !important;
        margin: 1rem 0 !important;
    }

    /* ── Headers — teal-white with soft glow ── */
    h1, h2, h3, h4, h5, h6 {
        color: #E0FFF8 !important;
        text-shadow: 0 0 24px rgba(0, 201, 167, 0.18), 0 0 48px rgba(0, 201, 167, 0.06) !important;
    }
    h1 {
        background: linear-gradient(135deg, #00C9A7 0%, #00E5FF 50%, #00C9A7 100%) !important;
        background-size: 200% 200% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 900 !important;
        letter-spacing: -0.6px !important;
        animation: gradientShift 6s ease infinite !important;
        padding-bottom: 4px !important;
    }
    h2 {
        font-weight: 700 !important;
        position: relative;
    }
    h3 {
        font-weight: 600 !important;
        color: #c0f0e8 !important;
    }
    h4 {
        color: #00C9A7 !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }

    /* ── Body text ── */
    .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stApp p, .stApp span, .stApp label {
        color: #CBD5E1 !important;
        line-height: 1.65 !important;
    }

    /* ── Buttons — teal-to-cyan gradient with hover darken ── */
    .stButton > button {
        background: linear-gradient(135deg, #00C9A7 0%, #00B4D8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.8rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 20px rgba(0, 201, 167, 0.3),
                    0 0 0 1px rgba(0, 201, 167, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 0; left: -100%; right: 0; bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        transition: left 0.5s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00A98D 0%, #0096B7 100%) !important;
        box-shadow: 0 8px 32px rgba(0, 201, 167, 0.4),
                    0 0 0 1px rgba(0, 201, 167, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:hover::after {
        left: 100%;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.98) !important;
        box-shadow: 0 2px 12px rgba(0, 201, 167, 0.25) !important;
    }

    /* ── File uploader — dark card with dashed teal border ── */
    .stFileUploader {
        background: #1A2235 !important;
        border: 2px dashed #00C9A7 !important;
        border-radius: 16px !important;
        padding: 8px !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: pulseGlow 4s ease-in-out infinite;
    }
    .stFileUploader:hover {
        box-shadow: 0 12px 40px rgba(0, 201, 167, 0.18),
                    0 0 0 1px rgba(0, 229, 255, 0.15) !important;
        border-color: #00E5FF !important;
        transform: translateY(-3px) !important;
        background: #1d2640 !important;
    }
    .stFileUploader label, .stFileUploader span,
    .stFileUploader p, .stFileUploader div {
        color: #94A3B8 !important;
    }
    /* Browse files button inside uploader */
    .stFileUploader button {
        background: linear-gradient(135deg, #00C9A7, #00B4D8) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.45rem 1.4rem !important;
        box-shadow: 0 3px 12px rgba(0, 201, 167, 0.25) !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader button:hover {
        background: linear-gradient(135deg, #00A98D, #0096B7) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 18px rgba(0, 201, 167, 0.35) !important;
    }

    /* ── Metrics — dark cards with teal accent + hover lift ── */
    [data-testid="stMetric"] {
        background: #16213E !important;
        border: 1px solid rgba(0, 201, 167, 0.25) !important;
        border-radius: 14px !important;
        padding: 20px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(0, 201, 167, 0.08) !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: fadeInUp 0.5s ease both;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 201, 167, 0.5) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4),
                    0 0 20px rgba(0, 201, 167, 0.1),
                    inset 0 1px 0 rgba(0, 201, 167, 0.12) !important;
        transform: translateY(-3px) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #00C9A7 !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    [data-testid="stMetricValue"] {
        color: #E0FFF8 !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricDelta"] {
        color: #00C9A7 !important;
    }

    /* ── Progress bars — teal gradient ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00C9A7, #00E5FF) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 12px rgba(0, 201, 167, 0.25) !important;
    }
    .stProgress > div {
        background: #1A2235 !important;
        border-radius: 10px !important;
    }

    /* ── Success / Info / Warning / Error boxes ── */
    .stSuccess, [data-testid="stNotification"][data-type="success"],
    div[data-testid="stAlert"][data-type="success"] {
        background-color: rgba(0, 201, 167, 0.06) !important;
        border-left: 4px solid #00C9A7 !important;
        color: #A7F3D0 !important;
        border-radius: 0 12px 12px 0 !important;
        backdrop-filter: blur(6px);
    }
    .stInfo, [data-testid="stNotification"][data-type="info"],
    div[data-testid="stAlert"][data-type="info"] {
        background-color: rgba(0, 180, 216, 0.06) !important;
        border-left: 4px solid #00B4D8 !important;
        color: #BAE6FD !important;
        border-radius: 0 12px 12px 0 !important;
        backdrop-filter: blur(6px);
    }
    .stWarning, [data-testid="stNotification"][data-type="warning"],
    div[data-testid="stAlert"][data-type="warning"],
    div[data-testid="stAlert"] {
        background-color: rgba(251, 191, 36, 0.06) !important;
        border-left: 4px solid #FBBF24 !important;
        color: #FDE68A !important;
        border-radius: 0 12px 12px 0 !important;
        backdrop-filter: blur(6px);
    }
    .stError, div[data-testid="stAlert"][data-type="error"] {
        background-color: rgba(239, 68, 68, 0.06) !important;
        border-left: 4px solid #EF4444 !important;
        color: #FCA5A5 !important;
        border-radius: 0 12px 12px 0 !important;
        backdrop-filter: blur(6px);
    }

    /* ── Cards / Expanders ── */
    .streamlit-expanderHeader {
        background: #16213E !important;
        border: 1px solid rgba(0, 201, 167, 0.18) !important;
        border-radius: 14px !important;
        color: #00C9A7 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: rgba(0, 201, 167, 0.4) !important;
        background: #1a2842 !important;
    }
    .streamlit-expanderContent {
        background: #111827 !important;
        border: 1px solid rgba(0, 201, 167, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 14px 14px !important;
    }
    details[data-testid="stExpander"] {
        background: #16213E !important;
        border: 1px solid rgba(0, 201, 167, 0.18) !important;
        border-radius: 14px !important;
        transition: all 0.3s ease !important;
        overflow: hidden;
    }
    details[data-testid="stExpander"]:hover {
        border-color: rgba(0, 201, 167, 0.35) !important;
    }
    details[data-testid="stExpander"] summary {
        color: #00C9A7 !important;
        font-weight: 600 !important;
    }
    details[data-testid="stExpander"] > div {
        background: #111827 !important;
    }

    /* ── Dataframe / Table ── */
    .stDataFrame {
        border: 1px solid rgba(0, 201, 167, 0.2) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }
    .stDataFrame table {
        background: #16213E !important;
    }
    .stDataFrame th {
        background: #1A2235 !important;
        color: #00C9A7 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid rgba(0, 201, 167, 0.25) !important;
        text-transform: uppercase !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.5px !important;
    }
    .stDataFrame td {
        color: #CBD5E1 !important;
        border-bottom: 1px solid rgba(0, 201, 167, 0.06) !important;
    }
    .stDataFrame tr:hover td {
        background: rgba(0, 201, 167, 0.04) !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00A98D 0%, #008B74 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(0, 201, 167, 0.25) !important;
        transition: all 0.3s ease !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #00C9A7 0%, #00A98D 100%) !important;
        box-shadow: 0 8px 24px rgba(0, 201, 167, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 201, 167, 0.2), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Caption / footer text ── */
    .stCaption, small {
        color: #64748B !important;
        font-size: 0.82rem !important;
    }

    /* ── Image display ── */
    .stImage {
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.35),
                    0 0 0 1px rgba(0, 201, 167, 0.08) !important;
        transition: all 0.3s ease !important;
    }
    .stImage:hover {
        box-shadow: 0 10px 36px rgba(0, 0, 0, 0.45),
                    0 0 16px rgba(0, 201, 167, 0.1) !important;
    }
    .stImage img {
        border-radius: 14px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #00C9A7 !important;
    }

    /* ── Text input / Select / TextArea ── */
    .stTextInput input, .stSelectbox select, .stTextArea textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #1A2235 !important;
        border: 1px solid rgba(0, 201, 167, 0.18) !important;
        border-radius: 10px !important;
        color: #E0FFF8 !important;
        transition: all 0.25s ease !important;
    }
    .stTextInput input:focus, .stSelectbox select:focus,
    .stTextArea textarea:focus {
        border-color: #00C9A7 !important;
        box-shadow: 0 0 0 3px rgba(0, 201, 167, 0.12) !important;
    }

    /* ── Bar chart / Line chart ── */
    .stBarChart, .stLineChart {
        background: #16213E !important;
        border: 1px solid rgba(0, 201, 167, 0.12) !important;
        border-radius: 14px !important;
        padding: 10px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2) !important;
    }

    /* ── Column containers (result cards) ── */
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
        background: transparent !important;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #16213E !important;
        border-radius: 12px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #E0FFF8 !important;
        background: rgba(0, 201, 167, 0.15) !important;
    }

    /* ── Custom scrollbar — thin teal ── */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-track {
        background: #0A0F1E;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(0, 201, 167, 0.35), rgba(0, 180, 216, 0.35));
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, rgba(0, 201, 167, 0.6), rgba(0, 180, 216, 0.6));
    }
    /* Firefox scrollbar */
    * {
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 201, 167, 0.35) #0A0F1E;
    }

    /* ── Markdown links ── */
    a {
        color: #00C9A7 !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
    }
    a:hover {
        color: #00E5FF !important;
        text-decoration: underline !important;
        text-shadow: 0 0 8px rgba(0, 229, 255, 0.2);
    }

    /* ── Tooltip ── */
    .stTooltipIcon {
        color: #00C9A7 !important;
    }

    /* ── Code blocks ── */
    code, .stCode {
        background: #1A2235 !important;
        color: #00E5FF !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
        font-size: 0.85em !important;
    }

    /* ── Selectbox dropdown ── */
    [data-baseweb="popover"] {
        background: #16213E !important;
        border: 1px solid rgba(0, 201, 167, 0.2) !important;
        border-radius: 12px !important;
    }
    [data-baseweb="menu"] {
        background: #16213E !important;
    }
    [data-baseweb="menu"] li {
        color: #CBD5E1 !important;
    }
    [data-baseweb="menu"] li:hover {
        background: rgba(0, 201, 167, 0.12) !important;
    }

    /* ── Generic block containers — result cards ── */
    div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {
        animation: fadeInUp 0.4s ease both;
    }

    /* ── Mobile responsive ── */
    @media (max-width: 768px) {
        .stApp > div > div {
            padding-left: 10px !important;
            padding-right: 10px !important;
        }
        [data-testid="stMetric"] {
            padding: 14px !important;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.25rem !important;
        }
        .stButton > button {
            padding: 0.55rem 1.2rem !important;
            font-size: 0.85rem !important;
        }
        .stFileUploader {
            border-radius: 12px !important;
        }
        .stImage {
            border-radius: 10px !important;
        }
        .stImage img {
            border-radius: 10px !important;
        }
    }

    @media (max-width: 480px) {
        .stApp > div > div {
            padding-left: 6px !important;
            padding-right: 6px !important;
        }
        h1 {
            font-size: 1.3rem !important;
        }
        [data-testid="stMetric"] {
            padding: 10px !important;
            border-radius: 10px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE — prediction history
# ─────────────────────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ─────────────────────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────────────────────
st.title("🩺 Skin Disease Detection Using Deep Learning")
st.write("Upload a clear image of the affected skin area to get an AI-based assessment.")

# ─────────────────────────────────────────────────────────────
# SIDEBAR — Dashboard + History + Model Comparison
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Prediction History ──
    st.markdown("### 📋 Session History")
    if st.session_state.prediction_history:
        for h in reversed(st.session_state.prediction_history[-10:]):
            st.markdown(f"**{h['disease']}** · {h['confidence']:.1f}%")
    else:
        st.caption("No predictions yet this session.")

    # ── Prediction Dashboard (Enhancement #4) ──
    if len(st.session_state.prediction_history) >= 2:
        st.markdown("---")
        st.markdown("### 📊 Session Dashboard")

        history = st.session_state.prediction_history

        # Session stats
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Scans", len(history))
        with col_s2:
            avg_conf = np.mean([h["confidence"] for h in history])
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")

        # Disease distribution bar chart
        from collections import Counter
        disease_counts = Counter(h["disease"] for h in history)
        st.bar_chart(disease_counts)

        # Confidence trend line chart
        conf_values = [h["confidence"] for h in history]
        st.caption("Confidence Trend")
        st.line_chart(conf_values)

    # ── Model Comparison Table (Enhancement #5) ──
    st.markdown("---")
    with st.expander("🧠 Model Architecture Comparison", expanded=False):
        import pandas as pd
        comparison_rows = get_comparison_table_rows()
        df = pd.DataFrame(comparison_rows)
        st.dataframe(df, hide_index=True)

        st.markdown(get_recommendation())

        # Strengths & Weaknesses
        for model_name, data in MODEL_COMPARISON.items():
            st.markdown(f"**{model_name}**")
            st.markdown("*Strengths:*")
            for s in data["strengths"]:
                st.markdown(f"- ✅ {s}")
            st.markdown("*Weaknesses:*")
            for w in data["weaknesses"]:
                st.markdown(f"- ⚠️ {w}")
            st.markdown("")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This tool uses a MobileNetV2 deep learning model "
        "trained on dermatological images to provide preliminary "
        "skin condition assessments."
    )

# ─────────────────────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload Skin Image",
    type=["jpg", "jpeg", "png"],
    help="Accepted: JPG, PNG. Max 10 MB. Use clear, well-lit images.",
)

# ─────────────────────────────────────────────────────────────
# FILE SIZE VALIDATION
# ─────────────────────────────────────────────────────────────
_MAX_SIZE_MB = 10
if uploaded_file is not None:
    _size_mb = uploaded_file.size / (1024 * 1024)
    if _size_mb > _MAX_SIZE_MB:
        st.error(f"File too large ({_size_mb:.1f} MB). Please upload an image under {_MAX_SIZE_MB} MB.")
        st.stop()

# ─────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Uploaded Image", width=400)

    with col2:
        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    # ── Save to temp file (predict.py needs a file path) ──
                    suffix = pathlib.Path(uploaded_file.name).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        uploaded_file.seek(0)
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # ── IMAGE QUALITY CHECK ──
                    is_quality_ok, quality_warnings = check_image_quality(tmp_path)
                    if quality_warnings:
                        for w in quality_warnings:
                            st.warning(f"⚠️ {w}")

                    # ── OOD DETECTION (skin check) ──
                    is_skin, skin_ratio = is_skin_image(tmp_path)
                    if not is_skin:
                        st.error(
                            "🚫 This image does not appear to contain human skin. "
                            "Please upload a clear photo of the affected skin area."
                        )
                        pathlib.Path(tmp_path).unlink(missing_ok=True)
                        st.stop()

                    # ── PREDICTION WITH TTA ──
                    tta_result = predict_single_image(tmp_path, use_tta=True)
                    top_results, avg_preds, all_pass_probs = tta_result

                    # ── Extract top result ──
                    disease_name = top_results[0]["disease"]
                    confidence   = top_results[0]["confidence"]   # already 0–100
                    tta_agreement = top_results[0].get("tta_agreement", 0)
                    latency_ms    = top_results[0].get("latency_ms", 0)

                    # ── RELIABILITY ANALYSIS (with TTA disagreement) ──
                    from predict import analyze_prediction_reliability
                    reliability = analyze_prediction_reliability(
                        avg_preds, confidence, all_pass_probs=all_pass_probs
                    )

                    if not reliability["is_reliable"]:
                        st.warning(
                            "🛡️ **Reliability Alert:** This prediction may not be accurate. "
                            "The model shows uncertainty about this image. "
                            "Please try a clearer, well-lit close-up photo or consult a dermatologist."
                        )
                        for w in reliability["warnings"]:
                            st.warning(f"⚠️ {w}")

                    # ── LOOKUP ──
                    info = get_disease_info(disease_name)
                    severity_label, severity_colour = get_severity(confidence, disease_name)
                    specialist = info.get("specialist", "Dermatologist")

                    # ── Save to history ──
                    st.session_state.prediction_history.append({
                        "disease": info.get("display_name", disease_name),
                        "confidence": confidence,
                    })

                    # ── RESULTS ──
                    if reliability["is_reliable"]:
                        st.success("✅ Analysis Complete")
                    else:
                        st.info("🔍 Analysis Complete — Low Reliability")

                    display_name = info.get("display_name", disease_name.upper())
                    icon = info.get("icon", "🔬")
                    st.markdown(f"### {icon} {display_name}")
                    st.write(info.get("description", ""))

                    st.markdown(f"**Confidence:** `{confidence:.1f}%`")
                    st.markdown(
                        f"**Severity:** <span style='color:{severity_colour}; font-weight:bold'>"
                        f"{severity_label}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Recommended Specialist:** {specialist}")

                    # ── Reliability + TTA metrics ──
                    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                    with col_r1:
                        st.metric("Confidence", f"{confidence:.1f}%")
                    with col_r2:
                        st.metric("TTA Agreement", f"{tta_agreement}%")
                    with col_r3:
                        st.metric("Entropy", f"{reliability['entropy']:.3f}")
                    with col_r4:
                        st.metric("Latency", f"{latency_ms:.0f}ms")

                    # ── Top 3 predictions ──
                    if len(top_results) > 1:
                        st.markdown("#### 📈 Top Predictions")
                        for r in top_results:
                            st.progress(
                                min(int(r["confidence"]), 100),
                                text=f"{r['disease'].capitalize()} — {r['confidence']:.1f}%",
                            )

                    # ── GRAD-CAM HEATMAP (Enhancement #1) ──
                    st.markdown("#### 🔥 Model Attention Heatmap (Grad-CAM++)")
                    try:
                        from preprocess import preprocess_single_image
                        img_array = preprocess_single_image(tmp_path)
                        predicted_class_idx = int(np.argmax(avg_preds))

                        gradcam_overlay = generate_gradcam_overlay(
                            model, img_array, predicted_class_idx
                        )

                        import cv2
                        gradcam_rgb = cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)

                        col_gc1, col_gc2 = st.columns(2)
                        with col_gc1:
                            st.image(image, caption="Original Image", width=400)
                        with col_gc2:
                            st.image(
                                gradcam_rgb,
                                caption="Grad-CAM++ Heatmap (where the model is looking)",
                                width=400,
                            )
                        st.caption(
                            "🔴 Red/Yellow = high attention regions | 🔵 Blue = low attention. "
                            "The model should be focusing on the skin lesion, not the background."
                        )
                    except Exception as gradcam_err:
                        st.info(
                            f"ℹ️ Grad-CAM heatmap could not be generated: {gradcam_err}. "
                            "Install `tf-keras-vis` for explainability support."
                        )

                    # ── Symptoms ──
                    st.markdown("#### 🔍 Common Symptoms")
                    for s in info.get("symptoms", []):
                        st.markdown(f"- {s}")

                    # ── Recommendations ──
                    st.markdown("#### 💊 Recommendations")
                    for r in info.get("recommendations", []):
                        st.markdown(f"- {r}")

                    # ── OTC Medications ──
                    disease_key = disease_name.strip().lower()
                    medications = MEDICATION_MAP.get(disease_key, [])
                    if medications:
                        st.markdown("#### 💊 Suggested OTC Medications")
                        for med in medications:
                            st.markdown(f"- **{med['name']}** — {med['use']}")

                    # ── Clean up temp files ──
                    pathlib.Path(tmp_path).unlink(missing_ok=True)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(get_disclaimer())
