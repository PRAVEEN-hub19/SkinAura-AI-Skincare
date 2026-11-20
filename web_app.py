"""SkinAura browser experience.

Run with: streamlit run web_app.py
"""

import glob
import os
from datetime import date, timedelta

import streamlit as st
from PIL import Image

from inference import get_model_warning, predict_skin_type_details
from recommend import (
    DISCLAIMER,
    PROFILE_OPTIONS,
    WEATHER_CONDITIONS,
    get_recommendation,
    get_weather_advice,
    resolve_profile,
)

st.set_page_config(
    page_title="SkinAura | Skin intelligence",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialize_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "routine_streak" not in st.session_state:
        st.session_state.routine_streak = 0
    if "routine_last_date" not in st.session_state:
        st.session_state.routine_last_date = None


def show_result(result, weather):
    recommendation = get_recommendation(result["label"], weather)
    if result.get("model_warning"):
        st.warning(result["model_warning"])
    profile_label = "SELECTED PROFILE" if result.get("manual_profile") else "PHOTO ESTIMATE"
    st.markdown(f'<p class="panel-title">{profile_label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="result-type">{result["label"].title()}</p>', unsafe_allow_html=True)
    if result.get("manual_profile"):
        st.caption(f'Self-assessed profile · photo model estimated {result["predicted_label"].title()}')
    st.markdown(f'<p class="confidence">{result["confidence"]:.1f}% photo-model confidence</p>', unsafe_allow_html=True)
    st.markdown("**Photo-model matches**")
    for match in result["prediction_breakdown"]:
        st.progress(min(match["confidence"] / 100, 1.0), text=f'{match["label"].title()} · {match["confidence"]:.1f}%')
    st.write(recommendation["summary"])
    st.markdown(
        f'<div class="info-block"><div class="info-label">ROUTINE STARTERS</div>'
        f'<div class="info-value">1. {recommendation["product_a"]}<br>2. {recommendation["product_b"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Morning tasks**")
    for task in recommendation["morning_tasks"]:
        st.write(f"- {task}")
    st.markdown("**Evening tasks**")
    for task in recommendation["evening_tasks"]:
        st.write(f"- {task}")
    st.markdown(
        f'<div class="info-block"><div class="info-label">PRODUCT LINKS</div>'
        f'<div class="info-value"><a href="{recommendation["product_a_url"]}" target="_blank" rel="noopener noreferrer">{recommendation["product_a"]}</a><br>'
        f'<a href="{recommendation["product_b_url"]}" target="_blank" rel="noopener noreferrer">{recommendation["product_b"]}</a></div></div>',
        unsafe_allow_html=True,
    )
    tips = "<br>".join(f"- {tip}" for tip in recommendation["tips"])
    st.markdown(
        f'<div class="info-block"><div class="info-label">DAILY NOTES</div>'
        f'<div class="info-value">{tips}</div></div>', unsafe_allow_html=True,
    )
    st.info(f'Hydration: {recommendation["hydration_message"]}')
    st.markdown("**Balanced eating**")
    for tip in recommendation["diet_tips"]:
        st.write(f"- {tip}")


initialize_state()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink: #18212b; --muted: #687780; --green: #2c765e; --mint: #e5f1eb; --paper: #f7f8f6; --line: #e5e9e5; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: var(--ink); }
    [data-testid="stSidebar"] * { color: #e5ece8 !important; }
    [data-testid="stSidebar"] .stButton button { background: #2c765e; border: 0; text-align: left; }
    h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif !important; color: var(--ink) !important; }
    p, label, .stMarkdown, .stCaption, button { font-family: 'DM Sans', sans-serif !important; }
    .eyebrow { color: var(--green); font: 700 0.72rem 'DM Sans', sans-serif; letter-spacing: .14em; margin: 0 0 .55rem; }
    .lead { color: var(--muted); font: 1.05rem 'DM Sans', sans-serif; margin-top: -1rem; }
    .panel { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 1.35rem; height: 100%; }
    .panel-title { color: var(--green); font: 700 .75rem 'DM Sans', sans-serif; letter-spacing: .12em; margin-bottom: 1rem; }
    .result-type { font: 700 2.6rem 'Playfair Display', Georgia, serif; color: var(--ink); margin: 0; }
    .confidence { color: var(--green); font: 600 .88rem 'DM Sans', sans-serif; margin: .2rem 0 1.2rem; }
    .info-block { background: #f5f8f5; border-radius: 12px; padding: .85rem 1rem; margin-top: .8rem; }
    .info-label { color: var(--green); font: 700 .68rem 'DM Sans', sans-serif; letter-spacing: .1em; }
    .info-value { color: var(--ink); font: .94rem 'DM Sans', sans-serif; margin-top: .35rem; }
    div[data-testid="stFileUploader"] { background: #f8e9de; border-radius: 12px; padding: .35rem; }
    .stButton button { background: var(--green); color: white; border: 0; border-radius: 9px; font-weight: 700; padding: .65rem 1rem; }
    .stButton button:hover { background: #225b48; color: white; }
    @media (max-width: 760px) { .result-type { font-size: 2.1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## SkinAura")
    st.caption("AI SKIN INSIGHTS")
    st.divider()
    st.caption("PRIVATE BY DESIGN")
    st.write("Images stay in this app session and are not saved by SkinAura.")
    if st.session_state.history and st.button("Clear analysis history", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.rerun()

st.markdown('<p class="eyebrow">SKINAURA / PERSONAL ANALYSIS</p>', unsafe_allow_html=True)
st.title("Understand your skin.")
st.markdown('<p class="lead">A considered starting point for a better daily routine.</p>', unsafe_allow_html=True)

analyze_tab, routine_tab, history_tab = st.tabs(["Analyze", "Routine builder", "History"])

with analyze_tab:
    upload_col, result_col = st.columns([5, 6], gap="large")
    with upload_col:
        st.markdown('<div class="panel"><p class="panel-title">YOUR PHOTO</p>', unsafe_allow_html=True)
        source = st.radio("Input source", ["Upload", "Camera", "Demo image"], horizontal=True, label_visibility="collapsed")
        selected_profile = st.selectbox(
            "Profile for product and routine suggestions",
            options=list(PROFILE_OPTIONS),
            format_func=lambda value: PROFILE_OPTIONS[value],
        )
        uploaded_file = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed") if source == "Upload" else None
        camera_file = st.camera_input("Take a face photo") if source == "Camera" else None
        demo_files = glob.glob(os.path.join("dataset", "train", "*", "*.png"))
        demo_path = st.selectbox("Choose a generated demo image", ["No demo image"] + demo_files) if source == "Demo image" else "No demo image"
        if camera_file:
            image, image_name = Image.open(camera_file).convert("RGB"), "Camera capture"
        elif uploaded_file:
            image, image_name = Image.open(uploaded_file).convert("RGB"), uploaded_file.name
        elif demo_path != "No demo image":
            image, image_name = Image.open(demo_path).convert("RGB"), os.path.basename(demo_path)
        else:
            image, image_name = None, None
        if image:
            st.image(image, use_container_width=True)
            st.caption(f"{image_name}  ·  {image.width} x {image.height} px")
        else:
            st.info("Use a clear, front-facing image for the best available reading.")
        analyze = st.button("Analyze my skin", disabled=image is None, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="panel"><p class="panel-title">YOUR SKIN PROFILE</p>', unsafe_allow_html=True)
        if analyze and image:
            with st.spinner("Reading your image locally..."):
                try:
                    prediction_breakdown = predict_skin_type_details(image)
                    label = prediction_breakdown[0]["label"]
                    confidence = prediction_breakdown[0]["confidence"]
                    profile, manual_profile = resolve_profile(selected_profile, label)
                    result = {"label": profile, "predicted_label": label, "manual_profile": manual_profile, "confidence": confidence, "prediction_breakdown": prediction_breakdown, "model_warning": get_model_warning(), "recommendation": get_recommendation(profile), "source": image_name}
                    st.session_state.last_result = result
                    st.session_state.history.insert(0, result)
                except FileNotFoundError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Analysis failed: {error}")
        if st.session_state.last_result:
            weather = st.selectbox(
                "Today's weather",
                options=list(WEATHER_CONDITIONS),
                format_func=lambda value: WEATHER_CONDITIONS[value],
            )
            show_result(st.session_state.last_result, weather)
            st.info(get_weather_advice(weather, st.session_state.last_result["label"]))
        else:
            st.markdown('<p class="result-type">Ready when you are</p>', unsafe_allow_html=True)
            st.caption("Choose Upload, Camera, or Demo image to begin.")
            st.markdown('<div class="info-block"><div class="info-label">WHAT YOU GET</div><div class="info-value">A predicted skin type, confidence score, routine starters, and practical daily notes.</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with routine_tab:
    st.subheader("Build a routine you can keep")
    st.caption("Adjust the routine around your preferences. These are general categories, not medical advice.")
    left, right = st.columns(2)
    with left:
        routine_goal = st.selectbox("Primary goal", ["Balance and consistency", "Reduce visible shine", "Support hydration", "Keep it minimal"])
        product_texture = st.select_slider("Preferred texture", options=["Rich cream", "Cream", "Lotion", "Gel"], value="Lotion")
    with right:
        routine_time = st.selectbox("Time available", ["2 minutes", "5 minutes", "10 minutes"])
        sunscreen = st.toggle("Include daily sunscreen", value=True)
    if st.button("Generate my routine", use_container_width=True):
        steps = ["Gentle cleanser", f"{product_texture} moisturizer"]
        if sunscreen:
            steps.append("Broad-spectrum SPF 30+")
        st.success(f"Your {routine_goal.lower()} routine ({routine_time}):")
        for number, step in enumerate(steps, 1):
            st.write(f"**{number}.** {step}")
    today = date.today()
    last_done = st.session_state.routine_last_date
    streak_is_active = last_done in (today, today - timedelta(days=1))
    if last_done == today:
        st.success(f"Routine complete today · {st.session_state.routine_streak}-day streak")
    elif st.button("Mark today's routine complete", use_container_width=True):
        st.session_state.routine_streak = (
            st.session_state.routine_streak + 1
            if last_done == today - timedelta(days=1)
            else 1
        )
        st.session_state.routine_last_date = today
        st.rerun()
    else:
        active_streak = st.session_state.routine_streak if streak_is_active else 0
        st.caption(f"Current routine streak: {active_streak} day(s)")

with history_tab:
    st.subheader("Your session history")
    if not st.session_state.history:
        st.info("Your completed analyses will appear here for comparison.")
    else:
        st.caption(f"{len(st.session_state.history)} analysis(ies) in this session")
        for result in st.session_state.history:
            with st.container(border=True):
                history_left, history_right = st.columns([2, 5])
                history_left.metric(result["label"].title(), f'{result["confidence"]:.1f}%')
                history_right.write(f'**Source:** {result["source"]}')
                history_right.write(result["recommendation"]["summary"])

st.caption(DISCLAIMER)
