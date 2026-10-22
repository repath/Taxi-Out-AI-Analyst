import os
import json
import re


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Import prediction helpers from predict.py
from predict import load_deployment_artifacts, prepare_flight_features
from src.rag_engine import ProjectRAGSystem
from src.analytics import build_monthly_statistics


CHAT_AVATAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "chat")
CHAT_AVATARS = {
    "assistant": os.path.join(CHAT_AVATAR_DIR, "ai-analyst-avatar.png"),
    "user": os.path.join(CHAT_AVATAR_DIR, "user-avatar.png"),
}


# Set Streamlit Page Config
st.set_page_config(
    page_title="aircraft Taxi-Out | Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme switch (off keeps the current dark appearance)
light_mode = st.sidebar.toggle("Light background", value=False)
plotly_template = "plotly_white" if light_mode else "plotly_dark"
chart_text_color = "#1f2937" if light_mode else "#e2e8f0"

@st.cache_resource
def get_cached_model_artifacts(model_type):
    return load_deployment_artifacts(model_type=model_type)


@st.cache_resource
def get_cached_rag_engine(base_dir, engine_version):
    return ProjectRAGSystem(base_dir)


@st.cache_data(max_entries=256)
def get_cached_prediction(flight_info, model_type):
    pipeline, encoders = get_cached_model_artifacts(model_type)
    prepared = prepare_flight_features(flight_info, encoders)
    predicted_taxi_out = float(pipeline.predict(prepared)[0])
    dep_time = flight_info.get("DEP_TIME", flight_info.get("CRS_DEP_TIME"))
    total_minutes = (int(dep_time) // 100 * 60 + int(dep_time) % 100 + int(round(predicted_taxi_out))) % 1440
    return {
        "predicted_taxi_out_minutes": round(predicted_taxi_out, 1),
        "estimated_wheels_off": f"{total_minutes // 60:02d}:{total_minutes % 60:02d}",
        "scheduled_dep": f"{int(flight_info.get('CRS_DEP_TIME')) // 100:02d}:{int(flight_info.get('CRS_DEP_TIME')) % 100:02d}",
        "actual_dep_time": f"{int(dep_time) // 100:02d}:{int(dep_time) % 100:02d}",
    }


@st.cache_data(max_entries=128)
def get_cached_sensitivity_curve(flight_info, model_type):
    pipeline, encoders = get_cached_model_artifacts(model_type)
    temp_flight = flight_info.copy()
    congestion_values = list(range(1, 36))
    prepared_inputs = []
    for congestion in congestion_values:
        temp_flight["CONGESTION_1H"] = congestion
        temp_flight["CONGESTION_30M"] = congestion // 2
        prepared_inputs.append(prepare_flight_features(temp_flight, encoders))
    combined_inputs = pd.concat(prepared_inputs, ignore_index=True)
    predictions = pipeline.predict(combined_inputs).astype(float).tolist()
    return pd.DataFrame({
        "Pushbacks / Hour": congestion_values,
        "Predicted Taxi-Out (min)": predictions,
    })

# AI Analyst Chat Dialog Definition
def clean_ai_answer(content):
    """Removes legacy file-path headings from displayed chat answers."""
    return re.sub(
        r"^\s*\*\*From `[^`]+` \([^\n]+?\):\*\*\s*",
        "",
        content,
    ).strip()


def refresh_ai_chat_topic():
    """Keeps follow-up context aligned with the remaining chat history."""
    latest_topic = next(
        (
            message.get("topic")
            for message in reversed(st.session_state.get("dialog_chat_history", []))
            if message.get("role") == "assistant" and message.get("topic")
        ),
        None,
    )
    if latest_topic:
        st.session_state.ai_last_dashboard_topic = latest_topic
    else:
        st.session_state.pop("ai_last_dashboard_topic", None)


def delete_ai_chat_exchange(assistant_index):
    """Deletes one assistant response and its immediately preceding user question."""
    history = st.session_state.get("dialog_chat_history", [])
    if not 0 <= assistant_index < len(history):
        return
    start_index = assistant_index
    if assistant_index > 0 and history[assistant_index - 1].get("role") == "user":
        start_index = assistant_index - 1
    del history[start_index:assistant_index + 1]
    refresh_ai_chat_topic()


def clear_ai_chat_history():
    """Clears all displayed questions, answers, and their conversational context."""
    st.session_state.dialog_chat_history = []
    st.session_state.pop("ai_last_dashboard_topic", None)


@st.fragment(key="ai_analyst_chat")
def show_ai_analyst_dialog():
    st.markdown(
        '<div class="ai-chat-drag-handle">AI Analyst <span>Drag to move · resize at corner</span></div>',
        unsafe_allow_html=True,
    )

    st.caption("Ask about data, cleaning, features, models, metrics, predictions, congestion, routes, or project methods.")

    rag_engine_version = "authoritative-knowledge-router-v18"
    if st.session_state.get("rag_engine_version") != rag_engine_version:
        with st.spinner("Loading AI Analyst results..."):
            st.session_state.rag_engine = get_cached_rag_engine(
                os.path.dirname(os.path.abspath(__file__)),
                rag_engine_version,
            )
            st.session_state.rag_engine_version = rag_engine_version

    if "dialog_chat_history" not in st.session_state:
        st.session_state.dialog_chat_history = [
            {"role": "assistant", "content": "Hello! I am your **AI Analyst**. What would you like to know about this project?"}
        ]
    if st.session_state.get("source_visibility_version") != 2:
        for message in st.session_state.dialog_chat_history:
            message["show_sources"] = False
        st.session_state.source_visibility_version = 2

    chat_box = st.container(height="stretch", key="ai_chat_messages")
    with chat_box:
        for message_index, msg in enumerate(st.session_state.dialog_chat_history):
            with st.chat_message(msg["role"], avatar=CHAT_AVATARS.get(msg["role"])):
                st.markdown(clean_ai_answer(msg["content"]))
                if msg.get("show_sources") and msg.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        for s in msg["sources"]:
                            st.caption(f"- {s}")
                if (
                    msg.get("role") == "assistant"
                    and message_index > 0
                    and st.session_state.dialog_chat_history[message_index - 1].get("role") == "user"
                ):
                    st.button(
                        "Delete Q&A",
                        key=f"ai_chat_delete_pair_{message_index}",
                        on_click=delete_ai_chat_exchange,
                        args=(message_index,),
                    )

    st.iframe(
        """
        <button id="speech-read">Read latest response</button>
        <button id="clear-all">Clear all</button>
        <style>
            html, body {
                display: flex; gap: 6px; margin: 0; padding: 0;
                background: transparent;
            }
            button {
                border: 1px solid #38bdf8; border-radius: 10px; padding: 3px 8px;
                background: #172033; color: #e5e7eb; font: 11px/1.15 "Segoe UI", sans-serif;
                cursor: pointer;
            }
            #clear-all {
                border-color: #f97316;
                background: #7c2d12;
            }
        </style>
        <script>
            class AIChatController {
                constructor(frameWindow, parentWindow, readButton, clearButton) {
                    this.frameWindow = frameWindow;
                    this.parentWindow = parentWindow;
                    this.parentDocument = parentWindow.document;
                    this.readButton = readButton;
                    this.clearButton = clearButton;
                    this.panel = null;
                    this.dragHandle = null;
                    this.resizeObserver = null;
                    this.dragging = false;
                    this.pointerId = null;
                    this.offsetX = 0;
                    this.offsetY = 0;
                    this.viewportMargin = 12;

                    this.readLatestResponse = this.readLatestResponse.bind(this);
                    this.clearAll = this.clearAll.bind(this);
                    this.startDragging = this.startDragging.bind(this);
                    this.movePanel = this.movePanel.bind(this);
                    this.stopDragging = this.stopDragging.bind(this);
                    this.keepPanelInViewport = this.keepPanelInViewport.bind(this);
                    this.destroy = this.destroy.bind(this);
                }

                init() {
                    if (this.parentDocument.querySelector(".light-chat-theme-override")) {
                        this.readButton.style.background = "#ffffff";
                        this.readButton.style.color = "#1f2937";
                        this.readButton.style.borderColor = "#0284c7";
                        this.clearButton.style.background = "#ffffff";
                        this.clearButton.style.color = "#9a3412";
                        this.clearButton.style.borderColor = "#f97316";
                    }

                    this.readButton.addEventListener("click", this.readLatestResponse);
                    this.clearButton.addEventListener("click", this.clearAll);
                    this.frameWindow.addEventListener("pagehide", this.destroy);

                    this.panel = this.parentDocument.querySelector(".st-key-ai_chat_panel");
                    this.dragHandle = this.panel
                        ? this.panel.querySelector(".ai-chat-drag-handle")
                        : null;
                    if (!this.panel || !this.dragHandle) {
                        return;
                    }

                    this.dragHandle.addEventListener("pointerdown", this.startDragging);
                    this.dragHandle.addEventListener("pointermove", this.movePanel);
                    this.dragHandle.addEventListener("pointerup", this.stopDragging);
                    this.dragHandle.addEventListener("pointercancel", this.stopDragging);

                    if (this.parentWindow.ResizeObserver) {
                        this.resizeObserver = new this.parentWindow.ResizeObserver(
                            this.keepPanelInViewport
                        );
                        this.resizeObserver.observe(this.panel);
                    }
                    this.parentWindow.addEventListener("resize", this.keepPanelInViewport);
                }

                readLatestResponse() {
                    const speech = this.parentWindow.speechSynthesis;
                    if (!speech) {
                        return;
                    }
                    if (speech.speaking || speech.pending) {
                        speech.cancel();
                        this.readButton.textContent = "Read latest response";
                        return;
                    }

                    const messages = this.panel
                        ? this.panel.querySelectorAll('[data-testid="stChatMessage"]')
                        : [];
                    if (!messages.length) {
                        return;
                    }

                    let message = messages[messages.length - 1];
                    for (let index = messages.length - 1; index >= 0; index -= 1) {
                        const label = (
                            messages[index].getAttribute("aria-label") || ""
                        ).toLowerCase();
                        if (label.includes("assistant")) {
                            message = messages[index];
                            break;
                        }
                    }

                    speech.cancel();
                    const utterance = new this.parentWindow.SpeechSynthesisUtterance(
                        message.innerText
                    );
                    utterance.rate = 1;
                    this.readButton.textContent = "Stop reading";
                    const resetButton = () => {
                        this.readButton.textContent = "Read latest response";
                    };
                    utterance.onend = resetButton;
                    utterance.onerror = resetButton;
                    speech.speak(utterance);
                }

                clearAll() {
                    if (this.parentWindow.speechSynthesis) {
                        this.parentWindow.speechSynthesis.cancel();
                    }
                    const hiddenClearButton = this.parentDocument.querySelector(
                        ".st-key-ai_chat_clear button"
                    );
                    if (hiddenClearButton) {
                        hiddenClearButton.click();
                    }
                }

                startDragging(event) {
                    event.preventDefault();
                    const rect = this.panel.getBoundingClientRect();
                    this.panel.style.setProperty("left", `${rect.left}px`, "important");
                    this.panel.style.setProperty("top", `${rect.top}px`, "important");
                    this.panel.style.setProperty("right", "auto", "important");
                    this.panel.style.setProperty("bottom", "auto", "important");
                    this.offsetX = event.clientX - rect.left;
                    this.offsetY = event.clientY - rect.top;
                    this.dragging = true;
                    this.pointerId = event.pointerId;
                    this.dragHandle.setPointerCapture(event.pointerId);
                    this.parentDocument.body.style.userSelect = "none";
                }

                movePanel(event) {
                    if (!this.dragging) {
                        return;
                    }
                    const bounds = this.getPanelBounds();
                    const left = Math.min(
                        bounds.maxLeft,
                        Math.max(bounds.minLeft, event.clientX - this.offsetX)
                    );
                    const top = Math.min(
                        bounds.maxTop,
                        Math.max(bounds.minTop, event.clientY - this.offsetY)
                    );
                    this.panel.style.setProperty("left", `${left}px`, "important");
                    this.panel.style.setProperty("top", `${top}px`, "important");
                }

                stopDragging() {
                    if (
                        this.dragHandle &&
                        this.pointerId !== null &&
                        this.dragHandle.hasPointerCapture(this.pointerId)
                    ) {
                        this.dragHandle.releasePointerCapture(this.pointerId);
                    }
                    this.dragging = false;
                    this.pointerId = null;
                    this.parentDocument.body.style.userSelect = "";
                }

                getPanelBounds() {
                    const visibleHandleWidth = Math.min(120, this.panel.offsetWidth);
                    const handleHeight = this.dragHandle.offsetHeight || 36;
                    return {
                        minLeft: this.viewportMargin - this.panel.offsetWidth + visibleHandleWidth,
                        maxLeft: this.parentWindow.innerWidth - visibleHandleWidth - this.viewportMargin,
                        minTop: this.viewportMargin,
                        maxTop: this.parentWindow.innerHeight - handleHeight - this.viewportMargin,
                    };
                }

                keepPanelInViewport() {
                    if (!this.panel || !this.dragHandle) {
                        return;
                    }
                    const rect = this.panel.getBoundingClientRect();
                    const bounds = this.getPanelBounds();
                    const left = Math.min(bounds.maxLeft, Math.max(bounds.minLeft, rect.left));
                    const top = Math.min(bounds.maxTop, Math.max(bounds.minTop, rect.top));
                    this.panel.style.setProperty("left", `${left}px`, "important");
                    this.panel.style.setProperty("top", `${top}px`, "important");
                    this.panel.style.setProperty("right", "auto", "important");
                    this.panel.style.setProperty("bottom", "auto", "important");
                }

                destroy() {
                    this.readButton.removeEventListener("click", this.readLatestResponse);
                    this.clearButton.removeEventListener("click", this.clearAll);
                    this.frameWindow.removeEventListener("pagehide", this.destroy);
                    this.parentWindow.removeEventListener("resize", this.keepPanelInViewport);

                    if (this.dragHandle) {
                        this.dragHandle.removeEventListener("pointerdown", this.startDragging);
                        this.dragHandle.removeEventListener("pointermove", this.movePanel);
                        this.dragHandle.removeEventListener("pointerup", this.stopDragging);
                        this.dragHandle.removeEventListener("pointercancel", this.stopDragging);
                    }
                    if (this.resizeObserver) {
                        this.resizeObserver.disconnect();
                        this.resizeObserver = null;
                    }
                    if (this.dragging) {
                        this.parentDocument.body.style.userSelect = "";
                    }
                    if (this.parentWindow.speechSynthesis) {
                        this.parentWindow.speechSynthesis.cancel();
                    }
                    if (this.parentWindow.__aiAnalystChatController === this) {
                        this.parentWindow.__aiAnalystChatController = null;
                    }
                }
            }

            const previousController = window.parent.__aiAnalystChatController;
            if (previousController && typeof previousController.destroy === "function") {
                previousController.destroy();
            }

            const controller = new AIChatController(
                window,
                window.parent,
                document.getElementById("speech-read"),
                document.getElementById("clear-all")
            );
            window.parent.__aiAnalystChatController = controller;
            controller.init();
        </script>
        """,
        height=28,
        alt="Read latest response",
    )
    st.button(
        "Clear all",
        key="ai_chat_clear",
        on_click=clear_ai_chat_history,
    )

    chat_query = st.chat_input(
        "Type your question here...",
        key="dialog_chat_input",
    )
    submitted_query = chat_query
    if submitted_query:
        show_sources = st.session_state.rag_engine._sources_requested(submitted_query)
        previous_user_query = next(
            (
                message["content"]
                for message in reversed(st.session_state.dialog_chat_history)
                if message.get("role") == "user"
            ),
            None,
        )
        last_dashboard_topic = st.session_state.get("ai_last_dashboard_topic")
        previous_assistant = next(
            (
                message
                for message in reversed(st.session_state.dialog_chat_history)
                if message.get("role") == "assistant" and message.get("sources")
            ),
            {},
        )
        if not last_dashboard_topic:
            for source in previous_assistant.get("sources", []):
                topic_marker = "ai_analyst_knowledge.md - "
                if topic_marker in source:
                    last_dashboard_topic = source.split(topic_marker, 1)[1].rstrip("]")
                    break

        st.session_state.dialog_chat_history.append({"role": "user", "content": submitted_query})
        with chat_box:
            with st.chat_message("user", avatar=CHAT_AVATARS["user"]):
                st.markdown(submitted_query)
            with st.chat_message("assistant", avatar=CHAT_AVATARS["assistant"]):
                with st.spinner("Analyzing project results..."):
                    selection_context = {
                        "carrier": carrier,
                        "origin": origin,
                        "destination": dest,
                        "flight_date": fl_date.strftime("%Y-%m-%d"),
                        "scheduled_departure": int(crs_dep_time),
                        "departure_delay": float(dep_delay),
                        "distance": float(distance),
                        "previous_user_query": previous_user_query,
                        "last_dashboard_topic": last_dashboard_topic,
                        "previous_sources": previous_assistant.get("sources", []),
                    }
                    res = st.session_state.rag_engine.answer_query(
                        submitted_query,
                        context=selection_context,
                        flight_data=analysis_df
                    )
                    cleaned_answer = clean_ai_answer(res["answer"])
                    st.markdown(cleaned_answer)
                    if show_sources and res.get("sources"):
                        with st.expander("📚 Sources", expanded=False):
                            for s in res["sources"]:
                                st.caption(f"- {s}")
                    if res.get("topic"):
                        st.session_state.ai_last_dashboard_topic = res["topic"]
                    st.session_state.dialog_chat_history.append({
                        "role": "assistant",
                        "content": cleaned_answer,
                        "sources": res.get("sources", []),
                        "show_sources": show_sources,
                        "topic": res.get("topic"),
                    })
                    assistant_index = len(st.session_state.dialog_chat_history) - 1
                    st.button(
                        "Delete Q&A",
                        key=f"ai_chat_delete_pair_{assistant_index}",
                        on_click=delete_ai_chat_exchange,
                        args=(assistant_index,),
                    )





def render_chart(fig):
    fig.update_layout(
        font=dict(family="Segoe UI, sans-serif", color=chart_text_color),
        legend=dict(font=dict(color=chart_text_color)),
        xaxis=dict(
            rangemode="tozero",
            title_font=dict(color=chart_text_color),
            tickfont=dict(color=chart_text_color)
        ),
        yaxis=dict(
            rangemode="tozero",
            title_font=dict(color=chart_text_color),
            tickfont=dict(color=chart_text_color)
        )
    )
    st.plotly_chart(fig, width="stretch")

def apply_hour_axis(fig, title):
    hour_values = list(range(24))
    hour_labels = [str(hour) for hour in range(1, 25)]
    fig.update_xaxes(
        tickmode="array",
        tickvals=hour_values,
        ticktext=hour_labels,
        range=[-0.5, 23.5],
        tickangle=0,
        title=title
    )
# Apply Custom Glassmorphism & Aviation Modern Theme Styling
st.markdown("""
<style>
    :root {
        --ai-chat-line-height: 1.15;
        --ai-chat-input-border: #f97316;
        --ai-chat-input-action: #0284c7;
    }

    /* Dark aviation theme styling */
    .stApp, .stApp button, .stApp input, .stApp textarea, .stApp select {
        font-family: "Segoe UI", sans-serif !important;
    }
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Reduce unused space above the dashboard */
    section.main > div.block-container {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }
    
    /* Header card */
    .main-header {
        width: fit-content;
        max-width: 100%;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 0.25rem 0.25rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 0.25rem;
    }
    .main-header h1 {
        color: #38bdf8;
        font-weight: 600;
        font-size: 1.55rem;
        margin-bottom: 0rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.74rem;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .metric-sub {
        color: #64748b;
        font-size: 0.68rem;
    }

    /* Congestion Badges */

    .badge-low { color: #4ade80; font-weight: bold; }
    .badge-mod { color: #facc15; font-weight: bold; }
    .badge-high { color: #fb923c; font-weight: bold; }
    .badge-severe { color: #f87171; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Light theme overrides
if light_mode:
    st.markdown("""
    <style class="light-theme-override">
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #f5f7fb !important;
            color: #1f2937 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            color: #1f2937 !important;
        }
        .main-header {
            background: linear-gradient(135deg, #ffffff 0%, #e8eef7 100%) !important;
            border-color: #d8dee9 !important;
            box-shadow: 0 8px 32px rgba(15, 23, 42, 0.10) !important;
        }
        .main-header h1, .metric-value {
            color: #0284c7 !important;
        }
        .main-header p, .metric-title {
            color: #526174 !important;
        }
        .metric-card {
            background: #ffffff !important;
            border-color: #d8dee9 !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.10) !important;
        }
        .badge-low { color: #16a34a !important; }
        .badge-mod { color: #ca8a04 !important; }
        .badge-high { color: #ea580c !important; }
        .badge-severe { color: #dc2626 !important; }
        .stApp [data-testid="stMarkdownContainer"] *,
        .stApp [data-testid="stWidgetLabel"] *,
        .stApp [data-testid="stCaptionContainer"] *,
        [data-baseweb="tab"] *, [role="tab"] *,
        label *,
        [data-baseweb="tab"], [role="tab"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stCaptionContainer"],
        label {
            color: #1f2937 !important;
        }
        [data-baseweb="select"] *,
        [data-baseweb="input"] *,
        [data-baseweb="textarea"] * {
            color: #1f2937 !important;
        }
        [data-testid="stHeader"] {
            background-color: #f5f7fb !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input {
            background-color: #ffffff !important;
            color: #1f2937 !important;
            border-color: #cbd5e1 !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="select"] span,
        [data-baseweb="input"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input {
            color: #1f2937 !important;
        }
        [data-baseweb="select"] svg {
            fill: #475569 !important;
        }
        [data-testid="stNumberInput"] button {
            background-color: #ffffff !important;
            color: #1f2937 !important;
            border-color: #cbd5e1 !important;
        }
        [data-baseweb="menu"],
        [role="listbox"],
        [role="option"] {
            background-color: #ffffff !important;
            color: #1f2937 !important;
        }
    </style>
    """, unsafe_allow_html=True)
# Load metrics json if available
@st.cache_data
def load_metrics_data():
    metrics_path = os.path.join("models", "model_evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None

metrics_json = load_metrics_data()

@st.cache_data
def load_cleaning_audit():
    audit_path = os.path.join("data", "processed", "cleaning_audit_summary.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            return json.load(f)
    return {}

data_audit = load_cleaning_audit()

@st.cache_data
def load_dashboard_data():
    data_path = os.path.join("data", "processed", "cleaned_flights.csv")
    if not os.path.exists(data_path):
        return {}, pd.DataFrame()

    columns = [
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "DISTANCE"
    ]
    raw_df = pd.read_csv(data_path, usecols=columns)

    exploration_df = raw_df[
        ["OP_UNIQUE_CARRIER", "ORIGIN", "CRS_DEP_TIME", "TAXI_OUT"]
    ].copy()
    exploration_df["TAXI_OUT"] = pd.to_numeric(exploration_df["TAXI_OUT"], errors="coerce")
    exploration_df["CRS_DEP_TIME"] = pd.to_numeric(exploration_df["CRS_DEP_TIME"], errors="coerce")
    exploration_df = exploration_df.dropna(subset=["TAXI_OUT", "CRS_DEP_TIME"]).copy()
    exploration_df["DEP_HOUR"] = (
        (exploration_df["CRS_DEP_TIME"] // 100).clip(0, 23).astype(int)
    )
    carrier_summary = (exploration_df.groupby("OP_UNIQUE_CARRIER", as_index=False)["TAXI_OUT"]
        .mean().rename(columns={"TAXI_OUT": "Mean Taxi-Out"})
        .sort_values("Mean Taxi-Out", ascending=True))
    hour_summary = (exploration_df.groupby("DEP_HOUR", as_index=False)["TAXI_OUT"]
        .mean().rename(columns={"TAXI_OUT": "Mean Taxi-Out"}))
    origin_summary = (exploration_df.groupby("ORIGIN", as_index=False)["TAXI_OUT"]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "Mean Taxi-Out", "count": "Flights"})
        .sort_values("Mean Taxi-Out", ascending=False).head(10)
        .sort_values("Mean Taxi-Out", ascending=True))
    exploration_data = {
        "carrier": carrier_summary,
        "hour": hour_summary,
        "origin": origin_summary,
    }

    df = raw_df
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], dayfirst=True, errors="coerce")
    for column in ["CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT", "DISTANCE"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["FL_DATE", "CRS_DEP_TIME", "DEP_DELAY", "TAXI_OUT"]).copy()
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int).mod(24)
    df["MONTH_NUM"] = df["FL_DATE"].dt.month
    df["MONTH"] = df["FL_DATE"].dt.strftime("%b")
    df["DAY_NAME"] = df["FL_DATE"].dt.day_name()
    df["ROUTE"] = df["ORIGIN"] + " → " + df["DEST"]
    df["HIGH_RISK"] = df["TAXI_OUT"] >= 30
    return exploration_data, df

exploration_data, analysis_df = load_dashboard_data()


@st.cache_data
def load_chart_aggregations():
    _, df = load_dashboard_data()
    if df.empty:
        return {}

    origin_stats = (
        df.groupby("ORIGIN", as_index=False)
        .agg(
            Flights=("TAXI_OUT", "size"),
            Mean_Taxi_Out=("TAXI_OUT", "mean"),
            Median_Taxi_Out=("TAXI_OUT", "median"),
            P95_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.95)),
        )
        .sort_values("Mean_Taxi_Out", ascending=True)
    )

    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    heat_df = (
        df.pivot_table(
            index="DAY_NAME",
            columns="DEP_HOUR",
            values="TAXI_OUT",
            aggfunc="mean",
        )
        .reindex(day_order)
    )

    route_stats = (
        df.groupby(["ORIGIN", "DEST"], as_index=False)
        .agg(
            Flights=("TAXI_OUT", "size"),
            Mean_Taxi_Out=("TAXI_OUT", "mean"),
            P95_Taxi_Out=("TAXI_OUT", lambda values: values.quantile(0.95)),
        )
    )
    route_stats["ROUTE"] = route_stats["ORIGIN"] + " → " + route_stats["DEST"]
    busiest_routes = (
        route_stats.sort_values("Flights", ascending=False)
        .head(12)
        .sort_values("Flights")
    )

    risk_by_hour = (
        df.groupby("DEP_HOUR", as_index=False)
        .agg(
            Flights=("TAXI_OUT", "size"),
            High_Risk_Rate=("HIGH_RISK", "mean"),
        )
    )
    risk_by_hour["High_Risk_Rate"] *= 100

    return {
        "monthly": build_monthly_statistics(df),
        "mean_taxi_out": df["TAXI_OUT"].mean(),
        "median_taxi_out": df["TAXI_OUT"].median(),
        "p95_taxi_out": df["TAXI_OUT"].quantile(0.95),
        "high_risk_pct": df["HIGH_RISK"].mean() * 100,
        "origin": origin_stats,
        "heat": heat_df,
        "routes": busiest_routes,
        "risk_by_hour": risk_by_hour,
        "correlation": df["DEP_DELAY"].corr(df["TAXI_OUT"]),
    }


chart_aggregations = load_chart_aggregations()



@st.cache_data
def load_model_test_predictions(model_type):
    test_path = os.path.join("data", "processed", "test_features.csv")
    if not os.path.exists(test_path):
        return pd.DataFrame()

    feature_columns = [
        "DEP_DELAY", "DISTANCE", "CONGESTION_1H", "CONGESTION_30M",
        "SIN_HOUR", "COS_HOUR", "ORIGIN_TAXI_MEAN", "CARRIER_TAXI_MEAN",
        "OP_UNIQUE_CARRIER", "ORIGIN", "DAY_OF_WEEK"
    ]
    test_df = pd.read_csv(test_path)
    if "TAXI_OUT" not in test_df.columns or any(column not in test_df.columns for column in feature_columns):
        return pd.DataFrame()

    pipeline, _ = get_cached_model_artifacts(model_type)
    predictions = pipeline.predict(test_df[feature_columns])
    return pd.DataFrame({
        "Actual Taxi-Out": pd.to_numeric(test_df["TAXI_OUT"], errors="coerce"),
        "Predicted Taxi-Out": predictions
    }).dropna()
# Sidebar Configuration
st.sidebar.title("🎛️ Flight Parameters")

model_choice = st.sidebar.radio(
    "Prediction Model Engine",
    [
        "Random Forest Regressor",
        "XGBoost Regressor (Recommended)",
        "Linear Regression Baseline"
    ]
)
model_type = (
    "rf" if "Random Forest" in model_choice
    else "xgb" if "XGBoost" in model_choice
    else "lr"
)


preset = st.sidebar.selectbox(
    "Load Scenario Preset",
    ["Custom", "DFW Morning Peak Push (AA to ORD)", "DFW Midday Off-Peak (AA to MIA)", "JFK Evening Congested Peak (B6 to LAX)"]
)

# Preset defaults
default_carrier = "AA"
default_origin = "DFW"
default_dest = "ORD"
default_crs_dep = 830
default_dep_time = 835
default_dep_delay = 5.0
default_distance = 802.0
default_cong_1h = 18
default_cong_30m = 10

if preset == "DFW Morning Peak Push (AA to ORD)":
    default_carrier, default_origin, default_dest = "AA", "DFW", "ORD"
    default_crs_dep, default_dep_time, default_dep_delay = 830, 835, 5.0
    default_distance, default_cong_1h, default_cong_30m = 802.0, 18, 10
elif preset == "DFW Midday Off-Peak (AA to MIA)":
    default_carrier, default_origin, default_dest = "AA", "DFW", "MIA"
    default_crs_dep, default_dep_time, default_dep_delay = 1400, 1400, 0.0
    default_distance, default_cong_1h, default_cong_30m = 1121.0, 5, 2
elif preset == "JFK Evening Congested Peak (B6 to LAX)":
    default_carrier, default_origin, default_dest = "B6", "JFK", "LAX"
    default_crs_dep, default_dep_time, default_dep_delay = 1845, 1910, 25.0
    default_distance, default_cong_1h, default_cong_30m = 2475.0, 22, 12

st.sidebar.subheader("Flight Identifiers")

carrier_names = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines"
}
airport_names = {
    "DFW": "DALLAS/FORT WORTH - DFW",
    "ATL": "ATLANTA - ATL",
    "ORD": "CHICAGO O'HARE - ORD",
    "DEN": "DENVER - DEN",
    "LAX": "LOS ANGELES - LAX",
    "CLT": "CHARLOTTE - CLT",
    "JFK": "NEW YORK (JFK) - JFK",
    "LGA": "NEW YORK (LAGUARDIA) - LGA",
    "SFO": "SAN FRANCISCO - SFO",
    "PHX": "PHOENIX - PHX"
}
carrier_options = list(carrier_names.values())
airport_options = list(airport_names.values())
carrier_code_by_label = {label: code for code, label in carrier_names.items()}
airport_code_by_label = {label: code for code, label in airport_names.items()}

flight_row1_col1, flight_row1_col2 = st.sidebar.columns(2)
with flight_row1_col1:
    carrier_label = st.selectbox(
        "Carrier",
        carrier_options,
        index=carrier_options.index(carrier_names.get(default_carrier, carrier_names["AA"]))
    )
    carrier = carrier_code_by_label[carrier_label]
with flight_row1_col2:
    fl_date = st.date_input("Flight Date", value=pd.to_datetime("2025-11-15"))

flight_row2_col1, flight_row2_col2 = st.sidebar.columns(2)
with flight_row2_col1:
    origin_label = st.selectbox(
        "Departure",
        airport_options,
        index=airport_options.index(airport_names.get(default_origin, airport_names["DFW"]))
    )
    origin = airport_code_by_label[origin_label]
with flight_row2_col2:
    dest_label = st.selectbox(
        "Arrival",
        airport_options,
        index=airport_options.index(airport_names.get(default_dest, airport_names["DFW"]))
    )
    dest = airport_code_by_label[dest_label]

st.sidebar.subheader("Timing & Airfield Congestion")
timing_col1, timing_col2, timing_col3 = st.sidebar.columns(3)
with timing_col1:
    crs_dep_time = st.number_input("Scheduled Departure (HHMM)", min_value=0, max_value=2359, value=int(default_crs_dep), step=5)
with timing_col2:
    dep_time = st.number_input("Actual Departure (HHMM)", min_value=0, max_value=2359, value=int(default_dep_time), step=5)
with timing_col3:
    dep_delay = st.number_input("Observed Dep. Delay (minutes)", value=float(default_dep_delay), step=1.0)
distance = st.sidebar.number_input("Stage Distance (miles)", value=float(default_distance), step=50.0)

st.sidebar.subheader("Surface Traffic Density")
cong_1h = st.sidebar.slider("Origin Pushbacks Past 1 Hr", min_value=1, max_value=40, value=int(default_cong_1h))
cong_30m = st.sidebar.slider("Origin Pushbacks Past 30 Min", min_value=0, max_value=25, value=int(default_cong_30m))


# Create flight input dictionary
flight_info = {
    "FL_DATE": fl_date.strftime("%Y-%m-%d"),
    "OP_UNIQUE_CARRIER": carrier,
    "ORIGIN": origin,
    "DEST": dest.upper(),
    "CRS_DEP_TIME": int(crs_dep_time),
    "DEP_TIME": int(dep_time),
    "DEP_DELAY": float(dep_delay),
    "DISTANCE": float(distance),
    "CONGESTION_1H": int(cong_1h),
    "CONGESTION_30M": int(cong_30m)
}

# Run Inference
try:
    prediction = get_cached_prediction(flight_info, model_type)
    pred_val = prediction["predicted_taxi_out_minutes"]
    wheels_off = prediction["estimated_wheels_off"]
    sched_dep_str = prediction["scheduled_dep"]
    act_dep_str = prediction["actual_dep_time"]
except Exception as e:
    st.error(f"Inference error: {e}")
    st.stop()

# Determine Congestion Level
if cong_1h < 8:
    cong_label = "Low (Free Flow)"
    cong_class = "badge-low"
elif cong_1h < 15:
    cong_label = "Moderate (Standard Push)"
    cong_class = "badge-mod"
elif cong_1h < 22:
    cong_label = "High (Peak Bank)"
    cong_class = "badge-high"
else:
    cong_label = "Severe Queueing"
    cong_class = "badge-severe"

# Compact top summary row
top_header, top_kpis = st.columns([1.1, 2.9])
with top_header:
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1>aircraft TAXI-OUT</h1>
                <p>Surface Trajectory & Congestion Predictions</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
with top_kpis:
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Taxi-Out</div>
            <div class="metric-value">{pred_val:.1f} <span style="font-size:1.1rem; color:#94a3b8;">min</span></div>
            <div class="metric-sub">Pushback to Wheels-off</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Actual Pushback</div>
            <div class="metric-value">{act_dep_str}</div>
            <div class="metric-sub">Sched: {sched_dep_str} ({dep_delay:+.0f}m)</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Est. Wheels-Off</div>
            <div class="metric-value" style="color: #34d399;">{wheels_off}</div>
            <div class="metric-sub">Airborne timestamp</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Surface Congestion</div>
            <div class="metric-value" style="font-size:1.05rem; padding-top:0.25rem;">
                <span class="{cong_class}">{cong_label}</span>
            </div>
            <div class="metric-sub">{cong_1h} pushbacks / hour</div>
        </div>
        """, unsafe_allow_html=True)
        

st.markdown("<div style=\"height: 0.25rem; clear: both;\"></div>", unsafe_allow_html=True)

# Main Layout Tabs
tab0, tab5, tab1, tab2, tab3 = st.tabs([
    "📋 Data Understanding",
    "Exploratory Analysis",
    "✈️ Dispatch Simulator",
    "📊 Model Performance & Benchmarks",
    "🚦 Congestion Analytics",
])

# ==============================================================================
# TAB 0: DATA UNDERSTANDING
# ====================================================================
with tab0:
    st.markdown("### 📋 Data Understanding & Quality Overview")
    st.caption("A quick view of the source data, cleaning pipeline, and taxi-out target before using the prediction and analysis tabs.")

    raw_count = int(data_audit.get("raw_record_count", 0))
    final_count = int(data_audit.get("final_cleaned_count", 0))
    retention = float(data_audit.get("retention_rate_pct", 0.0))
    target_stats = data_audit.get("post_clean_stats", {})
    percentile_stats = data_audit.get("percentile_inspection_before_outlier_treatment", {})

    data_kpi1, data_kpi2, data_kpi3, data_kpi4 = st.columns(4)
    with data_kpi1:
        st.metric("Raw Records", f"{raw_count:,}")
    with data_kpi2:
        st.metric("Clean Records", f"{final_count:,}")
    with data_kpi3:
        st.metric("Retention Rate", f"{retention:.2f}%")
    with data_kpi4:
        st.metric("Mean Taxi-Out", f"{float(target_stats.get('mean_taxi_out', 0.0)):.1f} min")

    data_left, data_right = st.columns(2)
    with data_left:
        st.subheader("Data Cleaning Funnel")
        after_cancelled = raw_count - int(data_audit.get("cancelled_flights_removed", 0))
        after_diverted = after_cancelled - int(data_audit.get("diverted_flights_removed", 0))
        after_invalid = after_diverted - int(data_audit.get("invalid_or_missing_taxi_out_removed", 0))
        funnel_df = pd.DataFrame({
            "Stage": ["Raw records", "After cancelled removal", "After diverted removal", "After invalid target removal", "Final cleaned records"],
            "Records": [raw_count, after_cancelled, after_diverted, after_invalid, final_count]
        })
        fig_funnel = px.bar(funnel_df, x="Records", y="Stage", orientation="h", text="Records", color="Records", color_continuous_scale="Blues")
        fig_funnel.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
        render_chart(fig_funnel)

    with data_right:
        st.subheader("Taxi-Out Target Reference")
        target_df = pd.DataFrame({
            "Reference": ["Minimum", "Median", "Mean", "P75", "P90", "P95", "Maximum"],
            "Minutes": [
                float(target_stats.get("min_taxi_out", 0.0)),
                float(target_stats.get("median_taxi_out", 0.0)),
                float(target_stats.get("mean_taxi_out", 0.0)),
                float(percentile_stats.get("p75", 0.0)),
                float(percentile_stats.get("p90", 0.0)),
                float(percentile_stats.get("p95", 0.0)),
                float(target_stats.get("max_taxi_out", 0.0))
            ]
        })
        fig_target = px.bar(target_df, x="Reference", y="Minutes", text="Minutes", color="Minutes", color_continuous_scale="Teal")
        fig_target.update_traces(texttemplate="%{text:.1f}")
        fig_target.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
        render_chart(fig_target)

    st.markdown("---")
    st.subheader("Exploratory Patterns")
    st.caption("These summaries show where taxi-out time tends to be higher across the cleaned flight records.")
    if exploration_data:
        exp_left, exp_right = st.columns(2)
        with exp_left:
            carrier_df = exploration_data["carrier"]
            fig_carrier = px.bar(carrier_df, x="Mean Taxi-Out", y="OP_UNIQUE_CARRIER", orientation="h", text="Mean Taxi-Out", color="Mean Taxi-Out", color_continuous_scale="Blues")
            fig_carrier.update_traces(texttemplate="%{text:.1f}")
            fig_carrier.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
            render_chart(fig_carrier)
        with exp_right:
            hour_df = exploration_data["hour"]
            fig_hour_explore = px.line(hour_df, x="DEP_HOUR", y="Mean Taxi-Out", markers=True, title="Mean Taxi-Out by Departure Hour")
            fig_hour_explore.update_traces(line_color="#38bdf8", line_width=3)
            fig_hour_explore.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
            apply_hour_axis(fig_hour_explore, "Departure hour")
            render_chart(fig_hour_explore)

        origin_left, origin_right = st.columns(2)
        with origin_left:
            st.subheader("Highest Mean Taxi-Out Origins")
            origin_df = exploration_data["origin"]
            fig_origin = px.bar(origin_df, x="Mean Taxi-Out", y="ORIGIN", orientation="h", text="Mean Taxi-Out", color="Mean Taxi-Out", color_continuous_scale="Teal")
            fig_origin.update_traces(texttemplate="%{text:.1f}", hovertemplate="Origin: %{y}<br>Mean taxi-out: %{x:.1f} min<br>Flights: %{customdata:,}<extra></extra>", customdata=origin_df["Flights"])
            fig_origin.update_layout(template=plotly_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)
            render_chart(fig_origin)

        with origin_right:
            st.subheader("Monthly Taxi-Out Trend")
            if analysis_df.empty:
                st.warning("Monthly analysis is unavailable because the cleaned flight data was not found.")
            else:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                monthly_stats = chart_aggregations["monthly"]
                fig_monthly = px.line(
                    monthly_stats,
                    x="MONTH",
                    y="Mean_Taxi_Out",
                    markers=True,
                    text="Mean_Taxi_Out",
                    category_orders={"MONTH": month_order},
                    labels={
                        "MONTH": "Month",
                        "Mean_Taxi_Out": "Mean taxi-out (minutes)"
                    },
                    hover_data=["Flights", "P25_Taxi_Out", "P50_Taxi_Out", "P75_Taxi_Out", "P90_Taxi_Out", "P95_Taxi_Out", "Max_Taxi_Out", "Min_Taxi_Out"]
                )
                fig_monthly.update_traces(line_color="#38bdf8", line_width=3, texttemplate="%{text:.1f}")
                fig_monthly.update_layout(
                    template=plotly_template,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=450,
                    xaxis_title="Month",
                    yaxis_title="Mean taxi-out (minutes)"
                )
                render_chart(fig_monthly)
                st.caption("Monthly averages; percentile, minimum, and maximum taxi-out values are available on hover.")
    else:
        st.warning("Cleaned flight data was not found, so exploratory charts are unavailable.")

    st.info("The cleaning funnel shows how cancelled, diverted, invalid, missing, and extreme taxi-out records are removed. The target chart shows the normal operating range and the upper-tail behavior that the models must learn.")

# TAB 1: DISPATCH SIMULATOR
# ==============================================================================
with tab1:
    st.markdown("<div style=\"height: 0.25rem;\"></div>", unsafe_allow_html=True)
    
    
    c_left, c_right = st.columns([1.2, 1])
    
    with c_left:
        st.subheader("⏱️ Pushback-to-Airborne Sequence Breakdown")
        
        # Calculate time components for waterfall/bar representation
        base_unimpeded = max(10.0, pred_val - (cong_1h * 0.45) - max(0, dep_delay * 0.1))
        queue_delay = max(0.0, pred_val - base_unimpeded)
        
        breakdown_df = pd.DataFrame([
            {"Stage": "Unimpeded Taxi Base", "Duration (min)": round(base_unimpeded, 1), "Type": "Base Transit"},
            {"Stage": "Congestion & Queue Delay", "Duration (min)": round(queue_delay, 1), "Type": "Airfield Queue"},
            {"Stage": "Total Predicted Taxi-Out", "Duration (min)": round(pred_val, 1), "Type": "Total"}
        ])
        
        fig_bar = px.bar(
            breakdown_df,
            x="Stage",
            y="Duration (min)",
            color="Type",
            text="Duration (min)",
            color_discrete_map={
                "Base Transit": "#38bdf8",
                "Airfield Queue": "#fb923c",
                "Total": "#34d399"
            },
            title=f"Flight Breakdown for {carrier} {origin} ➔ {dest}"
        )
        fig_bar.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, sans-serif"),
            showlegend=True,
            height=450
        )
        fig_bar.update_xaxes(
            tickmode="array",
            tickvals=breakdown_df["Stage"],
            ticktext=[
                "Unimpeded Taxi<br>Base",
                "Congestion &<br>Queue Delay",
                "Total Predicted<br>Taxi-Out"
            ],
            tickangle=0
        )

        render_chart(fig_bar)
        
    with c_right:
        st.subheader("📈 Live Congestion Sensitivity Curve")
        st.caption("How surface pushback density at origin affects taxi-out prediction:")
        
        # Generate the sensitivity curve once per flight scenario and model.
        curve_df = get_cached_sensitivity_curve(flight_info, model_type)
        fig_curve = px.line(
            curve_df,
            x="Pushbacks / Hour",
            y="Predicted Taxi-Out (min)",
            markers=True,
            title=f"Surface Traffic Sensitivity at {origin}"
        )
        fig_curve.add_vline(x=cong_1h, line_dash="dash", line_color="#facc15", annotation_text=f"Current: {cong_1h}")
        fig_curve.update_traces(line_color="#38bdf8", line_width=3)
        fig_curve.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, sans-serif"),
            height=450
        )
        render_chart(fig_curve)

# ==============================================================================

    st.markdown("---")
    st.subheader("Dispatch Decision Support")

    scenario_levels = [5, 15, 25, 35]
    scenario_labels = {
        5: "Low congestion",
        15: "Moderate congestion",
        25: "Severe congestion",
        35: "Extreme congestion"
    }
    scenario_df = curve_df[curve_df["Pushbacks / Hour"].isin(scenario_levels)].copy()
    scenario_df["Scenario"] = scenario_df["Pushbacks / Hour"].map(scenario_labels)

    low_congestion_prediction = float(
        curve_df.loc[curve_df["Pushbacks / Hour"] == 5, "Predicted Taxi-Out (min)"].iloc[0]
    )
    congestion_impact = pred_val - low_congestion_prediction

    support_col1, support_col2, support_col3, support_col4 = st.columns(4)
    support_col1.metric("Current prediction", f"{pred_val:.1f} min")
    support_col2.metric("Queue contribution", f"{queue_delay:.1f} min")
    support_col3.metric("Impact vs low congestion", f"{congestion_impact:+.1f} min")
    support_col4.metric("Operational status", cong_label)

    fig_scenario = px.bar(
        scenario_df,
        x="Scenario",
        y="Predicted Taxi-Out (min)",
        color="Predicted Taxi-Out (min)",
        text="Predicted Taxi-Out (min)",
        labels={
            "Scenario": "Congestion scenario",
            "Predicted Taxi-Out (min)": "Predicted taxi-out (minutes)"
        },
        color_continuous_scale="YlOrRd"
    )
    fig_scenario.update_traces(texttemplate="%{text:.1f}")
    fig_scenario.update_layout(
        template=plotly_template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        showlegend=False,
        xaxis_title="Congestion scenario",
        yaxis_title="Predicted taxi-out (minutes)"
    )
    render_chart(fig_scenario)

    if cong_1h < 8:
        dispatch_recommendation = "Normal pushback planning is appropriate; continue monitoring surface demand."
    elif cong_1h < 15:
        dispatch_recommendation = "Use standard pushback coordination and monitor the next departure bank."
    elif cong_1h < 22:
        dispatch_recommendation = "Coordinate closely with surface operations because queue growth is likely."
    else:
        dispatch_recommendation = "Consider departure metering or a short pushback hold to avoid adding surface queueing."
    st.info(f"Operational interpretation: {dispatch_recommendation}")
# TAB 2: MODEL PERFORMANCE & BENCHMARKS
# ==============================================================================
with tab2:
    c_m_left, c_m_right = st.columns(2)
    
    with c_m_left:
        st.subheader("Comparative Model Scores")
        model_metric_defaults = {
            "Linear_Regression_Baseline": {"label": "Linear Regression", "MAE_minutes": 4.17, "RMSE_minutes": 10.04, "R2_Score": 0.0166, "Pct_Within_3_Min": 54.6, "Pct_Within_5_Min": 78.91},
            "Random_Forest_Regressor": {"label": "Random Forest", "MAE_minutes": 4.04, "RMSE_minutes": 10.00, "R2_Score": 0.0239, "Pct_Within_3_Min": 57.48, "Pct_Within_5_Min": 82.60},
            "XGBoost_Regressor": {"label": "XGBoost", "MAE_minutes": 3.879, "RMSE_minutes": 9.89, "R2_Score": 0.0452, "Pct_Within_3_Min": 58.8, "Pct_Within_5_Min": 84.88}
        }
        score_rows = []
        for model_key, defaults in model_metric_defaults.items():
            test_metrics = (metrics_json or {}).get("models", {}).get(model_key, {}).get("test", {})
            score_rows.append({
                "Model": defaults["label"],
                "MAE (min)": test_metrics.get("MAE_minutes", defaults["MAE_minutes"]),
                "RMSE (min)": test_metrics.get("RMSE_minutes", defaults["RMSE_minutes"]),
                "R2": test_metrics.get("R2_Score", defaults["R2_Score"]),
                "Within +/-3 min": test_metrics.get("Pct_Within_3_Min", defaults["Pct_Within_3_Min"]),
                "Within +/-5 min": test_metrics.get("Pct_Within_5_Min", defaults["Pct_Within_5_Min"])
            })
        score_df = pd.DataFrame(score_rows)
        score_display = score_df.copy()
        score_display["MAE (min)"] = score_display["MAE (min)"].map(lambda value: f"{value:.2f}")
        score_display["RMSE (min)"] = score_display["RMSE (min)"].map(lambda value: f"{value:.2f}")
        score_display["R2"] = score_display["R2"].map(lambda value: f"{value:.4f}")
        score_display["Within +/-3 min"] = score_display["Within +/-3 min"].map(lambda value: f"{value:.1f}%")
        score_display["Within +/-5 min"] = score_display["Within +/-5 min"].map(lambda value: f"{value:.1f}%")
        score_table_style = score_display.style.set_properties(
            **{
                "background-color": "#ffffff",
                "color": "#0f172a",
                "border-color": "#d1d5db"
            }
        ).set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", "#0f172a"),
                    ("font-weight", "600"),
                    ("border-color", "#d1d5db")
                ]
            }
        ])
        st.dataframe(score_table_style, width="stretch", hide_index=True)
        st.caption("Scores use the November-December out-of-time test set. Lower MAE/RMSE and higher accuracy are better.")

        model_error_df = score_df.melt(
            id_vars="Model",
            value_vars=["MAE (min)", "RMSE (min)"],
            var_name="Metric",
            value_name="Minutes"
        )
        fig_model_scores = px.bar(
            model_error_df,
            x="Model",
            y="Minutes",
            color="Metric",
            barmode="group",
            text="Minutes",
            labels={"Minutes": "Error (minutes)"},
            color_discrete_sequence=["#38bdf8", "#f59e0b"]
        )
        fig_model_scores.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_model_scores.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            yaxis_title="Error (minutes)",
            xaxis_title="Model"
        )
        render_chart(fig_model_scores)
    with c_m_right:
        selected_model_name = {
            "rf": "Random Forest",
            "xgb": "XGBoost",
            "lr": "Linear Regression"
        }.get(model_type, "Selected model")
        st.subheader(f"Operational Risk Confusion Matrix ({selected_model_name})")
        test_prediction_df = load_model_test_predictions(model_type)
        if test_prediction_df.empty:
            st.warning("Test predictions are unavailable, so the operational risk matrix cannot be displayed.")
        else:
            risk_bins = [-np.inf, 20, 30, 45, np.inf]
            risk_labels = ["Normal", "Moderate", "High", "Severe"]
            actual_risk = pd.cut(test_prediction_df["Actual Taxi-Out"], bins=risk_bins, labels=risk_labels, right=False)
            predicted_risk = pd.cut(test_prediction_df["Predicted Taxi-Out"], bins=risk_bins, labels=risk_labels, right=False)
            risk_matrix = pd.crosstab(actual_risk, predicted_risk, dropna=False).reindex(index=risk_labels, columns=risk_labels, fill_value=0)
            fig_confusion = go.Figure(data=go.Heatmap(
                z=risk_matrix.values,
                x=risk_labels,
                y=risk_labels,
                text=risk_matrix.values,
                texttemplate="%{text}",
                colorscale="Blues",
                colorbar=dict(title="Flights")
            ))
            fig_confusion.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Predicted risk band",
                yaxis_title="Actual risk band"
            )
            render_chart(fig_confusion)
            st.caption("Risk bands: Normal <20 min, Moderate 20-30 min, High 30-45 min, Severe 45+ min. This is a classification view of the regression output.")
        st.subheader("Top Feature Importances (Random Forest)")
        
        rf_importances = (metrics_json or {}).get("feature_importances_rf", [])
        if not rf_importances:
            rf_importances = (
                (metrics_json or {})
                .get("models", {})
                .get("Random_Forest_Regressor", {})
                .get("top_importances", [])
            )
        if not rf_importances:
            rf_importances = [
                {"feature": "DISTANCE", "importance": 0.3003},
                {"feature": "DEP_DELAY", "importance": 0.1827},
                {"feature": "SIN_HOUR", "importance": 0.0881},
                {"feature": "CONGESTION_1H", "importance": 0.0673},
                {"feature": "COS_HOUR", "importance": 0.0642},
                {"feature": "CONGESTION_30M", "importance": 0.0523},
                {"feature": "ORIGIN_TAXI_MEAN", "importance": 0.0336},
                {"feature": "CARRIER_TAXI_MEAN", "importance": 0.0238},
            ]
        feature_imp_df = pd.DataFrame(rf_importances[:10]).rename(
            columns={"feature": "Feature", "importance": "Importance"}
        ).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            feature_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text="Importance",
            color="Importance",
            color_continuous_scale="Blues"
        )
        fig_imp.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450,
            showlegend=False
        )
        render_chart(fig_imp)

    st.markdown("### 📊 Model Evaluation & Operational Benchmarks")
    st.markdown("Evaluated on out-of-time test set (November–December 2025 BTS records).")
    
    benchmark_models = (metrics_json or {}).get("models", {})
    rf_benchmark = benchmark_models.get("Random_Forest_Regressor", {}).get("test", {})
    baseline_benchmark = benchmark_models.get("Linear_Regression_Baseline", {}).get("test", {})

    rf_mae = float(rf_benchmark.get("MAE_minutes", 4.04))
    rf_rmse = float(rf_benchmark.get("RMSE_minutes", 10.00))
    rf_within_3 = float(rf_benchmark.get("Pct_Within_3_Min", 57.48))
    rf_within_5 = float(rf_benchmark.get("Pct_Within_5_Min", 82.60))
    rf_r2 = float(rf_benchmark.get("R2_Score", 0.0239))
    baseline_mae = float(baseline_benchmark.get("MAE_minutes", 4.17))
    baseline_rmse = float(baseline_benchmark.get("RMSE_minutes", 10.04))
    baseline_within_3 = float(baseline_benchmark.get("Pct_Within_3_Min", 54.60))
    baseline_within_5 = float(baseline_benchmark.get("Pct_Within_5_Min", 78.91))
    baseline_r2 = float(baseline_benchmark.get("R2_Score", 0.0166))

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.metric("Random Forest MAE", f"{rf_mae:.2f} min", delta=f"{rf_mae - baseline_mae:+.2f} min vs Baseline", delta_color="inverse")
    with col_m2:
        st.metric("Random Forest RMSE", f"{rf_rmse:.2f} min", delta=f"{rf_rmse - baseline_rmse:+.2f} min vs Baseline", delta_color="inverse")
    with col_m3:
        st.metric("Within ±3 Min Tolerance", f"{rf_within_3:.1f}%", delta=f"{rf_within_3 - baseline_within_3:+.1f} pp vs Baseline", delta_color="normal")
    with col_m4:
        st.metric("Within ±5 Min Tolerance", f"{rf_within_5:.1f}%", delta=f"{rf_within_5 - baseline_within_5:+.1f} pp vs Baseline", delta_color="normal")
    with col_m5:
        st.metric("Model $R^2$ Score", f"{rf_r2:.4f}", delta=f"{rf_r2 - baseline_r2:+.4f} vs Baseline", delta_color="normal")

    st.markdown("---")
    
# ==============================================================================
# TAB 3: CONGESTION ANALYTICS
# ==============================================================================
with tab3:
    st.markdown("### 🚦 Surface Congestion & Airfield Hub Analytics")
    
    st.info("💡 **A-CDM Operational Insight:** Surface queue delays scale non-linearly during scheduled departure push banks at hub airports such as DFW, ATL, and JFK.")
    
    c_hub1, c_hub2 = st.columns(2)
    
    with c_hub1:
        st.subheader("Hub Airport Average Taxi-Out Times")
        hub_data = pd.DataFrame([
            {"Hub": "JFK (New York)", "Avg Taxi (min)": 28.4, "Category": "Constrained Runway"},
            {"Hub": "LGA (LaGuardia)", "Avg Taxi (min)": 26.1, "Category": "Constrained Runway"},
            {"Hub": "ORD (Chicago)", "Avg Taxi (min)": 21.8, "Category": "Major Hub"},
            {"Hub": "DFW (Dallas/Fort Worth)", "Avg Taxi (min)": 18.5, "Category": "Primary Scope Hub"},
            {"Hub": "ATL (Atlanta)", "Avg Taxi (min)": 17.9, "Category": "Major Hub"},
            {"Hub": "DEN (Denver)", "Avg Taxi (min)": 16.2, "Category": "Open Airfield Geometry"}
        ])
        
        fig_hubs = px.bar(
            hub_data,
            x="Hub",
            y="Avg Taxi (min)",
            color="Category",
            text="Avg Taxi (min)",
            color_discrete_map={
                "Primary Scope Hub": "#38bdf8",
                "Constrained Runway": "#f87171",
                "Major Hub": "#facc15",
                "Open Airfield Geometry": "#4ade80"
            }
        )
        fig_hubs.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        fig_hubs.update_xaxes(
            tickmode="array",
            tickvals=hub_data["Hub"],
            ticktext=[
                "JFK<br>(New York)",
                "LGA<br>(LaGuardia)",
                "ORD<br>(Chicago)",
                "DFW<br>(Dallas/Fort Worth)",
                "ATL<br>(Atlanta)",
                "DEN<br>(Denver)"
            ],
            tickangle=0
        )

        render_chart(fig_hubs)
        
    with c_hub2:
        st.subheader("Diurnal Departure Bank Wave Profile")
        
        hours = list(range(24))
        avg_taxi_by_hour = [14.2, 13.8, 13.5, 13.6, 14.1, 16.5, 21.4, 23.8, 22.1, 19.5, 18.2, 19.8, 20.4, 19.1, 18.5, 19.9, 22.5, 24.1, 21.2, 18.7, 16.9, 15.5, 14.8, 14.3]
        
        hour_df = pd.DataFrame({"Hour of Day (UTC/Local)": hours, "Average Taxi-Out (min)": avg_taxi_by_hour})
        
        fig_hour = px.area(
            hour_df,
            x="Hour of Day (UTC/Local)",
            y="Average Taxi-Out (min)",
            title="Taxi-Out Duration by Departure Hour"
        )
        fig_hour.update_traces(line_color="#38bdf8", fillcolor="rgba(56, 189, 248, 0.2)")
        fig_hour.update_layout(
            template=plotly_template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        apply_hour_axis(fig_hour, "Hour of day")

        render_chart(fig_hour)

# ==============================================================================

# ==============================================================================
# TAB 5: EXPLORATORY ANALYSIS
# ==============================================================================
with tab5:
    st.markdown("### Exploratory Analysis: Taxi-Out Operations")
    st.caption("Descriptive analysis of the cleaned 2025 flight records. Values describe association, not causation.")

    if analysis_df.empty:
        st.warning("The cleaned flight data was not found, so exploratory analysis is unavailable.")
    else:
        df_exp = analysis_df
        p50 = chart_aggregations["median_taxi_out"]
        p95 = chart_aggregations["p95_taxi_out"]
        high_risk_pct = chart_aggregations["high_risk_pct"]

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Flights analysed", f"{len(df_exp):,}")
        kpi2.metric("Mean taxi-out", f"{chart_aggregations['mean_taxi_out']:.1f} min")
        kpi3.metric("Median taxi-out", f"{p50:.1f} min")
        kpi4.metric("95th percentile", f"{p95:.1f} min")
        kpi5.metric("Above 30 minutes", f"{high_risk_pct:.1f}%")

        st.markdown("---")
        dist_left, dist_right = st.columns(2)

        with dist_left:
            st.subheader("Taxi-Out Distribution")
            fig_dist = px.histogram(
                df_exp,
                x="TAXI_OUT",
                nbins=45,
                labels={"TAXI_OUT": "Taxi-out time (minutes)"},
                color_discrete_sequence=["#38bdf8"]
            )
            fig_dist.add_vline(x=30, line_dash="dash", line_color="#ef4444", annotation_text="30-minute risk threshold")
            fig_dist.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Taxi-out time (minutes)",
                yaxis_title="Number of flights"
            )
            render_chart(fig_dist)

        with dist_right:
            st.subheader("Origin Airport Comparison")
            origin_stats = chart_aggregations["origin"]
            fig_origin_compare = px.bar(
                origin_stats,
                x="Mean_Taxi_Out",
                y="ORIGIN",
                orientation="h",
                text="Mean_Taxi_Out",
                color="P95_Taxi_Out",
                labels={
                    "Mean_Taxi_Out": "Mean taxi-out (minutes)",
                    "ORIGIN": "Origin airport",
                    "P95_Taxi_Out": "P95 taxi-out"
                },
                color_continuous_scale="YlOrRd",
                hover_data=["Flights", "Median_Taxi_Out", "P95_Taxi_Out"]
            )
            fig_origin_compare.update_traces(texttemplate="%{text:.1f}")
            fig_origin_compare.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Mean taxi-out (minutes)",
                yaxis_title="Origin airport"
            )
            render_chart(fig_origin_compare)

        st.markdown("---")
        heat_left, heat_right = st.columns(2)

        with heat_left:
            st.subheader("Congestion Heatmap by Day and Departure Hour")
            heat_df = chart_aggregations["heat"]
            fig_heat = go.Figure(data=go.Heatmap(
                z=heat_df.values,
                x=heat_df.columns,
                y=heat_df.index,
                colorscale="YlOrRd",
                colorbar=dict(title="Mean min")
            ))
            fig_heat.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Scheduled departure hour",
                yaxis_title="Day of week"
            )
            apply_hour_axis(fig_heat, "Scheduled departure hour")
            render_chart(fig_heat)

        with heat_right:
            st.subheader("Busiest Routes")
            busiest_routes = chart_aggregations["routes"]
            fig_routes = px.bar(
                busiest_routes,
                x="Flights",
                y="ROUTE",
                orientation="h",
                color="Mean_Taxi_Out",
                text="Flights",
                labels={
                    "Flights": "Number of flights",
                    "ROUTE": "Route",
                    "Mean_Taxi_Out": "Mean taxi-out"
                },
                color_continuous_scale="Blues",
                hover_data=["Mean_Taxi_Out", "P95_Taxi_Out"]
            )
            fig_routes.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Number of flights",
                yaxis_title="Route"
            )
            render_chart(fig_routes)

        st.markdown("---")
        risk_left, risk_right = st.columns(2)

        with risk_left:
            st.subheader("High-Risk Departures by Hour")
            risk_by_hour = chart_aggregations["risk_by_hour"]
            fig_risk = px.bar(
                risk_by_hour,
                x="DEP_HOUR",
                y="High_Risk_Rate",
                text="High_Risk_Rate",
                labels={
                    "DEP_HOUR": "Scheduled departure hour",
                    "High_Risk_Rate": "Flights above 30 minutes (%)"
                },
                color="High_Risk_Rate",
                color_continuous_scale="YlOrRd",
                hover_data=["Flights"]
            )
            fig_risk.update_traces(texttemplate="%{text:.1f}%")
            fig_risk.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Scheduled departure hour",
                yaxis_title="High-risk flights (%)"
            )
            apply_hour_axis(fig_risk, "Scheduled departure hour")
            render_chart(fig_risk)

        with risk_right:
            delay_title_col, delay_filter_col = st.columns([3.2, 1.2])
            with delay_title_col:
                st.subheader("Departure Delay vs Taxi-Out")
            with delay_filter_col:
                origin_codes = sorted(df_exp["ORIGIN"].dropna().astype(str).unique().tolist())
                origin_labels = ["ALL"] + [airport_names.get(code, code).rsplit(" - ", 1)[0] for code in origin_codes]
                origin_label_to_code = {label: code for code, label in zip(origin_codes, origin_labels[1:])}
                selected_origin_label = st.selectbox(
                    "Origin airport",
                    origin_labels,
                    index=0,
                    key="delay_origin_filter",
                    label_visibility="collapsed"
                )
                selected_origin = origin_label_to_code.get(selected_origin_label)

            delay_data = df_exp if selected_origin is None else df_exp[df_exp["ORIGIN"] == selected_origin]
            scatter_df = delay_data.sample(min(5000, len(delay_data)), random_state=42)
            fig_delay = px.scatter(
                scatter_df,
                x="DEP_DELAY",
                y="TAXI_OUT",
                color="ORIGIN",
                opacity=0.55,
                labels={
                    "DEP_DELAY": "Departure delay (minutes)",
                    "TAXI_OUT": "Taxi-out time (minutes)",
                    "ORIGIN": "Origin airport"
                },
                hover_data=["OP_UNIQUE_CARRIER", "DEST", "DISTANCE"]
            )
            fig_delay.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Departure delay (minutes)",
                yaxis_title="Taxi-out time (minutes)"
            )
            delay_y_max = int(np.ceil(delay_data["TAXI_OUT"].max() / 10.0) * 10)
            fig_delay.update_yaxes(
                tickmode="linear",
                tick0=0,
                dtick=10,
                range=[0, delay_y_max]
            )
            render_chart(fig_delay)
            correlation = chart_aggregations["correlation"]
            st.caption(f"Linear correlation in this dataset: {correlation:.3f}. This suggests departure delay and taxi-out should be analysed as separate operational signals.")

        st.markdown("---")
        distance_left, distance_right = st.columns(2)

        with distance_left:
            st.subheader("Flight Distance vs Departure Delay")
            distance_scatter = df_exp.sample(min(5000, len(df_exp)), random_state=42)
            fig_distance = px.scatter(
                distance_scatter,
                x="DISTANCE",
                y="DEP_DELAY",
                color="ORIGIN",
                opacity=0.55,
                labels={
                    "DISTANCE": "Flight distance (miles)",
                    "DEP_DELAY": "Departure delay (minutes)",
                    "ORIGIN": "Origin airport"
                },
                hover_data=["OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "TAXI_OUT"]
            )
            fig_distance.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                xaxis_title="Flight distance (miles)",
                yaxis_title="Departure delay (minutes)"
            )
            render_chart(fig_distance)

        with distance_right:
            st.subheader("Departure Delay by Distance Band")
            distance_bands = [0, 500, 1000, 1500, 2000, float("inf")]
            distance_labels = [
                "260–499 miles",
                "500–999 miles",
                "1,000–1,499 miles",
                "1,500–1,999 miles",
                "2,000+ miles"
            ]
            delay_band_df = df_exp.copy()
            delay_band_df["DISTANCE_BAND"] = pd.cut(
                delay_band_df["DISTANCE"],
                bins=distance_bands,
                labels=distance_labels,
                include_lowest=True
            )
            fig_delay_band = px.box(
                delay_band_df,
                x="DISTANCE_BAND",
                y="DEP_DELAY",
                color="DISTANCE_BAND",
                points=False,
                labels={
                    "DISTANCE_BAND": "Distance band",
                    "DEP_DELAY": "Departure delay (minutes)"
                }
            )
            fig_delay_band.update_layout(
                template=plotly_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                showlegend=False,
                xaxis_title="Distance band",
                yaxis_title="Departure delay (minutes)"
            )
            render_chart(fig_delay_band)
        st.info("The dataset is concentrated around DFW and American Airlines, so airport and carrier comparisons should be presented with flight volume and interpreted within this sample.")
# Floating AI Analyst button and chat panel
st.markdown("""
<style>
.st-key-ai_analyst_toggle {
    position: fixed !important;
    right: 28px !important;
    bottom: 28px !important;
    width: 180px !important;
    z-index: 999999 !important;
}
.st-key-ai_analyst_toggle button {
    width: 180px !important;
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 50px !important;
    padding: 14px 18px !important;
    font-family: "Segoe UI", sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.65) !important;
}
.st-key-ai_chat_clear {
    display: none !important;
}
[class*="st-key-ai_chat_delete_pair_"] {
    display: flex !important;
    justify-content: flex-end !important;
}
[class*="st-key-ai_chat_delete_pair_"] button {
    min-height: 0 !important;
    padding: 2px 7px !important;
    border: 1px solid #f97316 !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #f97316 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
}
.ai-chat-drag-handle {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 0 0 8px 0;
    padding: 6px 8px;
    color: #e5e7eb;
    border: 1px solid rgba(56, 189, 248, 0.45);
    border-radius: 8px;
    background: rgba(30, 41, 59, 0.75);
    cursor: move;
    touch-action: none;
    user-select: none;
    font-weight: 700;
}
.ai-chat-drag-handle span {
    color: #94a3b8;
    font-size: 10px;
    font-weight: 400;
}
.st-key-ai_chat_panel {
    position: fixed !important;
    right: 28px !important;
    bottom: 92px !important;
    width: 300px;
    height: min(590px, calc(100vh - 130px));
    min-width: 240px !important;
    min-height: 220px !important;
    max-width: calc(100vw - 24px) !important;
    max-height: calc(100vh - 24px) !important;
    overflow: hidden !important;
    resize: both !important;
    display: flex !important;
    flex-direction: column !important;
    box-sizing: border-box !important;
    z-index: 999998 !important;
    padding: 9px !important;
    background: #111827 !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 16px !important;
    box-shadow: 0 18px 45px rgba(2, 8, 23, 0.55) !important;
    font-size: 12px !important;
    line-height: var(--ai-chat-line-height) !important;
    font-weight: 400 !important;
}
.st-key-ai_chat_panel > div:first-child {
    flex: 1 1 auto !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
}
.st-key-ai_chat_panel > div:first-child > [data-testid="stVerticalBlock"] {
    height: 100% !important;
    min-height: 0 !important;
}
.st-key-ai_chat_messages {
    flex: 1 1 0 !important;
    min-height: 6rem !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}
.st-key-ai_chat_messages > [data-testid="stVerticalBlockBorderWrapper"],
.st-key-ai_chat_messages > [data-testid="stVerticalBlockBorderWrapper"] > div {
    height: 100% !important;
    min-height: 0 !important;
}
@media (max-width: 480px), (max-height: 520px) {
    .st-key-ai_chat_panel {
        min-width: 200px !important;
        min-height: 180px !important;
    }
}
.st-key-ai_chat_panel *,
.st-key-ai_chat_panel strong,
.st-key-ai_chat_panel b {
    font-size: inherit !important;
    line-height: var(--ai-chat-line-height) !important;
    font-weight: 400 !important;
}
.st-key-ai_chat_panel [data-testid="stChatMessage"] {
    padding: 0.25rem 0.15rem !important;
}
.st-key-ai_chat_panel [data-testid^="stChatMessageAvatar"] {
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
}
.st-key-ai_chat_panel [data-testid^="stChatMessageAvatar"] img {
    width: 100% !important;
    height: 100% !important;
    box-sizing: border-box !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    box-shadow: 0 2px 7px rgba(15, 23, 42, 0.32) !important;
}
.st-key-ai_chat_panel [data-testid="stChatMessage"] > img {
    width: 2rem !important;
    height: 2rem !important;
    flex: 0 0 2rem !important;
    box-sizing: border-box !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    box-shadow: 0 2px 7px rgba(15, 23, 42, 0.32) !important;
}
.st-key-ai_chat_panel [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > img {
    border: 2px solid #f97316 !important;
}
.st-key-ai_chat_panel [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) > img {
    border: 2px solid #38bdf8 !important;
}
.st-key-ai_chat_panel [data-testid="stChatMessageAvatarUser"] img {
    border: 2px solid #f97316 !important;
}
.st-key-ai_chat_panel [data-testid="stChatMessageAvatarAssistant"] img {
    border: 2px solid #38bdf8 !important;
}
.st-key-ai_chat_panel [data-testid="stMarkdownContainer"] p,
.st-key-ai_chat_panel [data-testid="stMarkdownContainer"] li {
    margin: 0 0 0.15rem 0 !important;
    line-height: var(--ai-chat-line-height) !important;
}
.st-key-ai_chat_panel [data-testid="stChatInput"] {
    position: relative !important;
    padding: 0 !important;
    border: 4px solid var(--ai-chat-input-border) !important;
    border-radius: 10px !important;
    background: #0f172a !important;
    overflow: hidden !important;
}
.st-key-ai_chat_panel [data-testid="stChatInput"] > div,
.st-key-ai_chat_panel [data-testid="stChatInput"] > div > div,
.st-key-ai_chat_panel [data-testid="stChatInput"] [data-baseweb="textarea"] {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
}
.st-key-ai_chat_panel [data-testid="stChatInput"] textarea {
    box-sizing: border-box !important;
    flex: 1 1 auto !important;
    min-width: 0 !important;
    width: 100% !important;
    min-height: 3.25rem !important;
    height: auto;
    max-height: 10.5rem !important;
    field-sizing: content;
    margin: 0 !important;
    padding: 0.4rem 3.5rem 0.4rem 0.5rem !important;
    border: 0 !important;
    border-radius: 0 !important;
    font-size: 12px !important;
    line-height: var(--ai-chat-line-height) !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    overflow-y: auto !important;
    resize: none !important;
}
.st-key-ai_chat_panel [data-testid="stChatInput"] button {
    position: absolute !important;
    top: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: 3rem !important;
    min-width: 3rem !important;
    height: auto !important;
    min-height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    border-left: 3px solid var(--ai-chat-input-border) !important;
    border-radius: 0 !important;
    background: var(--ai-chat-input-action) !important;
    color: #ffffff !important;
}
.st-key-ai_chat_panel [data-testid="stChatInput"] button svg {
    position: absolute !important;
    left: 50% !important;
    bottom: 1rem !important;
    margin: 0 !important;
    transform: translateX(-50%) !important;
}
</style>
""", unsafe_allow_html=True)

if light_mode:
    st.markdown("""
    <style class="light-chat-theme-override">
        .st-key-ai_chat_panel {
            background: #ffffff !important;
            color: #1f2937 !important;
            border-color: #0284c7 !important;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.20) !important;
            color-scheme: light !important;
        }
        .st-key-ai_chat_panel .ai-chat-drag-handle {
            color: #0f172a !important;
            background: #e8eef7 !important;
            border-color: #7dd3fc !important;
        }
        .st-key-ai_chat_panel .ai-chat-drag-handle span {
            color: #526174 !important;
        }
        .st-key-ai_chat_panel [data-testid="stChatMessage"],
        .st-key-ai_chat_panel [data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent !important;
        }
        .st-key-ai_chat_panel [data-testid="stMarkdownContainer"] *,
        .st-key-ai_chat_panel [data-testid="stChatMessage"] * {
            color: #1f2937 !important;
        }
        .st-key-ai_chat_panel [data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            color: #1f2937 !important;
            caret-color: #0284c7 !important;
        }
        .st-key-ai_chat_panel [data-testid="stChatInput"],
        .st-key-ai_chat_panel [data-testid="stChatInput"] > div,
        .st-key-ai_chat_panel [data-testid="stChatInput"] > div > div,
        .st-key-ai_chat_panel [data-testid="stChatInput"] [data-baseweb="textarea"] {
            background: #ffffff !important;
        }
        .st-key-ai_chat_panel [data-testid="stChatInput"] textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }
        .st-key-ai_chat_panel [data-testid="stChatInput"] button {
            background: var(--ai-chat-input-action) !important;
            color: #ffffff !important;
        }
        .st-key-ai_chat_panel details,
        .st-key-ai_chat_panel details summary {
            color: #1f2937 !important;
            background: #f8fafc !important;
            border-color: #cbd5e1 !important;
        }
    </style>
    """, unsafe_allow_html=True)
if "ai_chat_open" not in st.session_state:
    st.session_state.ai_chat_open = False

def toggle_ai_chat():
    st.session_state.ai_chat_open = not st.session_state.ai_chat_open

st.button(
    "Ask AI Analyst",
    key="ai_analyst_toggle",
    on_click=toggle_ai_chat,
)

if st.session_state.ai_chat_open:
    with st.container(key="ai_chat_panel"):
        show_ai_analyst_dialog()
