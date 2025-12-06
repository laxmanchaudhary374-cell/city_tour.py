# City Explorer Tours — Professional Chatbot (OpenAI)
# Streamlit app with secure secrets, rate limiting, caching, logging, and admin dashboard

import os
import time
import uuid
import json
import math
import traceback
from datetime import datetime, timedelta
from collections import defaultdict

import streamlit as st
import openai

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="City Explorer Tours",
    page_icon="🗺️",
    layout="wide"
)

# ==================== CONFIGURATION ====================
def get_api_key():
    """Get OpenAI API key from Streamlit secrets or environment."""
    # Prefer Streamlit Secrets (cloud)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # Fallback to local dev environment variable
    return os.getenv("OPENAI_API_KEY", "")

OPENAI_API_KEY = get_api_key()

# Rate limit: 10 requests per 60 seconds
RATE_LIMIT = 10
RATE_LIMIT_WINDOW = 60  # seconds
CACHE_TTL = 600  # seconds, cache chat responses
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "")  # optional admin pass

# ==================== TOUR CATALOG (CUSTOMIZE THIS) ====================
# Replace or extend this with your actual offerings. Keep it simple JSON.
TOUR_CATALOG = {
    "company": "City Explorer Tours",
    "contact": {
        "address": "789 Market Street, San Francisco, CA",
        "phone": "(650) 780-9123",
        "email": "hello@cityexplorertours.com"
    },
    "cities": [
        {
            "name": "San Francisco",
            "packages": [
                {
                    "title": "Golden Gate Highlights",
                    "code": "SF-GG-101",
                    "duration": "4 hours",
                    "price": 79,
                    "includes": ["Golden Gate Bridge", "Crissy Field", "Palace of Fine Arts"],
                    "schedule": "Daily at 10:00 and 14:00",
                    "notes": "Small groups. Ideal for first-time visitors."
                },
                {
                    "title": "Alcatraz & Waterfront",
                    "code": "SF-AL-202",
                    "duration": "6 hours",
                    "price": 129,
                    "includes": ["Alcatraz ferry", "Pier 39", "Fisherman's Wharf"],
                    "schedule": "Mon–Sat at 09:00",
                    "notes": "Passport required for booking."
                }
            ]
        },
        {
            "name": "Los Angeles",
            "packages": [
                {
                    "title": "Hollywood Essentials",
                    "code": "LA-HW-301",
                    "duration": "3 hours",
                    "price": 69,
                    "includes": ["Hollywood Sign view", "Walk of Fame", "TCL Chinese Theatre"],
                    "schedule": "Daily at 11:00 and 16:00",
                    "notes": "Photo stops included."
                }
            ]
        }
    ],
    "policies": {
        "refunds": "Full refund up to 48 hours before tour. 50% within 24–48 hours. No refund within 24 hours.",
        "weather": "Tours operate rain or shine. Severe weather may cause rescheduling.",
        "booking": "Online booking available. Call or email for group rates."
    }
}

# ==================== PROMPT ====================
SYSTEM_PROMPT = """You are City Explorer Tours' helpful assistant.
Answer using only the known catalog, contact info, packages, and policies provided.
If a question is outside the catalog, politely state what you know and invite the user to contact support.

Guidelines:
- Be clear, concise, and friendly.
- Include package codes when relevant.
- Offer next steps (booking, contact).
- If info is not available, say so transparently.
- Keep responses to 4–8 sentences unless listing package details."""

def build_user_prompt(user_question: str) -> str:
    catalog_json = json.dumps(TOUR_CATALOG, ensure_ascii=False)
    return f"""User question:
{user_question}

Known catalog (JSON):
{catalog_json}
"""

# ==================== SESSION STATE ====================
if "rate_limit_tracker" not in st.session_state:
    st.session_state.rate_limit_tracker = defaultdict(list)  # user_id -> timestamps list
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}  # (question hash) -> response dict
if "logs" not in st.session_state:
    st.session_state.logs = []  # list of dicts with usage, latency, errors
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# ==================== UTILITIES ====================
def check_rate_limit(user_id: str) -> bool:
    """Returns True if within rate limit, False otherwise."""
    now = time.time()
    timestamps = st.session_state.rate_limit_tracker[user_id]
    # Remove events older than window
    st.session_state.rate_limit_tracker[user_id] = [t for t in timestamps if now - t <= RATE_LIMIT_WINDOW]
    if len(st.session_state.rate_limit_tracker[user_id]) >= RATE_LIMIT:
        return False
    st.session_state.rate_limit_tracker[user_id].append(now)
    return True

def cache_key(question: str) -> str:
    return f"q::{hash(question)}"

def log_event(kind: str, **kwargs):
    entry = {
        "time": datetime.utcnow().isoformat(),
        "kind": kind,
        **kwargs
    }
    st.session_state.logs.append(entry)

def call_openai(question: str):
    """Calls OpenAI ChatCompletion with secure API key and returns content + usage."""
    openai.api_key = OPENAI_API_KEY
    start = time.time()
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question)}
            ],
            temperature=0.3,
            max_tokens=500
        )
        latency_ms = math.floor((time.time() - start) * 1000)
        content = resp.choices[0].message["content"]
        usage = resp.get("usage", {})
        log_event("success", question=question, latency_ms=latency_ms, usage=usage)
        return content, usage, latency_ms
    except openai.error.RateLimitError as e:
        latency_ms = math.floor((time.time() - start) * 1000)
        log_event("rate_limit", question=question, error=str(e), latency_ms=latency_ms)
        raise
    except openai.error.AuthenticationError as e:
        latency_ms = math.floor((time.time() - start) * 1000)
        log_event("auth_error", question=question, error=str(e), latency_ms=latency_ms)
        raise
    except Exception as e:
        latency_ms = math.floor((time.time() - start) * 1000)
        log_event("error", question=question, error=str(e), traceback=traceback.format_exc(), latency_ms=latency_ms)
        raise

# ==================== UI: SIDEBAR ====================
with st.sidebar:
    st.title("City Explorer Tours")
    st.caption("Professional chatbot with secure keys and admin tools.")
    st.divider()

    st.subheader("Contact")
    st.write(TOUR_CATALOG["contact"]["address"])
    st.write(f"{TOUR_CATALOG['contact']['phone']} | {TOUR_CATALOG['contact']['email']}")

    st.divider()
    st.subheader("Rate limit")
    st.write(f"{RATE_LIMIT} requests per {RATE_LIMIT_WINDOW}s per user")

    st.divider()
    mode = st.radio("Mode", ["Chat", "Admin"], horizontal=True)

# ==================== UI: CHAT MODE ====================
def chat_mode():
    st.header("Ask about our tours")
    st.write("Ask about cities, packages, schedules, prices, booking, and policies.")
    st.divider()

    question = st.text_input("Your question", placeholder="e.g., What does the Golden Gate Highlights tour include?")
    ask = st.button("Ask")

    if ask and question.strip():
        # Check rate limit
        if not check_rate_limit(st.session_state.user_id):
            st.warning("Too many requests. Please wait 1–2 minutes and try again.")
            return

        # Check cache
        key = cache_key(question.strip())
        cached = st.session_state.response_cache.get(key)
        if cached and (time.time() - cached["time"] <= CACHE_TTL):
            st.success("Served from cache")
            st.write(cached["content"])
            return

        # Validate API key
        if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith(("sk-", "sk-proj-")):
            st.error("Invalid or missing OpenAI API key. Please check App -> Secrets and try again.")
            log_event("missing_key", question=question)
            return

        # Call OpenAI with spinner and robust error handling
        with st.spinner("Thinking..."):
            try:
                content, usage, latency_ms = call_openai(question.strip())
                st.write(content)
                st.caption(f"Latency: {latency_ms} ms | Tokens: {usage.get('total_tokens', 'n/a')}")
                st.session_state.response_cache[key] = {"content": content, "time": time.time()}
            except openai.error.RateLimitError:
                st.error("We’re receiving too many requests right now. Please wait a moment and try again.")
            except openai.error.AuthenticationError:
                st.error("Authentication error with OpenAI. Check your API key in App -> Secrets.")
            except Exception:
                st.error("Something went wrong. Please try again later.")
                st.expander("Technical details").write(traceback.format_exc())

# ==================== UI: ADMIN MODE ====================
def admin_mode():
    st.header("Admin dashboard")
    # Protect dashboard (optional)
    if ADMIN_PASSWORD:
        pw = st.text_input("Admin password", type="password")
        if pw != ADMIN_PASSWORD:
            st.info("Enter admin password to view metrics.")
            return

    st.subheader("Usage metrics")
    logs = st.session_state.logs
    total_requests = sum(1 for l in logs if l["kind"] == "success")
    avg_latency = None
    latencies = [l["latency_ms"] for l in logs if "latency_ms" in l]
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
    recent_errors = [l for l in logs if l["kind"] in ("error", "auth_error", "rate_limit")]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total successful requests", total_requests)
    with col2:
        st.metric("Average latency (ms)", f"{avg_latency:.0f}" if avg_latency else "n/a")
    with col3:
        st.metric("Errors (recent)", len(recent_errors))

    st.subheader("Logs")
    st.write(logs if logs else "No logs yet.")

    st.download_button(
        "Download logs (JSON)",
        data=json.dumps(logs, indent=2),
        file_name=f"city_explorer_logs_{datetime.utcnow().date()}.json",
        mime="application/json"
    )

    st.subheader("Catalog preview")
    st.json(TOUR_CATALOG)

# ==================== RENDER ====================
if mode == "Chat":
    chat_mode()
else:
    admin_mode()

# ==================== FOOTER ====================
st.divider()
st.markdown(
    f"**{TOUR_CATALOG['company']}** • {TOUR_CATALOG['contact']['address']} • "
    f"{TOUR_CATALOG['contact']['phone']} • {TOUR_CATALOG['contact']['email']}"
)
