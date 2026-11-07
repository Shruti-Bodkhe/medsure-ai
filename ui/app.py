import streamlit as st
import requests
import json
from pathlib import Path

st.set_page_config(page_title="MedSure AI Clinical System", page_icon="🩺", layout="wide")

API_URL = "http://127.0.0.1:8000/query"
MULTIMODAL_API_URL = "http://127.0.0.1:8000/multimodal-query"
AUDIT_PATH = Path("data/audit/audit.jsonl")

st.sidebar.title("MedSure Navigation")
page = st.sidebar.radio("Go to", ["💬 Patient Chat & Multimodal OCR", "📊 Reviewer Governance Dashboard"])

if page == "💬 Patient Chat & Multimodal OCR":
    st.title("🩺 MedSure AI: Risk-Governed Health Assistant")
    st.caption("Supports English, Hindi, Marathi, Hinglish queries + Medicine Label / OCR Image analysis.")

    tab1, tab2 = st.tabs(["💬 Text Query / Voice Text", "📷 Upload Medicine Label / Health Doc (OCR)"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Type your symptom or health query in English, Hindi, Marathi, or Hinglish...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Assessing clinical risk, verifying evidence, and checking governance policy..."):
                    try:
                        resp = requests.post(API_URL, json={"question": prompt})
                        if resp.status_code == 200:
                            data = resp.json()
                            action = data["action"]
                            risk = data["risk_level"]
                            message = data["message"]
                            
                            badge_color = "red" if action == "ESCALATE" else ("orange" if action in ["ABSTAIN", "CLARIFY"] else "green")
                            st.markdown(f"**Governance Action:** `:{badge_color}[{action}]` | **Risk Level:** `{risk}` | **Language:** `{data['language']}`")
                            st.markdown(message)
                            
                            if data["sources"]:
                                with st.expander("📚 Verified Evidence Sources"):
                                    for src in data["sources"]:
                                        st.write(f"- {src}")
                                        
                            st.session_state.messages.append({"role": "assistant", "content": message})
                        else:
                            st.error(f"API Error: {resp.text}")
                    except Exception as e:
                        st.error(f"Backend connection failed. Make sure FastAPI is running. Error: {e}")

    with tab2:
        st.subheader("Upload Prescription, Lab Report, or Medicine Label Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        user_note = st.text_input("Add any specific question about this image:", "What are the indications and precautions for this medicine?")

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Medical Image", use_column_width=True)
            if st.button("Analyze & Govern Image Input"):
                with st.spinner("Performing OCR extraction and clinical risk evaluation..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data_form = {"prompt": user_note}
                    try:
                        res = requests.post(MULTIMODAL_API_URL, files=files, data=data_form)
                        if res.status_code == 200:
                            res_json = res.json()
                            st.success("Analysis Complete!")
                            st.markdown(f"### Governance Decision: `{res_json['action']}` (Risk: `{res_json['risk_level']}`)")
                            st.write("**Extracted Insight:**")
                            st.info(res_json["extracted_insight"])
                            st.write("**Governed Medical Response:**")
                            st.write(res_json["governed_message"])
                        else:
                            st.error(f"Multimodal API error: {res.text}")
                    except Exception as ex:
                        st.error(f"Failed to connect to backend: {ex}")

elif page == "📊 Reviewer Governance Dashboard":
    st.title("📊 IEEE Paper Evaluation & Audit Dashboard")
    st.caption("Inspect live audit traces, safety interventions, and decision distributions.")

    if AUDIT_PATH.exists():
        audit_lines = [json.loads(line) for line in open(AUDIT_PATH, encoding="utf-8")]
        st.metric("Total Evaluated Queries", len(audit_lines))
        
        col1, col2, col3 = st.columns(3)
        actions_count = {}
        for row in audit_lines:
            act = row['decision']['action']
            actions_count[act] = actions_count.get(act, 0) + 1
            
        col1.metric("Answers Given", actions_count.get("ANSWER", 0))
        col2.metric("Clarifications / Revisions", actions_count.get("CLARIFY", 0) + actions_count.get("REVISE", 0))
        col3.metric("Escalations / Abstentions", actions_count.get("ESCALATE", 0) + actions_count.get("ABSTAIN", 0))

        st.subheader("Recent Audit Traces")
        for i, row in enumerate(reversed(audit_lines[-25:])):
            with st.expander(f"[{row.get('time', 'N/A')}] Query: {row['question'][:60]}..."):
                st.write(f"**Action Decided:** `{row['decision']['action']}`")
                st.write(f"**Reason Code:** `{row['decision']['reason_code']}`")
                st.write(f"**Risk Level:** `{row['risk']['level']}`")
                st.write(f"**Detected Language:** `{row.get('language', 'en')}`")
                st.write(f"**Final System Message:** {row['decision']['message']}")
    else:
        st.info("No audit logs available yet. Try running queries in the chat tab first!")