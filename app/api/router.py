from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import shutil
import uuid
from pathlib import Path

from app.pipeline import run
from app.retrieval.search import Retriever
from app.core import llm
from app.database import get_db, hash_password, build_patient_longitudinal_context

router = APIRouter()
retriever = Retriever()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AuthRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    chronic_conditions: Optional[str] = ""
    allergies: Optional[str] = ""


class ProfileUpdateRequest(BaseModel):
    user_id: int
    full_name: str
    age: Optional[int] = None
    chronic_conditions: Optional[str] = ""
    allergies: Optional[str] = ""


class QueryRequest(BaseModel):
    question: str
    user_id: int = 1
    session_id: Optional[str] = None
    check_responsiveness: bool = True
    use_language_agent: bool = True


@router.post("/auth/login")
def login_or_register(req: AuthRequest):
    conn = get_db()
    email = req.email.strip().lower()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user:
        name = req.full_name or email.split("@")[0].replace(".", " ").title()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (email, password_hash, full_name, age, chronic_conditions, allergies) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (email, hash_password(req.password), name, req.age, req.chronic_conditions or "", req.allergies or "")
        )
        conn.commit()
    else:
        # Update profile fields if provided during sign-in
        name = req.full_name.strip() if req.full_name else user["full_name"]
        age = req.age if req.age is not None else user["age"]
        conds = req.chronic_conditions if req.chronic_conditions is not None else user["chronic_conditions"]
        allerg = req.allergies if req.allergies is not None else user["allergies"]
        conn.execute(
            "UPDATE users SET full_name = ?, age = ?, chronic_conditions = ?, allergies = ? WHERE email = ?",
            (name, age, conds, allerg, email)
        )
        conn.commit()

    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return {
        "user_id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "age": user["age"],
        "chronic_conditions": user["chronic_conditions"],
        "allergies": user["allergies"]
    }

@router.post("/auth/profile")
def update_profile(req: ProfileUpdateRequest):
    conn = get_db()
    conn.execute(
        "UPDATE users SET full_name = ?, age = ?, chronic_conditions = ?, allergies = ? WHERE id = ?",
        (req.full_name, req.age, req.chronic_conditions or "", req.allergies or "", req.user_id)
    )
    conn.commit()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (req.user_id,)).fetchone()
    conn.close()
    return dict(user) if user else {"status": "updated"}


@router.get("/chat/history/{user_id}")
def get_user_history(user_id: int):
    conn = get_db()
    sessions = conn.execute(
        "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    out_sessions = []
    for s in sessions:
        msgs = conn.execute(
            "SELECT id, role, content, action, risk_level, sources_json, created_at "
            "FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
            (s["id"],)
        ).fetchall()
        formatted_msgs = []
        for m in msgs:
            formatted_msgs.append({
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "action": m["action"],
                "risk_level": m["risk_level"],
                "sources": json.loads(m["sources_json"] or "[]"),
                "created_at": m["created_at"]
            })
        out_sessions.append({
            "session_id": s["id"],
            "title": s["title"],
            "updated_at": s["updated_at"],
            "messages": formatted_msgs
        })

    records = conn.execute(
        "SELECT id, filename, extracted_insight, governed_summary, created_at "
        "FROM medical_records WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    return {
        "sessions": out_sessions,
        "medical_records": [dict(r) for r in records]
    }


@router.post("/query")
def evaluate_query(req: QueryRequest):
    """Multilingual text health query evaluation with longitudinal patient memory."""
    try:
        conn = get_db()
        session_id = req.session_id or str(uuid.uuid4())

        existing = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        if not existing:
            title = req.question[:42] + ("..." if len(req.question) > 42 else "")
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)",
                (session_id, req.user_id, title)
            )
            conn.commit()

        # Retrieve longitudinal context (Saved OCR reports + Health Profile + Recent Chat)
        patient_ctx = build_patient_longitudinal_context(req.user_id, session_id)

        res = run(
            question=req.question,
            retriever=retriever,
            check_responsiveness=req.check_responsiveness,
            use_language_agent=req.use_language_agent,
            audit=True,
            patient_context=patient_ctx,
            allow_clinical_fallback=True
        )

        # Save User & Assistant messages to SQLite
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, action, risk_level) VALUES (?, 'user', ?, 'QUERY', ?)",
            (session_id, req.question, res.risk.level)
        )
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, action, risk_level, sources_json) VALUES (?, 'assistant', ?, ?, ?, ?)",
            (session_id, res.decision.message, res.decision.action, res.risk.level, json.dumps(res.sources))
        )
        conn.execute("UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()

        support_rate = getattr(res.report, 'overall_score', 1.0) if (res.report and getattr(res.report, 'overall_score', 1.0) is not None) else 1.0

        return {
            "session_id": session_id,
            "question": res.question,
            "action": res.decision.action,
            "reason_code": res.decision.reason_code,
            "risk_level": res.risk.level,
            "message": res.decision.message,
            "language": res.language,
            "translation_status": res.translation_status,
            "sources": res.sources,
            "support_rate": round(support_rate * 100, 1),
            "has_longitudinal_context": bool(patient_ctx),
            "seconds": round(res.seconds, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal-query")
async def evaluate_multimodal_query(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form("Analyze this medical document, prescription, or medicine label."),
    user_id: int = Form(1),
    session_id: Optional[str] = Form(None)
):
    """Handles Image/OCR (medicine labels, test reports) and saves to the Patient Longitudinal Vault."""
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        patient_ctx = build_patient_longitudinal_context(user_id, session_id)
        analysis_prompt = (
            "You are MedSure AI Clinical OCR & Document Analyzer.\n"
            f"Patient Context: {patient_ctx or 'None reported'}\n"
            f"User Question: {prompt}\n\n"
            "Carefully inspect this medical image (prescription, lab report, or medicine label). "
            "Provide a clear, structured breakdown:\n"
            "1. **Identified Medicines / Test Parameters**: Exact names, strengths (mg/ml), or lab values.\n"
            "2. **Indications & Usage Purpose**: What this medicine is for or what the test values indicate.\n"
            "3. **Key Precautions & Contraindications**: Important safety warnings, food/alcohol interactions, and side effects to watch for.\n"
            "4. **Longitudinal Cross-Check**: Note if anything interacts with the patient's known allergies or conditions."
        )
        image_bytes = file_path.read_bytes()

        extracted_response = llm.generate_multimodal(
            prompt=analysis_prompt,
            image_data=image_bytes,
            mime_type=file.content_type or "image/jpeg"
        )

        # Save to Medical Records Vault so future chat queries remember this report!
        conn = get_db()
        conn.execute(
            "INSERT INTO medical_records (user_id, filename, extracted_insight, governed_summary) VALUES (?, ?, ?, ?)",
            (user_id, file.filename, extracted_response, extracted_response)
        )

        active_session_id = session_id or str(uuid.uuid4())
        existing = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (active_session_id,)).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO chat_sessions (id, user_id, title) VALUES (?, ?, ?)",
                (active_session_id, user_id, f"ðŸ“„ Report: {file.filename[:25]}")
            )

        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, action, risk_level) VALUES (?, 'user', ?, 'OCR_UPLOAD', 'medium')",
            (active_session_id, f"ðŸ“Ž Uploaded Medical Document: {file.filename}\n{prompt}")
        )
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, action, risk_level) VALUES (?, 'assistant', ?, 'ANSWER', 'medium')",
            (active_session_id, extracted_response)
        )
        conn.commit()
        conn.close()

        return {
            "session_id": active_session_id,
            "filename": file.filename,
            "extracted_insight": extracted_response,
            "action": "ANSWER",
            "risk_level": "medium",
            "governed_message": extracted_response,
            "sources": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import Response
import io

@router.get("/chat/export-pdf/{session_id}")
def export_consultation_pdf(session_id: str):
    """Generates a clean clinical summary handover for a physician visit."""
    conn = get_db()
    session = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if not session:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
        
    user = conn.execute("SELECT * FROM users WHERE id = ?", (session["user_id"],)).fetchone()
    messages = conn.execute("SELECT role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
    conn.close()
    
    pdf_lines = [
        "=" * 60,
        "MEDSURE AI â€” CLINICAL CONSULTATION HANDOVER REPORT",
        "=" * 60,
        f"Session ID: {session['id']}",
        f"Date / Time: {session['updated_at']}",
        f"Consultation Title: {session['title']}"
    ]
    
    if user:
        pdf_lines.extend([
            "\nPATIENT PROFILE:",
            f"â€¢ Name: {user['full_name']}",
            f"â€¢ Age: {user['age'] or 'N/A'}",
            f"â€¢ Known Conditions: {user['chronic_conditions'] or 'None reported'}",
            f"â€¢ Known Allergies: {user['allergies'] or 'None reported'}"
        ])
    
    pdf_lines.extend([
        "\n" + "-" * 60,
        "CONSULTATION TRANSCRIPT & AI SYNTHESIS:",
        "-" * 60
    ])
    
    for m in messages:
        role_label = "PATIENT" if m["role"] == "user" else "MEDSURE AI CLINICAL ASSISTANT"
        pdf_lines.append(f"\n[{m['created_at']}] {role_label}:\n{m['content']}")
        
    pdf_lines.extend([
        "\n" + "=" * 60,
        "Generated by MedSure AI (IEEE-Standard Clinical RAG Architecture)",
        "Note: This summary is prepared to assist clinician handover and patient records."
    ])
    
    report_text = "\n".join(pdf_lines)
    
    return Response(
        content=report_text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=MedSure_Report_{session_id[:8]}.txt"}
    )
