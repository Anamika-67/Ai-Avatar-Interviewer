from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from motor.motor_asyncio import AsyncIOMotorClient # MongoDB Driver

# Import your helper scripts
from resume_parser import extract_text_from_pdf
from gemini_service import GeminiService

app = FastAPI()
gemini = GeminiService()

# --- MongoDB Setup ---
MONGO_DETAILS = "mongodb://localhost:27017" # Update with your MongoDB URI
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.interview_db
results_collection = database.get_collection("interview_results")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# Temporary in-memory store to track live sessions
# In a production app, you might store this in Redis or a 'sessions' collection
active_interviews = {}

@app.get("/")
def home():
    return {"message": "AI Avatar Interview System - Backend Active"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Step 0: Upload Resume & Start Session
    Condition: Differentiate role and set Question 1
    """
    session_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Extract and Analyze
    resume_text = extract_text_from_pdf(file_path)
    role_type = gemini.determine_role_type(resume_text)
    
    # 2. Initialize Session Data
    active_interviews[session_id] = {
        "current_step": 1,
        "resume_text": resume_text,
        "role_type": role_type,
        "responses": [] # To store Q&A pairs for the final report
    }

    # 3. Get first question (Condition: Introduce Yourself)
    first_question = gemini.get_interview_question(1, resume_text, role_type)

    return {
        "session_id": session_id,
        "role_detected": role_type,
        "question": first_question,
        "step": 1
    }

@app.post("/submit-answer")
async def submit_answer(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    Steps 1-5: Handle Answers & Progress Interview
    Condition: Track steps and save to MongoDB at the end
    """
    if session_id not in active_interviews:
        raise HTTPException(status_code=404, detail="Interview session not found")

    session = active_interviews[session_id]
    current_step = session["current_step"]
    
    # 1. Save and Transcribe Audio
    audio_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_step_{current_step}.wav")
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    transcription = gemini.transcribe_audio(audio_path)
    
    # 2. Get the question that was just asked to evaluate it
    current_question = gemini.get_interview_question(current_step, session["resume_text"], session["role_type"])
    evaluation = gemini.evaluate_response(current_question, transcription)

    # 3. Store the result for this step
    session["responses"].append({
        "step": current_step,
        "question": current_question,
        "answer": transcription,
        "score": evaluation.get("score"),
        "feedback": evaluation.get("feedback")
    })

    # 4. Progress to the Next Step
    next_step = current_step + 1
    session["current_step"] = next_step

    # 5. Check if Interview is finished
    if next_step > 5:
        # Save full interview to MongoDB
        await results_collection.insert_one({
            "session_id": session_id,
            "role_type": session["role_type"],
            "total_responses": session["responses"],
            "status": "Completed"
        })
        
        final_msg = "The interview is now complete. Thank you for your time! Your results have been saved."
        del active_interviews[session_id] # Clean up memory
        
        return {
            "message": final_msg,
            "is_complete": True,
            "evaluation": evaluation
        }

    # 6. Generate Next Question (Condition: Sequence based)
    next_question = gemini.get_interview_question(next_step, session["resume_text"], session["role_type"])

    return {
        "session_id": session_id,
        "next_step": next_step,
        "next_question": next_question,
        "interviewer_feedback": evaluation.get("interviewer_reply"),
        "is_complete": False
    }