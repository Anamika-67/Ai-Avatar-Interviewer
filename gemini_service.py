import os
import google.generativeai as genai
from dotenv import dotenv_values
import json

# Load environment configuration
env_config = dotenv_values(".env")
api_key = env_config.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

class GeminiService:
    def __init__(self):
        # Using flash for speed and efficiency in a real-time interview setting
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def determine_role_type(self, resume_text: str):
        """
        Condition: Differentiate if the role is Technical or Non-Technical.
        """
        try:
            prompt = f"""
            Analyze the following resume text. Based on the skills and experience, 
            classify the candidate's target role as either 'Technical' or 'Non-Technical'.
            Return ONLY the word 'Technical' or 'Non-Technical'.
            
            Resume: {resume_text}
            """
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return "Technical"  # Default fallback

    def get_interview_question(self, step: int, resume_text: str, role_type: str):
        """
        Condition: Generate 5 specific questions in the required sequence.
        """
        try:
            if step == 1:
                # Condition: Intro question
                return "Hello! I am your AI Avatar interviewer. To start our session, could you please introduce yourself?"
            
            elif step == 2:
                # Condition: Alignment question
                return "Thank you. How do you think your specific skills and experiences are perfectly aligned with this job?"
            
            elif step == 3:
                # Condition: Based on Resume
                prompt = f"Based on this {role_type} resume: {resume_text}, ask one specific, deep-dive question about a project or experience mentioned."
                response = self.model.generate_content(prompt)
                return response.text.strip()
            
            elif step == 4:
                # Condition: Scenario-based
                prompt = f"Based on the background in this {role_type} resume: {resume_text}, generate one realistic 'What would you do if...' scenario-based question."
                response = self.model.generate_content(prompt)
                return response.text.strip()
            
            elif step == 5:
                # Condition: Problem-solving based on a specific skill
                prompt = f"Identify the most prominent skill in this resume: {resume_text}. Ask a challenging problem-solving question or technical brain-teaser related to that skill."
                response = self.model.generate_content(prompt)
                return response.text.strip()

            return "The interview is now complete. Thank you for your time!"
            
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}"

    def evaluate_response(self, question: str, answer_text: str):
        """
        Evaluates answer and provides feedback/scores.
        """
        try:
            prompt = f"""
            As an expert interviewer, evaluate this response.
            Question: "{question}"
            Candidate Answer: "{answer_text}"
            
            Provide:
            1. A score out of 10.
            2. Constructive feedback.
            3. A short conversational transition to acknowledge the answer.

            Format as JSON:
            {{
                "score": 8,
                "feedback": "...",
                "interviewer_reply": "..."
            }}
            """
            response = self.model.generate_content(prompt)
            text_response = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(text_response)
        except Exception as e:
            return {"score": 0, "feedback": str(e), "interviewer_reply": "Got it, moving on."}

    def transcribe_audio(self, audio_path: str):
        """
        Uses Gemini's native audio capabilities to transcribe speech.
        """
        try:
            audio_file = genai.upload_file(path=audio_path)
            prompt = "Transcribe this audio exactly as spoken. Return ONLY the text."
            response = self.model.generate_content([prompt, audio_file])
            genai.delete_file(audio_file.name)
            return response.text.strip()
        except Exception as e:
            return f"Transcription error: {str(e)}"