import os
from typing import Dict, Any
from dotenv import load_dotenv
from msb_subjects_chapters import MSB_CURRICULUM

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Super Teacher application"""
    
    # Google Gemini Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = "gemini-2.0-flash-exp"
    GEMINI_TEMPERATURE = 0.7
    
    # Application Configuration
    APP_NAME = "Super Teacher"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Memory Configuration
    MEMORY_K = 5  # Number of previous interactions to remember
    
    # Chain Configuration
    MAX_TOKENS = 2000
    CHAIN_TIMEOUT = 30
    
    # Supported Standards (Only 9th and 10th Grade for Maharashtra Board)
    SUPPORTED_STANDARDS = [
        "9th", "10th"
    ]
    
    # Maharashtra Board Curriculum Structure
    MAHARASHTRA_CURRICULUM = MSB_CURRICULUM
    
    # Get all subjects from the curriculum
    @classmethod
    def get_all_subjects(cls) -> list:
        """Get all unique subjects from the curriculum"""
        subjects = set()
        for standard in cls.MAHARASHTRA_CURRICULUM.values():
            subjects.update(standard.keys())
        return sorted(list(subjects))
    
    SUPPORTED_SUBJECTS = property(get_all_subjects)
    
    # Chain Types
    CHAIN_TYPES = {
        "explain": "Explanation Chain",
        "quiz": "Quiz Chain", 
        "socratic": "Socratic Chain",
        "fiveyo": "Five-Year-Old Chain",
        "hook": "Hook Chain"
    }
    
    # Action Types
    ACTION_TYPES = {
        "explain": "Explain a topic",
        "quiz": "Generate a quiz",
        "socratic": "Start Socratic dialogue",
        "fiveyo": "Explain like I'm 5",
        "hook": "Generate a hook",
        "clarify": "Clarify a concept",
        "examples": "Provide examples",
        "step_by_step": "Step-by-step explanation",
        "objectives": "Learning objectives",
        "assessment": "Assessment questions",
        "activities": "Fun activities",
        "stories": "Create stories",
        "songs": "Create songs/rhymes",
        "experiments": "Simple experiments"
    }
    
    # Quiz Configuration
    QUIZ_DIFFICULTIES = ["easy", "medium", "hard"]
    QUIZ_TYPES = ["multiple_choice", "true_false", "fill_blank", "short_answer"]
    DEFAULT_QUIZ_QUESTIONS = 5
    MAX_QUIZ_QUESTIONS = 20
    
    # Socratic Dialogue Configuration
    SOCRATIC_MAX_QUESTIONS = 10
    SOCRATIC_TIMEOUT = 300  # 5 minutes
    
    # Memory Configuration
    MEMORY_RETENTION_DAYS = 30
    MEMORY_MAX_ENTRIES = 1000
    
    # Response Configuration
    MAX_RESPONSE_LENGTH = 2000
    RESPONSE_TIMEOUT = 60
    
    # File Upload Configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES = ['.txt', '.pdf', '.doc', '.docx']
    
    # Session Configuration
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_SESSIONS_PER_USER = 5
    
    # Error Messages
    ERROR_MESSAGES = {
        "missing_api_key": "Google API key is required. Please set GOOGLE_API_KEY environment variable.",
        "invalid_standard": "Invalid standard selected. Please choose from supported standards.",
        "invalid_subject": "Invalid subject selected. Please choose from supported subjects.",
        "missing_topic": "Topic is required for this action.",
        "chain_timeout": "Request timed out. Please try again.",
        "memory_full": "Memory is full. Oldest entries will be removed.",
        "file_too_large": "File size exceeds maximum allowed size.",
        "invalid_file_type": "File type not supported.",
        "session_expired": "Session has expired. Please start a new session."
    }
    
    # Success Messages
    SUCCESS_MESSAGES = {
        "explanation_generated": "Explanation generated successfully!",
        "quiz_generated": "Quiz generated successfully!",
        "memory_saved": "Memory saved successfully!",
        "memory_loaded": "Memory loaded successfully!",
        "session_created": "New session created successfully!",
        "file_uploaded": "File uploaded successfully!"
    }
    
    # Prompt Templates
    PROMPT_TEMPLATES = {
        "explanation": """
        You are a knowledgeable teacher for {grade_level} {subject}. 
        Explain the topic "{topic}" in a clear, engaging way that students can understand.
        Use examples, analogies, and step-by-step explanations when helpful.
        """,
        "quiz": """
        Generate a {difficulty} level quiz with {num_questions} questions about "{topic}" 
        for {grade_level} {subject} students.
        Include multiple choice questions with clear explanations for correct answers.
        """,
        "socratic": """
        Start a Socratic dialogue about "{topic}" for {grade_level} {subject} students.
        Ask thought-provoking questions that guide students to discover the answer themselves.
        """,
        "fiveyo": """
        Explain "{topic}" from {subject} as if you're talking to a 5-year-old child.
        Use simple language, fun analogies, and relatable examples.
        Make it engaging and easy to understand.
        """
    }
    
    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get Gemini configuration"""
        return {
            "api_key": cls.GOOGLE_API_KEY,
            "model": cls.GEMINI_MODEL,
            "temperature": cls.GEMINI_TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "timeout": cls.CHAIN_TIMEOUT
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError(cls.ERROR_MESSAGES["missing_api_key"])
        return True
    
    @classmethod
    def get_subjects_for_standard(cls, standard: str) -> list:
        """Get subjects available for a specific standard"""
        if standard not in cls.MAHARASHTRA_CURRICULUM:
            return []
        return list(cls.MAHARASHTRA_CURRICULUM[standard].keys())
    
    @classmethod
    def get_chapters_for_subject(cls, standard: str, subject: str) -> list:
        """Get chapters available for a specific standard and subject"""
        if standard not in cls.MAHARASHTRA_CURRICULUM:
            return []
        if subject not in cls.MAHARASHTRA_CURRICULUM[standard]:
            return []
        return list(cls.MAHARASHTRA_CURRICULUM[standard][subject].keys())
    
    @classmethod
    def get_topics_for_chapter(cls, standard: str, subject: str, chapter: str) -> list:
        """Get topics available for a specific standard, subject, and chapter"""
        if standard not in cls.MAHARASHTRA_CURRICULUM:
            return []
        if subject not in cls.MAHARASHTRA_CURRICULUM[standard]:
            return []
        if chapter not in cls.MAHARASHTRA_CURRICULUM[standard][subject]:
            return []
        return cls.MAHARASHTRA_CURRICULUM[standard][subject][chapter] 