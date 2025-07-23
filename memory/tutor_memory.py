from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import json
import os
from datetime import datetime

class TutorMemory:
    """Memory management for Super Teacher application"""
    
    def __init__(self, k: int = 5):
        self.memory = ConversationBufferWindowMemory(k=k, return_messages=True)
        self.student_profiles: Dict[str, Dict[str, Any]] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
        
    def add_message(self, message: str, is_human: bool = True, session_id: str = "default"):
        """Add a message to the conversation memory"""
        if is_human:
            self.memory.chat_memory.add_user_message(message)
        else:
            self.memory.chat_memory.add_ai_message(message)
            
        # Store in session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
        self.session_data[session_id]["messages"].append({
            "content": message,
            "type": "human" if is_human else "ai",
            "timestamp": datetime.now().isoformat()
        })
        self.session_data[session_id]["last_updated"] = datetime.now().isoformat()
    
    def get_messages(self, session_id: str = "default") -> List[BaseMessage]:
        """Get conversation messages"""
        return self.memory.chat_memory.messages
    
    def get_session_history(self, session_id: str = "default") -> List[Dict[str, Any]]:
        """Get session conversation history"""
        if session_id in self.session_data:
            return self.session_data[session_id]["messages"]
        return []
    
    def create_student_profile(self, student_id: str, grade_level: str, subject: str, 
                             topics: List[str] = None, preferences: Dict[str, Any] = None):
        """Create or update a student profile"""
        self.student_profiles[student_id] = {
            "grade_level": grade_level,
            "subject": subject,
            "topics": topics or [],
            "preferences": preferences or {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "progress": {},
            "sessions": []
        }
    
    def update_student_progress(self, student_id: str, topic: str, progress_data: Dict[str, Any]):
        """Update student progress for a specific topic"""
        if student_id in self.student_profiles:
            self.student_profiles[student_id]["progress"][topic] = {
                **progress_data,
                "last_updated": datetime.now().isoformat()
            }
            self.student_profiles[student_id]["last_updated"] = datetime.now().isoformat()
    
    def get_student_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Get student profile"""
        return self.student_profiles.get(student_id)
    
    def add_session_to_profile(self, student_id: str, session_id: str, session_data: Dict[str, Any]):
        """Add session data to student profile"""
        if student_id in self.student_profiles:
            session_info = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                **session_data
            }
            self.student_profiles[student_id]["sessions"].append(session_info)
    
    def get_student_sessions(self, student_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a student"""
        if student_id in self.student_profiles:
            return self.student_profiles[student_id]["sessions"]
        return []
    
    def clear_memory(self):
        """Clear current conversation memory"""
        self.memory.clear()
    
    def save_to_file(self, filename: str = "tutor_memory.json"):
        """Save memory data to file"""
        data = {
            "student_profiles": self.student_profiles,
            "session_data": self.session_data,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str = "tutor_memory.json"):
        """Load memory data from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.student_profiles = data.get("student_profiles", {})
                self.session_data = data.get("session_data", {})
    
    def get_context_for_topic(self, student_id: str, topic: str) -> Dict[str, Any]:
        """Get relevant context for a specific topic"""
        profile = self.get_student_profile(student_id)
        if not profile:
            return {}
        
        context = {
            "grade_level": profile.get("grade_level"),
            "subject": profile.get("subject"),
            "previous_knowledge": profile.get("progress", {}).get(topic, {}),
            "learning_preferences": profile.get("preferences", {}),
            "recent_sessions": profile.get("sessions", [])[-3:]  # Last 3 sessions
        }
        
        return context 