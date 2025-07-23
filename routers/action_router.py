from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
import re

class ActionRouter:
    """Router to determine which chain to use based on user input"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            temperature=config.get("temperature", 0.7),
            model=config.get("model", "gemini-2.0-flash-exp"),
            max_output_tokens=config.get("max_tokens", 1000),
            google_api_key=config.get("api_key")
        )
        
        # Define available actions
        self.available_actions = {
            "explain": "Provide clear explanations of concepts",
            "quiz": "Generate quizzes and assessments",
            "socratic": "Use Socratic questioning and Bloom's taxonomy",
            "fiveyo": "Create 5-year-old level explanations and activities",
            "clarify": "Clarify specific concepts or questions",
            "examples": "Provide examples and applications",
            "step_by_step": "Give step-by-step explanations",
            "objectives": "Generate learning objectives",
            "activities": "Create hands-on learning activities",
            "stories": "Create educational stories",
            "songs": "Create educational songs or rhymes",
            "experiments": "Create simple experiments",
            "assessment": "Assess current understanding level"
        }
        
        # Create routing prompt
        self.routing_prompt = PromptTemplate(
            input_variables=["user_input", "grade_level", "subject", "topic", "available_actions"],
            template="""
            You are an intelligent router for a Super Teacher application. Based on the user's input, determine the most appropriate action to take.
            
            User Input: {user_input}
            Grade Level: {grade_level}
            Subject: {subject}
            Topic: {topic}
            
            Available Actions:
            {available_actions}
            
            Analyze the user's request and choose the most appropriate action. Consider:
            1. What the user is explicitly asking for
            2. The context (grade level, subject, topic)
            3. The user's learning needs
            4. The most effective teaching approach for their request
            
            Return your response as JSON with this structure:
            {{
                "action": "chosen_action",
                "confidence": 0.95,
                "reasoning": "Why this action was chosen",
                "parameters": {{
                    "additional_info": "Any additional parameters needed"
                }}
            }}
            
            Choose the action that best matches the user's request.
            """
        )
    
    def route_action(self, user_input: str, grade_level: str, subject: str, 
                    topic: str = "", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route user input to the appropriate action"""
        
        # Prepare available actions string
        actions_str = "\n".join([f"- {action}: {description}" 
                                for action, description in self.available_actions.items()])
        
        # Prepare context
        context_data = {
            "user_input": user_input,
            "grade_level": grade_level,
            "subject": subject,
            "topic": topic or "general",
            "available_actions": actions_str
        }
        
        try:
            # Get routing decision
            response = self.llm.predict(self.routing_prompt.format(**context_data))
            
            # Parse JSON response
            routing_decision = json.loads(response)
            
            # Validate the action
            if routing_decision["action"] not in self.available_actions:
                # Fallback to explanation if action not found
                routing_decision = {
                    "action": "explain",
                    "confidence": 0.5,
                    "reasoning": "Action not recognized, defaulting to explanation",
                    "parameters": {}
                }
            
            return routing_decision
            
        except Exception as e:
            # Fallback routing based on keywords
            return self._fallback_route(user_input, grade_level, subject, topic)
    
    def _fallback_route(self, user_input: str, grade_level: str, subject: str, topic: str) -> Dict[str, Any]:
        """Fallback routing based on keyword matching"""
        
        user_input_lower = user_input.lower()
        
        # Keyword-based routing
        if any(word in user_input_lower for word in ["quiz", "test", "question", "assessment"]):
            return {
                "action": "quiz",
                "confidence": 0.8,
                "reasoning": "User requested quiz or assessment",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["socratic", "question", "think", "discuss"]):
            return {
                "action": "socratic",
                "confidence": 0.8,
                "reasoning": "User requested Socratic questioning or discussion",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["simple", "easy", "young", "child", "kid", "5", "five"]):
            return {
                "action": "fiveyo",
                "confidence": 0.9,
                "reasoning": "User requested simple or child-friendly explanation",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["clarify", "confused", "don't understand", "help"]):
            return {
                "action": "clarify",
                "confidence": 0.8,
                "reasoning": "User needs clarification",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["example", "instance", "case"]):
            return {
                "action": "examples",
                "confidence": 0.8,
                "reasoning": "User requested examples",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["step", "process", "how to"]):
            return {
                "action": "step_by_step",
                "confidence": 0.8,
                "reasoning": "User requested step-by-step explanation",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["objective", "goal", "learn"]):
            return {
                "action": "objectives",
                "confidence": 0.8,
                "reasoning": "User requested learning objectives",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["activity", "hands-on", "experiment", "fun"]):
            return {
                "action": "activities",
                "confidence": 0.8,
                "reasoning": "User requested hands-on activity",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["story", "tale", "narrative"]):
            return {
                "action": "stories",
                "confidence": 0.8,
                "reasoning": "User requested story or narrative",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["song", "rhyme", "music", "sing"]):
            return {
                "action": "songs",
                "confidence": 0.8,
                "reasoning": "User requested song or rhyme",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["experiment", "lab", "try", "test"]):
            return {
                "action": "experiments",
                "confidence": 0.8,
                "reasoning": "User requested experiment or hands-on activity",
                "parameters": {}
            }
        elif any(word in user_input_lower for word in ["assess", "evaluate", "check", "level"]):
            return {
                "action": "assessment",
                "confidence": 0.8,
                "reasoning": "User requested assessment or evaluation",
                "parameters": {}
            }
        else:
            # Default to explanation
            return {
                "action": "explain",
                "confidence": 0.6,
                "reasoning": "Default action - provide explanation",
                "parameters": {}
            }
    
    def get_action_description(self, action: str) -> str:
        """Get description of an action"""
        return self.available_actions.get(action, "Unknown action")
    
    def get_all_actions(self) -> Dict[str, str]:
        """Get all available actions and their descriptions"""
        return self.available_actions.copy()
    
    def suggest_actions(self, grade_level: str, subject: str, topic: str) -> List[str]:
        """Suggest appropriate actions based on context"""
        
        suggestions = []
        
        # Always suggest explanation
        suggestions.append("explain")
        
        # Add context-specific suggestions
        if "Kindergarten" in grade_level or "1st Grade" in grade_level or "2nd Grade" in grade_level:
            suggestions.extend(["fiveyo", "stories", "songs", "activities"])
        elif "3rd Grade" in grade_level or "4th Grade" in grade_level or "5th Grade" in grade_level:
            suggestions.extend(["examples", "activities", "stories", "quiz"])
        else:
            suggestions.extend(["examples", "quiz", "socratic", "objectives"])
        
        # Subject-specific suggestions
        if subject.lower() in ["science", "physics", "chemistry", "biology"]:
            suggestions.extend(["experiments", "step_by_step"])
        elif subject.lower() in ["mathematics", "math"]:
            suggestions.extend(["step_by_step", "examples"])
        elif subject.lower() in ["english", "literature", "language"]:
            suggestions.extend(["stories", "examples"])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for action in suggestions:
            if action not in seen:
                seen.add(action)
                unique_suggestions.append(action)
        
        return unique_suggestions[:5]  # Limit to 5 suggestions 