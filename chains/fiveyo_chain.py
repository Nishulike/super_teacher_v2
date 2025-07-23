from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict, Any, List, Optional
import json

class FiveYoChain:
    """Chain for creating 5-year-old level explanations and activities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            temperature=config.get("temperature", 0.8),
            model=config.get("model", "gemini-2.0-flash-exp"),
            max_output_tokens=config.get("max_tokens", 2000),
            google_api_key=config.get("api_key")
        )
        
        self.memory = ConversationBufferWindowMemory(
            k=config.get("memory_k", 5),
            return_messages=True
        )
        
        # Create a simple, engaging prompt for young learners
        self.prompt = PromptTemplate(
            input_variables=["topic", "subject", "activity_type"],
            template="""
            You are a super friendly teacher for 5-year-olds! You make learning super fun and easy to understand.
            
            Topic: {topic}
            Subject: {subject}
            Activity Type: {activity_type}
            
            Remember:
            - Use simple words a 5-year-old knows
            - Make everything fun and exciting
            - Use lots of examples they can see and touch
            - Keep explanations short and sweet
            - Use colors, animals, and everyday things in examples
            - Be encouraging and positive
            - Make learning feel like playing a game
            
            Create something amazing for our little learner!
            """
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=config.get("debug", False)
        )
    
    def explain_like_five(self, topic: str, subject: str) -> str:
        """Explain a topic like you're talking to a 5-year-old"""
        
        context = {
            "topic": topic,
            "subject": subject,
            "activity_type": "explanation"
        }
        try:
            # Use invoke instead of run (future-proof)
            response = self.chain.invoke(context)
            return response
        except Exception as e:
            # Fallback: format prompt manually and use llm.predict
            try:
                prompt_text = self.prompt.format(**context)
                response = self.llm.predict(prompt_text)
                return response
            except Exception as e2:
                return f"Oops! Something went wrong while explaining {topic}: {str(e2)}"
    
    def create_fun_activity(self, topic: str, subject: str) -> Dict[str, Any]:
        """Create a fun, hands-on activity for young learners"""
        
        activity_prompt = f"""
        Create a super fun activity for a 5-year-old to learn about "{topic}" in {subject}.
        
        The activity should:
        1. Be hands-on and interactive
        2. Use everyday materials they can find at home
        3. Be safe for young children
        4. Take 10-15 minutes to complete
        5. Include simple instructions
        6. Have a fun learning outcome
        
        Provide the activity in this format:
        {{
            "title": "Fun activity name",
            "materials": ["item 1", "item 2"],
            "instructions": ["step 1", "step 2"],
            "learning_goal": "What they will learn",
            "fun_fact": "Something cool to share"
        }}
        """
        
        try:
            response = self.llm.predict(activity_prompt)
            activity_data = json.loads(response)
            return activity_data
        except Exception as e:
            return {
                "title": f"Fun {topic} Activity",
                "materials": ["paper", "crayons"],
                "instructions": [f"Draw something about {topic}"],
                "learning_goal": f"Learn about {topic}",
                "fun_fact": f"{topic} is really cool!"
            }
    
    def create_story(self, topic: str, subject: str) -> str:
        """Create a simple story about the topic"""
        
        story_prompt = f"""
        Write a short, fun story for a 5-year-old about "{topic}" in {subject}.
        
        The story should:
        1. Be about 3-4 sentences long
        2. Have a happy ending
        3. Include a character they can relate to
        4. Use simple words
        5. Teach something about the topic
        6. Be engaging and fun to read
        
        Make it feel like a bedtime story that teaches!
        """
        
        try:
            response = self.llm.predict(story_prompt)
            return response
        except Exception as e:
            return f"Once upon a time, there was a little friend who learned about {topic}. It was so much fun!"
    
    def create_simple_quiz(self, topic: str, subject: str) -> Dict[str, Any]:
        """Create a very simple quiz for young learners"""
        
        quiz_prompt = f"""
        Create a super simple quiz for a 5-year-old about "{topic}" in {subject}.
        
        Make it:
        1. Only 3 questions
        2. Use pictures or simple words
        3. Have easy, fun answers
        4. Be encouraging and positive
        5. Include a reward message at the end
        
        Format as JSON:
        {{
            "questions": [
                {{
                    "question": "Simple question here?",
                    "options": ["A: option 1", "B: option 2"],
                    "correct": "A",
                    "explanation": "Simple explanation"
                }}
            ],
            "reward_message": "Great job! You're amazing!"
        }}
        """
        
        try:
            response = self.llm.predict(quiz_prompt)
            quiz_data = json.loads(response)
            return quiz_data
        except Exception as e:
            return {
                "questions": [
                    {
                        "question": f"What is {topic}?",
                        "options": ["A: Something cool", "B: Something boring"],
                        "correct": "A",
                        "explanation": f"{topic} is something really cool!"
                    }
                ],
                "reward_message": "You did great! High five! 🖐️"
            }
    
    def create_song_or_rhyme(self, topic: str, subject: str) -> str:
        """Create a simple song or rhyme about the topic"""
        
        song_prompt = f"""
        Create a simple, fun song or rhyme for a 5-year-old about "{topic}" in {subject}.
        
        Make it:
        1. Easy to sing or say
        2. About 4-6 lines long
        3. Include the topic name
        4. Be catchy and fun
        5. Use simple words
        6. Have a good rhythm
        
        Make it something they'll want to sing over and over!
        """
        
        try:
            response = self.llm.predict(song_prompt)
            return response
        except Exception as e:
            return f"🎵 {topic}, {topic}, so much fun! Learning about {topic} is number one! 🎵"
    
    def create_coloring_activity(self, topic: str, subject: str) -> Dict[str, Any]:
        """Create a coloring activity with learning"""
        
        coloring_prompt = f"""
        Create a coloring activity for a 5-year-old about "{topic}" in {subject}.
        
        Include:
        1. A simple drawing description
        2. What colors to use and why
        3. A fun fact about the topic
        4. What they can learn while coloring
        
        Format as JSON:
        {{
            "drawing_description": "What to draw",
            "colors": ["color 1", "color 2"],
            "color_reasons": "Why these colors",
            "fun_fact": "Something cool to learn",
            "learning_tip": "What they learn while coloring"
        }}
        """
        
        try:
            response = self.llm.predict(coloring_prompt)
            coloring_data = json.loads(response)
            return coloring_data
        except Exception as e:
            return {
                "drawing_description": f"Draw a picture of {topic}",
                "colors": ["blue", "green"],
                "color_reasons": "These are fun colors!",
                "fun_fact": f"{topic} is really interesting!",
                "learning_tip": f"While you color, think about {topic}"
            }
    
    def create_movement_activity(self, topic: str, subject: str) -> Dict[str, Any]:
        """Create a movement-based learning activity"""
        
        movement_prompt = f"""
        Create a fun movement activity for a 5-year-old to learn about "{topic}" in {subject}.
        
        The activity should:
        1. Get them moving and active
        2. Teach something about the topic
        3. Be safe and simple
        4. Use their whole body
        5. Be fun and silly
        6. Include simple instructions
        
        Format as JSON:
        {{
            "activity_name": "Fun name",
            "instructions": ["step 1", "step 2"],
            "learning_connection": "How it teaches the topic",
            "safety_note": "Any safety tips",
            "fun_factor": "Why it's fun"
        }}
        """
        
        try:
            response = self.llm.predict(movement_prompt)
            movement_data = json.loads(response)
            return movement_data
        except Exception as e:
            return {
                "activity_name": f"{topic} Dance",
                "instructions": [f"Jump around like {topic}", "Spin in circles"],
                "learning_connection": f"Moving helps you remember {topic}",
                "safety_note": "Make sure you have space to move!",
                "fun_factor": "Dancing is always fun!"
            }
    
    def create_simple_experiment(self, topic: str, subject: str) -> Dict[str, Any]:
        """Create a simple, safe experiment for young learners"""
        
        experiment_prompt = f"""
        Create a simple, safe experiment for a 5-year-old to learn about "{topic}" in {subject}.
        
        The experiment should:
        1. Use safe, everyday materials
        2. Be supervised by an adult
        3. Have a clear learning outcome
        4. Be exciting and fun
        5. Include simple steps
        6. Have a "wow" factor
        
        Format as JSON:
        {{
            "title": "Experiment name",
            "materials": ["item 1", "item 2"],
            "steps": ["step 1", "step 2"],
            "what_happens": "What they will see",
            "why_it_happens": "Simple explanation",
            "safety_note": "Safety reminder"
        }}
        """
        
        try:
            response = self.llm.predict(experiment_prompt)
            experiment_data = json.loads(response)
            return experiment_data
        except Exception as e:
            return {
                "title": f"Fun {topic} Experiment",
                "materials": ["water", "cup"],
                "steps": ["Pour water in cup", "Watch what happens"],
                "what_happens": "You'll see something cool!",
                "why_it_happens": f"This teaches us about {topic}",
                "safety_note": "Ask an adult for help!"
            }
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear() 