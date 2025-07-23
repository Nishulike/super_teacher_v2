from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict, Any, Optional
import os

class ExplanationChain:
    """Chain for providing clear explanations to students"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            temperature=config.get("temperature", 0.7),
            model=config.get("model", "gemini-2.0-flash-exp"),
            max_output_tokens=config.get("max_tokens", 2000),
            google_api_key=config.get("api_key")
        )
        
        # Load explanation prompt
        prompt_path = "prompts/explain_prompt.txt"
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
        
        self.prompt = PromptTemplate(
            input_variables=["grade_level", "subject", "topic", "previous_knowledge"],
            template=prompt_template
        )
        
        self.memory = ConversationBufferWindowMemory(
            k=config.get("memory_k", 5),
            return_messages=True
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=config.get("debug", False)
        )
    
    def explain_topic(self, topic: str, grade_level: str, subject: str, 
                     previous_knowledge: str = "", student_id: str = None, custom_prompt: str = None) -> str:
        """Generate an explanation for a specific topic using the prompt template file only"""
        try:
            if custom_prompt:
                # Use custom prompt for specific questions with enhanced context
                custom_explanation_prompt = f"""
                You are a Super Teacher helping a {grade_level} student with {subject}.
                
                {custom_prompt}
                
                Please provide a helpful response that directly addresses their question.
                Use age-appropriate language and make sure your answer is clear and engaging.
                If they're asking for examples, provide relevant examples. If they want a brief explanation, keep it concise.
                If they want applications, show real-world uses. Address exactly what they're asking for.
                
                Make your response comprehensive but easy to understand for a {grade_level} student.
                """
                response = self.llm.predict(custom_explanation_prompt)
            else:
                # Use standard prompt template
                prompt_text = self.prompt.format(
                    grade_level=grade_level,
                    subject=subject,
                    topic=topic,
                    previous_knowledge=previous_knowledge or "No previous knowledge specified"
                )
                response = self.llm.predict(prompt_text)
            
            if student_id:
                self.memory.chat_memory.add_user_message(f"Explain {topic} for {grade_level} {subject}")
                self.memory.chat_memory.add_ai_message(response)
            # Remove all double asterisks, single asterisks, and backticks from the response to avoid markdown
            response = response.replace('**', '')
            response = response.replace('*', '')
            response = response.replace('`', '')
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while explaining {topic}: {str(e)}"
    
    def explain_with_examples(self, topic: str, grade_level: str, subject: str,
                            num_examples: int = 3, difficulty: str = "medium") -> str:
        """Generate explanation with multiple examples"""
        
        examples_prompt = f"""
        You are a Super Teacher creating engaging examples for students.
        
        Please explain {topic} for a {grade_level} student studying {subject}.
        Include {num_examples} practical examples that are appropriate for {difficulty} difficulty level.
        Make sure the examples are engaging and help reinforce the concept.
        Use age-appropriate language and real-world scenarios that students can relate to.
        """
        
        try:
            response = self.llm.predict(examples_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while creating examples for {topic}: {str(e)}"
    
    def explain_step_by_step(self, topic: str, grade_level: str, subject: str) -> str:
        """Generate a step-by-step explanation"""
        
        step_prompt = f"""
        Please provide a step-by-step explanation of {topic} for a {grade_level} student studying {subject}.
        Break down the concept into clear, sequential steps that build upon each other.
        Include checkpoints where the student can verify their understanding.
        """
        
        try:
            response = self.llm.predict(step_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while creating step-by-step explanation for {topic}: {str(e)}"
    
    def clarify_concept(self, topic: str, grade_level: str, subject: str,
                       student_question: str) -> str:
        """Clarify a concept based on student's specific question"""
        
        clarification_prompt = f"""
        A {grade_level} student studying {subject} is asking about {topic}:
        Student's question: {student_question}
        
        Please provide a clear, helpful clarification that addresses their specific question.
        Use age-appropriate language and examples that will help them understand.
        """
        
        try:
            response = self.llm.predict(clarification_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while clarifying {topic}: {str(e)}"
    
    def get_learning_objectives(self, topic: str, grade_level: str, subject: str) -> str:
        """Generate learning objectives for a topic"""
        
        objectives_prompt = f"""
        For a {grade_level} student studying {subject}, what are the key learning objectives for {topic}?
        Please list 3-5 clear, measurable learning objectives that a student should achieve.
        Make them specific and appropriate for the grade level.
        """
        
        try:
            response = self.llm.predict(objectives_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while generating learning objectives for {topic}: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear() 