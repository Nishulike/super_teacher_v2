from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict, Any, List, Optional
import json
import re

class SocraticChain:
    """Chain for Socratic questioning and Bloom's Taxonomy-based learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            temperature=config.get("temperature", 0.7),
            model=config.get("model", "gemini-2.0-flash-exp"),
            max_output_tokens=config.get("max_tokens", 2000),
            google_api_key=config.get("api_key")
        )
        
        # Load Bloom's taxonomy prompt
        prompt_path = "prompts/bloom_prompt.txt"
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
        
        self.prompt = PromptTemplate(
            input_variables=["grade_level", "subject", "topic", "current_understanding"],
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
        
        # Bloom's Taxonomy levels
        self.bloom_levels = {
            "remember": "Recall facts and basic concepts",
            "understand": "Explain ideas and concepts", 
            "apply": "Use information in new situations",
            "analyze": "Draw connections and identify patterns",
            "evaluate": "Make judgments and justify decisions",
            "create": "Produce new work or original ideas"
        }
    
    def start_socratic_dialogue(self, topic: str, grade_level: str, subject: str,
                              current_understanding: str = "") -> str:
        """Start a Socratic dialogue on a topic"""
        
        context = {
            "grade_level": grade_level,
            "subject": subject,
            "topic": topic,
            "current_understanding": current_understanding or "No previous understanding specified"
        }
        
        try:
            response = self.chain.run(context)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while starting the Socratic dialogue on {topic}: {str(e)}"
    
    def ask_clarifying_question(self, topic: str, grade_level: str, subject: str,
                               student_response: str) -> str:
        """Ask a clarifying question based on student's response"""
        
        clarifying_prompt = f"""
        A {grade_level} student studying {subject} made this response about {topic}:
        "{student_response}"
        
        Ask a clarifying question that will help the student think more deeply about their response.
        Use Socratic questioning techniques to guide them toward better understanding.
        Keep the question age-appropriate and encouraging.
        """
        
        try:
            response = self.llm.predict(clarifying_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while formulating a clarifying question: {str(e)}"
    
    def ask_analysis_question(self, topic: str, grade_level: str, subject: str,
                            current_discussion: str) -> str:
        """Ask an analysis question to deepen understanding"""
        
        analysis_prompt = f"""
        A {grade_level} student studying {subject} is discussing {topic}.
        Current discussion: "{current_discussion}"
        
        Ask an analysis question that will help the student:
        1. Compare and contrast different aspects of the topic
        2. Identify patterns or relationships
        3. Break down complex ideas into parts
        4. Examine the underlying structure or assumptions
        
        Make the question engaging and appropriate for their grade level.
        """
        
        try:
            response = self.llm.predict(analysis_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while formulating an analysis question: {str(e)}"
    
    def ask_evaluation_question(self, topic: str, grade_level: str, subject: str,
                              student_analysis: str) -> str:
        """Ask an evaluation question to assess understanding"""
        
        evaluation_prompt = f"""
        A {grade_level} student studying {subject} provided this analysis of {topic}:
        "{student_analysis}"
        
        Ask an evaluation question that will help the student:
        1. Assess the quality or validity of their analysis
        2. Consider different perspectives or viewpoints
        3. Justify their reasoning or conclusions
        4. Reflect on the implications of their understanding
        
        Encourage critical thinking while being supportive.
        """
        
        try:
            response = self.llm.predict(evaluation_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while formulating an evaluation question: {str(e)}"
    
    def ask_creation_question(self, topic: str, grade_level: str, subject: str,
                            student_evaluation: str) -> str:
        """Ask a creation question to apply knowledge"""
        
        creation_prompt = f"""
        A {grade_level} student studying {subject} provided this evaluation of {topic}:
        "{student_evaluation}"
        
        Ask a creation question that will help the student:
        1. Apply their knowledge in a new way
        2. Create something original based on their understanding
        3. Synthesize their learning into a new format
        4. Demonstrate mastery through creative application
        
        Make it engaging and appropriate for their age and skill level.
        """
        
        try:
            response = self.llm.predict(creation_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while formulating a creation question: {str(e)}"
    
    def generate_bloom_questions(self, topic: str, grade_level: str, subject: str,
                               target_level: str = "all") -> Dict[str, List[str]]:
        """Generate questions for each Bloom's taxonomy level with improved JSON handling"""
        
        # Use the prompt template that was loaded in the constructor
        context = {
            "grade_level": grade_level,
            "subject": subject,
            "topic": topic,
            "current_understanding": ""
        }
        
        # Create a more specific prompt for generating questions only
        bloom_prompt = f"""
You are a Super Teacher using Bloom's Taxonomy to create educational questions.

Context:
- Student's Grade Level: {grade_level}
- Subject: {subject}
- Topic: {topic}

Generate exactly 6 questions, one for each Bloom's Taxonomy level, in this order:
1. Remember: Recall a basic fact about {topic}
2. Understand: Explain {topic} in simple words
3. Apply: How would you use knowledge of {topic} in a real-life situation?
4. Analyze: What are the different parts or aspects of {topic}?
5. Evaluate: What do you think about {topic} and why?
6. Create: How could you use {topic} to create something new?

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "questions": [
    {{"level": "remember", "question": "Your remember question here"}},
    {{"level": "understand", "question": "Your understand question here"}},
    {{"level": "apply", "question": "Your apply question here"}},
    {{"level": "analyze", "question": "Your analyze question here"}},
    {{"level": "evaluate", "question": "Your evaluate question here"}},
    {{"level": "create", "question": "Your create question here"}}
  ]
}}

Do not include any text before or after the JSON. Only return the JSON object.
"""
        
        try:
            response = self.llm.predict(bloom_prompt)
            
            # Clean the response and extract JSON
            import re, json
            
            # Remove any leading/trailing whitespace and newlines
            response = response.strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Extract questions
            questions = {}
            for q in data.get("questions", []):
                level = q.get("level", "").lower()
                question = q.get("question", "")
                if level and question:
                    if level not in questions:
                        questions[level] = []
                    questions[level].append(question)
            
            # Ensure all levels are present
            all_levels = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
            for level in all_levels:
                if level not in questions:
                    questions[level] = [f"Error: Could not generate {level} question"]
            
            return questions
            
        except Exception as e:
            # Return fallback questions for each level
            return {
                "remember": [f"Can you recall a basic fact about {topic}?"],
                "understand": [f"Can you explain {topic} in your own words?"],
                "apply": [f"How would you use your knowledge of {topic} in everyday life?"],
                "analyze": [f"What are the different parts or aspects of {topic}?"],
                "evaluate": [f"What do you think about {topic} and why?"],
                "create": [f"How could you use {topic} to create something new?"]
            }
    
    def assess_bloom_level(self, topic: str, grade_level: str, subject: str,
                          student_response: str) -> Dict[str, Any]:
        """Assess which Bloom's taxonomy level the student's response demonstrates and score it using the rubric"""
        
        # Use LLM for accurate assessment with better error handling
        assessment_prompt = f'''
You are a Super Teacher evaluating a student's answer for Bloom's Taxonomy.
Topic: {topic}
Grade Level: {grade_level}
Subject: {subject}
Student's Answer: "{student_response}"

Score the answer using this rubric:
- Perfect answer: 100
- Mostly correct: 80
- Partial/vague: 60
- Incomplete effort: 40
- Hesitant ("Maybe", "I think"): 20
- Wrong/no idea/off-topic: 0

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "level": "(bloom level)",
  "score": (0|20|40|60|80|100),
  "explanation": "short explanation for the score"
}}

Do not include any text before or after the JSON. Only return the JSON object.
'''
        
        try:
            response = self.llm.predict(assessment_prompt)
            
            # Clean the response and extract JSON
            import re, json
            
            # Remove any leading/trailing whitespace and newlines
            response = response.strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
            
            # Parse the JSON
            assessment = json.loads(json_str)
            
            # Validate the assessment
            if not isinstance(assessment, dict):
                raise ValueError("Assessment is not a dictionary")
            
            # Ensure required fields are present
            if "score" not in assessment:
                assessment["score"] = 60  # Default score
            
            if "level" not in assessment:
                assessment["level"] = "unknown"
            
            if "explanation" not in assessment:
                assessment["explanation"] = "No explanation provided"
            
            # Ensure score is a number
            try:
                assessment["score"] = int(assessment["score"])
            except (ValueError, TypeError):
                assessment["score"] = 60
            
            return assessment
            
        except Exception as e:
            print(f"Error in assess_bloom_level: {e}")
            print(f"Response was: {response if 'response' in locals() else 'No response'}")
            
            # If API fails, use a simple fallback based on response length and content
            response_lower = student_response.lower().strip()
            
            if not response_lower or len(response_lower) < 5:
                return {
                    "level": "unknown",
                    "score": 0,
                    "explanation": "No response or very short answer (API unavailable)"
                }
            
            # Check for hesitant responses
            hesitant_words = ["maybe", "i think", "i guess", "probably", "possibly", "not sure", "don't know", "i don't know"]
            if any(word in response_lower for word in hesitant_words):
                return {
                    "level": "unknown",
                    "score": 20,
                    "explanation": "Hesitant response (API unavailable)"
                }
            
            # Simple length-based scoring as fallback
            if len(response_lower) > 100:
                score = 80
            elif len(response_lower) > 50:
                score = 60
            elif len(response_lower) > 20:
                score = 40
            else:
                score = 20
            
            return {
                "level": "unknown",
                "score": score,
                "explanation": f"Fallback scoring due to API error: {str(e)}"
            }
    
    def provide_guidance(self, topic: str, grade_level: str, subject: str,
                        student_struggles: str) -> str:
        """Provide guidance when student is struggling"""
        
        guidance_prompt = f"""
        A {grade_level} student studying {subject} is struggling with {topic}.
        Their specific difficulty: "{student_struggles}"
        
        Provide supportive guidance that:
        1. Acknowledges their effort and progress
        2. Breaks down the concept into smaller, manageable parts
        3. Offers a different perspective or approach
        4. Encourages them to think about it in a new way
        5. Suggests a simpler starting point if needed
        
        Be encouraging and patient. Remember that struggle is part of learning.
        """
        
        try:
            response = self.llm.predict(guidance_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while providing guidance: {str(e)}"
    
    def summarize_learning(self, topic: str, grade_level: str, subject: str,
                          conversation_history: List[str]) -> str:
        """Summarize what the student has learned"""
        
        summary_prompt = f"""
        A {grade_level} student studying {subject} has been engaged in a Socratic dialogue about {topic}.
        
        Conversation highlights:
        {chr(10).join(conversation_history)}
        
        Provide a brief summary of:
        1. Key concepts the student has explored
        2. Their demonstrated understanding
        3. Areas where they showed growth
        4. Suggestions for further exploration
        
        Keep it encouraging and focused on their learning journey.
        """
        
        try:
            response = self.llm.predict(summary_prompt)
            return response
        except Exception as e:
            return f"Sorry, I encountered an error while summarizing the learning: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear() 

    def get_next_weak_level_question(self, topic: str, grade_level: str, subject: str, level: str) -> str:
        """Generate a short, clear, focused Socratic question for a weak Bloom's level (max 20 words)."""
        prompt = (
            f"Generate a short, clear, and focused Socratic question (max 20 words) "
            f"for the {level} Bloom's level on {topic} for grade {grade_level} {subject}. "
            f"The question should directly help the student understand the concept."
        )
        return self.llm.predict(prompt)

    def mastery_loop(self, topic: str, grade_level: str, subject: str, student_answers: dict, weak_levels: list = None, retry_index: int = 0, prev_scores: dict = None):
        """Mastery loop: assess answers, identify weak levels, and generate next question for one weak level at a time."""
        # Assess all answers
        breakdown = {}
        total_score = 0
        for level, answer in student_answers.items():
            assessment = self.assess_bloom_level(topic, grade_level, subject, answer)
            breakdown[level] = assessment['score']
            total_score += assessment['score']
        overall_score = total_score / len(breakdown) if breakdown else 0
        if overall_score >= 80:
            return {
                "mastery": True,
                "overall_score": overall_score,
                "breakdown": breakdown,
                "weak_levels": [],
                "next_weak_level": None,
                "next_question": None,
                "retry_index": None
            }
        # Find weak levels
        if weak_levels is None:
            weak_levels = [level for level, score in breakdown.items() if score < 80]
        # If all weak levels have been retried, return for re-evaluation
        if retry_index >= len(weak_levels):
            return {
                "mastery": False,
                "overall_score": overall_score,
                "breakdown": breakdown,
                "weak_levels": weak_levels,
                "next_weak_level": None,
                "next_question": None,
                "retry_index": retry_index
            }
        # Otherwise, return the next weak-level question
        next_level = weak_levels[retry_index]
        next_question = self.get_next_weak_level_question(topic, grade_level, subject, next_level)
        return {
            "mastery": False,
            "overall_score": overall_score,
            "breakdown": breakdown,
            "weak_levels": weak_levels,
            "next_weak_level": next_level,
            "next_question": next_question,
            "retry_index": retry_index
        } 

    def analyze_diagnostic_answers(self, topic: str, grade_level: str, subject: str, answers: list) -> dict:
        """
        Analyze student answers to diagnostic Bloom-level questions.
        Returns per-level scores, overall score, misconceptions, and suggested next steps.
        Each answer in 'answers' should be a dict: {"level": ..., "answer": ...}
        """
        results = []
        total_score = 0
        misconceptions = []
        for entry in answers:
            level = entry.get("level")
            answer = entry.get("answer", "")
            assessment = self.assess_bloom_level(topic, grade_level, subject, answer)
            score = assessment.get("score", 0)
            explanation = assessment.get("explanation", "")
            results.append({
                "level": level,
                "answer": answer,
                "score": score,
                "explanation": explanation
            })
            total_score += score
            # Simple misconception detection: if score <= 40, flag as misconception
            if score <= 40:
                misconceptions.append({"level": level, "answer": answer, "reason": explanation})
        overall_score = total_score / len(results) if results else 0
        # Find weakest level(s)
        weak_levels = [r["level"] for r in results if r["score"] < 80]
        # Suggest next steps
        if overall_score >= 80:
            next_steps = "Great job! You have strong understanding. Ready to move to the next topic."
        elif overall_score < 50:
            next_steps = "Let's try an 'Explain Like I'm 5' approach or use more visuals/examples."
        elif weak_levels:
            next_steps = f"Let's focus on more practice for: {', '.join(weak_levels)}."
        else:
            next_steps = "Keep practicing and ask questions if you're unsure."
        return {
            "results": results,
            "overall_score": overall_score,
            "misconceptions": misconceptions,
            "weak_levels": weak_levels,
            "next_steps": next_steps
        }
    
    def diagnostic_dialogue(self, topic: str, grade_level: str, subject: str, num_questions: int = 4) -> list:
        """Generate a diagnostic Socratic dialogue: 3–5 open-ended, progressively deeper Bloom-level questions."""
        # Bloom's taxonomy levels in order
        bloom_levels = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
        # Generate questions for all levels
        questions = self.generate_bloom_questions(topic, grade_level, subject)
        # Select the first num_questions levels
        selected_levels = bloom_levels[:num_questions]
        dialogue = []
        for level in selected_levels:
            q_list = questions.get(level, [])
            question = q_list[0] if q_list else f"Could you share your thoughts about {topic}?"
            dialogue.append({"level": level, "question": question})
        return dialogue 

    def real_time_feedback(self, topic: str, grade_level: str, subject: str, answer: str, current_level: str) -> dict:
        """
        Analyze a student's answer for hesitation/confusion and provide adaptive feedback.
        Returns a dict with feedback, suggested action, and (if needed) a simpler or scaffolded question.
        """
        # Detect hesitation/confusion
        hesitation_words = ["maybe", "i think", "i guess", "probably", "possibly", "not sure", "don't know", "um", "uh"]
        answer_lower = answer.lower()
        is_hesitant = any(word in answer_lower for word in hesitation_words)
        feedback = ""
        suggested_action = "continue"
        scaffolded_question = None
        if is_hesitant or len(answer_lower.strip()) < 10:
            feedback = (
                "I noticed you seem unsure or hesitant. That's totally okay! Let's try a simpler question or use an example."
            )
            # Backtrack to a simpler Bloom level if possible
            bloom_order = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
            if current_level in bloom_order and bloom_order.index(current_level) > 0:
                simpler_level = bloom_order[bloom_order.index(current_level) - 1]
                scaffolded_question = self.get_next_weak_level_question(topic, grade_level, subject, simpler_level)
                suggested_action = f"backtrack_to_{simpler_level}"
            else:
                # Offer a multiple-choice or hint
                scaffolded_question = f"Here's a hint: Think about the basics of {topic}. What comes to mind first?"
                suggested_action = "offer_hint"
        else:
            feedback = "Great effort! If you want to try a different approach or need a hint, just ask."
        return {
            "feedback": feedback,
            "suggested_action": suggested_action,
            "scaffolded_question": scaffolded_question
        }

    def get_feedback_and_next_question(self, topic: str, grade_level: str, subject: str, student_answer: str, current_level: str) -> dict:
        """
        Given a student's answer and the current Bloom's level, return feedback and the next question.
        - If the answer is hesitant, provide supportive feedback and a simplified next question (using real_time_feedback).
        - If confident, provide positive feedback and the next Bloom's level question as normal.
        Always proceed to the next question (do not repeat the same level).
        """
        feedback_data = self.real_time_feedback(topic, grade_level, subject, student_answer, current_level)
        # Get LLM-based assessment for richer feedback
        assessment = self.assess_bloom_level(topic, grade_level, subject, student_answer)
        score = assessment.get('score', 60)
        explanation = assessment.get('explanation', '')
        if feedback_data["suggested_action"] != "continue":
            # Hesitant: Give feedback and simplified next question
            feedback = feedback_data["feedback"]
            next_question = feedback_data["scaffolded_question"]
        else:
            # LLM-driven feedback based on score
            if score >= 80:
                feedback = f"Excellent! {explanation}"
            elif score >= 60:
                feedback = f"Good try! {explanation} If you want to improve, think about the details or examples."
            elif score >= 40:
                feedback = f"You're on the right track, but let's try to be more specific. {explanation}"
            elif score >= 20:
                feedback = f"Don't worry! Many students find this tricky. {explanation} Want a hint or another example?"
            else:
                feedback = f"It's okay to make mistakes! {explanation} Let's review the basics and try again."
            # Determine the next Bloom's level
            bloom_order = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
            try:
                next_index = bloom_order.index(current_level) + 1
                if next_index < len(bloom_order):
                    next_level = bloom_order[next_index]
                    next_question = self.get_next_weak_level_question(topic, grade_level, subject, next_level)
                else:
                    next_question = None  # No more levels
            except ValueError:
                next_question = None
        return {
            "feedback": feedback,
            "next_question": next_question
        }

    def reflection_and_next_steps(self, topic: str, grade_level: str, subject: str, session_summary: str = "") -> dict:
        """
        Generate reflection prompts and personalized next-step suggestions after a learning session.
        Returns a dict with reflection questions and recommended actions.
        """
        # Reflection prompts (as per Bloom flow Step 7)
        reflection_prompts = [
            "What surprised you the most while learning about this topic?",
            "Where do you feel less confident or want more practice?",
            "Would you like to go deeper into this topic or shift to something new?"
        ]
        # Personalized next steps (simple logic, can be expanded)
        recommended_actions = [
            "If you found Evaluate questions hard, try a debate or discussion activity.",
            "If you haven't tried a Create-level task, consider a small project or creative challenge related to this topic.",
            "Review your answers and try to explain the topic to someone else (or back to me!)."
        ]
        # Optionally, use session_summary to tailor suggestions
        return {
            "reflection_prompts": reflection_prompts,
            "recommended_actions": recommended_actions
        } 