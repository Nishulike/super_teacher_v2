from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any, List, Optional
import json
import re

class QuizChain:
    """Chain for generating quizzes and assessments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            temperature=config.get("temperature", 0.7),
            model=config.get("model", "gemini-2.0-flash-exp"),
            max_output_tokens=config.get("max_tokens", 2000),
            google_api_key=config.get("api_key")
        )
        
        # Load quiz prompt
        prompt_path = "prompts/quiz_prompt.txt"
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
        
        self.prompt = PromptTemplate(
            input_variables=["grade_level", "subject", "topic", "num_questions", "difficulty"],
            template=prompt_template
        )
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=config.get("debug", False)
        )
    
    def generate_quiz(self, topic: str, grade_level: str, subject: str,
                     num_questions: int = 5, difficulty: str = "medium") -> Dict[str, Any]:
        """Generate a complete quiz with questions and answers"""
        
        context = {
            "grade_level": grade_level,
            "subject": subject,
            "topic": topic,
            "num_questions": num_questions,
            "difficulty": difficulty
        }
        
        try:
            # Generate quiz content
            response = self.chain.run(context)
            
            # Parse the response into structured format
            quiz_data = self._parse_quiz_response(response, topic, grade_level, subject)
            
            return {
                "topic": topic,
                "grade_level": grade_level,
                "subject": subject,
                "difficulty": difficulty,
                "questions": quiz_data["questions"],
                "total_questions": len(quiz_data["questions"]),
                "quiz_instructions": quiz_data.get("instructions", ""),
                "time_limit": self._calculate_time_limit(num_questions, difficulty)
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate quiz for {topic}: {str(e)}",
                "topic": topic,
                "questions": []
            }
    
    def generate_multiple_choice_quiz(self, topic: str, grade_level: str, subject: str,
                                    num_questions: int = 5) -> Dict[str, Any]:
        """Generate a multiple choice only quiz"""
        mc_prompt = f"""
        Create a {num_questions}-question multiple choice quiz on "{topic}" for a {grade_level} student studying {subject}.
        
        For each question, provide:
        1. Question text
        2. Four options (A, B, C, D)
        3. Correct answer (A, B, C, or D)
        4. Brief explanation of why the answer is correct
        
        IMPORTANT: Return ONLY valid JSON. Do not include any text before or after the JSON. The response MUST start with '{{' and be valid JSON.
        
        Format the response as JSON with this structure:
        {{
            "questions": [
                {{
                    "question": "Question text",
                    "options": {{
                        "A": "Option A",
                        "B": "Option B", 
                        "C": "Option C",
                        "D": "Option D"
                    }},
                    "correct_answer": "A",
                    "explanation": "Explanation of why this is correct"
                }}
            ]
        }}
        """
        try:
            response = self.llm.predict(mc_prompt)
            try:
                quiz_data = json.loads(response)
            except Exception:
                # Try to extract JSON substring
                import re
                match = re.search(r'\{[\s\S]*\}', response)
                if match:
                    quiz_data = json.loads(match.group(0))
                else:
                    raise ValueError("No valid JSON found in LLM response.")
            questions = quiz_data.get("questions", [])
            if not isinstance(questions, list) or len(questions) == 0:
                return {
                    "error": "Sorry, I couldn't generate a quiz for this topic. Please try again or ask another question.",
                    "topic": topic,
                    "questions": []
                }
            return {
                "topic": topic,
                "grade_level": grade_level,
                "subject": subject,
                "quiz_type": "multiple_choice",
                "questions": questions,
                "total_questions": len(questions)
            }
        except Exception as e:
            return {
                "error": f"Sorry, I couldn't generate a quiz for this topic. Please try again or ask another question. (Error: {str(e)})",
                "topic": topic,
                "questions": []
            }
    
    def generate_true_false_quiz(self, topic: str, grade_level: str, subject: str,
                               num_questions: int = 5) -> Dict[str, Any]:
        """Generate a true/false quiz"""
        
        tf_prompt = f"""
        Create a {num_questions}-question true/false quiz on "{topic}" for a {grade_level} student studying {subject}.
        
        For each question, provide:
        1. Statement (true or false)
        2. Correct answer (True or False)
        3. Brief explanation of why the statement is true or false
        
        Format the response as JSON with this structure:
        {{
            "questions": [
                {{
                    "statement": "Statement text",
                    "correct_answer": "True",
                    "explanation": "Explanation of why this is true or false"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm.predict(tf_prompt)
            quiz_data = json.loads(response)
            
            return {
                "topic": topic,
                "grade_level": grade_level,
                "subject": subject,
                "quiz_type": "true_false",
                "questions": quiz_data["questions"],
                "total_questions": len(quiz_data["questions"])
            }
            
        except Exception as e:
            return {
                "error": f"Failed to generate true/false quiz: {str(e)}",
                "topic": topic,
                "questions": []
            }
    
    def grade_quiz(self, quiz_data: Dict[str, Any], student_answers: Dict[int, str]) -> Dict[str, Any]:
        """Grade a completed quiz"""
        
        # Convert keys to int if they are strings (fix for frontend JSON keys)
        student_answers = {int(k): v for k, v in student_answers.items()}
        
        total_questions = len(quiz_data["questions"])
        correct_answers = 0
        detailed_results = []
        
        for i, question in enumerate(quiz_data["questions"]):
            student_answer = student_answers.get(i, "")
            is_correct = self._check_answer(question, student_answer)
            
            if is_correct:
                correct_answers += 1
            
            detailed_results.append({
                "question_number": i + 1,
                "question": question.get("question", question.get("statement", "")),
                "student_answer": student_answer,
                "correct_answer": self._get_correct_answer(question),
                "is_correct": is_correct,
                "explanation": question.get("explanation", "")
            })
        
        score_percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
        
        return {
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "score_percentage": score_percentage,
            "grade": self._calculate_grade(score_percentage),
            "detailed_results": detailed_results,
            "feedback": self._generate_feedback(score_percentage, detailed_results)
        }
    
    def _parse_quiz_response(self, response: str, topic: str, grade_level: str, subject: str) -> Dict[str, Any]:
        """Parse the LLM response into structured quiz format"""
        
        # Try to extract JSON if present
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: parse manually
        questions = []
        lines = response.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                if current_question:
                    questions.append(current_question)
                current_question = {
                    "question": line[line.find('.')+1:].strip(),
                    "options": {},
                    "correct_answer": "",
                    "explanation": ""
                }
            elif line.startswith(('A.', 'B.', 'C.', 'D.')) and current_question:
                option_letter = line[0]
                option_text = line[line.find('.')+1:].strip()
                current_question["options"][option_letter] = option_text
        
        if current_question:
            questions.append(current_question)
        
        return {
            "questions": questions,
            "instructions": f"Answer all questions about {topic} for {grade_level} {subject}."
        }
    
    def _check_answer(self, question: Dict[str, Any], student_answer: str) -> bool:
        """Check if student answer is correct"""
        correct_answer = self._get_correct_answer(question)
        return student_answer.strip().lower() == correct_answer.strip().lower()
    
    def _get_correct_answer(self, question: Dict[str, Any]) -> str:
        """Extract correct answer from question"""
        return question.get("correct_answer", "")
    
    def _calculate_grade(self, score_percentage: float) -> str:
        """Calculate letter grade based on percentage"""
        if score_percentage >= 90:
            return "A"
        elif score_percentage >= 80:
            return "B"
        elif score_percentage >= 70:
            return "C"
        elif score_percentage >= 60:
            return "D"
        else:
            return "F"
    
    def _calculate_time_limit(self, num_questions: int, difficulty: str) -> int:
        """Calculate suggested time limit for quiz"""
        base_time = 2  # minutes per question
        if difficulty == "easy":
            multiplier = 0.8
        elif difficulty == "hard":
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        return int(num_questions * base_time * multiplier)
    
    def _generate_feedback(self, score_percentage: float, detailed_results: List[Dict[str, Any]]) -> str:
        """Generate feedback based on quiz performance"""
        
        if score_percentage >= 90:
            feedback = "Excellent work! You have a strong understanding of this topic."
        elif score_percentage >= 80:
            feedback = "Good job! You understand most of the concepts well."
        elif score_percentage >= 70:
            feedback = "You're on the right track! Review the incorrect answers to improve."
        elif score_percentage >= 60:
            feedback = "You need more practice with this topic. Focus on the areas you missed."
        else:
            feedback = "This topic needs more study. Consider reviewing the material before retaking the quiz."
        
        # Add specific feedback for incorrect answers
        incorrect_questions = [r for r in detailed_results if not r["is_correct"]]
        if incorrect_questions:
            feedback += f"\n\nFocus on reviewing these concepts: "
            topics_to_review = []
            for result in incorrect_questions[:3]:  # Limit to 3 topics
                topics_to_review.append(f"Question {result['question_number']}")
            feedback += ", ".join(topics_to_review)
        
        return feedback 