from flask import Flask, render_template, request, jsonify, session, send_from_directory
from typing import Dict, Any, Optional
import os
import uuid
from datetime import datetime
import time

# Import our custom modules
from config import Config
from memory.tutor_memory import TutorMemory
from chains.explanation_chain import ExplanationChain
from chains.quiz_chain import QuizChain
from chains.socratic_chain import SocraticChain
from chains.fiveyo_chain import FiveYoChain
from chains.hook_chain import HookChain
from routers.action_router import ActionRouter

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'super-teacher-secret-key')

# Initialize components
config = Config.get_gemini_config()
memory = TutorMemory(k=Config.MEMORY_K)

# Initialize chains
explanation_chain = ExplanationChain(config)
quiz_chain = QuizChain(config)
socratic_chain = SocraticChain(config)
fiveyo_chain = FiveYoChain(config)
hook_chain = HookChain(config)

# Initialize router
action_router = ActionRouter(config)

# In-memory Socratic session state: {student_id: {questions, answers, scores, times, current_index}}
socratic_sessions = {}

class SuperTeacher:
    """Main Super Teacher application class"""
    
    def __init__(self):
        self.memory = memory
        self.chains = {
            "explain": explanation_chain,
            "quiz": quiz_chain,
            "socratic": socratic_chain,
            "fiveyo": fiveyo_chain
        }
        self.router = action_router
        self.hook_chain = hook_chain
    
    def process_request(self, user_input: str, grade_level: str, subject: str, 
                       topic: str = "", student_id: str = None, action: str = None) -> Dict[str, Any]:
        """Process a user request and return appropriate response"""
        try:
            if action == "yes":
                response = self.chains["explain"].explain_topic(topic, grade_level, subject, student_id=student_id)
                return {"response": response, "type": "explanation"}
            elif action == "understand":
                appreciation_prompt = f"""
                The student just confirmed they understood the explanation about {topic} in {subject} (grade {grade_level}).
                Respond with a short, enthusiastic appreciation message (with emojis) and ask if they want to ask another question.
                """
                appreciation = self.chains["explain"].llm.predict(appreciation_prompt)
                return {"response": appreciation, "type": "appreciation"}
            elif action == "quiz":
                quiz_data = self.chains["quiz"].generate_multiple_choice_quiz(topic, grade_level, subject, num_questions=5)
                return {"response": quiz_data, "type": "quiz"}
            elif action == "fiveyo":
                response = self.chains["fiveyo"].explain_like_five(topic, subject)
                return {"response": response, "type": "fiveyo"}
            elif action == "fiveyo_understand":
                appreciation_prompt = f"""
                The student just confirmed they understood the 5-year-old explanation about {topic} in {subject} (grade {grade_level}).
                Respond with a short, enthusiastic appreciation message (with emojis) and ask if they want to ask another question.
                """
                appreciation = self.chains["fiveyo"].llm.predict(appreciation_prompt)
                return {"response": appreciation, "type": "appreciation"}
            elif action == "fiveyo_another_example":
                # Use the same chain for another example (could randomize prompt for more variety)
                response = self.chains["fiveyo"].explain_like_five(topic, subject)
                return {"response": response, "type": "fiveyo_example"}
            elif action == "fiveyo_example_yes":
                appreciation_prompt = f"""
                The student just confirmed they understood the additional 5-year-old example for {topic} in {subject} (grade {grade_level}).
                Respond with a short, enthusiastic appreciation message (with emojis) and ask if they want to ask another question.
                """
                appreciation = self.chains["fiveyo"].llm.predict(appreciation_prompt)
                return {"response": appreciation, "type": "appreciation"}
            elif action == "fiveyo_example_no":
                return {"response": "Don't worry, your teacher will reach out to help you! Do you want to ask another question?", "type": "fiveyo_no"}
            elif action == "different_approach":
                response = self.chains["socratic"].start_socratic_dialogue(topic, grade_level, subject)
                return {"response": response, "type": "different_approach"}
            elif action == "no":
                return {"response": "No worries! You can ask about another topic whenever you're ready.", "type": "end"}
            elif action == "hook":
                # Check if this is a specific question from "Ask Another Question" flow
                if user_input and "The student wants to ask another question" in user_input:
                    # Extract the specific question and context from the prompt
                    import re
                    question_match = re.search(r'Student\'s specific question/prompt: "(.*?)"', user_input)
                    if question_match:
                        specific_question = question_match.group(1)
                        # Extract additional context if available
                        standard_match = re.search(r'Standard: (\w+)', user_input)
                        chapter_match = re.search(r'Chapter: ([^\n]+)', user_input)
                        
                        # Create enhanced context for the LLM
                        enhanced_context = f"""
                        Context for the student's question:
                        - Standard: {standard_match.group(1) if standard_match else grade_level}
                        - Subject: {subject}
                        - Chapter: {chapter_match.group(1) if chapter_match else 'Not specified'}
                        - Topic: {topic}
                        - Student's specific question: {specific_question}
                        
                        Please provide a comprehensive response that addresses their question with full context.
                        """
                        
                        # Process the specific question directly with enhanced context
                        response = self.chains["explain"].explain_topic(topic, grade_level, subject, student_id=student_id, custom_prompt=enhanced_context)
                        return {"response": response, "type": "explanation"}
                
                # Default hook generation
                hook = self.hook_chain.generate_hook(topic, grade_level, subject)
                return {"response": hook, "type": "hook"}
            
            # Default: generate hook
            hook = self.hook_chain.generate_hook(topic, grade_level, subject)
            return {"response": hook, "type": "hook"}
        except Exception as e:
            return {
                "error": f"Error processing request: {str(e)}"
            }
    
    def _process_with_chain(self, chain, action: str, user_input: str, 
                           grade_level: str, subject: str, topic: str, 
                           student_id: str = None) -> Dict[str, Any]:
        """Process request with the appropriate chain"""
        
        # Special handling for fiveyo chain and activities
        if action == "fiveyo":
            # Only use explain_like_five for fiveyo explanations
            response = chain.explain_like_five(topic, subject)
            return {"response": response, "type": "fiveyo"}
        elif action == "activities":
            # If the chain is FiveYoChain, use create_fun_activity
            if hasattr(chain, "create_fun_activity"):
                response = chain.create_fun_activity(topic, subject)
            else:
                response = chain.explain_with_examples(topic, grade_level, subject)
            return {"response": response, "type": "activities"}
        elif action == "stories":
            if hasattr(chain, "create_story"):
                response = chain.create_story(topic, subject)
            else:
                response = chain.explain_topic(topic, grade_level, subject)
            return {"response": response, "type": "stories"}
        elif action == "songs":
            if hasattr(chain, "create_song_or_rhyme"):
                response = chain.create_song_or_rhyme(topic, subject)
            else:
                response = chain.explain_topic(topic, grade_level, subject)
            return {"response": response, "type": "songs"}
        elif action == "experiments":
            if hasattr(chain, "create_simple_experiment"):
                response = chain.create_simple_experiment(topic, subject)
            else:
                response = chain.explain_topic(topic, grade_level, subject)
            return {"response": response, "type": "experiments"}
        # All other actions remain unchanged
        if action == "explain":
            response = chain.explain_topic(topic, grade_level, subject, student_id=student_id)
            return {"response": response, "type": "explanation"}
        elif action == "quiz":
            # Extract quiz parameters from user input
            num_questions = 5
            difficulty = "medium"
            if "easy" in user_input.lower():
                difficulty = "easy"
            elif "hard" in user_input.lower():
                difficulty = "hard"
            import re
            num_match = re.search(r'(\d+)\s*questions?', user_input.lower())
            if num_match:
                num_questions = int(num_match.group(1))
            quiz_data = chain.generate_quiz(topic, grade_level, subject, num_questions, difficulty)
            return {"response": quiz_data, "type": "quiz"}
        elif action == "socratic":
            # If user_input is provided, use it as current_understanding
            response = chain.start_socratic_dialogue(
                topic, grade_level, subject, current_understanding=user_input or ""
            )
            return {"response": response, "type": "socratic"}
        elif action == "clarify":
            response = chain.clarify_concept(topic, grade_level, subject, user_input)
            return {"response": response, "type": "clarification"}
        elif action == "examples":
            response = chain.explain_with_examples(topic, grade_level, subject)
            return {"response": response, "type": "examples"}
        elif action == "step_by_step":
            response = chain.explain_step_by_step(topic, grade_level, subject)
            return {"response": response, "type": "step_by_step"}
        elif action == "objectives":
            response = chain.get_learning_objectives(topic, grade_level, subject)
            return {"response": response, "type": "objectives"}
        elif action == "assessment":
            response = chain.start_socratic_dialogue(topic, grade_level, subject)
            return {"response": response, "type": "assessment"}
        else:
            response = chain.explain_topic(topic, grade_level, subject, student_id=student_id)
            return {"response": response, "type": "explanation"}

# Initialize Super Teacher
super_teacher = SuperTeacher()

@app.route('/')
def index():
    """Main page with the form"""
    return render_template('index.html', 
                         standards=Config.SUPPORTED_STANDARDS,
                         subjects=Config.SUPPORTED_SUBJECTS)

@app.route('/test')
def test_frontend():
    """Test page for cascading dropdowns"""
    return send_from_directory(os.getcwd(), 'test_frontend.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    """API endpoint to process user requests"""
    try:
        data = request.get_json()
        user_input = data.get('user_input', '')
        grade_level = data.get('grade_level', '')
        subject = data.get('subject', '')
        topic = data.get('topic', '')
        student_id = data.get('student_id', str(uuid.uuid4()))
        action = data.get('action', None)

        # Validate required fields
        if not grade_level or not subject:
            return jsonify({
                'error': 'Missing required fields: grade_level, subject'
            }), 400

        # Process the request
        response = super_teacher.process_request(
            user_input, grade_level, subject, topic, student_id, action
        )
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """Get suggested actions based on context"""
    try:
        data = request.get_json()
        grade_level = data.get('grade_level', '')
        subject = data.get('subject', '')
        topic = data.get('topic', '')
        
        suggestions = action_router.suggest_actions(grade_level, subject, topic)
        
        return jsonify({
            'suggestions': suggestions,
            'actions': action_router.get_all_actions()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/subjects', methods=['POST'])
def get_subjects():
    """Get subjects available for a specific standard"""
    try:
        data = request.get_json()
        standard = data.get('standard', '')
        
        if not standard:
            return jsonify({
                'error': 'Standard is required'
            }), 400
        
        subjects = Config.get_subjects_for_standard(standard)
        
        return jsonify({
            'subjects': subjects
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/chapters', methods=['POST'])
def get_chapters():
    """Get chapters available for a specific standard and subject"""
    try:
        data = request.get_json()
        standard = data.get('standard', '')
        subject = data.get('subject', '')
        
        if not standard or not subject:
            return jsonify({
                'error': 'Standard and subject are required'
            }), 400
        
        chapters = Config.get_chapters_for_subject(standard, subject)
        
        return jsonify({
            'chapters': chapters
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/topics', methods=['POST'])
def get_topics():
    """Get topics available for a specific chapter"""
    try:
        data = request.get_json()
        standard = data.get('standard', '')
        subject = data.get('subject', '')
        chapter = data.get('chapter', '')
        
        if not standard or not subject or not chapter:
            return jsonify({
                'error': 'Standard, subject, and chapter are required'
            }), 400
        
        topics = Config.get_topics_for_chapter(standard, subject, chapter)
        
        return jsonify({
            'topics': topics
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/quiz/grade', methods=['POST'])
def grade_quiz():
    """Grade a completed quiz"""
    try:
        data = request.get_json()
        quiz_data = data.get('quiz_data', {})
        student_answers = data.get('student_answers', {})
        
        if not quiz_data or not student_answers:
            return jsonify({
                'error': 'Missing quiz_data or student_answers'
            }), 400
        
        # Grade the quiz
        results = quiz_chain.grade_quiz(quiz_data, student_answers)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/api/memory/save', methods=['POST'])
def save_memory():
    """Save memory data to file"""
    try:
        memory.save_to_file()
        return jsonify({'message': 'Memory saved successfully'})
    except Exception as e:
        return jsonify({
            'error': f'Failed to save memory: {str(e)}'
        }), 500

@app.route('/api/memory/load', methods=['POST'])
def load_memory():
    """Load memory data from file"""
    try:
        memory.load_from_file()
        return jsonify({'message': 'Memory loaded successfully'})
    except Exception as e:
        return jsonify({
            'error': f'Failed to load memory: {str(e)}'
        }), 500

@app.route('/api/student/profile', methods=['POST'])
def create_student_profile():
    """Create or update a student profile"""
    try:
        data = request.get_json()
        student_id = data.get('student_id', str(uuid.uuid4()))
        grade_level = data.get('grade_level', '')
        subject = data.get('subject', '')
        topics = data.get('topics', [])
        preferences = data.get('preferences', {})
        
        memory.create_student_profile(student_id, grade_level, subject, topics, preferences)
        
        return jsonify({
            'message': 'Student profile created successfully',
            'student_id': student_id
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to create student profile: {str(e)}'
        }), 500

@app.route('/api/student/profile/<student_id>', methods=['GET'])
def get_student_profile(student_id):
    """Get a student profile"""
    try:
        profile = memory.get_student_profile(student_id)
        
        if not profile:
            return jsonify({
                'error': 'Student profile not found'
            }), 404
        
        return jsonify(profile)
        
    except Exception as e:
        return jsonify({
            'error': f'Failed to get student profile: {str(e)}'
        }), 500

@app.route('/api/socratic/start', methods=['POST'])
def socratic_start():
    data = request.get_json()
    student_id = data.get('student_id')
    grade_level = data.get('grade_level')
    subject = data.get('subject')
    topic = data.get('topic')
    if not (student_id and grade_level and subject and topic):
        return jsonify({'error': 'Missing required fields'}), 400
    # Generate 6 Bloom's questions
    socratic_chain = super_teacher.chains['socratic']
    bloom_questions = socratic_chain.generate_bloom_questions(topic, grade_level, subject)
    # Flatten to list in Bloom's order
    bloom_order = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    questions = []
    for level in bloom_order:
        qs = bloom_questions.get(level, [])
        if qs:
            questions.append({'level': level, 'question': qs[0]})
        else:
            questions.append({'level': level, 'question': f'No question for {level}.'})
    # Store session
    socratic_sessions[student_id] = {
        'questions': questions,
        'answers': [None]*6,
        'scores': [None]*6,
        'times': [None]*6,
        'current_index': 0,
        'start_times': [None]*6,
        'topic': topic,
        'grade_level': grade_level,
        'subject': subject
    }
    # Mark start time for first question
    socratic_sessions[student_id]['start_times'][0] = time.time()
    return jsonify({'question': questions[0]['question'], 'level': questions[0]['level'], 'index': 0})

@app.route('/api/socratic/answer', methods=['POST'])
def socratic_answer():
    data = request.get_json()
    student_id = data.get('student_id')
    answer = data.get('answer')
    index = data.get('index')
    # For mastery retry flow
    weak_levels = data.get('weak_levels')
    retry_index = data.get('retry_index', 0)
    all_answers = data.get('all_answers')
    socratic_chain = super_teacher.chains['socratic']
    bloom_order = ['remember', 'understand', 'apply', 'analyze', 'evaluate', 'create']
    if all_answers is not None:
        # Mastery retry flow: update answers for weak levels
        session = socratic_sessions.get(student_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 400
        # Update answers for weak levels
        for level in all_answers:
            if level in bloom_order:
                idx = bloom_order.index(level)
                session['answers'][idx] = all_answers[level]
        # Run mastery loop for the next weak level
        mastery_result = socratic_chain.mastery_loop(
            session['topic'], session['grade_level'], session['subject'],
            {bloom_order[i]: session['answers'][i] for i in range(6)},
            weak_levels=weak_levels, retry_index=retry_index
        )
        # Add timing and scores to result
        mastery_result['avg_time'] = sum(session['times'])/6
        mastery_result['scores'] = session['scores']
        mastery_result['times'] = session['times']
        return jsonify(mastery_result)
    # Normal flow for first 6 questions
    if student_id not in socratic_sessions:
        return jsonify({'error': 'Session not found'}), 400
    session = socratic_sessions[student_id]
    if index != session['current_index']:
        return jsonify({'error': 'Out of order answer'}), 400
    # Calculate time taken
    end_time = time.time()
    start_time = session['start_times'][index] or end_time
    time_taken = max(5, int(end_time - start_time))
    session['times'][index] = time_taken
    session['answers'][index] = answer
    # Score the answer using SocraticChain.assess_bloom_level
    q = session['questions'][index]
    assessment = socratic_chain.assess_bloom_level(session['topic'], session['grade_level'], session['subject'], answer)
    # Use the score from the assessment
    score = assessment.get('score', 60)  # Default to 60 if no score provided
    session['scores'][index] = score
    # Next question or results
    if index < 5:
        session['current_index'] += 1
        session['start_times'][index+1] = time.time()
        next_q = session['questions'][index+1]
        # Use adaptive feedback and next question
        feedback_and_next = socratic_chain.get_feedback_and_next_question(
            session['topic'], session['grade_level'], session['subject'], answer, q['level']
        )
        return jsonify({
            'question': feedback_and_next['next_question'] or next_q['question'],
            'level': next_q['level'],
            'index': index+1,
            'score': score,
            'time_taken': time_taken,
            'feedback': feedback_and_next['feedback']
        })
    # All done: calculate results and run mastery loop for first time
    avg_scores = session['scores']
    avg_score = sum(avg_scores)/6
    avg_time = sum(session['times'])/6
    breakdown = {bloom_order[i]: avg_scores[i] for i in range(6)}
    student_answers = {bloom_order[i]: session['answers'][i] for i in range(6)}
    mastery_result = socratic_chain.mastery_loop(session['topic'], session['grade_level'], session['subject'], student_answers)
    mastery_result['avg_time'] = avg_time
    mastery_result['scores'] = avg_scores
    mastery_result['times'] = session['times']
    return jsonify(mastery_result)

if __name__ == '__main__':
    # Validate configuration
    try:
        Config.validate_config()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set the GOOGLE_API_KEY environment variable")
        exit(1)
    
    # Load existing memory if available
    try:
        memory.load_from_file()
        print("✅ Memory loaded successfully")
    except Exception as e:
        print(f"⚠️  Could not load memory: {e}")
    
    print("🚀 Starting Super Teacher application...")
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=8000) 