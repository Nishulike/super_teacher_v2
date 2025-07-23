from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any

class HookChain:
    """Chain for generating curiosity/surprise hook questions about a topic"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatGoogleGenerativeAI(
            temperature=config.get("temperature", 0.8),
            model=config.get("model", "gemini-2.0-flash-exp"),
            max_output_tokens=config.get("max_tokens", 512),
            google_api_key=config.get("api_key")
        )
        self.prompt = PromptTemplate(
            input_variables=["topic", "grade_level", "subject"],
            template=(
                """
                You are an expert educator skilled in the Socratic method and cognitive psychology. Your goal is to create provocative, counter-intuitive hooks that make learners question assumptions about a topic, sparking deep curiosity and self-driven exploration. Prioritize simplicity and clarity to minimize cognitive load.\n\n
                Topic: {topic}
                Grade Level: {grade_level}
                Subject: {subject}\n\n
                Instructions:
                - Create a 1–2 sentence, counter-intuitive scenario or paradox that contradicts common intuition. Use vivid, relatable analogies.
                  Example: 'What if I told you [shocking fact]? Imagine [simple analogy]... but here’s the twist: [contradiction].'
                - Frame a question forcing learners to confront a gap in their understanding. Use phrases like: 'Why doesn’t [obvious expectation] happen?' or 'How can [seemingly impossible outcome] be true?'
                - Use everyday language (no jargon).
                - Anchor to ONE core concept.
                - Max 3 sentences total.\n\n
                - End with a yes/no question that invites the learner to dive deeper or explore the explanation (e.g., "Do you want to find out why?").

                Output: Write only the hook (max 3 sentences).
                """
            )
        )

    def generate_hook(self, topic: str, grade_level: str, subject: str) -> str:
        context = {
            "topic": topic,
            "grade_level": grade_level,
            "subject": subject
        }
        try:
            prompt_text = self.prompt.format(**context)
            response = self.llm.predict(prompt_text)
            return response.strip()
        except Exception as e:
            return f"Let's get curious about {topic}! Ready to learn? Let's dive in!" 