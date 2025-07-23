#!/usr/bin/env python3
"""
Test assessment functionality
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append('.')

from chains.socratic_chain import SocraticChain
from config import Config

def test_assessment():
    """Test the assessment functionality"""
    
    # Validate configuration
    try:
        Config.validate_config()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Create SocraticChain instance
    config = Config.get_gemini_config()
    socratic_chain = SocraticChain(config)
    
    # Test parameters
    topic = "photosynthesis"
    grade_level = "5th Grade"
    subject = "Science"
    
    # Test different types of answers
    test_answers = [
        "Photosynthesis is the process where plants make food using sunlight, water, and carbon dioxide.",
        "I think maybe plants do something with sunlight...",
        "I don't know",
        "Plants use sunlight to make food and oxygen.",
        "Photosynthesis is when plants take in carbon dioxide and water, then use sunlight to convert them into glucose and oxygen."
    ]
    
    print(f"Testing assessment for topic: {topic}")
    print(f"Grade Level: {grade_level}")
    print(f"Subject: {subject}")
    print("-" * 50)
    
    for i, answer in enumerate(test_answers, 1):
        print(f"\nTest {i}:")
        print(f"Answer: {answer}")
        
        try:
            assessment = socratic_chain.assess_bloom_level(topic, grade_level, subject, answer)
            print(f"Assessment: {assessment}")
            print(f"Score: {assessment.get('score', 'N/A')}")
            print(f"Level: {assessment.get('level', 'N/A')}")
            print(f"Explanation: {assessment.get('explanation', 'N/A')}")
        except Exception as e:
            print(f"Error: {e}")
    
    return True

if __name__ == "__main__":
    success = test_assessment()
    if success:
        print("\n✅ Assessment test completed!")
    else:
        print("\n❌ Assessment test failed!")
        sys.exit(1) 