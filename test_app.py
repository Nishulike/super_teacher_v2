#!/usr/bin/env python3
"""
Test script for Super Teacher application
"""

import os
import sys
import json
from typing import Dict, Any

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from config import Config
        print("✅ Config imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Config: {e}")
        return False
    
    try:
        from memory.tutor_memory import TutorMemory
        print("✅ TutorMemory imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TutorMemory: {e}")
        return False
    
    try:
        from chains.explanation_chain import ExplanationChain
        print("✅ ExplanationChain imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ExplanationChain: {e}")
        return False
    
    try:
        from chains.quiz_chain import QuizChain
        print("✅ QuizChain imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import QuizChain: {e}")
        return False
    
    try:
        from chains.socratic_chain import SocraticChain
        print("✅ SocraticChain imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SocraticChain: {e}")
        return False
    
    try:
        from chains.fiveyo_chain import FiveYoChain
        print("✅ FiveYoChain imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import FiveYoChain: {e}")
        return False
    
    try:
        from routers.action_router import ActionRouter
        print("✅ ActionRouter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ActionRouter: {e}")
        return False
    
    return True

def test_config():
    """Test configuration"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import Config
        
        # Test configuration validation
        if not Config.GOOGLE_API_KEY or Config.GOOGLE_API_KEY == "your-google-api-key-here":
            print("⚠️  Google API key not set (this is expected for testing)")
        else:
            print("✅ Google API key is set")
        
        print(f"✅ Supported standards: {len(Config.SUPPORTED_STANDARDS)}")
        print(f"✅ Supported subjects: {len(Config.SUPPORTED_SUBJECTS)}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_memory():
    """Test memory functionality"""
    print("\n🧪 Testing memory...")
    
    try:
        from memory.tutor_memory import TutorMemory
        
        memory = TutorMemory(k=3)
        
        # Test basic memory operations
        memory.add_message("Hello", is_human=True)
        memory.add_message("Hi there!", is_human=False)
        
        messages = memory.get_messages()
        if len(messages) == 2:
            print("✅ Memory operations work correctly")
            return True
        else:
            print("❌ Memory operations failed")
            return False
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_router():
    """Test action router"""
    print("\n🧪 Testing action router...")
    
    try:
        from routers.action_router import ActionRouter
        from config import Config
        
        config = Config.get_gemini_config()
        router = ActionRouter(config)
        
        # Test routing
        result = router.route_action(
            "Explain photosynthesis",
            "5th Grade",
            "Science",
            "photosynthesis"
        )
        
        if result and "action" in result:
            print(f"✅ Router works correctly (action: {result['action']})")
            return True
        else:
            print("❌ Router test failed")
            return False
            
    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False

def test_prompts():
    """Test prompt loading"""
    print("\n🧪 Testing prompts...")
    
    prompt_files = [
        "prompts/explain_prompt.txt",
        "prompts/quiz_prompt.txt", 
        "prompts/bloom_prompt.txt"
    ]
    
    for prompt_file in prompt_files:
        try:
            with open(prompt_file, 'r') as f:
                content = f.read()
                if len(content) > 0:
                    print(f"✅ {prompt_file} loaded successfully")
                else:
                    print(f"❌ {prompt_file} is empty")
                    return False
        except FileNotFoundError:
            print(f"❌ {prompt_file} not found")
            return False
        except Exception as e:
            print(f"❌ Error reading {prompt_file}: {e}")
            return False
    
    return True

def test_template():
    """Test HTML template"""
    print("\n🧪 Testing template...")
    
    try:
        with open("templates/index.html", 'r') as f:
            content = f.read()
            if len(content) > 0 and "Super Teacher" in content:
                print("✅ HTML template loaded successfully")
                return True
            else:
                print("❌ HTML template is invalid")
                return False
    except FileNotFoundError:
        print("❌ HTML template not found")
        return False
    except Exception as e:
        print(f"❌ Error reading template: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Super Teacher tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_memory,
        test_router,
        test_prompts,
        test_template
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Super Teacher is ready to use.")
        print("\n📋 To start the application:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run: python app.py")
        print("3. Open http://localhost:5000")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 