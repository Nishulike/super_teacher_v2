#!/usr/bin/env python3
"""
Setup script for Super Teacher
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print("✅ Python version is compatible")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = ".env"
    if not os.path.exists(env_file):
        print("📝 Creating .env file...")
        with open(env_file, "w") as f:
            f.write("# Super Teacher Environment Variables\n")
            f.write("# Add your Google API key below\n\n")
            f.write("GOOGLE_API_KEY=your-google-api-key-here\n")
            f.write("DEBUG=False\n")
            f.write("SECRET_KEY=super-teacher-secret-key\n")
        print("✅ .env file created. Please add your Google API key.")
    else:
        print("✅ .env file already exists")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ["templates", "prompts", "chains", "memory", "routers"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up Super Teacher...")
    print("=" * 50)
    
    check_python_version()
    create_directories()
    create_env_file()
    install_dependencies()
    
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    print("\n🎓 Happy learning with Super Teacher!")

if __name__ == "__main__":
    main() 