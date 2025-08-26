#!/usr/bin/env python3
"""
Launch script for Iowa Wells RAG Chat Interface
Automatically installs dependencies and starts the Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages."""
    print("🔧 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists."""
    env_file = Path(__file__).parent.parent / "8_vectordb" / ".env"
    
    if not env_file.exists():
        print("⚠️  Warning: .env file not found in 8_vectordb folder!")
        print("   Please ensure you have OPENAI_API_KEY and PINECONE_API_KEY configured.")
        return False
    
    print("✅ Environment file found!")
    return True

def launch_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Starting Iowa Wells RAG Chat Interface...")
    
    app_file = Path(__file__).parent / "iowa_wells_chat.py"
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped by user.")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        launch_streamlit()

def main():
    """Main function."""
    print("=" * 60)
    print("🏔️  IOWA WELLS RAG CHAT INTERFACE LAUNCHER")
    print("=" * 60)
    
    # Check environment
    print("\n📋 Checking environment...")
    check_env_file()
    
    # Install requirements
    print("\n📦 Setting up dependencies...")
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually.")
        return
    
    # Launch app
    print("\n🌐 Launching web interface...")
    print("   ➡️  The chat interface will open in your browser")
    print("   ➡️  URL: http://localhost:8501")
    print("   ➡️  Press Ctrl+C to stop")
    print("-" * 60)
    
    launch_streamlit()

if __name__ == "__main__":
    main()
