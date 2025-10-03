#!/usr/bin/env python3
"""Development setup script for qbacktester."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main() -> int:
    """Main setup function."""
    print("🚀 Setting up qbacktester development environment...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Commands to run
    commands = [
        ("python -m pip install --upgrade pip", "Upgrading pip"),
        ("pip install -e .[dev]", "Installing package in development mode"),
        ("pre-commit install", "Installing pre-commit hooks"),
        ("python -c 'import qbacktester; print(f\"qbacktester {qbacktester.__version__} installed successfully\")'", "Verifying installation"),
    ]
    
    # Run all commands
    success = True
    for command, description in commands:
        if not run_command(command, description):
            success = False
    
    if success:
        print("\n🎉 Development environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run 'make test' to run the test suite")
        print("2. Run 'make lint' to check code quality")
        print("3. Run 'make format' to format code")
        print("4. Run 'qbt --help' to see available commands")
        print("\nHappy coding! 🐍")
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

