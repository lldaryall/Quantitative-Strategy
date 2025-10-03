#!/usr/bin/env python3
"""Code quality check script for qbacktester."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main() -> int:
    """Main quality check function."""
    print("🔍 Running code quality checks for qbacktester...")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Quality checks to run
    checks = [
        ("black --check src/ tests/", "Code formatting check (Black)"),
        ("isort --check-only src/ tests/", "Import sorting check (isort)"),
        ("flake8 src/ tests/", "Linting check (flake8)"),
        ("mypy src/qbacktester", "Type checking (mypy)"),
        ("bandit -r src/ -ll", "Security check (bandit)"),
        ("pytest --cov=qbacktester --cov-report=term-missing", "Test coverage"),
    ]
    
    # Run all checks
    success = True
    for command, description in checks:
        if not run_command(command, description):
            success = False
    
    if success:
        print("\n🎉 All quality checks passed!")
        print("Your code is ready for commit! 🚀")
    else:
        print("\n❌ Some quality checks failed.")
        print("Please fix the issues above before committing.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

