import os
import sys
import subprocess
import venv
from pathlib import Path

def run_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    print("Setting up COVID-19 Data Tracker...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create virtual environment
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create("venv", with_pip=True)
    
    # Determine the path to pip
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
        python_path = "venv\\Scripts\\python"
    else:
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    # Install requirements
    print("Installing requirements...")
    success, output = run_command(f"{pip_path} install -r requirements.txt")
    if not success:
        print("Error installing requirements:", output)
        sys.exit(1)
    
    print("\nSetup complete! You can now run the application.")
    print("\nAvailable commands:")
    print(f"1. Run data analysis: {python_path} project.py")
    print(f"2. Run dashboard: {python_path} -m streamlit run dashboard.py")
    print("3. Run tests: pytest")
    
    # Create a runner script for Windows
    if sys.platform == "win32":
        with open("run.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Choose an option:\n")
            f.write("echo 1. Run data analysis\n")
            f.write("echo 2. Run dashboard\n")
            f.write("echo 3. Run tests\n")
            f.write("set /p choice=Enter your choice (1-3): \n")
            f.write("if %choice%==1 venv\\Scripts\\python project.py\n")
            f.write("if %choice%==2 venv\\Scripts\\python -m streamlit run dashboard.py\n")
            f.write("if %choice%==3 venv\\Scripts\\pytest\n")
            f.write("pause\n")
        print("\nA run.bat file has been created for easy execution.")

if __name__ == "__main__":
    main() 