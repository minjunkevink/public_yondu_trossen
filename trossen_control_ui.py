#!/usr/bin/env python3
import os
import sys
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QPushButton, QTextEdit, QFormLayout,
    QDoubleSpinBox, QGroupBox
)
from PySide6.QtCore import Qt

class TrossenControlUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trossen AI Solo Control")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Create form for parameters
        form_group = QGroupBox("Control Parameters")
        form_layout = QFormLayout()
        
        # Get HF_USER from environment
        self.hf_user = os.environ.get('HF_USER', '')
        
        # User section
        user_group = QGroupBox("User Settings")
        user_layout = QFormLayout()
        
        # HF_USER input - use environment variable if available
        self.hf_user_input = QLineEdit(self.hf_user)
        self.hf_user_input.setPlaceholderText("Your HuggingFace username")
        self.hf_user_input.textChanged.connect(self.update_hf_user)
        user_layout.addRow(QLabel("HuggingFace Username:"), self.hf_user_input)
        
        if not self.hf_user:
            # Add warning if HF_USER not set
            warning_label = QLabel("HF_USER not found in environment! Please enter it above.")
            warning_label.setStyleSheet("color: red")
            user_layout.addRow(warning_label)
            
        user_group.setLayout(user_layout)
        main_layout.addWidget(user_group)
        
        # Repo ID (without HF_USER)
        self.repo_name_input = QLineEdit("trossen_ai_solo_test")
        form_layout.addRow(QLabel(f"Repo Name (will be {self.hf_user}/...)"), self.repo_name_input)
        
        # Task description
        self.task_input = QLineEdit("Test recording episode using Trossen AI Solo.")
        form_layout.addRow(QLabel("Task Description:"), self.task_input)
        
        # Tags as comma-separated values
        self.tags_input = QLineEdit("tutorial")
        form_layout.addRow(QLabel("Tags (comma-separated):"), self.tags_input)
        
        # Episode time
        self.episode_time_input = QDoubleSpinBox()
        self.episode_time_input.setRange(1, 60)
        self.episode_time_input.setValue(10)
        self.episode_time_input.setSuffix(" seconds")
        form_layout.addRow(QLabel("Episode Time:"), self.episode_time_input)
        
        # Reset time
        self.reset_time_input = QDoubleSpinBox()
        self.reset_time_input.setRange(1, 30)
        self.reset_time_input.setValue(5)
        self.reset_time_input.setSuffix(" seconds")
        form_layout.addRow(QLabel("Reset Time:"), self.reset_time_input)
        
        # Number of episodes
        self.num_episodes_input = QSpinBox()
        self.num_episodes_input.setRange(1, 100)
        self.num_episodes_input.setValue(5)
        form_layout.addRow(QLabel("Number of Episodes:"), self.num_episodes_input)
        
        # Finish form group
        form_group.setLayout(form_layout)
        main_layout.addWidget(form_group)
        
        # Create group for command preview and execution
        command_group = QGroupBox("Command")
        command_layout = QVBoxLayout()
        
        # Command preview
        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        command_layout.addWidget(self.command_preview)
        
        # Run button
        self.run_button = QPushButton("Run Command")
        self.run_button.clicked.connect(self.run_command)
        command_layout.addWidget(self.run_button)
        
        # Output log
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        output_layout.addWidget(self.output_log)
        output_group.setLayout(output_layout)
        
        # Finish command group
        command_group.setLayout(command_layout)
        main_layout.addWidget(command_group)
        main_layout.addWidget(output_group)
        
        # Connect signals
        self.repo_name_input.textChanged.connect(self.update_command)
        self.task_input.textChanged.connect(self.update_command)
        self.tags_input.textChanged.connect(self.update_command)
        self.episode_time_input.valueChanged.connect(self.update_command)
        self.reset_time_input.valueChanged.connect(self.update_command)
        self.num_episodes_input.valueChanged.connect(self.update_command)
        
        # Initial command preview
        self.update_command()
    
    def update_hf_user(self):
        """Update the HF_USER variable when the input changes"""
        self.hf_user = self.hf_user_input.text().strip()
        self.update_command()
        
    def update_command(self):
        """Update the command preview based on current inputs"""
        # Ensure we have an HF_USER, use a placeholder if empty
        hf_user = self.hf_user if self.hf_user else "YOUR_USERNAME"
        
        # Format tags as a proper JSON array string
        tags = [tag.strip() for tag in self.tags_input.text().split(',')]
        tags_str = str(tags).replace("'", '"') if tags and tags[0] else '["tutorial"]'
        
        # Construct the command
        command = f"python lerobot/scripts/control_robot.py " \
                  f"--robot.type=trossen_ai_solo " \
                  f"--robot.max_relative_target=null " \
                  f"--control.type=record " \
                  f"--control.fps=30 " \
                  f"--control.single_task=\"{self.task_input.text()}\" " \
                  f"--control.repo_id={hf_user}/{self.repo_name_input.text()} " \
                  f"--control.tags='{tags_str}' " \
                  f"--control.warmup_time_s=5 " \
                  f"--control.episode_time_s={self.episode_time_input.value()} " \
                  f"--control.reset_time_s={self.reset_time_input.value()} " \
                  f"--control.num_episodes={self.num_episodes_input.value()} " \
                  f"--control.push_to_hub=true " \
                  f"--control.num_image_writer_threads_per_camera=8 " \
                  f"--control.display_cameras=false"
        
        self.command_preview.setText(command)
    
    def run_command(self):
        """Execute the generated command"""
        # Check if we have a valid HF_USER
        if not self.hf_user:
            self.output_log.setText("Error: HuggingFace username is required before running the command.")
            return
            
        command = self.command_preview.toPlainText()
        self.output_log.append(f"Running command: {command}\n")
        
        # Clear previous output
        self.output_log.clear()
        self.output_log.append("Starting execution...\n")
        
        try:
            # Run the command
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Show initial indication
            self.output_log.append("Command is running...\n")
            self.output_log.append("Check terminal for full output\n")
            
            # Disable run button during execution
            self.run_button.setEnabled(False)
            self.run_button.setText("Running...")
            
            # Process will continue in background
            # In a real application, you would use QProcess and handle
            # stdout/stderr in real-time, but for simplicity we'll just
            # indicate the command is running
            
            # Re-enable run button (the process will continue in background)
            self.run_button.setEnabled(True)
            self.run_button.setText("Run Command")
            
        except Exception as e:
            self.output_log.append(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Check if HF_USER is set
    hf_user = os.environ.get('HF_USER')
    if not hf_user:
        print("Notice: HF_USER environment variable is not set.")
        print("You will need to enter your HuggingFace username in the UI.")
        print("To avoid this in the future, set the environment variable:")
        print("  export HF_USER=your_huggingface_username")
    else:
        print(f"Using HuggingFace username from environment: {hf_user}")
    
    print("\nStarting Trossen AI Solo Control UI...")
    print("This UI allows you to easily configure and run the control_robot.py script")
    print("with custom parameters for data collection.")
    
    window = TrossenControlUI()
    window.show()
    sys.exit(app.exec()) 