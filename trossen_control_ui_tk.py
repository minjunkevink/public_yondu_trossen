#!/usr/bin/env python3
import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, scrolledtext

class TrossenControlUI:
    def __init__(self, root):
        self.root = root
        root.title("Trossen AI Solo Control")
        root.geometry("700x750")  # Wider and taller to accommodate all controls
        
        # Get HF_USER from environment
        self.hf_user = os.environ.get('HF_USER', '')
        
        # Create main container with padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # User Settings Frame
        user_frame = ttk.LabelFrame(main_frame, text="User Settings")
        user_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(user_frame, text="HuggingFace Username:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.hf_user_var = tk.StringVar(value=self.hf_user)
        self.hf_user_entry = ttk.Entry(user_frame, textvariable=self.hf_user_var, width=40)
        self.hf_user_entry.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        self.hf_user_entry.bind("<KeyRelease>", lambda e: self.update_hf_user())
        
        if not self.hf_user:
            ttk.Label(user_frame, text="HF_USER not found in environment! Please enter it above.", 
                    foreground="red").grid(column=0, row=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(main_frame, text="Control Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Repo Name
        ttk.Label(params_frame, text="Repo Name:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
        self.repo_name_var = tk.StringVar(value="trossen_ai_solo_test")
        self.repo_name_entry = ttk.Entry(params_frame, textvariable=self.repo_name_var, width=40)
        self.repo_name_entry.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)
        self.repo_name_entry.bind("<KeyRelease>", lambda e: self.update_command())
        
        # Task Description
        ttk.Label(params_frame, text="Task Description:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
        self.task_var = tk.StringVar(value="Test recording episode using Trossen AI Solo.")
        self.task_entry = ttk.Entry(params_frame, textvariable=self.task_var, width=40)
        self.task_entry.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)
        self.task_entry.bind("<KeyRelease>", lambda e: self.update_command())
        
        # Tags
        ttk.Label(params_frame, text="Tags (comma-separated):").grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.tags_var = tk.StringVar(value="tutorial")
        self.tags_entry = ttk.Entry(params_frame, textvariable=self.tags_var, width=40)
        self.tags_entry.grid(column=1, row=2, sticky=tk.W, padx=5, pady=5)
        self.tags_entry.bind("<KeyRelease>", lambda e: self.update_command())
        
        # Episode Time
        ttk.Label(params_frame, text="Episode Time (seconds):").grid(column=0, row=3, sticky=tk.W, padx=5, pady=5)
        self.episode_time_var = tk.DoubleVar(value=10.0)
        self.episode_time_spinbox = ttk.Spinbox(params_frame, from_=1.0, to=60.0, 
                                   textvariable=self.episode_time_var, width=10, 
                                   increment=0.5, command=self.update_command)
        self.episode_time_spinbox.grid(column=1, row=3, sticky=tk.W, padx=5, pady=5)
        self.episode_time_spinbox.bind("<KeyRelease>", lambda e: self.update_command())
        
        # Reset Time
        ttk.Label(params_frame, text="Reset Time (seconds):").grid(column=0, row=4, sticky=tk.W, padx=5, pady=5)
        self.reset_time_var = tk.DoubleVar(value=5.0)
        self.reset_time_spinbox = ttk.Spinbox(params_frame, from_=1.0, to=30.0, 
                                 textvariable=self.reset_time_var, width=10, 
                                 increment=0.5, command=self.update_command)
        self.reset_time_spinbox.grid(column=1, row=4, sticky=tk.W, padx=5, pady=5)
        self.reset_time_spinbox.bind("<KeyRelease>", lambda e: self.update_command())
        
        # Number of Episodes
        ttk.Label(params_frame, text="Number of Episodes:").grid(column=0, row=5, sticky=tk.W, padx=5, pady=5)
        self.num_episodes_var = tk.IntVar(value=5)
        self.num_episodes_spinbox = ttk.Spinbox(params_frame, from_=1, to=100, 
                                    textvariable=self.num_episodes_var, width=10, 
                                    increment=1, command=self.update_command)
        self.num_episodes_spinbox.grid(column=1, row=5, sticky=tk.W, padx=5, pady=5)
        self.num_episodes_spinbox.bind("<KeyRelease>", lambda e: self.update_command())
        
        # Command Preview Frame
        cmd_frame = ttk.LabelFrame(main_frame, text="Command")
        cmd_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.command_text = scrolledtext.ScrolledText(cmd_frame, height=6, wrap=tk.WORD)
        self.command_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.command_text.config(state=tk.DISABLED)
        
        # Run Button
        self.run_button = ttk.Button(cmd_frame, text="Run Command", command=self.run_command)
        self.run_button.pack(padx=5, pady=5)
        
        # Output Log Frame
        log_frame = ttk.LabelFrame(main_frame, text="Output")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_log = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD)
        self.output_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_log.config(state=tk.DISABLED)
        
        # Initial command update
        self.update_command()
        
        # Focus the first field
        self.hf_user_entry.focus_set()
    
    def update_hf_user(self):
        """Update the HF_USER variable when the input changes"""
        self.hf_user = self.hf_user_var.get().strip()
        self.update_command()
    
    def update_command(self):
        """Update the command preview based on current inputs"""
        # Ensure we have an HF_USER, use a placeholder if empty
        hf_user = self.hf_user if self.hf_user else "YOUR_USERNAME"
        
        # Format tags as a proper JSON array string
        tags = [tag.strip() for tag in self.tags_var.get().split(',')]
        tags_str = str(tags).replace("'", '"') if tags and tags[0] else '["tutorial"]'
        
        # Get numeric values safely
        try:
            episode_time = float(self.episode_time_var.get())
        except:
            episode_time = 10.0
            
        try:
            reset_time = float(self.reset_time_var.get())
        except:
            reset_time = 5.0
            
        try:
            num_episodes = int(self.num_episodes_var.get())
        except:
            num_episodes = 5
        
        # Construct the command
        command = f"python lerobot/scripts/control_robot.py " \
                  f"--robot.type=trossen_ai_solo " \
                  f"--robot.max_relative_target=null " \
                  f"--control.type=record " \
                  f"--control.fps=30 " \
                  f"--control.single_task=\"{self.task_var.get()}\" " \
                  f"--control.repo_id={hf_user}/{self.repo_name_var.get()} " \
                  f"--control.tags='{tags_str}' " \
                  f"--control.warmup_time_s=5 " \
                  f"--control.episode_time_s={episode_time} " \
                  f"--control.reset_time_s={reset_time} " \
                  f"--control.num_episodes={num_episodes} " \
                  f"--control.push_to_hub=true " \
                  f"--control.num_image_writer_threads_per_camera=8 " \
                  f"--control.display_cameras=false"
        
        # Update the text widget
        self.command_text.config(state=tk.NORMAL)
        self.command_text.delete(1.0, tk.END)
        self.command_text.insert(tk.END, command)
        self.command_text.config(state=tk.DISABLED)
    
    def run_command(self):
        """Execute the generated command"""
        # Check if we have a valid HF_USER
        if not self.hf_user:
            self.write_to_log("Error: HuggingFace username is required before running the command.")
            return
        
        # Get the command from the preview
        command = self.command_text.get(1.0, tk.END).strip()
        
        # Clear and update log
        self.write_to_log(f"Running command: {command}\n\nStarting execution...\n")
        
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
            
            # Update log
            self.write_to_log("Command is running...\n\nCheck terminal for full output")
            
            # Disable run button temporarily  
            self.run_button.config(state=tk.DISABLED)
            self.root.update()
            
            # Re-enable run button after a short delay
            self.root.after(1000, lambda: self.run_button.config(state=tk.NORMAL))
            
        except Exception as e:
            self.write_to_log(f"Error: {str(e)}")
    
    def write_to_log(self, text):
        """Write to the output log"""
        self.output_log.config(state=tk.NORMAL)
        self.output_log.delete(1.0, tk.END)
        self.output_log.insert(tk.END, text)
        self.output_log.config(state=tk.DISABLED)


if __name__ == "__main__":
    # Check if HF_USER is set
    hf_user = os.environ.get('HF_USER')
    if not hf_user:
        print("Notice: HF_USER environment variable is not set.")
        print("You will need to enter your HuggingFace username in the UI.")
        print("To avoid this in the future, set the environment variable:")
        print("  export HF_USER=your_huggingface_username")
    else:
        print(f"Using HuggingFace username from environment: {hf_user}")
    
    print("\nStarting Trossen AI Solo Control UI (Tkinter version)...")
    print("This UI allows you to easily configure and run the control_robot.py script")
    print("with custom parameters for data collection.")
    
    # Create the Tkinter window
    root = tk.Tk()
    app = TrossenControlUI(root)
    root.mainloop() 