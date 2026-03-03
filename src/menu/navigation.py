"""
Navigation utilities for Style Transfer AI.
Handles common navigation patterns and user interactions.
"""

import os
import sys
from datetime import datetime


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title, width=60):
    """
    Print a formatted header.
    
    Args:
        title (str): Header title
        width (int): Header width
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_section(title, width=50):
    """
    Print a section separator.
    
    Args:
        title (str): Section title
        width (int): Section width
    """
    print("\n" + "-"*width)
    print(title)
    print("-"*width)


def confirm_action(message, default='n'):
    """
    Get user confirmation for an action.
    
    Args:
        message (str): Confirmation message
        default (str): Default choice ('y' or 'n')
        
    Returns:
        bool: True if confirmed, False otherwise
    """
    choices = "Y/n" if default.lower() == 'y' else "y/N"
    response = input(f"{message} ({choices}): ").strip().lower()
    
    if not response:
        return default.lower() == 'y'
    
    return response in ['y', 'yes']


def get_user_input(prompt, default=None, input_type=str, validator=None):
    """
    Get validated user input.
    
    Args:
        prompt (str): Input prompt
        default: Default value if user presses Enter
        input_type (type): Expected input type (str, int, float)
        validator (callable): Optional validation function
        
    Returns:
        User input converted to the specified type
    """
    while True:
        try:
            if default is not None:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                if not user_input:
                    return default
            else:
                user_input = input(f"{prompt}: ").strip()
                if not user_input:
                    print("Input required. Please try again.")
                    continue
            
            # Convert to specified type
            if input_type != str:
                converted_input = input_type(user_input)
            else:
                converted_input = user_input
            
            # Apply validator if provided
            if validator:
                if validator(converted_input):
                    return converted_input
                else:
                    print("Invalid input. Please try again.")
                    continue
            
            return converted_input
            
        except ValueError:
            print(f"Invalid {input_type.__name__}. Please try again.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled.")
            return None


def display_loading_animation(message="Processing", duration=2):
    """
    Display a simple loading animation.
    
    Args:
        message (str): Loading message
        duration (int): Animation duration in seconds
    """
    import time
    
    spinner = "|/-\\"
    end_time = time.time() + duration
    
    print(f"\n{message}", end="")
    
    while time.time() < end_time:
        for char in spinner:
            print(f"\r{message} {char}", end="", flush=True)
            time.sleep(0.1)
    
    print(f"\r{message} Γ£ô")


def pause_for_user(message="Press Enter to continue..."):
    """
    Pause execution and wait for user input.
    
    Args:
        message (str): Message to display
    """
    try:
        input(f"\n{message}")
    except KeyboardInterrupt:
        print("\n")


def format_file_size(size_bytes):
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def format_timestamp(timestamp=None):
    """
    Format timestamp in readable format.
    
    Args:
        timestamp (datetime): Timestamp to format (uses current time if None)
        
    Returns:
        str: Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def create_numbered_menu(options, title="Select an option"):
    """
    Create a numbered menu and get user selection.
    
    Args:
        options (list): List of menu options
        title (str): Menu title
        
    Returns:
        int: Selected option index (0-based), or -1 if cancelled
    """
    print_section(title)
    
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    print("0. Cancel/Return")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (0-{len(options)}): ").strip()
            
            if choice == "0":
                return -1
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return choice_num - 1
            else:
                print(f"Invalid choice. Please enter 0-{len(options)}.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nCancelled.")
            return -1


def display_progress_bar(current, total, width=50, title="Progress"):
    """
    Display a progress bar.
    
    Args:
        current (int): Current progress value
        total (int): Total progress value
        width (int): Progress bar width
        title (str): Progress bar title
    """
    if total == 0:
        return
    
    percentage = (current / total) * 100
    filled_width = int(width * current // total)
    bar = "Γûê" * filled_width + "-" * (width - filled_width)
    
    print(f"\r{title}: |{bar}| {percentage:.1f}% ({current}/{total})", end="", flush=True)
    
    if current == total:
        print()  # New line when complete


def handle_keyboard_interrupt(func):
    """
    Decorator to handle keyboard interrupts gracefully.
    
    Args:
        func (callable): Function to wrap
        
    Returns:
        callable: Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n\nOperation interrupted by user.")
            return None
    
    return wrapper


def validate_file_path(file_path):
    """
    Validate that a file path exists and is readable.
    
    Args:
        file_path (str): Path to validate
        
    Returns:
        bool: True if path is valid, False otherwise
    """
    try:
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)
    except Exception:
        return False


def get_file_paths_interactive(prompt="Enter file paths (one per line, empty line to finish):"):
    """
    Interactively collect file paths from user.
    
    Args:
        prompt (str): Input prompt
        
    Returns:
        list: List of validated file paths
    """
    print(f"\n{prompt}")
    file_paths = []
    
    while True:
        try:
            path = input("File path: ").strip()
            
            if not path:  # Empty line ends input
                break
            
            if validate_file_path(path):
                file_paths.append(path)
                print(f"Γ£ô Added: {os.path.basename(path)}")
            else:
                print(f"Γ£ù Invalid or inaccessible file: {path}")
                retry = confirm_action("Try again?", default='y')
                if not retry:
                    break
                    
        except KeyboardInterrupt:
            print("\n\nFile collection cancelled.")
            break
    
    return file_paths


def display_table(data, headers, title=None):
    """
    Display data in a formatted table.
    
    Args:
        data (list): List of rows (each row is a list of values)
        headers (list): List of column headers
        title (str): Optional table title
    """
    if not data:
        print("No data to display.")
        return
    
    # Calculate column widths
    col_widths = [len(str(header)) for header in headers]
    for row in data:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create format string
    row_format = " | ".join(f"{{:<{width}}}" for width in col_widths)
    separator = "-+-".join("-" * width for width in col_widths)
    
    # Display table
    if title:
        print_section(title)
    
    print(row_format.format(*headers))
    print(separator)
    
    for row in data:
        # Pad row with empty strings if needed
        padded_row = list(row) + [""] * (len(headers) - len(row))
        print(row_format.format(*padded_row[:len(headers)]))


def safe_exit(exit_code=0):
    """
    Safely exit the application with cleanup.
    
    Args:
        exit_code (int): Exit code
    """
    print("\nThank you for using Style Transfer AI!")
    sys.exit(exit_code)


if __name__ == "__main__":
    # Test navigation utilities
    print_header("Navigation Utilities Test")
    
    # Test user input
    name = get_user_input("Enter your name", default="User")
    age = get_user_input("Enter your age", input_type=int, validator=lambda x: x > 0)
    
    print(f"\nHello {name}, you are {age} years old.")
    
    # Test menu
    options = ["Option 1", "Option 2", "Option 3"]
    choice = create_numbered_menu(options, "Test Menu")
    
    if choice >= 0:
        print(f"You selected: {options[choice]}")
    else:
        print("No selection made.")
    
    pause_for_user()