"""
User profile management utilities.
Handles collection and validation of user background information.
"""
import os


def get_user_profile():
    """
    Collect essential user profile information for contextual analysis.
    
    Returns:
        dict: User profile data
    """
    print("\n" + "="*60)
    print("USER PROFILE COLLECTION")
    print("="*60)
    print("To provide personalized stylometric analysis, please share some")
    print("background information. This helps interpret cultural and linguistic")
    print("patterns in your writing style.")
    print("="*60)
    
    profile = {}
    
    # Essential information
    profile['name'] = input("\n1. Your name (for file naming): ").strip() or "Anonymous_User"
    
    # Language background
    print("\nLANGUAGE BACKGROUND:")
    profile['native_language'] = input("2. Native language: ").strip() or "Not provided"
    profile['english_fluency'] = input("3. English fluency (Beginner/Intermediate/Advanced/Native): ").strip() or "Not provided"
    profile['other_languages'] = input("4. Other languages you speak: ").strip() or "Not provided"
    
    # Cultural context
    print("\nCULTURAL CONTEXT:")
    profile['nationality'] = input("5. Nationality: ").strip() or "Not provided"
    profile['cultural_background'] = input("6. Cultural background (brief): ").strip() or "Not provided"
    
    # Educational background
    print("\nEDUCATIONAL BACKGROUND:")
    profile['education_level'] = input("7. Education level (High School/Bachelor's/Master's/PhD/Other): ").strip() or "Not provided"
    profile['field_of_study'] = input("8. Field of study/work: ").strip() or "Not provided"
    
    # Writing experience
    print("\nWRITING EXPERIENCE:")
    profile['writing_experience'] = input("9. Writing experience (Beginner/Intermediate/Advanced/Professional): ").strip() or "Not provided"
    profile['writing_frequency'] = input("10. How often do you write? (Daily/Weekly/Monthly/Rarely): ").strip() or "Not provided"
    
    print(f"\nThank you, {profile['name']}! This information will help provide")
    print("more accurate and personalized stylometric analysis.")
    
    return profile


def get_file_paths():
    """
    Get file paths from user input.
    
    Returns:
        list: List of file paths to analyze
    """
    from ..config.settings import DEFAULT_FILE_PATHS
    
    print("\n" + "="*50)
    print("FILE SELECTION")
    print("="*50)
    print("Select text files for analysis:")
    
    # Show available sample files with validation
    print("\nAvailable sample files:")
    valid_defaults = []
    for i, file_path in enumerate(DEFAULT_FILE_PATHS, 1):
        if os.path.exists(file_path):
            # Get file size info
            size = os.path.getsize(file_path)
            print(f"  ✓ {i}. {file_path} ({size} bytes)")
            valid_defaults.append(file_path)
        else:
            print(f"  ✗ {i}. {file_path} (not found)")
    
    print(f"\nOptions:")
    print("1. Use sample files (recommended for testing)")
    print("2. Specify your own file paths")
    print("3. Enter custom text directly (no files needed)")
    print("="*50)
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                if valid_defaults:
                    print(f"✓ Using {len(valid_defaults)} sample files")
                    return valid_defaults
                else:
                    print("⚠️ No sample files found, please specify your own files.")
                    choice = "2"  # Fall through to custom files
            
            if choice == "2":
                file_paths = []
                print("\nEnter file paths (one per line, empty line to finish):")
                while True:
                    path = input("File path: ").strip()
                    if not path:
                        break
                    if os.path.exists(path):
                        file_paths.append(path)
                        print(f"✓ Added: {os.path.basename(path)}")
                    else:
                        print(f"✗ File not found: {path}")
                
                if file_paths:
                    return file_paths
                else:
                    print("No valid files provided. Using sample files as fallback.")
                    return valid_defaults if valid_defaults else DEFAULT_FILE_PATHS
            
            elif choice == "3":
                return get_custom_text_input()
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return valid_defaults if valid_defaults else []


def get_custom_text_input():
    """
    Get custom text input directly from the user.
    
    Returns:
        dict: Dictionary containing custom text with special marker
    """
    print("\n" + "="*50)
    print("CUSTOM TEXT INPUT")
    print("="*50)
    print("Enter your text for analysis. You can:")
    print("• Paste a single paragraph or document")
    print("• Enter multiple paragraphs (empty line to finish)")
    print("• Minimum recommended: 100+ words for meaningful analysis")
    print("="*50)
    
    text_lines = []
    print("\nEnter your text (press Enter twice to finish):")
    print("-" * 50)
    
    empty_lines = 0
    while True:
        try:
            line = input()
            if not line.strip():
                empty_lines += 1
                if empty_lines >= 2:  # Two consecutive empty lines to finish
                    break
                text_lines.append("")  # Keep single empty lines for formatting
            else:
                empty_lines = 0
                text_lines.append(line)
        except KeyboardInterrupt:
            print("\n\nInput cancelled by user.")
            return []
    
    combined_text = "\n".join(text_lines).strip()
    
    if not combined_text:
        print("⚠️ No text entered. Please try again.")
        return []
    
    word_count = len(combined_text.split())
    print(f"\n✓ Text captured: {word_count} words, {len(combined_text)} characters")
    
    if word_count < 50:
        print("⚠️ Warning: Text is quite short. Consider adding more content for better analysis.")
        confirm = input("Continue with this text? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            return []
    
    # Return a special format to indicate this is custom text, not file paths
    return {
        'type': 'custom_text',
        'text': combined_text,
        'word_count': word_count,
        'source': 'direct_input'
    }


def get_cloud_storage_preference():
    """
    Get user's preference for cloud storage.
    
    Returns:
        bool: Always returns False (local storage only)
    """
    print("\n" + "="*50)
    print("STORAGE PREFERENCE")
    print("="*50)
    print("Using local storage only")
    print("="*50)
    
    return False


def ask_to_continue():
    """
    Ask user if they want to continue with another analysis.
    
    Returns:
        bool: True if user wants to continue, False otherwise
    """
    print("\n" + "="*50)
    print("CONTINUE ANALYSIS")
    print("="*50)
    
    while True:
        try:
            choice = input("Do you want to analyze more files? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return False