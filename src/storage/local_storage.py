"""
Local file storage module for Style Transfer AI.
Handles saving and loading of analysis results to local files.
"""

import json
import os
from datetime import datetime
from ..config.settings import TIMESTAMP_FORMAT
from ..utils.formatters import format_human_readable_output, save_dual_format
from ..utils.text_processing import sanitize_filename


def save_style_profile_locally(style_profile, base_filename="user_style_profile_enhanced"):
    """
    Save style profile in dual format (JSON + TXT) locally.
    
    Args:
        style_profile (dict): The complete style profile data
        base_filename (str): Base filename without extension
        
    Returns:
        dict: Save result with success status and file paths
    """
    try:
        # Extract user name from profile for personalized filename
        user_name = "Anonymous_User"
        if 'user_profile' in style_profile and 'name' in style_profile['user_profile']:
            user_name = sanitize_filename(style_profile['user_profile']['name'])
        
        # Save using dual format utility
        json_filename, txt_filename = save_dual_format(style_profile, base_filename, user_name)
        
        return {
            'success': True,
            'json_file': json_filename,
            'txt_file': txt_filename,
            'message': f"Personal stylometric profile saved locally for {user_name}:\nΓÇó JSON: {json_filename}\nΓÇó TXT: {txt_filename}"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error saving files locally: {e}"
        }


def cleanup_old_reports(patterns=None):
    """
    Clean up old analysis reports based on filename patterns.
    
    Args:
        patterns (list): List of filename patterns to match for deletion
        
    Returns:
        dict: Cleanup result with count of deleted files
    """
    if patterns is None:
        patterns = [
            "*_stylometric_profile_*.json",
            "*_stylometric_profile_*.txt",
            "user_style_profile_enhanced_*.json",
            "user_style_profile_enhanced_*.txt"
        ]
    
    deleted_count = 0
    try:
        import glob
        
        # Check both main directory and stylometry fingerprints directory
        directories_to_check = [".", "stylometry fingerprints"]
        
        for directory in directories_to_check:
            for pattern in patterns:
                search_pattern = pattern if directory == "." else os.path.join(directory, pattern)
                files = glob.glob(search_pattern)
                for file in files:
                    try:
                        os.remove(file)
                        deleted_count += 1
                    except Exception as e:
                        print(f"Warning: Could not delete {file}: {e}")
        
        return {
            'success': True,
            'deleted_count': deleted_count,
            'message': f"Cleaned up {deleted_count} old report files"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error during cleanup: {e}"
        }


def list_local_profiles(pattern="*_stylometric_profile_*.json"):
    """
    List existing local profile files.
    
    Args:
        pattern (str): Glob pattern to match profile files
        
    Returns:
        list: List of found profile files with metadata
    """
    try:
        import glob
        
        # Look for profiles in both the main directory and the stylometry fingerprints directory
        patterns_to_check = [
            pattern,  # Main directory
            os.path.join("stylometry fingerprints", pattern)  # Stylometry fingerprints directory
        ]
        
        profiles = []
        
        for search_pattern in patterns_to_check:
            files = glob.glob(search_pattern)
            for file in files:
                try:
                    stat = os.stat(file)
                    profiles.append({
                        'filename': file,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
                except Exception:
                    continue
        
        # Remove duplicates (in case a file exists in both locations) and sort
        unique_profiles = {p['filename']: p for p in profiles}.values()
        return sorted(unique_profiles, key=lambda x: x['modified'], reverse=True)
        
    except Exception:
        return []


def delete_profile(filepath):
    """
    Delete a style profile and its companion file (JSON+TXT pair).

    Args:
        filepath (str): Path to either the .json or .txt profile file.

    Returns:
        dict: Result with 'success', 'deleted' list, and 'message' or 'error'.
    """
    deleted = []
    try:
        base, ext = os.path.splitext(filepath)
        pair = [base + ".json", base + ".txt"]
        for path in pair:
            if os.path.isfile(path):
                os.remove(path)
                deleted.append(os.path.basename(path))
        if deleted:
            return {
                "success": True,
                "deleted": deleted,
                "message": f"Deleted: {', '.join(deleted)}",
            }
        return {
            "success": False,
            "error": f"No profile files found for {os.path.basename(filepath)}",
        }
    except Exception as e:
        return {"success": False, "error": f"Error deleting profile: {e}"}


def load_local_profile(filename):
    """
    Load a local profile file.
    
    Args:
        filename (str): Path to the profile file
        
    Returns:
        dict: Loaded profile data or error information
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        
        return {
            'success': True,
            'profile': profile,
            'message': f"Profile loaded from {filename}"
        }
        
    except FileNotFoundError:
        return {
            'success': False,
            'error': f"File not found: {filename}"
        }
    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f"Invalid JSON in file {filename}: {e}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error loading file {filename}: {e}"
        }