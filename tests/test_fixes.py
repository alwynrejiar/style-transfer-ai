#!/usr/bin/env python3
"""
Quick validation script for Style Transfer AI model selection fixes.
Tests key components without running full analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all critical imports work correctly."""
    print("Testing imports...")
    
    try:
        from src.menu.model_selection import (
            SELECTED_MODEL, USE_LOCAL_MODEL, USER_CHOSEN_API_KEY,
            reset_model_selection, display_model_menu, validate_model_selection
        )
        print("✓ Model selection imports successful")
        
        from src.analysis.analyzer import analyze_style, create_enhanced_style_profile
        print("✓ Analyzer imports successful")
        
        from src.models.ollama_client import check_ollama_connection, analyze_with_ollama
        from src.models.openai_client import setup_openai_client, analyze_with_openai
        from src.models.gemini_client import setup_gemini_client, analyze_with_gemini
        print("✓ Model client imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_state():
    """Test model state management."""
    print("\nTesting model state management...")
    
    try:
        from src.menu.model_selection import (
            SELECTED_MODEL, USE_LOCAL_MODEL, USER_CHOSEN_API_KEY,
            reset_model_selection
        )
        
        # Test initial state
        print(f"Initial SELECTED_MODEL: {SELECTED_MODEL}")
        print(f"Initial USE_LOCAL_MODEL: {USE_LOCAL_MODEL}")
        print(f"Initial USER_CHOSEN_API_KEY: {USER_CHOSEN_API_KEY}")
        
        # Test reset
        reset_model_selection()
        print("✓ Model reset successful")
        print(f"After reset SELECTED_MODEL: {SELECTED_MODEL}")
        
        return True
    except Exception as e:
        print(f"✗ Model state error: {e}")
        return False

def test_analyzer_validation():
    """Test that analyzer functions properly validate parameters."""
    print("\nTesting analyzer parameter validation...")
    
    try:
        from src.analysis.analyzer import analyze_style, create_enhanced_style_profile
        
        # Test that analyze_style requires model_name for local use
        try:
            analyze_style("test text", use_local=True, model_name=None)
            print("✗ analyze_style should have failed without model_name")
            return False
        except ValueError as e:
            print(f"✓ analyze_style correctly validates parameters: {e}")
        
        # Test that create_enhanced_style_profile requires model_name for local use
        try:
            create_enhanced_style_profile(["test.txt"], use_local=True, model_name=None)
            print("✗ create_enhanced_style_profile should have failed without model_name")
            return False
        except ValueError as e:
            print(f"✓ create_enhanced_style_profile correctly validates parameters: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Analyzer validation error: {e}")
        return False

def test_menu_imports():
    """Test main menu imports."""
    print("\nTesting main menu imports...")
    
    try:
        from src.menu.main_menu import display_main_menu, handle_analyze_style
        print("✓ Main menu imports successful")
        
        # Test that display_main_menu can access SELECTED_MODEL
        display_main_menu()
        print("✓ Main menu display works")
        
        return True
    except Exception as e:
        print(f"✗ Main menu error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=== Style Transfer AI Validation ===")
    
    tests = [
        test_imports,
        test_model_state,
        test_analyzer_validation,
        test_menu_imports
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All validation tests passed! The model selection system should work correctly.")
        print("\nKey fixes implemented:")
        print("• Removed auto-selection behavior")
        print("• Fixed function import errors")
        print("• Added proper parameter validation")
        print("• Consolidated imports to avoid conflicts")
        print("• Model selection is now required for each analysis")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)