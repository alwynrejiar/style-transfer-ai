"""
Test script for the modular Style Transfer AI implementation.
Validates that all modules can be imported and basic functionality works.
"""

import sys
import os
import traceback

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing module imports...")
    
    test_results = []
    
    # Test config module
    try:
        from src.config import settings
        print("✓ Config module imported successfully")
        test_results.append(("Config", True, None))
    except Exception as e:
        print(f"✗ Config module import failed: {e}")
        test_results.append(("Config", False, str(e)))
    
    # Test utils modules
    utils_modules = ['text_processing', 'formatters', 'user_profile']
    for module in utils_modules:
        try:
            exec(f"from src.utils import {module}")
            print(f"✓ Utils.{module} imported successfully")
            test_results.append((f"Utils.{module}", True, None))
        except Exception as e:
            print(f"✗ Utils.{module} import failed: {e}")
            test_results.append((f"Utils.{module}", False, str(e)))
    
    # Test model modules
    model_modules = ['ollama_client', 'openai_client', 'gemini_client']
    for module in model_modules:
        try:
            exec(f"from src.models import {module}")
            print(f"✓ Models.{module} imported successfully")
            test_results.append((f"Models.{module}", True, None))
        except Exception as e:
            print(f"✗ Models.{module} import failed: {e}")
            test_results.append((f"Models.{module}", False, str(e)))
    
    # Test analysis modules
    analysis_modules = ['prompts', 'metrics', 'analyzer']
    for module in analysis_modules:
        try:
            exec(f"from src.analysis import {module}")
            print(f"✓ Analysis.{module} imported successfully")
            test_results.append((f"Analysis.{module}", True, None))
        except Exception as e:
            print(f"✗ Analysis.{module} import failed: {e}")
            test_results.append((f"Analysis.{module}", False, str(e)))
    
    # Test storage modules
    storage_modules = ['local_storage']
    for module in storage_modules:
        try:
            exec(f"from src.storage import {module}")
            print(f"✓ Storage.{module} imported successfully")
            test_results.append((f"Storage.{module}", True, None))
        except Exception as e:
            print(f"✗ Storage.{module} import failed: {e}")
            test_results.append((f"Storage.{module}", False, str(e)))
    
    # Test menu modules
    menu_modules = ['main_menu', 'model_selection', 'navigation']
    for module in menu_modules:
        try:
            exec(f"from src.menu import {module}")
            print(f"✓ Menu.{module} imported successfully")
            test_results.append((f"Menu.{module}", True, None))
        except Exception as e:
            print(f"✗ Menu.{module} import failed: {e}")
            test_results.append((f"Menu.{module}", False, str(e)))
    
    # Test main module
    try:
        from src import main
        print("✓ Main module imported successfully")
        test_results.append(("Main", True, None))
    except Exception as e:
        print(f"✗ Main module import failed: {e}")
        test_results.append(("Main", False, str(e)))
    
    return test_results


def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\nTesting basic functionality...")
    
    test_results = []
    
    # Test configuration access
    try:
        from src.config.settings import APPLICATION_NAME, VERSION, AVAILABLE_MODELS
        print(f"✓ Configuration loaded: {APPLICATION_NAME} v{VERSION}")
        print(f"✓ Available models: {len(AVAILABLE_MODELS)} configured")
        test_results.append(("Configuration Access", True, None))
    except Exception as e:
        print(f"✗ Configuration access failed: {e}")
        test_results.append(("Configuration Access", False, str(e)))
    
    # Test text processing utilities
    try:
        from src.utils.text_processing import sanitize_filename, count_syllables
        
        # Test sanitize_filename
        test_filename = sanitize_filename("Test File!@# Name.txt")
        assert "Test_File_Name" in test_filename or "Test_File" in test_filename
        
        # Test syllable counting
        syllables = count_syllables("hello world")
        assert syllables > 0
        
        print("✓ Text processing utilities working")
        test_results.append(("Text Processing", True, None))
    except Exception as e:
        print(f"✗ Text processing test failed: {e}")
        test_results.append(("Text Processing", False, str(e)))
    
    # Test formatting utilities
    try:
        from src.utils.formatters import format_human_readable_output
        
        # Test with minimal profile data
        test_profile = {
            'user_profile': {'name': 'Test User'},
            'metadata': {
                'analysis_date': '2024-01-01',
                'model_used': 'test-model',
                'total_samples': 1,
                'combined_text_length': 100,
                'file_info': []
            },
            'statistical_analysis': {'total_words': 100}
        }
        
        formatted_output = format_human_readable_output(test_profile)
        assert "Test User" in formatted_output
        
        print("✓ Formatting utilities working")
        test_results.append(("Formatting", True, None))
    except Exception as e:
        print(f"✗ Formatting test failed: {e}")
        test_results.append(("Formatting", False, str(e)))
    
    # Test model availability checking (without actual connections)
    try:
        from src.menu.model_selection import get_current_model_info
        
        model_info = get_current_model_info()
        assert isinstance(model_info, dict)
        
        print("✓ Model selection utilities working")
        test_results.append(("Model Selection", True, None))
    except Exception as e:
        print(f"✗ Model selection test failed: {e}")
        test_results.append(("Model Selection", False, str(e)))
    
    return test_results


def test_entry_points():
    """Test that entry points are properly configured."""
    print("\nTesting entry points...")
    
    test_results = []
    
    # Test main entry point
    try:
        from src.main import cli_entry_point, main
        print("✓ Main entry points accessible")
        test_results.append(("Entry Points", True, None))
    except Exception as e:
        print(f"✗ Entry points test failed: {e}")
        test_results.append(("Entry Points", False, str(e)))
    
    # Test run.py compatibility
    try:
        # Check if run.py exists and can be imported
        if os.path.exists('run.py'):
            # Read the file to check it references the new structure
            with open('run.py', 'r') as f:
                content = f.read()
                if 'src.main' in content:
                    print("✓ run.py updated for modular structure")
                    test_results.append(("run.py", True, None))
                else:
                    print("✗ run.py not updated for modular structure")
                    test_results.append(("run.py", False, "Not updated"))
        else:
            print("⚠ run.py not found")
            test_results.append(("run.py", False, "File not found"))
    except Exception as e:
        print(f"✗ run.py test failed: {e}")
        test_results.append(("run.py", False, str(e)))
    
    return test_results


def print_test_summary(all_results):
    """Print a summary of all test results."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(all_results)
    passed_tests = sum(1 for _, success, _ in all_results if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print("\nFAILED TESTS:")
        for test_name, success, error in all_results:
            if not success:
                print(f"  ✗ {test_name}: {error}")
    
    print("\nRECOMMENDATIONS:")
    
    # Check for missing dependencies
    import_failures = [name for name, success, error in all_results 
                      if not success and "could not be resolved" in str(error)]
    
    if import_failures:
        print("• Install missing dependencies:")
        print("  pip install requests openai google-generativeai")
    
    if passed_tests == total_tests:
        print("• ✓ All tests passed! The modular refactoring is complete.")
        print("• Ready to run: python run.py or python -m src.main")
    elif passed_tests >= total_tests * 0.8:
        print("• ✓ Most tests passed. Minor issues may be due to missing optional dependencies.")
        print("• Core functionality should work correctly.")
    else:
        print("• ⚠ Several tests failed. Review the implementation.")
    
    return passed_tests == total_tests


def main():
    """Run all tests and print results."""
    print("="*60)
    print("STYLE TRANSFER AI - MODULAR IMPLEMENTATION TEST")
    print("="*60)
    
    all_results = []
    
    # Run all test suites
    all_results.extend(test_imports())
    all_results.extend(test_basic_functionality())
    all_results.extend(test_entry_points())
    
    # Print summary
    success = print_test_summary(all_results)
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)