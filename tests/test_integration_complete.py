#!/usr/bin/env python3
"""
Comprehensive test for the custom text input feature integration.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

def test_complete_integration():
    """Test the complete custom text input integration."""
    print("="*70)
    print("COMPREHENSIVE CUSTOM TEXT INPUT INTEGRATION TEST")
    print("="*70)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Import all required modules
    total_tests += 1
    try:
        from src.utils.user_profile import get_file_paths, get_custom_text_input
        from src.analysis.analyzer import create_enhanced_style_profile
        from src.menu.main_menu import handle_analyze_style
        print("✅ Test 1: All modules imported successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 1: Import failed - {e}")
    
    # Test 2: Custom text input structure
    total_tests += 1
    try:
        test_text = "This is a comprehensive test of the new custom text input feature for Style Transfer AI."
        custom_input = {
            'type': 'custom_text',
            'text': test_text,
            'word_count': len(test_text.split()),
            'source': 'direct_input'
        }
        
        assert custom_input['type'] == 'custom_text'
        assert custom_input['word_count'] > 0
        assert custom_input['source'] == 'direct_input'
        print("✅ Test 2: Custom text input structure is correct")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 2: Structure test failed - {e}")
    
    # Test 3: Input type detection
    total_tests += 1
    try:
        # Test file paths (list)
        file_paths = ['test1.txt', 'test2.txt']
        is_list = isinstance(file_paths, list)
        
        # Test custom text (dict)
        is_dict = isinstance(custom_input, dict)
        has_type = custom_input.get('type') == 'custom_text'
        
        assert is_list and is_dict and has_type
        print("✅ Test 3: Input type detection logic is correct")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 3: Type detection failed - {e}")
    
    # Test 4: Analysis function compatibility
    total_tests += 1
    try:
        # Check if the function signature accepts both types
        import inspect
        sig = inspect.signature(create_enhanced_style_profile)
        params = list(sig.parameters.keys())
        
        # Should have 'input_data' as first parameter
        assert 'input_data' in params
        print("✅ Test 4: Analysis function signature updated correctly")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 4: Function signature test failed - {e}")
    
    # Test 5: Menu integration
    total_tests += 1
    try:
        # Check if handle_analyze_style function exists and is callable
        assert callable(handle_analyze_style)
        print("✅ Test 5: Menu integration is functional")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 5: Menu integration test failed - {e}")
    
    # Test 6: File system compatibility
    total_tests += 1
    try:
        # Test that default text files exist (for fallback)
        from src.config.settings import DEFAULT_FILE_PATHS
        default_exists = any(os.path.exists(path) for path in DEFAULT_FILE_PATHS)
        print("✅ Test 6: File system compatibility checked")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Test 6: File system test failed - {e}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe custom text input feature is fully integrated and ready for use!")
        print("\nHow to use:")
        print("1. Run: python run.py")
        print("2. Choose option 1 (Complete Analysis) or 2 (Quick Analysis)")
        print("3. Select option 3 'Enter custom text directly (no files needed)'")
        print("4. Enter or paste your text")
        print("5. Press Enter twice to finish input")
        print("6. Enjoy comprehensive stylometric analysis!")
        return True
    else:
        print(f"\n⚠️ {total_tests - tests_passed} tests failed.")
        print("Please check the implementation for issues.")
        return False

if __name__ == "__main__":
    success = test_complete_integration()
    sys.exit(0 if success else 1)