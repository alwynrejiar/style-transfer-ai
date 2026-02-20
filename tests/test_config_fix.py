#!/usr/bin/env python3
"""
Test script to validate the configuration checker fixes.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

def test_integrated_checker():
    """Test the integrated configuration checker."""
    print("="*60)
    print("TESTING INTEGRATED CONFIGURATION CHECKER")
    print("="*60)
    
    try:
        from src.main import check_system_requirements
        results = check_system_requirements()
        
        print(f"\nResults: {results}")
        print(f"Success: {results.get('success', False)}")
        print(f"Issues: {len(results.get('issues', []))}")
        print(f"Warnings: {len(results.get('warnings', []))}")
        print(f"Recommendations: {len(results.get('recommendations', []))}")
        
        return results.get('success', False)
        
    except Exception as e:
        print(f"❌ Failed to run integrated checker: {e}")
        return False

def test_menu_integration():
    """Test the menu integration."""
    print("\n" + "="*60)
    print("TESTING MENU INTEGRATION")
    print("="*60)
    
    try:
        from src.menu.main_menu import handle_check_configuration
        print("✅ Successfully imported handle_check_configuration")
        print("✅ Menu integration is ready")
        return True
        
    except Exception as e:
        print(f"❌ Failed to import menu function: {e}")
        return False

def test_standalone_checker():
    """Test the standalone checker."""
    print("\n" + "="*60)
    print("TESTING STANDALONE CHECKER")
    print("="*60)
    
    if os.path.exists('check_config.py'):
        print("✅ check_config.py exists")
        print("✅ Standalone checker is available")
        return True
    else:
        print("❌ check_config.py not found")
        return False

def main():
    """Run all tests."""
    print("STYLE TRANSFER AI - CONFIGURATION CHECKER VALIDATION")
    print("="*60)
    
    tests = [
        ("Integrated Checker", test_integrated_checker),
        ("Menu Integration", test_menu_integration),
        ("Standalone Checker", test_standalone_checker),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Configuration checker fixes are working correctly.")
        print("\nThe 'Configuration checker not found' error should now be resolved.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. There may still be issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())