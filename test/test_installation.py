#!/usr/bin/env python3
"""
Installation Test Script for Style Transfer AI
Run this after installation to verify everything works correctly.
"""

import sys
import os
import importlib

def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    if sys.version_info >= (3, 7):
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (Requires 3.7+)")
        return False

def test_dependencies():
    """Test core dependencies."""
    print("\n📦 Testing core dependencies...")
    
    core_deps = ['requests']
    optional_deps = ['openai', 'google.generativeai']
    
    success = True
    
    # Test core dependencies
    for dep in core_deps:
        try:
            importlib.import_module(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} (REQUIRED)")
            success = False
    
    # Test optional dependencies
    for dep in optional_deps:
        try:
            importlib.import_module(dep)
            print(f"   ✅ {dep} (optional)")
        except ImportError:
            print(f"   ⚠️  {dep} (optional - install if needed)")
    
    return success

def test_module_imports():
    """Test main module imports."""
    print("\n🔧 Testing module imports...")
    
    modules_to_test = [
        'src.main',
        'src.config.settings',
        'src.models.ollama_client',
        'src.analysis.analyzer',
        'src.utils.text_processing'
    ]
    
    success = True
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module} - {e}")
            success = False
    
    return success

def test_sample_files():
    """Test if sample files exist."""
    print("\n📄 Testing sample files...")
    
    sample_files = [
        'default text/about_my_pet.txt',
        'default text/about_my_pet_1.txt',
        'default text/about_my_pet_2.txt'
    ]
    
    found_files = 0
    for file_path in sample_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
            found_files += 1
        else:
            print(f"   ⚠️  {file_path} (not found)")
    
    if found_files > 0:
        print(f"   📊 Found {found_files} sample files for testing")
        return True
    else:
        print("   ⚠️  No sample files found (you can still use your own files)")
        return True

def test_cli_entry_point():
    """Test CLI entry point."""
    print("\n🖥️  Testing CLI entry point...")
    
    try:
        from src.main import cli_entry_point
        print("   ✅ CLI entry point accessible")
        return True
    except ImportError as e:
        print(f"   ❌ CLI entry point failed - {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Style Transfer AI - Installation Test")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_dependencies,
        test_module_imports,
        test_sample_files,
        test_cli_entry_point
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    if all(results):
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ Your installation is ready!")
        print("Run: python run.py  OR  style-transfer-ai")
    else:
        print("⚠️  Some tests failed or have warnings.")
        print("\n🔧 Next steps:")
        print("1. Install missing dependencies: pip install -r install/requirements.txt")
        print("2. Or install as package: pip install -e .")
        print("3. For optional features, install: pip install openai google-generativeai")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)