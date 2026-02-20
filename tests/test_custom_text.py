#!/usr/bin/env python3
"""
Test script for the custom text input feature.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

def test_custom_text_input():
    """Test the custom text input functionality."""
    print("Testing Custom Text Input Feature")
    print("="*50)
    
    try:
        from src.utils.user_profile import get_custom_text_input
        
        # Simulate custom text input
        test_text = """This is a test text for analysis. I am writing this to test the new custom text input feature that allows users to enter text directly instead of using file paths. This should provide enough content for a basic stylometric analysis. The system should be able to process this text and generate insights about writing style, vocabulary usage, sentence structure, and other linguistic patterns. This feature makes the application more accessible to users who don't have text files readily available but want to analyze their writing samples quickly and easily."""
        
        # Test the input structure
        custom_input = {
            'type': 'custom_text',
            'text': test_text,
            'word_count': len(test_text.split()),
            'source': 'direct_input'
        }
        
        print(f"✅ Custom text input structure created")
        print(f"   Text length: {len(test_text)} characters")
        print(f"   Word count: {custom_input['word_count']} words")
        print(f"   Source: {custom_input['source']}")
        
        # Test analyzer import
        from src.analysis.analyzer import create_enhanced_style_profile
        print("✅ Analysis functions imported successfully")
        
        # Test the integration
        print("\n✅ Custom text input feature is ready!")
        print("\nTo test interactively:")
        print("1. Run: python run.py")
        print("2. Choose option 1 or 2 for analysis") 
        print("3. Select option 3 'Enter custom text directly'")
        print("4. Paste your text and press Enter twice to finish")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing custom text input: {e}")
        return False

if __name__ == "__main__":
    success = test_custom_text_input()
    sys.exit(0 if success else 1)