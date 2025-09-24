#!/usr/bin/env python3
"""
Configuration checker for Style Transfer AI.
This is a standalone script that can be run independently or called from the main application.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.main import check_system_requirements
    
    def main():
        """Main entry point for the configuration checker."""
        print("="*60)
        print("STYLE TRANSFER AI - CONFIGURATION CHECKER")
        print("="*60)
        print()
        
        try:
            results = check_system_requirements()
            
            print("\n" + "="*60)
            print("SYSTEM CONFIGURATION CHECK RESULTS")
            print("="*60)
            
            if results['success']:
                print("✅ All system requirements are met!")
                print("\nYour Style Transfer AI installation is ready to use!")
            else:
                print("❌ Some issues were found:")
                for issue in results.get('issues', []):
                    print(f"  • {issue}")
            
            if results.get('warnings'):
                print("\n⚠️  Warnings:")
                for warning in results['warnings']:
                    print(f"  • {warning}")
            
            if results.get('recommendations'):
                print("\n💡 Recommendations:")
                for rec in results['recommendations']:
                    print(f"  • {rec}")
            
            print("\n" + "="*60)
            
            return 0 if results['success'] else 1
            
        except Exception as e:
            print(f"❌ Error during configuration check: {e}")
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print("❌ Error: Cannot import required modules.")
    print(f"Details: {e}")
    print("\nThis usually means the package is not properly installed.")
    print("Try installing with: pip install -e .")
    sys.exit(1)