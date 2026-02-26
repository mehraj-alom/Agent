#!/usr/bin/env python3
"""
Test script to verify error handling improvements in the blog generation pipeline.
Run this to validate that errors are properly caught and reported.
"""

import sys
from pathlib import Path

# Add reasoning_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from reasoning_agent.Core.core import run
from datetime import date

def test_error_handling():
    """Test that errors are properly caught and reported"""
    
    print("=" * 60)
    print("Testing Blog Generation Error Handling")
    print("=" * 60)
    
    # Test with minimal configuration
    test_cases = [
        {
            "name": "Normal Case",
            "topic": "Python Basics",
            "audience": "Beginners",
            "tone": "friendly",
            "blog_kind": "tutorial"
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 40)
        
        # Define a progress callback to see messages
        messages = []
        def progress_callback(msg):
            messages.append(msg)
            print(f"  ✓ {msg}")
        
        try:
            result = run(
                topic=test_case["topic"],
                audience=test_case["audience"],
                tone=test_case["tone"],
                blog_kind=test_case["blog_kind"],
                as_of=date.today().isoformat(),
                progress_callback=progress_callback
            )
            
            # Check result structure
            print(f"\n  Result Keys: {list(result.keys())}")
            
            if result.get("error"):
                print(f"  ❌ ERROR CAPTURED: {result.get('error')}")
            else:
                print(f"  ✓ Pipeline succeeded")
                if result.get("final"):
                    print(f"  ✓ Content generated: {len(result.get('final', ''))} chars")
            
            print(f"\n  Progress messages: {len(messages)}")
            
        except Exception as e:
            print(f"  ❌ Unexpected exception: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("Error Handling Tests Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_error_handling()
