import sys
import asyncio
from pathlib import Path

# Ensure project package path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.ai.handler import AdvancedAIHandler, AIProviderExhausted


async def main():
    h = AdvancedAIHandler()
    print("Testing basic generation...")
    try:
        r = await h.generate('hello world')
        print('generate ok ->', r)
    except AIProviderExhausted as e:
        print('provider exhausted', e)
    
    # Test tool execution if tools are available
    if h.tools:
        print("\nTesting tool execution...")
        try:
            # Test get_profile tool (this will likely fail without proper phone number)
            tool_result = await h.execute_tool("get_profile", phone="+1234567890")
            print('tool execution result ->', tool_result)
        except Exception as e:
            print('tool execution failed ->', str(e))
    else:
        print("\nTools not available for testing")


if __name__ == '__main__':
    asyncio.run(main())
