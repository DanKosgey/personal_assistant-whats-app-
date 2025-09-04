import sys
import asyncio
from pathlib import Path

# Ensure project package path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.ai.handler import AdvancedAIHandler, AIProviderExhausted


async def main():
    h = AdvancedAIHandler()
    try:
        r = await h.generate('hello world')
        print('generate ok ->', r)
    except AIProviderExhausted as e:
        print('provider exhausted', e)


if __name__ == '__main__':
    asyncio.run(main())
