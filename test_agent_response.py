#!/usr/bin/env python3
"""Test agent API response format"""

import requests
import json

# Test the agent API
url = "http://localhost:7000/u/test_debug/chat?session_id=debug_session"
# params = {
#     "user_id": "test_debug",
#     "session_id": "debug_session"
# }
data = {
    "message": "Use given calculator tool to calculate 50 * 59 / 32 + 14"
}

print("=" * 80)
print("Testing Agent API Response Format")
print("=" * 80)

try:
    response = requests.post(url, params={}, json=data, timeout=60)
    response.raise_for_status()

    result = response.json()

    print("\n1. Full API Response:")
    print(json.dumps(result, indent=2)[:1000])

    print("\n2. Response Structure:")
    print(f"   Keys: {list(result.keys())}")

    if 'result' in result:
        result_data = result['result']['raw_response'] if 'raw_response' in result['result'] else result['result']
        print(f"   result keys: {list(result_data.keys())}")

        if 'response_text' in result_data:
            print(f"   ✓ response_text found: {result_data['response_text'][:100]}...")

        if 'messages' in result_data:
            print(f"   ✓ messages found: {len(result_data['messages'])} messages")

            # Check for tool calls
            tool_call_count = 0
            for i, msg in enumerate(result_data['messages']):
                if isinstance(msg, dict):
                    msg_type = msg.get('type', 'unknown')
                    has_tool_calls = 'tool_calls' in msg

                    print(f"      Message {i}: type={msg_type}, has_tool_calls={has_tool_calls}")

                    if has_tool_calls:
                        tool_call_count += len(msg['tool_calls'])
                        print(f"         Tool calls: {msg['tool_calls']}")

            print(f"\n   Total tool calls found: {tool_call_count}")
        else:
            print(f"   ✗ messages NOT found")

except Exception as e:
    print(f"\nError: {e}")

print("\n" + "=" * 80)
