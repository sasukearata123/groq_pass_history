import os
import json
from openai import OpenAI


# -------------------- TOOL FUNCTION --------------------
def get_weather(location: str, unit: str = "celsius"):
    return f"The current weather in {location} is 20 degrees {unit}."


# -------------------- CLIENT --------------------
client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1",
)


# -------------------- TOOLS --------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]


# -------------------- MESSAGE STATE --------------------
messages = [
    {
        "role": "system",
        "content": "You are a weather assistant. Use tools when needed."
    }
]


# -------------------- MAIN LOOP --------------------
def main():
    print("Type 'stop' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "stop":
            print("Goodbye!")
            break

        # Add user message
        messages.append({
            "role": "user",
            "content": user_input
        })

        # First call (model may request a tool)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use a valid Groq model
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message
        
        # Check if the model wants to call a tool
        if assistant_message.tool_calls:
            # Add assistant's tool call request to messages
            messages.append(assistant_message)
            
            # Process each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call the actual function
                if function_name == "get_weather":
                    tool_result = get_weather(
                        location=function_args["location"],
                        unit=function_args.get("unit", "celsius")
                    )
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })
            
            # Second call with tool results
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            
            assistant_message = response.choices[0].message

        # Print assistant reply
        print("Assistant:", assistant_message.content)

        # Add assistant reply to messages
        messages.append({
            "role": "assistant",
            "content": assistant_message.content
        })


# -------------------- RUN --------------------
if __name__ == "__main__":
    main()
