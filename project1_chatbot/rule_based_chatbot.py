"""
Project 1: Rule-Based AI Chatbot
DecodeLabs - Artificial Intelligence Industrial Training Kit
"""

# STEP 1: Knowledge Base
responses = {
    "hello": "Hi there! How can I help you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bunch of code, but I'm doing great!",
    "what is your name": "I'm RuleBot, your friendly rule-based assistant.",
    "help": "I can chat with you! Try saying hello, or ask about my name.",
    "what can you do": "I answer simple predefined questions using rule-based logic.",
    "who made you": "I was built as part of the DecodeLabs AI training kit.",
}

exit_commands = ["exit", "bye", "quit", "goodbye"]


def get_response(user_text):
    clean_input = user_text.lower().strip()
    reply = responses.get(clean_input, "I do not understand that yet. Try 'help'!")
    return clean_input, reply


def run_chatbot():
    print("Bot: Hello! I'm RuleBot. Type 'exit' or 'bye' anytime to quit.")

    while True:
        user_input = input("You: ")
        clean_input, reply = get_response(user_input)

        if clean_input in exit_commands:
            print("Bot: Goodbye! Have a great day!")
            break

        print("Bot:", reply)


if __name__ == "__main__":
    run_chatbot()