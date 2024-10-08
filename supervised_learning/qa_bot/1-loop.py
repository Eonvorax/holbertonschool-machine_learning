#!/usr/bin/env python3
"""
Basic QA loop
"""


def qa_loop():
    """
    Continuously prompts the user for input until one of the predefined
    farewell words is entered.
    """
    farewells = ["exit", "quit", "goodbye", "bye"]

    user_input = ""
    while True:
        user_input = input("Q: ")

        # Check user input for farewell string (case insensitive)
        if user_input.lower() in farewells:
            print("A: Goodbye")
            exit()

        # Response to any other input
        print("A: ")


if __name__ == "__main__":
    qa_loop()
