import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key= "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"

# Function to generate a response using GPT-3
def generate_response(comment_text):
    statements = """
    A is on a plate.
    D is on a plate.
    F is on a plate.
    J is on a plate.

    C is on D.
    B is on C. 
    E is on F.
    I is on J.
    H is on I.
    G is on H.

    A is to the left of D.
    D is to the left of F.
    F is to the left of J.

    The color of A is Red.
    The color of B is White.
    The color of C is Brown.
    The color of D is Grey.
    The color of E is Brown.
    The color of F is Red.
    The color of G is Brown.
    The color of H is Red.
    The color of I is Blue.
    The color of J is Yellow.

    A is a type of Brick.
    B is a type of Tile.
    C is a type of Plate.
    D is a type of Brick.
    E is a type of Tile.
    F is a type of Brick.
    G is a type of Tile.
    H is a type of Brick.
    I is a type of Brick.
    J is a type of Plate.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a Lego world problem solver. Solve the problem based on these expressions( Note, The expressions are relateed to one another in a way):\n{statements}."},
            {"role": "user", "content": comment_text}
        ],
    )

    return response.choices[0].message["content"].strip()

# Interpret and respond to different types of statements

while True:
    user_input = input("\nAsk me a question about the lego world(Type in 'exit' To end session): \n")
    if user_input.lower() == 'exit':
        break
    response = generate_response(user_input)
    print(f"Computer: {response}")
