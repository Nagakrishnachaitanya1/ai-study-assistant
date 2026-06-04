
from google.colab import userdata
from google import genai
from google.genai import types
from pprint import pprint

client = genai.Client(api_key=userdata.get('GEMINI_API'))

personalities = {
    "friendly": "you are an ai student assistant your tone should be friendly and casual as a friend explaining to another",
    "academic": "you are an ai student assistant your tone should be like a lecturer explaining to a student"
}

memory = []

def student_assistant(user_question, persona):
    personality = personalities[persona]

    memory.append({
        "role": "user",
        "parts": [{"text": user_question}]
    })

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        config=types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=2000,
            system_instruction=personality
        ),
        contents=memory
    )

    memory.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })

    return response.text

while True:
    question = input("here : ")

    if question == "break":
        break

    output = student_assistant(question, "friendly")
    pprint(output)
