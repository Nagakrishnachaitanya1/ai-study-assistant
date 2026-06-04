from google.colab import userdata
from google import genai
from google.genai import types
from pprint import pprint
import gradio as gr

client = genai.Client(api_key=userdata.get('GEMINI_API'))

personalities = {
    "friendly": "you are an ai student assistant your tone should be friendly and casual as a friend explaining to another",
    "academic": "you are an ai student assistant your tone should be like a lecturer explaining to a student"
}

memory = {
    "friendly" : [],
    "academic" : []
}

def student_assistant(user_question, persona):
    personality = personalities[persona]

    memory[persona].append({
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
        contents=memory[persona]
    )

    memory[persona].append({
        "role": "model",
        "parts": [{"text": response.text}]
    })

    return response.text

demo = gr.Interface(
    fn = student_assistant,
    inputs=[
        gr.Textbox(
            lines=4,
            placeholder="Enter your question"
        ),

        gr.Dropdown(
            choices=["friendly", "academic"],
            value="friendly",
            label="Choose Personality"
        )
    ],
    outputs = gr.Textbox(lines = 10 , placeholder = "AI response"),
    title="🎓 AI Study Assistant",
    description="Ask any academic question and learn with AI."
)

demo.launch(debug=True)
