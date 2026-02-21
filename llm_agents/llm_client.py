import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def call_llm(prompt):
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",  # ✅ MISTRAL-7B
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical AI assistant. "
                    "Do not provide treatment or medication. "
                    "Always advise consulting a qualified doctor."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content