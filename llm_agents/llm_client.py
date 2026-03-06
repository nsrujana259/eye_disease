import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
MODEL_CANDIDATES = [
    "mistralai/mistral-7b-instruct",
    "mistralai/mistral-7b-instruct:free",
    "openai/gpt-3.5-turbo",
]

def call_llm(prompt):
    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            response = client.chat.completions.create(
                model=model_name,
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
        except Exception as exc:
            last_error = exc
            continue

    return (
        "AI explanation/report service is temporarily unavailable. "
        "Please consult a qualified doctor for clinical interpretation. "
        f"(LLM error: {last_error})"
    )
    return response.choices[0].message.content