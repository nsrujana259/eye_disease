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

# Keep response size modest to avoid OpenRouter credit-limit failures.
MAX_TOKENS_CANDIDATES = [700, 350]
MAX_PROMPT_CHARS = 6000

def call_llm(prompt):
    last_error = None
    safe_prompt = str(prompt or "")[:MAX_PROMPT_CHARS]
    last_error = None

    for model_name in MODEL_CANDIDATES:
        for max_tokens in MAX_TOKENS_CANDIDATES:
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
                        {"role": "user", "content": safe_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()
                # If not a credit/token-related issue, don't retry smaller token size for same model.
                if "402" not in error_text and "credit" not in error_text and "max_tokens" not in error_text:
                    break
                continue

    return (
        "AI explanation/report service is temporarily unavailable. "
        "Please consult a qualified doctor for clinical interpretation. "
        f"(LLM error: {last_error})"
    )
