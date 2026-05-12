"""
PARSON LLM HELPER
Uses Gemini API for:
1. Natural recommendation explanations
2. Intelligent fallback responses
"""

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)
genai.configure(
    api_key=GEMINI_API_KEY
)

model = None
def get_gemini_model():
    global model

    if model is None:
        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    return model

def generate_book_explanations(query, books):

    books_text = ""

    for i, book in enumerate(books, start=1):

        books_text += f"""
BOOK {i}
TITLE: {book['title']}
SYNOPSIS: {str(book.get('synopsis', ''))[:1000]}

"""

    prompt = f"""
You are an intelligent book recommendation assistant.

USER SEARCH:
{query}

BOOKS:
{books_text}

TASK:
Generate ONE short personalized explanation for EACH book.

IMPORTANT RULES:
- Return ONLY valid JSON
- No markdown
- No code blocks
- Keep explanations under 2 sentences
- Make explanations natural and human
- Mention themes from the query
- Mention relevant story/world elements
- Avoid generic wording
- Do NOT mention AI or ranking systems

OUTPUT FORMAT:
{{
  "1": "explanation...",
  "2": "explanation..."
}}
"""

    try:

        response = get_gemini_model().generate_content(prompt)

        text = response.text.strip()
        print("\nGEMINI RAW RESPONSE:\n")
        print(text)

        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            text = text[start:end]
        parsed = json.loads(text)

        return parsed

    except Exception as e:

        print("Batch Gemini Error:", e)

        fallback = {}

        for i in range(len(books)):
            fallback[str(i + 1)] = (
                "A potentially relevant recommendation based on your search themes."
                "based on its themes and story."
            )

        return fallback

def generate_fallback_response(
    query
):
    """
    Generates intelligent fallback response
    when recommendation confidence is weak.
    """

    prompt = f"""
A user searched for:

{query}

But PARSON has weak dataset coverage
for this topic.

Generate a polite helpful response.

RULES:
- Friendly tone
- Professional
- Short
- Mention future improvements naturally
- Suggest refining the query if useful
"""

    try:

        response = get_gemini_model().generate_content(
            prompt
        )

        return response.text.strip()

    except Exception as e:
        print("Gemini Error:", e)

        return (
            "A potentially relevant recommendation based on your search themes."
        )