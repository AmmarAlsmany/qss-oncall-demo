"""
Service quality analysis using GPT-4o.
Receives the labeled transcript + tone data and returns a structured evaluation.
"""

import json
from groq import Groq


def analyze_conversation(
    transcript: str,
    tone_per_speaker: dict,
    groq_api_key: str,
) -> dict:
    """
    Send transcript and tone data to GPT-4o for analysis.

    Returns a structured dict:
    {
        "employee_speaker": "SPEAKER_00",
        "customer_speaker": "SPEAKER_01",
        "employee_identification_reason": "...",
        "score": 8,
        "verdict": "Good",
        "employee_tone": "Neutral / Professional",
        "customer_tone": "Frustrated",
        "strengths": ["...", "..."],
        "improvements": ["...", "..."],
        "summary": "...",
        "key_moments": [{"moment": "...", "assessment": "positive|negative|neutral"}, ...]
    }
    """
    client = Groq(api_key=groq_api_key)

    # Format tone info for the prompt
    tone_summary = "\n".join(
        f"  - {speaker}: {data['dominant_tone']} (confidence: {data['confidence']})"
        for speaker, data in tone_per_speaker.items()
    )

    system_prompt = """You are an expert Quality Assurance evaluator for airline customer service.
Your job is to analyze recorded conversations between airline employees and customers.

You must:
1. Identify which speaker is the EMPLOYEE and which is the CUSTOMER based on conversational context
   (employees greet first, use professional language, offer solutions, etc.)
2. Evaluate the employee's performance based on:
   - Politeness and professionalism
   - Problem-solving ability
   - Accuracy of information provided
   - Empathy and patience
   - Proper greeting and closing
   - Following up and ensuring customer satisfaction
3. Consider the tone/emotion data provided from audio analysis

Always respond with a single valid JSON object — no extra text, no markdown.

JSON schema:
{
  "employee_speaker": "<SPEAKER_XX>",
  "customer_speaker": "<SPEAKER_XX>",
  "employee_identification_reason": "<brief explanation>",
  "score": <integer 1-10>,
  "verdict": "<Excellent|Good|Needs Improvement|Poor>",
  "employee_tone": "<tone description>",
  "customer_tone": "<tone description>",
  "strengths": ["<point>", ...],
  "improvements": ["<point>", ...],
  "summary": "<2-3 sentence overall summary>",
  "key_moments": [
    {"moment": "<description>", "assessment": "<positive|negative|neutral>"},
    ...
  ]
}

Verdict scale: Excellent=9-10, Good=7-8, Needs Improvement=5-6, Poor=1-4."""

    user_message = f"""Analyze this airline customer service conversation.

=== TRANSCRIPT ===
{transcript}

=== TONE ANALYSIS (from audio) ===
{tone_summary}

Provide your evaluation as a JSON object."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)
    return result
