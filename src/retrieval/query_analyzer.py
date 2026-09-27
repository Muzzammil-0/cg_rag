"""
LLM-based act and section extraction from a user query.

Used as a fallback tier in retriever.py when the act classifier
is not confident enough to route.

Example:
    extract_act_section("What are the penalties for carrying an unlicensed gun?")
    -> ("Arms Act", "25")
"""

import os
import re
import json
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


# ------------------------------------------------------------------
# Canonical act names → source fragments in the retriever
# ------------------------------------------------------------------
ACT_NAME_TO_FRAGMENTS = {
    "IPC": ["IPC_eng"],
    "Indian Penal Code": ["IPC_eng"],
    "BNS": ["BNS"],
    "Bharatiya Nyaya Sanhita": ["BNS"],
    "BNSS": ["Bharatiya_Nagarik"],
    "Bharatiya Nagarik Suraksha Sanhita": ["Bharatiya_Nagarik"],
    "BSA": ["BSA"],
    "Bharatiya Sakshya Adhiniyam": ["BSA"],
    "CrPC": ["criminal_procedure"],
    "Criminal Procedure Code": ["criminal_procedure"],
    "CPC": ["civil_procedure"],
    "Civil Procedure Code": ["civil_procedure"],
    "Indian Evidence Act": ["iea_1872"],
    "Evidence Act": ["iea_1872"],
    "Constitution": ["constitution"],
    "Hindu Marriage Act": ["Hindu_Marriage"],
    "Special Marriage Act": ["Special_Marriage"],
    "Dowry Prohibition Act": ["Dowry_Prohibition"],
    "Domestic Violence Act": ["Domestic_Violence"],
    "Protection of Women from Domestic Violence Act": ["Domestic_Violence"],
    "Prohibition of Child Marriage Act": ["Child_Marriage"],
    "Child Marriage Act": ["Child_Marriage"],
    "Negotiable Instruments Act": ["Negotiable"],
    "NI Act": ["Negotiable"],
    "Motor Vehicles Act": ["Motor"],
    "MV Act": ["Motor"],
    "NDPS Act": ["Narcotic"],
    "POCSO Act": ["pocso"],
    "Protection of Children from Sexual Offences Act": ["pocso"],
    "Arms Act": ["ArmsAct"],
    "Prevention of Corruption Act": ["Corruption"],
    "PC Act": ["Corruption"],
    "SC/ST (PoA) Act": ["Scheduled"],
    "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act": ["Scheduled"],
    "Juvenile Justice Act": ["Juvenile"],
    "Maintenance and Welfare of Parents Act": ["Maintenance_and_Welfare"],
    "PESA": ["pesa"],
    "Panchayats (Extension to Scheduled Areas) Act": ["pesa"],
    "Forest Rights Act Rules": ["FRA"],
    "CG Land Revenue Code": ["Land_Revenue"],
    "Chhattisgarh Land Revenue Code": ["Land_Revenue"],
    "CG Excise Act": ["Excise"],
    "Chhattisgarh Excise Act": ["Excise"],
    "CG Rent Control Act": ["Rent_Control"],
    "Chhattisgarh Rent Control Act": ["Rent_Control"],
    "CG Municipalities Act": ["Municipalities"],
    "Chhattisgarh Municipalities Act": ["Municipalities"],
    "CG Panchayat Raj Act": ["Panchayat"],
    "Chhattisgarh Panchayat Raj Act": ["Panchayat"],
    "CG Tendu Patta Act": ["tendu"],
    "Chhattisgarh Tendu Patta Act": ["tendu"],
    "CG Vanopaj Act": ["vanopaj"],
    "Chhattisgarh Vanopaj Act": ["vanopaj"],
    "CG Lok Sewa Guarantee Act": ["Lok-Sewa"],
    "Chhattisgarh Lok Sewa Guarantee Act": ["Lok-Sewa"],
    "CG Special Courts Act": ["Special Courts"],
    "CG Cattle Preservation Rules": ["Cattle"],
}

VALID_ACT_NAMES = sorted(ACT_NAME_TO_FRAGMENTS.keys())


# ------------------------------------------------------------------
# Prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a legal query analyzer for Indian and Chhattisgarh law.

Given a user query, identify:
1. The act it is most likely about (pick the exact name from the list below)
2. The specific section number if you can identify it with confidence, or null

Valid act names (use EXACTLY as written):
""" + "\n".join(f"- {name}" for name in VALID_ACT_NAMES) + """

Rules:
- If you are not confident about the act, set "act" to null.
- If you are confident about the act but not the section, set "section" to null.
- Do not invent act names outside the list.
- Return ONLY a JSON object with keys "act" and "section". No thinking, no markdown fences.

Examples:
Query: "What is the punishment for murder under BNS?"
{"act": "BNS", "section": "103"}

Query: "How do I get a protection order against my abusive husband?"
{"act": "Domestic Violence Act", "section": "18"}

Query: "What happens if a cheque gets dishonoured?"
{"act": "Negotiable Instruments Act", "section": "138"}

Query: "What is the tax rate on cryptocurrency?"
{"act": null, "section": null}
"""


# ------------------------------------------------------------------
# JSON extraction
# ------------------------------------------------------------------
def _extract_json(text):
    text = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.DOTALL)
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```', '', text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ------------------------------------------------------------------
# Public: extract act + section from query
# ------------------------------------------------------------------
@lru_cache(maxsize=512)
def extract_act_section(query, model="openai/gpt-oss-20b", timeout=10):
    """
    Returns (act_name, section) where either may be None.

    Cached per (query, model) so repeated queries skip the LLM.
    """
    if not query or len(query.strip()) < 5:
        return (None, None)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.choices[0].message.content
        parsed = _extract_json(content)

        if not parsed:
            return (None, None)

        act = parsed.get("act")
        section = parsed.get("section")

        # Validate act against known names (case-insensitive match)
        if act:
            if act not in ACT_NAME_TO_FRAGMENTS:
                lower = {k.lower(): k for k in ACT_NAME_TO_FRAGMENTS}
                act = lower.get(str(act).lower())

        # Validate section is a short identifier
        if section is not None:
            section = str(section).strip()
            if not re.match(r'^[0-9]+[A-Za-z]?(\([0-9]+\))?$', section):
                section = None

        return (act, section)

    except Exception as e:
        print(f"[query_analyzer] {type(e).__name__}: {str(e)[:120]}")
        return (None, None)


def get_fragments_for_act(act_name):
    """Return source fragments for an act name, or None."""
    if not act_name:
        return None
    return ACT_NAME_TO_FRAGMENTS.get(act_name)

# ------------------------------------------------------------------
# Reverse map: source fragment → act name(s)
# ------------------------------------------------------------------
FRAGMENT_TO_ACT_NAMES = {}
for _act, _frags in ACT_NAME_TO_FRAGMENTS.items():
    for _f in _frags:
        FRAGMENT_TO_ACT_NAMES.setdefault(_f, []).append(_act)


SECTION_REFINE_PROMPT = """You are a legal section identifier for Indian and Chhattisgarh law.

Given a user query and the specific act that applies, identify the SINGLE section number of that act that most directly answers the query.

Rules:
- Return the section number as a plain string like "25" or "138" or "498A".
- If you are not confident, set "section" to null.
- Only return a section that exists in this act.
- Return ONLY a JSON object with key "section". No thinking, no markdown fences.

Examples:
Query: "What are the penalties for carrying an unlicensed gun?"
Act: Arms Act
{"section": "25"}

Query: "Punishment for dishonour of cheque"
Act: Negotiable Instruments Act
{"section": "138"}

Query: "How do I file for divorce?"
Act: Hindu Marriage Act
{"section": "13"}

Query: "What is the process for something I don't know about?"
Act: Motor Vehicles Act
{"section": null}
"""


@lru_cache(maxsize=512)
def refine_section_with_llm(query, act_name, model="openai/gpt-oss-20b"):
    """
    Ask the LLM which section of a GIVEN act answers the query.
    Returns section string or None.
    """
    if not query or not act_name:
        return None

    try:
        client = _get_client()
        user_prompt = f"Query: {query}\nAct: {act_name}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SECTION_REFINE_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content
        parsed = _extract_json(content)

        if not parsed:
            return None

        section = parsed.get("section")
        if section is None:
            return None

        section = str(section).strip()
        if not re.match(r'^[0-9]+[A-Za-z]?(\([0-9]+\))?$', section):
            return None

        return section

    except Exception as e:
        print(f"[query_analyzer.refine] {type(e).__name__}: {str(e)[:120]}")
        return None


def act_name_for_fragment(fragment):
    """Return the primary act name for a source fragment, or None."""
    names = FRAGMENT_TO_ACT_NAMES.get(fragment)
    return names[0] if names else None