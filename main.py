"""
AI Teamwork Voice Interviewer
Run:  uvicorn main:app --reload
Open: http://localhost:8000
"""
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from groq import Groq

app = FastAPI(title="AI Teamwork Interviewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

CHAT_SYSTEM = """You are Priya, a warm and friendly AI interviewer. Your ONLY task: assess TEAMWORK skills through a short voice conversation.

ALWAYS return ONLY valid JSON. No other text. Format:
{"message": "your response here", "interview_complete": false}

To end the interview after 4-5 good exchanges:
{"message": "closing message", "interview_complete": true}

CANDIDATE PROFILE: Fresh graduate or under 2 years experience from Tier 2 or Tier 3 city in India.

RULES:
- Be warm, encouraging, patient
- Keep each response to 2-3 sentences MAX (this is a voice interview)
- Use simple English — no corporate jargon
- Ask about: college group projects, hostel life, sports teams, college clubs, NCC/NSS, community events, family — NOT corporate jobs
- Never repeat a question you already asked
- After 4-5 meaningful exchanges, wrap up and set interview_complete: true

INTERVIEW FLOW:
1. Greet warmly, introduce yourself as Priya, ask for their name
2. Use their name — ask about a team experience from college or life
3. Probe HOW they worked together: did they share info, check in on teammates, take on extra?
4. Ask about a disagreement or conflict in that team — what happened, what did they do?
5. Ask if they ever helped someone who was stuck, or stepped up when the group was blocked
6. Close warmly using their name, set interview_complete: true

THE 12 TEAMWORK BEHAVIORS TO LISTEN FOR (adapted for college context):
1. Proactive info sharing — did they keep teammates updated without being asked?
2. Saying they're stuck quickly — did they ask for help early instead of suffering silently?
3. Quality handoffs — when passing work to a teammate, did they explain it clearly?
4. Active listening — did they really hear others, ask questions, build on what was said?
5. Productive disagreement — did they push back on ideas respectfully with reasons? (HIGHEST SIGNAL)
6. Giving credit precisely — did they name specific teammates for their contributions?
7. Receiving feedback without defending — did they accept input gracefully?
8. Unblocking others — did they notice a teammate was stuck and help them? (HIGH SIGNAL)
9. Reliable delivery — did they do what they promised, on time, so the team could depend on them?
10. Documenting decisions — did they write things down, share notes, keep the group aligned?
11. Working across functions — did they coordinate with people outside their immediate group?
12. Covering without being asked — did they pick up work that wasn't theirs when the team needed it?

PRIORITY PROBES (fit these naturally into the 4-5 exchanges):
- #5 Productive disagreement: "Did anyone in the group disagree? How did that go?"
- #8 Unblocking others: "Was there a moment someone on your team was stuck? What did you do?"
- #1 Proactive info sharing: "How did you keep everyone in the loop?"

RETURN JSON ONLY. Always."""

ASSESS_SYSTEM = """You are an expert assessor evaluating TEAMWORK for entry-level candidates from Tier 2/3 Indian cities.

Your job is to score ACCURATELY — not to be conservative. If the evidence is there, give the high score. Underscore only if evidence is genuinely absent.

THE 12 BEHAVIORS (look for ANY evidence, even brief mentions):
1. Proactive info sharing — updated teammates without being asked (e.g. shared doc, sent updates)
2. Saying stuck quickly — asked for help early instead of struggling silently
3. Quality handoffs — explained context when passing work to others
4. Active listening — heard others out, asked questions, built on ideas
5. Productive disagreement — challenged ideas respectfully with reasoning, not just emotion (HIGHEST SIGNAL)
6. Giving credit precisely — named a specific person for a specific contribution
7. Receiving feedback without defending — accepted input gracefully
8. Unblocking others — proactively helped a stuck teammate without being asked (HIGH SIGNAL)
9. Reliable delivery — did what they promised on time
10. Documenting decisions — wrote things down, kept group aligned (e.g. Notion, shared doc)
11. Working across functions — coordinated outside their immediate group
12. Covering without being asked — took on extra work when team needed it

SCORING RUBRIC — read carefully:
1 - No teamwork at all: only solo work mentioned, zero behaviors observed
2 - Passive: was in a group but only followed instructions, no behaviors demonstrated
3 - Active: 2-3 behaviors with some specific detail (decent but unremarkable)
4 - Strong: 4-5 behaviors with specific examples, INCLUDING #5 OR #8
5 - Champion: 5+ behaviors with vivid specific examples, INCLUDING BOTH #5 AND #8

CALIBRATION EXAMPLES:
- Score 3: "We divided work and met weekly. I did my part on time." → reliable delivery + basic coordination only
- Score 4: "I helped a stuck teammate (unblocking) and we had a disagreement about tech stack that I resolved calmly (productive disagreement)" → both high-signal behaviors present
- Score 5: Candidate demonstrates productive disagreement AND unblocking others AND proactive info sharing AND giving credit AND at least 2 more behaviors, all with specific named people and concrete situations

IMPORTANT: Count behaviors generously. A candidate who mentions making a comparison doc (proactive sharing + documenting), helped a stuck teammate (unblocking + covering), named teammates by name (giving credit), and handled a tech disagreement (productive disagreement) has demonstrated 5+ behaviors and scores a 5.

RETURN ONLY this JSON (no other text):
{
  "name": "<candidate's first name from the transcript, or 'Candidate' if not mentioned>",
  "score": <1-5>,
  "label": "<one of: Needs Development | Basic Team Member | Active Collaborator | Strong Team Player | Team Champion>",
  "summary": "<2-3 sentences describing what you specifically observed about their teamwork>",
  "strengths": ["specific strength tied to one of the 12 behaviors", "another specific strength"],
  "areas_to_grow": ["specific behavior from the 12 they showed no evidence of"],
  "encouragement": "<warm, specific 1-2 sentence message addressing them by name>",
  "behaviors_observed": [
    {"behavior": "<behavior name>", "evidence": "<what they said that showed this>"}
  ]
}"""


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```\s*", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Brace-matching fallback for finding outermost JSON object
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except Exception:
                        break
    return {"message": text.strip(), "interview_complete": False}


class Message(BaseModel):
    role: str
    content: str = ""


LANG_NAMES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "bn-IN": "Bengali",
    "mr-IN": "Marathi",
    "gu-IN": "Gujarati",
    "pa-IN": "Punjabi",
}

class ChatReq(BaseModel):
    messages: List[Message]
    elapsed: int = 0
    language: str = "en-IN"


class AssessReq(BaseModel):
    messages: List[Message]


@app.post("/chat")
async def chat(req: ChatReq):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages if m.content.strip()]

    # Conversation must start with a user message
    if not msgs or msgs[0]["role"] == "assistant":
        msgs.insert(0, {"role": "user", "content": "Please start the interview."})

    lang_name = LANG_NAMES.get(req.language, "English")
    system = CHAT_SYSTEM
    if req.language != "en-IN":
        system += f"\n\nLANGUAGE: Conduct the ENTIRE interview in {lang_name}. All your responses must be in {lang_name} only."
    if req.elapsed >= 150:
        system += "\n\nTIME NOTE: Interview has reached 2.5 minutes. Please wrap up warmly NOW and set interview_complete: true."

    resp = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=600,
        messages=[{"role": "system", "content": system}, *msgs],
    )
    raw = resp.choices[0].message.content
    print("RAW CHAT:", raw[:200])
    return parse_json(raw)


@app.post("/assess")
async def assess(req: AssessReq):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    # Build readable transcript, filtering out internal kickoff messages
    INTERNAL = {"Please start the interview.", "Begin."}
    transcript_lines = [
        f"{'PRIYA' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in msgs
        if m.get("content") not in INTERNAL
    ]
    conversation = "\n".join(transcript_lines)

    resp = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=900,
        messages=[
            {"role": "system", "content": ASSESS_SYSTEM},
            {"role": "user", "content": f"Interview transcript:\n\n{conversation}\n\nProvide the teamwork assessment as JSON."},
        ],
    )

    raw = resp.choices[0].message.content or ""
    print("RAW ASSESS:", raw[:500])
    result = parse_json(raw)
    if "score" not in result:
        return {
            "score": 3,
            "label": "Active Collaborator",
            "summary": "Interview completed successfully.",
            "strengths": ["Participated in the interview"],
            "areas_to_grow": ["Practice describing specific teamwork examples"],
            "encouragement": "Thank you for completing the interview! Every experience helps you grow.",
            "behaviors_observed": [],
        }
    return result


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
