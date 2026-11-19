import json
from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    content = {
        "clips": [
            {
                "start_sid": 5,
                "end_sid": 9,
                "title": "The ninety day customer loop",
                "hook": "Here is the real problem.",
                "summary": "Most people skip the boring part: talking to customers.",
                "why": "Contrarian claim plus a concrete number.",
                "scores": {"hook": 88, "retention": 81, "emotion": 70, "value": 92, "quotability": 84},
                "risks": ["Setup depends on the previous sentence"],
            },
            {
                "start_sid": 20,
                "end_sid": 24,
                "title": "Four focused hours beat fourteen",
                "hook": "Now, the second thing nobody tells you is about energy.",
                "summary": "Working fewer focused hours shipped more.",
                "why": "Before/after contrast with a specific claim.",
                "scores": {"hook": 74, "retention": 83, "emotion": 66, "value": 78, "quotability": 72},
                "risks": [],
            },
        ]
    }
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps(content)}}],
        "model": body.get("model"),
    }
