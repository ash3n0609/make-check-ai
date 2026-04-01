"""
Maker-Checker Demo
==================
Maker  : allenai/OLMo-2-1124-7B-Instruct  (~7B, ungated, Apache 2.0)
Checker: Qwen/Qwen3-4B                    (~4B, ungated)

Models are loaded ONCE into GPU memory via @app.cls + @modal.enter,
then reused across all requests — no per-call weight loading.

Usage
-----
  modal run  path.py                          # CLI test
  modal run  path.py --prompt "your question" # custom prompt
  modal serve path.py                         # live endpoint
"""

import re
import modal
import requests
import json
import os
import time
import threading
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────

app = modal.App("maker-checker-demo")

# Online API Endpoints
KIMI_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.48.0",
        "torch>=2.3.0",
        "accelerate",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
        "requests",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

VOLUME_PATH      = "/root/.cache/huggingface"
MAKER_MODEL_ID   = "allenai/OLMo-2-1124-7B-Instruct"
CHECKER_MODEL_ID = "Qwen/Qwen3-4B"

CHECKER_SYSTEM = """You are a rigorous fact-checker and quality reviewer.

You will receive:
  ORIGINAL PROMPT - what the user asked
  DRAFT RESPONSE  - an answer produced by another AI model

Your job is to:
1. Identify any factual errors, logical gaps, or missing nuance.
2. Note anything that is well-done.
3. Give an overall verdict: PASS, PASS_WITH_NOTES, or FAIL.
4. If FAIL or PASS_WITH_NOTES, provide a concise improved response.

Format your reply EXACTLY using the following XML tags:

<verdict>PASS</verdict>

<issues>
- bullet list of issues, or "None" if PASS
</issues>

<strengths>
- bullet list of strengths
</strengths>

<improved>
revised answer, or "N/A" if PASS
</improved>
"""


# ═══════════════════════════════════════════════════════════
# Maker class — model loaded once, reused across all calls
# ═══════════════════════════════════════════════════════════

SCALEDOWN_SECONDS = 240  # 4 minutes warm window


def _start_idle_countdown(label: str, warm_seconds: int, get_last_active):
    """Daemon thread: prints idle countdown every 30 s to the container log."""
    def _run():
        while True:
            time.sleep(30)
            idle = time.time() - get_last_active()
            remaining = max(0, warm_seconds - idle)
            if remaining > 0:
                print(f"[{label}] idle {idle:.0f}s — shutting down in {remaining:.0f}s if no requests")
            else:
                print(f"[{label}] idle {idle:.0f}s — container scaling down now")
    t = threading.Thread(target=_run, daemon=True)
    t.start()


@app.cls(
    gpu="A100", # Use A100 instead of H100 to increase availability, H100 quota often hit
    image=image,
    volumes={VOLUME_PATH: hf_cache_vol},
    timeout=600,
    scaledown_window=SCALEDOWN_SECONDS,
)
class Maker:
    # ⚠️  Do NOT use modal.parameter here.
    # Parameterised classes get a unique container key per value — meaning
    # Maker(model_id="X") and Maker() are routed to DIFFERENT containers and
    # each cold-starts (reloads weights) independently.
    # Hard-coding the model ID gives Modal a single stable container identity
    # that stays warm between requests.

    @modal.enter()
    def load(self):
        """Called ONCE when the container starts. Weights stay in VRAM."""
        t0 = time.time()
        from transformers import pipeline
        print(f"[Maker] Loading {MAKER_MODEL_ID} into GPU memory...")
        self.pipe = pipeline(
            "text-generation",
            model=MAKER_MODEL_ID,
            device_map="cuda",
            model_kwargs={"dtype": "auto"},
        )
        self._last_active = time.time()
        self.load_time = time.time() - t0
        _start_idle_countdown("Maker", SCALEDOWN_SECONDS, lambda: self._last_active)
        print(f"[Maker] Ready. Will stay warm for {SCALEDOWN_SECONDS}s after last request.")

    @modal.method()
    def generate(self, messages: list) -> dict:
        """
        Accept the full conversation history so the model has context
        from previous turns.  Each element is {"role": ..., "content": ...}.
        """
        self._last_active = time.time()  # reset idle clock
        t0 = time.time()
        result = self.pipe(
            messages,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
        inf_time = time.time() - t0
        self._last_active = time.time()  # reset again after inference
        draft = result[0]["generated_text"][-1]["content"]
        print(f"[Maker] Draft ({len(draft)} chars):\n{draft}\n")
        
        load_time = getattr(self, "load_time", 0.0)
        self.load_time = 0.0
        
        return {
            "draft": draft,
            "load_time": load_time,
            "inference_time": inf_time
        }


# ═══════════════════════════════════════════════════════════
# Checker class — model loaded once, reused across all calls
# ═══════════════════════════════════════════════════════════

@app.cls(
    gpu="A100",
    image=image,
    volumes={VOLUME_PATH: hf_cache_vol},
    timeout=600,
    scaledown_window=SCALEDOWN_SECONDS,
)
class Checker:
    # Same reasoning as Maker — no modal.parameter so the container
    # identity is stable and the warm container is always reused.

    @modal.enter()
    def load(self):
        """Called ONCE when the container starts. Weights stay in VRAM."""
        t0 = time.time()
        from transformers import pipeline
        print(f"[Checker] Loading {CHECKER_MODEL_ID} into GPU memory...")
        self.pipe = pipeline(
            "text-generation",
            model=CHECKER_MODEL_ID,
            device_map="cuda",
            model_kwargs={"dtype": "auto"},
        )
        self._last_active = time.time()
        self.load_time = time.time() - t0
        _start_idle_countdown("Checker", SCALEDOWN_SECONDS, lambda: self._last_active)
        print(f"[Checker] Ready. Will stay warm for {SCALEDOWN_SECONDS}s after last request.")

    @modal.method()
    def review(self, original_prompt: str, draft: str) -> dict:
        self._last_active = time.time()  # reset idle clock
        t0 = time.time()
        user_message = (
            f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
            f"DRAFT RESPONSE:\n{draft}"
        )
        messages = [
            {"role": "system", "content": CHECKER_SYSTEM},
            {"role": "user",   "content": user_message},
        ]
        result = self.pipe(messages, max_new_tokens=2048, do_sample=False)
        inf_time = time.time() - t0
        self._last_active = time.time()  # reset again after inference
        review = result[0]["generated_text"][-1]["content"]
        print(f"[Checker] Review:\n{review}\n")
        
        load_time = getattr(self, "load_time", 0.0)
        self.load_time = 0.0
        
        return {
            "review": review,
            "load_time": load_time,
            "inference_time": inf_time
        }


# ═══════════════════════════════════════════════════════════
# Response parser
# ═══════════════════════════════════════════════════════════

def strip_think(text: str) -> str:
    """Remove <think>…</think> chain-of-thought blocks from any model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_checker_output(review: str) -> dict:
    """Strip <think> block, parse XML tags."""
    # Extract chain-of-thought content before stripping it
    think_match = re.search(r"<think>(.*?)</think>", review, flags=re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""

    clean = strip_think(review)

    def extract(tag: str) -> str:
        # Match <tag>content</tag> across multiple lines
        m = re.search(rf"<{tag}>(.*?)</{tag}>", clean, re.DOTALL | re.IGNORECASE)
        # If the model forgot the closing tag, try to match to the next opening tag or end of string
        if not m:
             m = re.search(rf"<{tag}>(.*?)(?:<\w+>|$)", clean, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return {
        "verdict":   extract("verdict"),
        "issues":    extract("issues"),
        "strengths": extract("strengths"),
        "improved":  extract("improved"),
        "think":     think_content,
        "raw":       clean,
    }

# ═══════════════════════════════════════════════════════════
# Online Model Client
# ═══════════════════════════════════════════════════════════

def call_online_model(model_id: str, messages: list) -> str:
    """
    Synchronously call DeepSeek or Kimi (NVIDIA) API.
    Used within the Modal environment.
    """
    if model_id.startswith("online/deepseek"):
        url = DEEPSEEK_ENDPOINT
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        actual_model = model_id.split("/")[-1] # e.g. "deepseek-chat"
    elif model_id.startswith("online/kimi"):
        url = KIMI_ENDPOINT
        api_key = os.environ.get("KIMI_API_KEY")
        actual_model = "moonshotai/kimi-k2.5"
    elif model_id.startswith("online/gemini"):
        url = GEMINI_ENDPOINT
        api_key = os.environ.get("GEMINI_API_KEY")
        actual_model = model_id.split("/")[-1] # e.g. "gemini-3.1-flash-image-preview"
    else:
        raise ValueError(f"Unknown online model: {model_id}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "model": actual_model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": False
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        error_msg = f"API Error ({url}): {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f" | Body: {e.response.text}"
        raise RuntimeError(error_msg)


# ═══════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════

@app.function(image=image, secrets=[modal.Secret.from_dotenv()], timeout=1200)
async def run_pipeline(prompt: str, maker_model_id: str = MAKER_MODEL_ID, checker_model_id: str = CHECKER_MODEL_ID) -> dict:
    """Call Maker then Checker via their warm class instances (CLI helper)."""
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)

    maker   = Maker()
    checker = Checker()

    # Single-turn: wrap prompt as a one-message history
    messages = [{"role": "user", "content": prompt}]
    maker_res = await maker.generate.remote.aio(messages)
    draft = maker_res["draft"]
    checker_res = await checker.review.remote.aio(prompt, draft)
    review = checker_res["review"]

    print(f"\n[Maker  - {maker_model_id}]\n{draft}\n")
    print(f"\n[Checker - {checker_model_id}]\n{review}\n")

    return {"prompt": prompt, "draft": draft, "review": review}


# ═══════════════════════════════════════════════════════════
# FastAPI web endpoint  (modal serve)
# ═══════════════════════════════════════════════════════════

from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

web_app = modal.App("maker-checker-web")
from fastapi import FastAPI
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.function(image=image, timeout=1200)
@modal.wsgi_app()
def fastapi_wrapper():
    return fastapi_app

@app.function(image=image, secrets=[modal.Secret.from_dotenv()], timeout=1200)
@modal.fastapi_endpoint(method="POST", label="maker-checker")
async def web_check(body: dict):
    """
    POST /maker-checker
    Body:    {"messages": [{"role":"user","content":"..."},...], "maker": "...", "checker": "..."}
             OR legacy: {"prompt": "...", "maker": "...", "checker": "..."}
    Returns: NDJSON stream

    The `messages` list is the full conversation history (user + assistant turns).
    The Maker receives the entire history so it can answer in context.
    The Checker is always single-turn — it only sees the current prompt + draft.
    """
    # Accept either the full messages list or a bare prompt string (backwards compat)
    messages: list = body.get("messages")
    if not messages:
        prompt = body.get("prompt", "").strip()
        if not prompt:
            return {"error": "Provide 'messages' (list) or 'prompt' (string)."}
        messages = [{"role": "user", "content": prompt}]
    else:
        # Derive the current prompt from the last user message for the Checker
        prompt = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            ""
        )

    maker_model_id = body.get("maker") or MAKER_MODEL_ID
    checker_model_id = body.get("checker") or CHECKER_MODEL_ID

    async def generate():
        try:
            # Yield an initial comment to flush reverse-proxy buffers
            yield f": {' ' * 2048}\n\n"

            # ── MAKER ──────────────────────────────────────────────────
            # Pass the full conversation history so the model can answer in context.
            if maker_model_id.startswith("online/"):
                t0 = time.time()
                draft = call_online_model(maker_model_id, messages)
                m_load_time = 0.0
                m_inf_time = time.time() - t0
            else:
                maker = Maker()
                maker_res = await maker.generate.remote.aio(messages)
                draft = maker_res["draft"]
                m_load_time = maker_res["load_time"]
                m_inf_time = maker_res["inference_time"]

            # Strip chain-of-thought tokens before rendering or passing to checker
            # (works for both on-prem models like Qwen3 and online models like Gemini)
            draft = strip_think(draft)

            yield f"data: {json.dumps({'step': 'maker', 'draft': draft, 'model': maker_model_id})}\n\n"

            # ── CHECKER ────────────────────────────────────────────────
            # The Checker is always single-turn: it only reviews the current (clean) draft.
            checker_user_msg = (
                f"ORIGINAL PROMPT:\n{prompt}\n\n"
                f"DRAFT RESPONSE:\n{draft}"
            )
            checker_messages = [
                {"role": "system", "content": CHECKER_SYSTEM},
                {"role": "user",   "content": checker_user_msg},
            ]
            if checker_model_id.startswith("online/"):
                t0 = time.time()
                review = call_online_model(checker_model_id, checker_messages)
                c_load_time = 0.0
                c_inf_time = time.time() - t0
            else:
                checker = Checker()
                checker_res = await checker.review.remote.aio(prompt, draft)
                review = checker_res["review"]
                c_load_time = checker_res["load_time"]
                c_inf_time = checker_res["inference_time"]

            parsed = parse_checker_output(review)
            yield f"data: {json.dumps({'step': 'checker', 'review': parsed, 'model': checker_model_id})}\n\n"

            yield f"data: {json.dumps({'step': 'metrics', 'maker_load_time': m_load_time, 'maker_inference_time': m_inf_time, 'checker_load_time': c_load_time, 'checker_inference_time': c_inf_time, 'closing_time': SCALEDOWN_SECONDS})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════
# Local entrypoint  (modal run maker_checker_demo.py)
# ═══════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(prompt: str = "In two sentences, what is the capital of Australia and why is it not Sydney?"):
    import asyncio
    result  = asyncio.get_event_loop().run_until_complete(run_pipeline.remote.aio(prompt))
    checker = parse_checker_output(result["review"])

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(f"\nPrompt  : {result['prompt']}")
    print(f"\nMaker ({MAKER_MODEL_ID}):\n{result['draft']}")
    print(f"\nVerdict : {checker['verdict']}")
    print(f"\nChecker ({CHECKER_MODEL_ID}):\n{checker['raw']}")