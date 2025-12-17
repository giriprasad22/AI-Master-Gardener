import os
from pathlib import Path
from typing import List, Dict

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from supabase import create_client
from ollama import Client as OllamaClient
import numpy as np
import hashlib
import csv
from math import isfinite
import io
from PIL import Image
import re
import json
import base64

try:
    import tensorflow as tf  # heavy, optional until image detection/disease used
except Exception:
    tf = None


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / "kb" / ".env")  # reuse KB env

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))


# ========== Organic & Sustainable Agriculture Constants ==========
ORGANIC_TREATMENT_PRIORITY = """
🌱 SUSTAINABLE TREATMENT PHILOSOPHY:

Priority 1: PREVENTION (Best approach!)
- Companion planting for natural pest/disease resistance
- Crop rotation to break disease cycles
- Proper spacing, ventilation, and drainage
- Healthy soil = healthy plants

Priority 2: NATURAL/ORGANIC REMEDIES (Primary solution)
- Neem oil, garlic spray, insecticidal soap
- Beneficial insects (ladybugs, lacewings, parasitic wasps)
- Compost tea, bacterial biocontrols (Bacillus)
- Organic fungicides (copper, sulfur, baking soda)

Priority 3: CULTURAL PRACTICES (Long-term fix)
- Remove and destroy infected plant parts
- Improve soil drainage and air circulation
- Adjust watering (avoid wetting foliage)
- Mulching, balanced organic fertilization

Priority 4: CHEMICAL CONTROLS (LAST RESORT)
- Only if disease is severe and spreading rapidly
- Use least toxic, targeted options
- Follow all safety protocols and PPE requirements
- Consider environmental impact
"""

# Disease-specific companion planting recommendations
COMPANION_DISEASE_SOLUTIONS = {
    "bacterial_spot": {
        "companions": ["basil", "marigold", "garlic", "chives"],
        "benefits": "Basil repels aphids that spread bacteria; garlic has natural antibacterial properties",
        "spacing": "Plant companions 12-18 inches away"
    },
    "early_blight": {
        "companions": ["marigold", "borage", "nasturtium", "garlic"],
        "benefits": "Marigolds deter nematodes; nasturtiums trap aphids; garlic boosts plant immunity",
        "spacing": "Interplant throughout garden bed"
    },
    "late_blight": {
        "companions": ["marigold", "chamomile", "garlic", "onion"],
        "benefits": "Chamomile strengthens plant immunity; alliums have antifungal properties",
        "spacing": "Border planting recommended"
    },
    "leaf_mold": {
        "companions": ["basil", "parsley", "dill", "marigold"],
        "benefits": "Herbs improve air circulation; marigolds reduce fungal spores",
        "spacing": "Plant in alternating rows"
    },
    "mosaic_virus": {
        "companions": ["marigold", "nasturtium", "radish", "tansy"],
        "benefits": "Nasturtiums trap aphid vectors; tansy repels many insects",
        "spacing": "Companion border around affected area"
    },
    "septoria_leaf_spot": {
        "companions": ["basil", "oregano", "thyme", "marigold"],
        "benefits": "Aromatic herbs confuse pests; improve air circulation",
        "spacing": "Interplant or border planting"
    },
    "spider_mites": {
        "companions": ["coriander", "dill", "garlic", "onion", "chives"],
        "benefits": "Attract predatory insects; strong scents repel mites",
        "spacing": "Plant throughout garden"
    },
    "target_spot": {
        "companions": ["marigold", "calendula", "garlic", "chives"],
        "benefits": "Marigolds release compounds that suppress fungal growth",
        "spacing": "Interplant 12 inches apart"
    },
    "yellow_leaf_curl": {
        "companions": ["marigold", "nasturtium", "basil", "mint"],
        "benefits": "Repel whiteflies (virus vector); trap crop strategy",
        "spacing": "Dense companion planting around tomatoes"
    },
    "healthy": {
        "companions": ["basil", "marigold", "parsley", "carrots", "onion"],
        "benefits": "Preventive companions boost overall plant health and pest resistance",
        "spacing": "Standard companion spacing"
    }
}

# Organic treatment database
ORGANIC_TREATMENTS = {
    "fungal": {
        "primary": [
            {"name": "Neem Oil Spray", "recipe": "2 tsp neem oil + 1 tsp mild soap per liter water", "frequency": "Every 7-14 days", "timing": "Early morning or evening"},
            {"name": "Baking Soda Solution", "recipe": "1 tbsp baking soda + 1 tbsp oil + 1 tsp soap per liter", "frequency": "Weekly", "timing": "Avoid hot sun"},
            {"name": "Copper Fungicide", "recipe": "Follow organic product label", "frequency": "Every 7-10 days", "timing": "Preventive application"}
        ],
        "secondary": [
            {"name": "Compost Tea", "recipe": "Brew mature compost in water 24-48 hours", "frequency": "Bi-weekly foliar spray", "timing": "Morning application"},
            {"name": "Garlic Spray", "recipe": "Blend 1 bulb garlic + 1L water, strain", "frequency": "Every 3-5 days", "timing": "Evening application"}
        ]
    },
    "bacterial": {
        "primary": [
            {"name": "Copper-based Spray", "recipe": "Organic copper product per label", "frequency": "Every 5-7 days", "timing": "Preventive, before rain"},
            {"name": "Hydrogen Peroxide", "recipe": "1 part 3% H2O2 + 10 parts water", "frequency": "Weekly", "timing": "Avoid direct sun"},
            {"name": "Remove Infected Parts", "recipe": "Sterilize tools with 70% alcohol", "frequency": "Immediately upon detection", "timing": "Dry conditions"}
        ],
        "secondary": [
            {"name": "Garlic & Chili Spray", "recipe": "5 cloves garlic + 2 chili + 1L water, strain", "frequency": "Every 3-4 days", "timing": "Evening"},
            {"name": "Baking Soda Spray", "recipe": "1 tsp baking soda per liter water", "frequency": "Twice weekly", "timing": "Morning"}
        ]
    },
    "viral": {
        "primary": [
            {"name": "Vector Control", "recipe": "Neem oil spray for aphids/whiteflies", "frequency": "Every 5-7 days", "timing": "Early detection crucial"},
            {"name": "Remove Infected Plants", "recipe": "Dig up and destroy (don't compost)", "frequency": "Immediately", "timing": "Prevent spread"},
            {"name": "Reflective Mulch", "recipe": "Silver/aluminum reflective mulch", "frequency": "Season-long", "timing": "Installation at planting"}
        ],
        "secondary": [
            {"name": "Milk Spray", "recipe": "1 part milk + 9 parts water", "frequency": "Weekly", "timing": "May reduce virus activity"},
            {"name": "Companion Barriers", "recipe": "Plant marigold/nasturtium borders", "frequency": "Season-long", "timing": "Trap crop strategy"}
        ]
    },
    "pest": {
        "primary": [
            {"name": "Neem Oil", "recipe": "2 tsp neem oil + 1 tsp soap per liter", "frequency": "Every 7 days", "timing": "Evening application"},
            {"name": "Insecticidal Soap", "recipe": "2 tbsp liquid soap per liter water", "frequency": "Every 3-5 days", "timing": "Direct spray on pests"},
            {"name": "Beneficial Insects", "recipe": "Release ladybugs, lacewings, parasitic wasps", "frequency": "As needed", "timing": "Early season best"}
        ],
        "secondary": [
            {"name": "Diatomaceous Earth", "recipe": "Food-grade DE, dust on plants", "frequency": "After rain/dew", "timing": "Dry conditions"},
            {"name": "Hot Pepper Spray", "recipe": "2 tbsp cayenne + 1 tsp soap per liter", "frequency": "Weekly", "timing": "Repels many pests"}
        ]
    }
}


def _fake_embed(text: str, dim: int = 768) -> List[float]:
    seed = int(hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v.tolist()


def embed_query(text: str) -> List[float]:
    # Default embedding model aligned with available local models
    model = os.getenv("EMBEDDING_MODEL", "gemma3:latest")
    if os.getenv("FAKE_EMBEDDINGS", "0") == "1":
        return _fake_embed(text)
    ollama = OllamaClient()  # local client, no URL
    resp = ollama.embeddings(model=model, prompt=text)
    emb = resp.get("embedding")
    if not isinstance(emb, list):
        raise RuntimeError("Ollama embeddings response missing 'embedding' list")
    return emb


# ========== Agent planner and memory ==========
MEM_DIR = BASE_DIR / "memory"
MEM_DIR.mkdir(exist_ok=True)


def _load_thread_state(thread_id: str) -> Dict:
    if not thread_id:
        return {"summary": "", "facts": {}, "tool_traces": [], "messages": []}
    p = MEM_DIR / f"{thread_id}.json"
    if not p.exists():
        return {"summary": "", "facts": {}, "tool_traces": [], "messages": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"summary": "", "facts": {}, "tool_traces": [], "messages": []}


def _save_thread_state(thread_id: str, state: Dict):
    if not thread_id:
        return
    p = MEM_DIR / f"{thread_id}.json"
    try:
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _select_best_model(oc: OllamaClient, has_image: bool = False, needs_reasoning: bool = False) -> str:
    """Select the single supported model (gemma3) if installed.

    Policy:
    - Always prefer gemma3 for all tasks (planning, search synthesis, chat).
    - If gemma3 isn't available locally, raise a clear error.
    """
    try:
        available = [m.get("model") for m in (oc.list().get("models") or [])]
        if not available:
            raise RuntimeError("No Ollama models available")
        # Find gemma3 among installed models
        for m in available:
            if isinstance(m, str) and "gemma3" in m.lower():
                return m
        raise RuntimeError("Required model 'gemma3' not found in Ollama. Please pull it (e.g., gemma3:latest).")
    except Exception as e:
        raise RuntimeError(f"Failed to select model: {e}")


AGENT_SYSTEM_PROMPT = """You are an advanced AI agricultural assistant with deep expertise in farming, plant care, and agronomy.

🎯 MISSION: Help farmers and agricultural professionals by intelligently routing their requests to the right tool or providing direct assistance.

📋 AVAILABLE TOOLS:
1. detect_crop - Identifies crop/plant species from images
2. detect_disease - Diagnoses plant diseases from images  
3. companions - Finds beneficial/harmful companion plants
4. yield - Predicts crop yield based on conditions
5. search - Searches knowledge base for agricultural information
6. reply - Direct conversational response (for greetings, simple questions)

🧠 INTENT CLASSIFICATION SYSTEM:

Your PRIMARY task is to classify user intent into ONE of these 5 categories:

**CATEGORY 1: crop_detection**
- User wants to IDENTIFY what crop/plant is in an image
- Keywords: "what is this", "identify this", "what crop", "what plant", "which crop"
- Requirements: Image MUST be present + explicit identification request
- Action: "detect_crop"

**CATEGORY 2: disease_detection**  
- User wants to DIAGNOSE plant health issues from images
- Keywords: "what's wrong", "disease", "sick", "problem", "spots", "dying", "infected", "unhealthy"
- Requirements: Image present + health concern mentioned
- Action: "detect_disease"

**CATEGORY 3: companion_planting**
- User asks about which plants grow well together
- Keywords: "companion", "plant with", "grow with", "goes well with", "plant together"
- Extract plant name from query or use last_crop from memory
- Action: "companions"

**CATEGORY 4: yield_prediction**
- User wants to estimate harvest amounts
- Keywords: "yield", "harvest", "how much", "production", "output"
- Needs: crop name, temperature, rainfall data
- Action: "yield"

**CATEGORY 5: general_query** 
- ANY other agricultural question (soil, fertilizer, techniques, seasons, etc.)
- Examples: "How to prepare soil?", "What is crop rotation?", "Best organic fertilizer?"
- For follow-up questions about previous results (e.g., "how to treat it?")
- Action: "search" (searches KB) OR "reply" (uses LLM knowledge as fallback)

🔄 FALLBACK STRATEGY:

When KB search returns insufficient results (<60% relevance, <100 chars, or empty):
- Use action: "reply" with your own agricultural knowledge
- Still provide helpful, accurate information
- Indicate uncertainty if the topic is very specialized

📝 OUTPUT FORMAT:

YOU MUST RESPOND WITH ONLY VALID JSON. NO OTHER TEXT.
{
  "reasoning": "Brief analysis of user intent and why you chose this action",
  "action": "detect_crop|detect_disease|companions|yield|search|reply",
  "params": {<required parameters for the action>},
  "confidence": 0.0-1.0,
  "kb_fallback_ok": true/false
}

⚠️ CRITICAL RULES:

1. **For IMAGE queries:**
   - Check if user explicitly asks to identify crop → "detect_crop"
   - Check if user mentions disease/health problem → "detect_disease"  
   - If just showing image without specific request → "reply" (discuss image)

2. **For TEXT questions:**
   - Agriculture topic (soil, water, pests, etc.) → "search"
   - Follow-up on previous detection → "search" using last_disease/last_crop from memory
   - Greeting/thanks/casual → "reply"

3. **Parameter extraction:**
   - For companions: extract plant name or use facts.last_crop
   - For yield: extract crop, temp, rain from query
   - For search: use full user question as query

4. **Context awareness:**
   - Use conversation memory (facts.last_crop, facts.last_disease)
   - For "how to treat it?" → search using last_disease from memory
   - For "what companions?" → use last_crop from memory

5. **KB Fallback:**
   - Set "kb_fallback_ok": true if question can be answered with general knowledge
   - Set false for highly specific/regional questions that need KB data

🎓 EXAMPLES:

User: [image] "What crop is this?"
→ {"action": "detect_crop", "reasoning": "Explicit crop identification request with image"}

User: [image] "My plant leaves have brown spots, what's wrong?"
→ {"action": "detect_disease", "reasoning": "Health issue mentioned with image"}

User: "What should I plant with tomatoes?"
→ {"action": "companions", "params": {"plant": "tomatoes"}}

User: "How to prepare soil for planting?"
→ {"action": "search", "params": {"query": "soil preparation for planting"}, "kb_fallback_ok": true}

User: [after disease detection] "How do I treat it?"
→ {"action": "search", "params": {"query": "[last_disease] treatment prevention"}, "kb_fallback_ok": false}

User: "Hello"
→ {"action": "reply", "params": {"answer": "Hello! I'm your agricultural assistant..."}}

REMEMBER: 
- Prioritize KB search for agricultural questions
- Enable fallback for general knowledge questions
- Use tools only when explicitly needed
- Be conversational and helpful!
"""


def _agent_plan(user_text: str, image_b64: str | None, memory: Dict) -> Dict:
    """Enhanced agentic planning with 5-category intent classification and KB fallback awareness.

    Single-model policy: gemma3 only (no vision model usage). Images are never sent to the LLM; they only
    influence downstream handlers (crop/disease classifiers)."""
    try:
        oc = OllamaClient()

        # Select model (gemma3 only)
        has_image = image_b64 is not None
        model = _select_best_model(oc, has_image=has_image, needs_reasoning=True)
        supports_vision = False  # retained variable for backward compatibility logic branches
        
        # Build rich context for agent
        mem_ctx = {
            "summary": memory.get("summary", ""),
            "facts": memory.get("facts", {}),
            "recent_messages": memory.get("messages", [])[-4:],
            "tool_history": memory.get("tool_traces", [])[-3:],
        }
        
        context_str = json.dumps(mem_ctx, ensure_ascii=False, indent=2)
        user_prompt = f"""CONVERSATION MEMORY:
{context_str}

USER REQUEST: "{user_text or ""}"

IMAGE ATTACHED: {"Yes - User provided an image for analysis" if has_image else "No"}

🎯 YOUR TASK: 
1. Analyze the user's intent deeply
2. Classify into one of 5 categories: crop_detection, disease_detection, companion_planting, yield_prediction, general_query
3. Decide best action: detect_crop, detect_disease, companions, yield, search, or reply
4. Extract necessary parameters
5. Assess if KB fallback is acceptable

Think step by step and respond ONLY with JSON."""

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Image is never sent to LLM (no multimodal model); classifiers handle image content.
        
        print(f"[AGENT] Using model: {model} | Image: {has_image} | Query: {user_text[:60]}...")
        
        # Call Ollama with settings for reliable JSON planning
        call_kwargs = {
            "model": model,
            "messages": messages,
            "options": {"temperature": 0.2, "num_predict": 220}  # allow a bit more room for JSON
        }
        
        # Add format="json" only for text models (not vision models)
        if not supports_vision:
            call_kwargs["format"] = "json"
        
        # Try the primary model first, then allow a single reliability retry if content is empty
        resp = None
        try:
            resp = oc.chat(**call_kwargs)
        except Exception as e:
            # If the model fails to load (OOM or similar), try alternatives from the local Ollama registry
            print(f"[AGENT] Primary model chat call failed: {e}")
            try:
                available = [m.get("model") for m in (oc.list().get("models") or [])]
            except Exception as list_err:
                print(f"[AGENT] Could not list Ollama models: {list_err}")
                raise

            # Only gemma3 is supported now; try any installed gemma3 variant
            preferred_order = ["gemma3"]
            tried = set([call_kwargs.get("model")])

            for cand in available:
                if cand in tried:
                    continue
                # Heuristic: choose candidates that match preferred prefixes but are not the same huge model
                try_pref = False
                for p in preferred_order:
                    if p in (cand or "").lower():
                        try_pref = True
                        break
                # If none of the preferred prefixes match, still allow trying it as a last resort
                if not try_pref:
                    continue

                print(f"[AGENT] Attempting fallback model: {cand}")
                call_kwargs["model"] = cand
                try:
                    resp = oc.chat(**call_kwargs)
                    print(f"[AGENT] Fallback model succeeded: {cand}")
                    break
                except Exception as e2:
                    print(f"[AGENT] Fallback model {cand} failed: {e2}")
                    tried.add(cand)

            if resp is None:
                # All fallbacks failed; re-raise original exception for visibility
                print("[AGENT] All model fallbacks failed; raising original error")
                raise
        
        content = (resp.get("message") or {}).get("content") or "{}"

        # If we got an empty JSON ({}), try a reliability retry:
        if content.strip() in ("{}", "", None):
            try:
                # Retry 1: remove JSON forcing, increase tokens with same gemma3 model
                retry_kwargs = dict(call_kwargs)
                retry_kwargs.pop("format", None)
                retry_kwargs["options"] = {"temperature": 0.2, "num_predict": 280}
                print(f"[AGENT] Empty JSON, retrying planning with model={retry_kwargs['model']} (no format)")
                resp = oc.chat(**retry_kwargs)
                content = (resp.get("message") or {}).get("content") or content
            except Exception as retry_err:
                print(f"[AGENT] Retry after empty content failed: {retry_err}")
        
        # Try to parse JSON, but be flexible (especially for vision models)
        try:
            # Try direct JSON parse first
            plan = json.loads(content)
        except:
            # Vision models often produce malformed JSON, try multiple extraction methods
            try:
                # Method 1: Extract from markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group(1))
                else:
                    # Method 2: Find JSON-like structure and fix common issues
                    json_match = re.search(r'\{.*?"action".*?\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        # Fix common JSON issues from vision models
                        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                        plan = json.loads(json_str)
                    else:
                        # Method 3: For vision models, if we can't parse JSON, use smart defaults
                        if has_image:
                            print(f"[AGENT] Vision model produced non-JSON response, using smart defaults")
                            # Check for disease keywords
                            if any(word in user_text.lower() for word in ["disease", "wrong", "sick", "problem", "spot", "dying", "infected"]):
                                plan = {
                                    "action": "detect_disease",
                                    "params": {},
                                    "reasoning": "Image with disease keywords detected",
                                    "confidence": 0.75,
                                    "kb_fallback_ok": True
                                }
                            # Check for crop identification keywords
                            elif any(word in user_text.lower() for word in ["what plant", "what crop", "identify", "which plant", "which crop"]):
                                plan = {
                                    "action": "detect_crop",
                                    "params": {},
                                    "reasoning": "Image with crop identification request",
                                    "confidence": 0.75,
                                    "kb_fallback_ok": True
                                }
                            else:
                                # Default: conversational reply about the image
                                plan = {
                                    "action": "reply",
                                    "params": {},
                                    "reasoning": "Image discussion request",
                                    "confidence": 0.7,
                                    "kb_fallback_ok": True
                                }
                        else:
                            raise ValueError(f"No valid JSON found in response: {content[:200]}")
            except Exception as parse_error:
                print(f"[AGENT] JSON parsing failed with: {parse_error}")
                raise ValueError(f"Could not parse JSON from: {content[:200]}")
        
        # Validate plan structure
        if not isinstance(plan, dict) or "action" not in plan:
            raise ValueError(f"Invalid plan structure: {content}")
        
        # Validate action type
        valid_actions = {"reply", "detect_crop", "detect_disease", "companions", "yield", "search"}
        if plan.get("action") not in valid_actions:
            raise ValueError(f"Invalid action: {plan.get('action')}")
        
        # Add kb_fallback_ok flag if not present (default True for flexibility)
        if "kb_fallback_ok" not in plan:
            plan["kb_fallback_ok"] = True
        
        print(f"[AGENT] 🎯 Intent: {plan.get('action')} | Confidence: {plan.get('confidence', 0):.2f}")
        print(f"[AGENT] 💭 Reasoning: {plan.get('reasoning', '')[:120]}")
        print(f"[AGENT] 🔄 KB Fallback OK: {plan.get('kb_fallback_ok', True)}")
        
        return plan
        
    except Exception as e:
        import traceback
        print(f"[AGENT ERROR] Planning failed: {e}")
        traceback.print_exc()
        # Fallback: try to make intelligent guess
        return _fallback_plan(user_text, image_b64, memory, str(e))


def _fallback_plan(user_text: str, image_b64: str | None, memory: Dict, error_msg: str) -> Dict:
    """Enhanced fallback when agent planning fails - smart keyword-based routing."""
    t = (user_text or "").lower()
    facts = memory.get("facts", {})
    
    print(f"[FALLBACK] Analyzing: image={image_b64 is not None}, text='{t[:60]}...'")
    
    # Check for follow-up questions about previous detection
    if any(word in t for word in ["reduce", "treat", "cure", "fix", "prevent", "control", "manage", "stop"]):
        last_disease = facts.get("last_disease", "")
        last_crop = facts.get("last_crop", "")
        
        if last_disease:
            search_query = f"{last_disease} treatment prevention management control"
            print(f"[FALLBACK] → search (follow-up about {last_disease})")
            return {
                "action": "search",
                "params": {"query": search_query},
                "reasoning": f"Fallback: follow-up question about {last_disease} treatment",
                "confidence": 0.8,
                "kb_fallback_ok": True
            }
        elif last_crop:
            search_query = f"{last_crop} care {user_text}"
            print(f"[FALLBACK] → search (follow-up about {last_crop})")
            return {
                "action": "search",
                "params": {"query": search_query},
                "reasoning": f"Fallback: follow-up question about {last_crop}",
                "confidence": 0.7,
                "kb_fallback_ok": True
            }
    
    # If image present - improved keyword detection
    if image_b64:
        # Disease detection keywords (more comprehensive)
        disease_keywords = ["disease", "sick", "infection", "fungus", "pest", "dying", "wrong", "problem", 
                           "spot", "spots", "yellow", "brown", "damaged", "infected", "unhealthy", "issue"]
        disease_questions = ["what", "why", "is this", "diagnose", "got"]
        
        has_disease_keyword = any(word in t for word in disease_keywords)
        has_question = any(word in t for word in disease_questions)
        
        # Crop identification keywords
        crop_keywords = ["what plant", "what crop", "which plant", "which crop", "identify", 
                        "what is this", "plant it is", "crop it is"]
        has_crop_keyword = any(phrase in t for phrase in crop_keywords)
        
        # Decision logic for images
        if has_disease_keyword or (has_question and ("it" in t or "this" in t)):
            print(f"[FALLBACK] → detect_disease (keywords: disease-related)")
            return {
                "action": "detect_disease",
                "params": {},
                "reasoning": f"Fallback: disease-related keywords detected",
                "confidence": 0.7,
                "kb_fallback_ok": True
            }
        elif has_crop_keyword:
            print(f"[FALLBACK] → detect_crop (keywords: identification request)")
            return {
                "action": "detect_crop",
                "params": {},
                "reasoning": f"Fallback: crop identification keywords detected",
                "confidence": 0.7,
                "kb_fallback_ok": True
            }
        else:
            print(f"[FALLBACK] → reply (image discussion)")
            return {
                "action": "reply",
                "params": {"answer": ""},
                "reasoning": f"Fallback: conversational image discussion",
                "confidence": 0.6,
                "kb_fallback_ok": True
            }
    
    # Text-only queries - improved keyword detection
    
    # Companion planting keywords
    if any(phrase in t for phrase in ["companion", "grow with", "plant with", "goes well", "plant together"]):
        plant = memory.get("facts", {}).get("last_crop", "")
        print(f"[FALLBACK] → companions (plant: {plant or 'unknown'})")
        return {
            "action": "companions",
            "params": {"plant": plant},
            "reasoning": f"Fallback: companion planting keywords",
            "confidence": 0.7,
            "kb_fallback_ok": True
        }
    
    # Yield prediction keywords
    if any(word in t for word in ["yield", "harvest", "production", "how much"]):
        print(f"[FALLBACK] → yield")
        return {
            "action": "yield",
            "params": {},
            "reasoning": f"Fallback: yield prediction keywords",
            "confidence": 0.6,
            "kb_fallback_ok": True
        }
    
    # Greetings and casual conversation
    if any(word in t for word in ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"]):
        greetings = {
            "hello": "Hello! I'm your agriculture assistant. I can help with crop identification, disease diagnosis, companion planting, and agricultural questions. What would you like to know?",
            "hi": "Hi there! How can I help with your agriculture questions today?",
            "hey": "Hey! Ready to help with your farming or gardening questions!",
            "thanks": "You're welcome! Feel free to ask more questions anytime.",
            "thank you": "You're very welcome! Happy to help with agriculture questions.",
            "bye": "Goodbye! Come back anytime for agriculture advice. Happy farming!",
            "goodbye": "Take care! Feel free to return whenever you need agriculture help."
        }
        answer = next((v for k, v in greetings.items() if k in t), "Hello! How can I assist you with agriculture today?")
        print(f"[FALLBACK] → reply (greeting)")
        return {
            "action": "reply",
            "params": {"answer": answer},
            "reasoning": "Fallback: greeting/casual conversation",
            "confidence": 0.95,
            "kb_fallback_ok": True
        }
    
    # Default: use search for all other agricultural questions
    print(f"[FALLBACK] → search (default for agriculture question)")
    return {
        "action": "search",
        "params": {"query": user_text},
        "reasoning": f"Fallback: general agriculture question, using search",
        "confidence": 0.65,
        "kb_fallback_ok": True
    }


@app.post("/api/companions")
def api_companions():
    data = request.get_json(force=True) or {}
    plant = str(data.get("plant", "")).strip().lower()
    if not plant:
        return jsonify({"error": "plant is required"}), 400

    help_csv = BASE_DIR.parent / "companion_plants" / "help_network.csv"
    avoid_csv = BASE_DIR.parent / "companion_plants" / "avoid_network.csv"

    def read_edges(path: Path):
        edges = []
        if not path.exists():
            return edges
        with open(path, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            # Support alternate column names just in case
            for row in r:
                a = (row.get('Source Node') or row.get('Source') or '').strip().lower()
                b = (row.get('Destination Node') or row.get('Destination') or '').strip().lower()
                if a and b:
                    edges.append((a, b))
        return edges

    try:
        help_edges = read_edges(help_csv)
        avoid_edges = read_edges(avoid_csv)

        # Build node set
        nodes = set()
        for a, b in help_edges + avoid_edges:
            nodes.add(a); nodes.add(b)

        def candidate_nodes(name: str) -> List[str]:
            """Return possible node names in graph matching the input (handles simple plural/synonyms)."""
            cand = set()
            if name in nodes:
                cand.add(name)
            # Simple plural/singular forms
            forms = set([name])
            if name.endswith('ies'): forms.add(name[:-3] + 'y')
            if name.endswith('y'): forms.add(name[:-1] + 'ies')
            if name.endswith('oes'): forms.add(name[:-3] + 'o')
            if name.endswith('o'): forms.add(name + 'es')
            if name.endswith('s'): forms.add(name[:-1])
            else: forms.add(name + 's')
            # Synonyms
            syn = {
                'tomato': ['tomatoes'],
                'potato': ['potatoes'],
                'pepper': ['peppers', 'capsicum', 'capsicums'],
                'chilli': ['chillies', 'chili', 'chilies', 'peppers'],
                'bean': ['beans'],
                'allium': ['alliums', 'onion', 'garlic', 'onions', 'garlics'],
            }
            for k, vs in syn.items():
                if name == k or name in vs:
                    forms.update([k, *vs])
            for f in forms:
                if f in nodes: cand.add(f)
            # Fuzzy contains match (last resort)
            if not cand:
                for n in nodes:
                    if name in n or n in name:
                        cand.add(n)
            return sorted(cand)

        targets = candidate_nodes(plant)
        # Aggregate neighbors for all target aliases
        good_set = set()
        bad_set = set()
        for t in targets:
            good_set |= {b for a, b in help_edges if a == t}
            good_set |= {a for a, b in help_edges if b == t}
            bad_set |= {b for a, b in avoid_edges if a == t}
            bad_set |= {a for a, b in avoid_edges if b == t}

        good = sorted(good_set)
        bad = sorted(bad_set)
        return jsonify({
            "input": {"plant": plant, "resolved": targets},
            "output": {"good": good[:50], "avoid": bad[:50]}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/companions/plants")
def api_companion_plants():
    """Return a list of unique plant names from the companion CSVs for autocomplete."""
    help_csv = BASE_DIR.parent / "companion_plants" / "help_network.csv"
    avoid_csv = BASE_DIR.parent / "companion_plants" / "avoid_network.csv"

    def read_nodes(path: Path):
        nodes = set()
        if not path.exists():
            return nodes
        with open(path, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                a = (row.get('Source Node') or row.get('Source') or '').strip().lower()
                b = (row.get('Destination Node') or row.get('Destination') or '').strip().lower()
                if a: nodes.add(a)
                if b: nodes.add(b)
        return nodes

    nodes = set()
    nodes |= read_nodes(help_csv)
    nodes |= read_nodes(avoid_csv)
    items = sorted(nodes)
    # return a reasonable number to avoid overly large payloads
    return jsonify({"plants": items[:1000]})


# ===== Yield prediction (using Yield_prediction/yield_df.csv) =====
_yield_cache = {"data": None, "coeffs": {}}


def _load_yield_rows():
    if _yield_cache["data"] is not None:
        return _yield_cache["data"]
    rows = []
    path = BASE_DIR.parent / "Yield_prediction" / "yield_df.csv"
    if not path.exists():
        _yield_cache["data"] = []
        return _yield_cache["data"]
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                item = (row.get('Item') or '').strip()
                y = float(row['hg/ha_yield'])/10000.0  # to t/ha
                t = float(row['avg_temp'])
                rf = float(row['average_rain_fall_mm_per_year'])
                if item and isfinite(y) and isfinite(t) and isfinite(rf):
                    rows.append({"item": item, "y": y, "temp": t, "rain": rf})
            except Exception:
                continue
    _yield_cache["data"] = rows
    return rows


def _match_item(crop: str, items: list[str]) -> str | None:
    c = (crop or '').strip().lower()
    if not c:
        return None
    # exact case-insensitive match
    for it in items:
        if it.lower() == c:
            return it
    # simple plural/singular forms
    forms = {c}
    if c.endswith('ies'): forms.add(c[:-3] + 'y')
    if c.endswith('y'): forms.add(c[:-1] + 'ies')
    if c.endswith('oes'): forms.add(c[:-3] + 'o')
    if c.endswith('o'): forms.add(c + 'es')
    if c.endswith('s'): forms.add(c[:-1])
    else: forms.add(c + 's')
    # synonyms for common crops
    syn = {
        'tomato': ['tomatoes'],
        'potato': ['potatoes'],
        'pepper': ['peppers', 'capsicum', 'capsicums'],
        'chilli': ['chillies', 'chili', 'chilies', 'peppers'],
        'cucumber': ['cucumbers', 'gherkin', 'gherkins'],
    }
    for k, vs in syn.items():
        if c == k or c in vs:
            forms.update([k, *vs])
    # prefer first form that exists
    for f in forms:
        for it in items:
            if it.lower() == f:
                return it
    # contains fallback
    for it in items:
        low = it.lower()
        if c in low or low in c:
            return it
    return None


def _fit_coeffs(item: str):
    # Fit OLS y = a + b*temp + c*rain for the given item
    if item in _yield_cache["coeffs"]:
        return _yield_cache["coeffs"][item]
    rows = [r for r in _load_yield_rows() if r['item'] == item]
    if len(rows) >= 6:
        X = np.array([[1.0, r['temp'], r['rain']] for r in rows], dtype=np.float64)
        y = np.array([r['y'] for r in rows], dtype=np.float64)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        _yield_cache["coeffs"][item] = beta.tolist()
        return _yield_cache["coeffs"][item]
    # fallback to mean-only model
    if rows:
        mean_y = float(np.mean([r['y'] for r in rows]))
        _yield_cache["coeffs"][item] = [mean_y, 0.0, 0.0]
        return _yield_cache["coeffs"][item]
    return None


@app.post("/api/yield")
def api_yield():
    data = request.get_json(force=True) or {}
    crop = str(data.get('crop', '')).strip()
    try:
        temp = float(data.get('temp'))
        rain = float(data.get('rain'))
    except Exception:
        return jsonify({"error": "temp and rain must be numbers"}), 400

    rows = _load_yield_rows()
    items = sorted({r['item'] for r in rows})
    item = _match_item(crop, items)
    resolved = item or None
    # Predict
    pred = None
    if item:
        coeffs = _fit_coeffs(item)
        if coeffs is not None:
            a, b, c = coeffs
            pred = float(max(0.1, a + b*temp + c*rain))
    if pred is None:
        # Baseline heuristic if item not matched or data missing
        base = 5.0
        adj = (temp - 22.0)*0.05 + (rain - 70.0)*0.01
        pred = float(max(0.1, round(base + adj, 2)))

    return jsonify({
        "input": {"crop": crop, "resolved": resolved, "temp": temp, "rain": rain},
        "output": {"yield_t_ha": round(pred, 3)}
    })


# ===== Image detection (crop classification) =====
_det_model = None
_det_labels = None
_dis_model = None
_dis_labels = None


def _ensure_tf():
    if tf is None:
        raise RuntimeError("TensorFlow not installed. Please add tensorflow to app/requirements.txt and install.")


def _load_pickle(path: Path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_detection_model():
    global _det_model, _det_labels
    if _det_model is not None:
        return
    _ensure_tf()
    preferred = BASE_DIR.parent / "crop_detection" / "best_crop_detection_model.keras"
    fallback = BASE_DIR.parent / "crop_detection" / "crop_detection_model.keras"
    model_path = preferred if preferred.exists() else fallback
    labels_path = BASE_DIR.parent / "crop_detection" / "class_indices.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Detection model not found at {model_path}")
    _det_model = tf.keras.models.load_model(str(model_path))
    idx = _load_pickle(labels_path)
    # invert mapping: index -> class name
    _det_labels = {v: k for k, v in idx.items()}


def _preprocess_image(file_bytes: bytes, size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((size, size))
    arr = np.asarray(img).astype('float32')/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.post("/api/detect")
def api_detect():
    if 'image' not in request.files:
        return jsonify({"error": "image file is required"}), 400
    f = request.files['image']
    raw = f.read()
    try:
        _load_detection_model()
        x = _preprocess_image(raw, 224)
        preds = _det_model.predict(x)
        if preds.ndim != 2 or preds.shape[0] != 1:
            raise RuntimeError(f"Unexpected prediction shape: {preds.shape}")
        probs = preds[0]
        cls = int(np.argmax(probs))
        conf = float(np.max(probs))
        label = _det_labels.get(cls, str(cls))

        # Build top-3
        top_idx = np.argsort(probs)[-3:][::-1]
        top3 = [{
            "label": _det_labels.get(int(i), str(int(i))),
            "confidence": float(round(float(probs[int(i)]), 4))
        } for i in top_idx]

        note = None
        if _det_labels and len(_det_labels) != probs.shape[-1]:
            note = f"Warning: labels ({len(_det_labels)}) do not match model outputs ({probs.shape[-1]})."

        return jsonify({
            "input": {"mode": "detect", "filename": f.filename},
            "output": {"label": label, "confidence": round(conf, 4), "top3": top3, "note": note}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== Disease detection (using crop_disease) =====
def _load_disease_model():
    global _dis_model, _dis_labels
    if _dis_model is not None:
        return
    _ensure_tf()
    # Load ONLY the newly trained 18-epoch model (TF 2.16 compatible .h5 format)
    model_path = BASE_DIR.parent / "crop_disease" / "trained_model_18epochs.h5"
    labels_path = BASE_DIR.parent / "crop_disease" / "class_indices.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Disease model not found: {model_path}")
    
    _dis_model = tf.keras.models.load_model(str(model_path), compile=False)
    try:
        print(f"[DISEASE] Loaded model {model_path.name} IO={_dis_model.input_shape}->{_dis_model.output_shape}")
    except Exception:
        pass

    if labels_path.exists():
        try:
            idx = _load_pickle(labels_path)
            _dis_labels = {v: k for k, v in idx.items()}
            return
        except Exception as e:
            print(f"[DISEASE] Failed to load labels pickle, will attempt reconstruction: {e}")

    # Reconstruction fallback: prefer known 38-class order from Streamlit main.py
    try:
        _dis_labels = _dis_labels or {}
        if not _dis_labels:
            streamlit_class_names = [
                'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
                'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy',
                'Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight',
                'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
            ]
            if len(streamlit_class_names) == int(getattr(_dis_model.output_shape, '__iter__', lambda: [0])()[-1] if hasattr(_dis_model, 'output_shape') else 38):
                _dis_labels = {i: name for i, name in enumerate(streamlit_class_names)}
                print(f"[DISEASE] Using built-in 38-class label list from Streamlit app")
            else:
                # As last resort, infer from dataset on disk if available
                data_root = BASE_DIR.parent / "crop_disease" / "Plant_Disease_Dataset" / "train"
                class_dirs = [p.name for p in data_root.iterdir() if p.is_dir()] if data_root.exists() else []
                if class_dirs:
                    class_dirs = sorted(class_dirs)
                    _dis_labels = {i: name for i, name in enumerate(class_dirs)}
                    print(f"[DISEASE] Reconstructed {len(_dis_labels)} labels from dataset directory names")
                else:
                    _dis_labels = {}
    except Exception as recon_err:
        print(f"[DISEASE] Label reconstruction failed: {recon_err}")
        _dis_labels = {}


@app.post("/api/disease")
def api_disease():
    if 'image' not in request.files:
        return jsonify({"error": "image file is required"}), 400
    notes = request.form.get('notes', '')
    f = request.files['image']
    raw = f.read()
    try:
        _load_disease_model()
        # Disease model expects raw 0-255 pixel values (NO normalization), matching main.py
        img = Image.open(io.BytesIO(raw)).convert('RGB').resize((128, 128))
        input_arr = np.asarray(img, dtype='float32')  # Keep raw 0-255 range
        x = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        preds = _dis_model.predict(x)
        if preds.ndim != 2 or preds.shape[0] != 1:
            raise RuntimeError(f"Unexpected prediction shape: {preds.shape}")
        probs = preds[0]
        cls = int(np.argmax(probs))
        conf = float(np.max(probs))
        label = _dis_labels.get(cls, str(cls))

        # Top-3 and Top-5
        order = np.argsort(probs)[::-1]
        top3 = [{
            "label": _dis_labels.get(int(i), str(int(i))),
            "confidence": float(round(float(probs[int(i)]), 4))
        } for i in order[:3]]
        top5 = [{
            "label": _dis_labels.get(int(i), str(int(i))),
            "confidence": float(round(float(probs[int(i)]), 4))
        } for i in order[:5]]

        note = None
        if _dis_labels and len(_dis_labels) != probs.shape[-1]:
            note = f"Warning: labels ({len(_dis_labels)}) do not match model outputs ({probs.shape[-1]})."

        # Full probability list (optional) for debugging
        full_probs = [
            {
                "label": _dis_labels.get(int(i), str(int(i))),
                "confidence": float(round(float(probs[int(i)]), 6))
            } for i in order
        ]

        return jsonify({
            "input": {"mode": "disease", "filename": f.filename, "notes": notes},
            "output": {
                "label": label,
                "confidence": round(conf, 4),
                "top3": top3,
                "top5": top5,
                "note": note,
                "labels": [_dis_labels.get(i, str(i)) for i in range(len(probs))],
                "probabilities": full_probs
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/")
def index():
    # Keep the old template accessible
    return render_template("index.html")

@app.get("/about")
def about():
    # About page explaining the AI Master Gardener
    return render_template("about.html")


def _kb_search(query: str, k: int = 5) -> List[Dict]:
    try:
        # Reuse existing kb/search.py logic if available
        from kb.search import search as kb_search
        return kb_search(query, k=k)
    except Exception:
        return []


def _check_kb_quality(results: List[Dict], min_similarity: float = 0.6, min_length: int = 100) -> Dict:
    """
    Check if KB search results are sufficient quality.
    Returns: {"sufficient": bool, "reason": str, "best_similarity": float, "total_length": int}
    """
    if not results or len(results) == 0:
        return {"sufficient": False, "reason": "no_results", "best_similarity": 0.0, "total_length": 0}
    
    # Check similarity scores
    similarities = [r.get("similarity", 0.0) for r in results if r.get("similarity") is not None]
    best_sim = max(similarities) if similarities else 0.0
    
    if best_sim < min_similarity:
        return {"sufficient": False, "reason": "low_similarity", "best_similarity": best_sim, "total_length": 0}
    
    # Check content length
    total_length = sum(len(r.get("content", "")) for r in results)
    
    if total_length < min_length:
        return {"sufficient": False, "reason": "insufficient_content", "best_similarity": best_sim, "total_length": total_length}
    
    return {"sufficient": True, "reason": "good_quality", "best_similarity": best_sim, "total_length": total_length}


def _generate_ollama_response(prompt: str, system_msg: str = None, image_b64: str = None, temperature: float = 0.4, max_tokens: int = 300) -> str:
    """Generate a response using local LLM.

    Notes (model policy update):
    - Single-model policy: ONLY gemma3 is supported.
    - Vision model removed; if an image is supplied we ask gemma3 for a textual description based on user context.
    - If gemma3 is missing locally, a clear error will be raised earlier in model selection.
    """
    try:
        oc = OllamaClient()
        model = _select_best_model(oc, has_image=(image_b64 is not None), needs_reasoning=False)

        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        user_content = prompt
        if image_b64:
            # Provide a brief inline instruction since we no longer use a vision model.
            user_content = (
                "The user provided an image (base64 omitted here for brevity). "
                "Please infer and describe relevant agricultural details (crop traits, health) from context if possible. "
                + prompt
            )
        messages.append({"role": "user", "content": user_content})

        resp = oc.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        content = (resp.get("message") or {}).get("content") or ""
        if not content.strip():
            # Retry once with gemma3 (reliable) if initial was deepseek
            if "deepseek" in str(model).lower():
                try:
                    alt = _select_best_model(oc, has_image=(image_b64 is not None), needs_reasoning=True)
                    if alt != model:
                        print(f"[OLLAMA] Empty response from {model}, retrying with {alt}")
                        resp2 = oc.chat(
                            model=alt,
                            messages=messages,
                            options={"temperature": temperature, "num_predict": max_tokens + 50}
                        )
                        content = (resp2.get("message") or {}).get("content") or content
                except Exception as retry_err:
                    print(f"[OLLAMA] Retry after empty content failed: {retry_err}")
        if not content.strip():
            content = "I'm here to help with agriculture questions! Ask me about crops, diseases, companions, or yield."
        return content
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        return "I'm having trouble generating a response. Please try again."


def _get_organic_treatments(disease_label: str, disease_type: str = "fungal") -> Dict:
    """
    Get organic treatment recommendations for a specific disease.
    Returns structured treatment plan prioritizing natural methods.
    """
    try:
        # Normalize disease name for lookup
        disease_key = disease_label.lower().replace(" ", "_").replace("-", "_")
        
        # Determine disease type if not provided
        if "bacterial" in disease_label.lower():
            disease_type = "bacterial"
        elif "virus" in disease_label.lower() or "mosaic" in disease_label.lower() or "curl" in disease_label.lower():
            disease_type = "viral"
        elif "mite" in disease_label.lower() or "aphid" in disease_label.lower():
            disease_type = "pest"
        else:
            disease_type = "fungal"  # default
        
        treatments = ORGANIC_TREATMENTS.get(disease_type, ORGANIC_TREATMENTS["fungal"])
        
        return {
            "disease_type": disease_type,
            "primary_treatments": treatments.get("primary", []),
            "secondary_treatments": treatments.get("secondary", []),
            "philosophy": ORGANIC_TREATMENT_PRIORITY
        }
    except Exception as e:
        print(f"[ORGANIC TREATMENT ERROR] {e}")
        return {
            "disease_type": "unknown",
            "primary_treatments": [],
            "secondary_treatments": [],
            "philosophy": ORGANIC_TREATMENT_PRIORITY
        }


def _get_companion_disease_solutions(disease_label: str, crop_name: str = None) -> Dict:
    """
    Get companion planting recommendations to help prevent/manage specific disease.
    Links disease detection with companion planting strategies.
    """
    try:
        # Normalize disease name for lookup
        disease_key = disease_label.lower().replace(" ", "_").replace("-", "_")
        
        # Try exact match first
        for key, data in COMPANION_DISEASE_SOLUTIONS.items():
            if key in disease_key or disease_key in key:
                return {
                    "found": True,
                    "disease": disease_label,
                    "companions": data["companions"],
                    "benefits": data["benefits"],
                    "spacing": data["spacing"],
                    "strategy": "disease-specific"
                }
        
        # Fallback to general healthy companions
        default = COMPANION_DISEASE_SOLUTIONS.get("healthy", {})
        return {
            "found": False,
            "disease": disease_label,
            "companions": default.get("companions", ["basil", "marigold", "parsley"]),
            "benefits": default.get("benefits", "General pest resistance and plant health"),
            "spacing": default.get("spacing", "Standard spacing"),
            "strategy": "general-prevention"
        }
    except Exception as e:
        print(f"[COMPANION DISEASE ERROR] {e}")
        return {
            "found": False,
            "disease": disease_label,
            "companions": ["marigold", "basil"],
            "benefits": "General plant health support",
            "spacing": "12-18 inches",
            "strategy": "fallback"
        }


def _analyze_disease_severity(confidence: float, disease_label: str) -> Dict:
    """
    Analyze disease severity and recommend treatment approach.
    Returns severity level and recommended treatment escalation.
    """
    try:
        # Severity based on confidence and disease type
        if confidence >= 0.85:
            severity = "high_confidence"
            urgency = "immediate"
        elif confidence >= 0.70:
            severity = "moderate_confidence"
            urgency = "prompt"
        else:
            severity = "low_confidence"
            urgency = "monitor"
        
        # Check if disease is typically severe
        severe_diseases = ["late_blight", "bacterial_wilt", "fusarium", "verticillium"]
        is_severe = any(sd in disease_label.lower() for sd in severe_diseases)
        
        if is_severe and confidence >= 0.70:
            approach = "aggressive"
            stage = "moderate_to_severe"
        elif confidence >= 0.80:
            approach = "active"
            stage = "moderate"
        else:
            approach = "preventive"
            stage = "mild"
        
        return {
            "severity": severity,
            "urgency": urgency,
            "approach": approach,
            "stage": stage,
            "confidence": confidence,
            "is_severe_disease": is_severe
        }
    except Exception as e:
        print(f"[SEVERITY ANALYSIS ERROR] {e}")
        return {
            "severity": "unknown",
            "urgency": "monitor",
            "approach": "preventive",
            "stage": "mild",
            "confidence": confidence,
            "is_severe_disease": False
        }


def _companions_merge(plant: str) -> Dict:
    # Build CSV graph directly (avoid calling the HTTP route)
    help_csv = BASE_DIR.parent / "companion_plants" / "help_network.csv"
    avoid_csv = BASE_DIR.parent / "companion_plants" / "avoid_network.csv"

    def read_edges(path: Path):
        edges = []
        if not path.exists():
            return edges
        with open(path, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                a = (row.get('Source Node') or row.get('Source') or '').strip().lower()
                b = (row.get('Destination Node') or row.get('Destination') or '').strip().lower()
                if a and b:
                    edges.append((a, b))
        return edges

    help_edges = read_edges(help_csv)
    avoid_edges = read_edges(avoid_csv)
    nodes = set([n for ab in help_edges + avoid_edges for n in ab])

    def candidate_nodes(name: str) -> List[str]:
        cand = set(); name = (name or '').strip().lower()
        if not name:
            return []
        if name in nodes: cand.add(name)
        forms = {name}
        if name.endswith('ies'): forms.add(name[:-3] + 'y')
        if name.endswith('y'): forms.add(name[:-1] + 'ies')
        if name.endswith('oes'): forms.add(name[:-3] + 'o')
        if name.endswith('o'): forms.add(name + 'es')
        if name.endswith('s'): forms.add(name[:-1])
        else: forms.add(name + 's')
        syn = {
            'tomato': ['tomatoes'], 'potato': ['potatoes'], 'pepper': ['peppers', 'capsicum', 'capsicums'],
            'chilli': ['chillies', 'chili', 'chilies', 'peppers'], 'bean': ['beans'], 'allium': ['alliums', 'onion', 'garlic', 'onions', 'garlics']
        }
        for k, vs in syn.items():
            if name == k or name in vs:
                forms.update([k, *vs])
        for f in forms:
            if f in nodes: cand.add(f)
        if not cand:
            for n in nodes:
                if name in n or n in name:
                    cand.add(n)
        return sorted(cand)

    targets = candidate_nodes(plant)
    good_set = set(); bad_set = set()
    for tname in targets:
        good_set |= {b for a, b in help_edges if a == tname}
        good_set |= {a for a, b in help_edges if b == tname}
        bad_set |= {b for a, b in avoid_edges if a == tname}
        bad_set |= {a for a, b in avoid_edges if b == tname}
    data = {"input": {"plant": plant, "resolved": targets}, "output": {"good": sorted(good_set)[:50], "avoid": sorted(bad_set)[:50]}}
    
    # KB retrieve and simple synthesis prompt
    kb_q = f"companion planting for {plant} helpful and avoid with reasons"
    kb_results = _kb_search(kb_q, k=3)
    citations = []
    kb_texts = []
    for r in kb_results:
        content = (r.get("content") or "").strip()
        if content:
            kb_texts.append(content[:300])
            citations.append({"doc_id": r.get("doc_id"), "similarity": r.get("similarity")})
    
    merged_note = None
    if kb_texts and len(targets) > 0:
        # Ask Ollama to synthesize KB results with CSV data
        try:
            oc = OllamaClient()
            model = _select_best_model(oc, has_image=False, needs_reasoning=False)
            if model:
                prompt = (
                    f"Plant: {plant}\n"
                    f"CSV helps: {', '.join(data.get('output',{}).get('good', [])[:12])}\n"
                    f"CSV avoid: {', '.join(data.get('output',{}).get('avoid', [])[:12])}\n"
                    "Sources (free text):\n" + "\n---\n".join(kb_texts[:3]) + "\n\n"
                    "In 2-3 sentences, explain why the top companions help and which to avoid."
                )
                resp = oc.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
                merged_note = (resp.get("message") or {}).get("content")
        except Exception as e:
            print(f"KB synthesis error: {e}")
            merged_note = None
    
    return {"graph": data, "citations": citations, "note": merged_note}


@app.post("/api/search")
def api_search():
    data = request.get_json(force=True) or {}
    q = data.get("query", "").strip()
    k = int(data.get("k", 5))
    if not q:
        return jsonify({"error": "query is required"}), 400

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not SUPABASE_URL or not KEY:
        return jsonify({"error": "Supabase env not configured"}), 500

    try:
        q_emb = embed_query(q)
    except Exception as e:
        return jsonify({"error": f"embedding failed: {e}"}), 500

    supa = create_client(SUPABASE_URL, KEY)
    rpc = supa.rpc("match_chunks", {"query_embedding": q_emb, "match_count": k}).execute()
    results: List[Dict] = rpc.data or []
    # normalize output
    norm = []
    for r in results:
        sim = r.get("similarity")
        content = r.get("content") or ""
        norm.append({
            "id": r.get("id"),
            "doc_id": r.get("doc_id"),
            "similarity": float(sim) if isinstance(sim, (int, float)) else None,
            "content": content,
        })
    return jsonify({"results": norm})


@app.post("/api/chat")
def api_chat():
    """Agent chat: accepts text and/or image and a thread_id; plans and executes tools."""
    try:
        thread_id = request.form.get("thread_id", "").strip() or "default"
        message = request.form.get("message", "")
        img_file = request.files.get("image")
        fast_mode = request.form.get("fast_mode", "true").lower() == "true"  # Enable fast mode by default
        
        image_b64 = None
        raw_img = None
        if img_file:
            raw_img = img_file.read()
            try:
                image_b64 = base64.b64encode(raw_img).decode("utf-8")
            except Exception:
                image_b64 = None

        state = _load_thread_state(thread_id)
        # Append user message to memory
        state.setdefault("messages", []).append({"role": "user", "content": (message or "")[:400], "ts": int(__import__('time').time()*1000)})

        # FAST MODE: Skip intent classification for common patterns
        plan = None
        if fast_mode and image_b64:
            t = message.lower()
            # Fast-track disease detection (most common use case)
            if any(word in t for word in ["disease", "wrong", "sick", "problem", "spot", "dying", "what", "got", "infected"]):
                print(f"[FAST MODE] → detect_disease (skipping intent classification)")
                plan = {
                    "action": "detect_disease",
                    "params": {},
                    "reasoning": "Fast mode: disease keywords detected",
                    "confidence": 0.9,
                    "kb_fallback_ok": True
                }
            # Fast-track crop identification
            elif any(phrase in t for phrase in ["what plant", "what crop", "identify", "which plant", "which crop"]):
                print(f"[FAST MODE] → detect_crop (skipping intent classification)")
                plan = {
                    "action": "detect_crop",
                    "params": {},
                    "reasoning": "Fast mode: crop identification keywords",
                    "confidence": 0.9,
                    "kb_fallback_ok": True
                }
        
        # Use agentic planner only if fast mode didn't match
        if not plan:
            print(f"[CHAT] Processing message: '{message[:50]}...' | Image: {image_b64 is not None}")
            plan = _agent_plan(message, image_b64, state)
        
        agent_info = {
            "plan": plan,
            "reasoning": plan.get("reasoning", ""),
            "confidence": plan.get("confidence", 0.0),
            "fast_mode": plan is not None and "Fast mode" in plan.get("reasoning", "")
        }
        out_message = ""
        data = None
        citations = None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Pre-execution error: {str(e)}"}), 500

    try:
        if plan["action"] == "reply":
            # Direct conversational reply (greetings, casual conversation, or image discussion)
            out_message = plan.get("params", {}).get("answer") or ""
            if not out_message:
                # Generate response using Ollama
                kb_context = ""
                
                # For agricultural questions, try KB enrichment first (unless it's just a greeting)
                if not any(word in message.lower() for word in ["hello", "hi", "thanks", "thank", "bye"]):
                    try:
                        facts = state.get("facts", {})
                        search_query = message
                        if facts.get("last_disease"):
                            search_query = f"{facts['last_disease']} {message}"
                        elif facts.get("last_crop"):
                            search_query = f"{facts['last_crop']} {message}"
                        
                        kb_results = _kb_search(search_query, k=2)
                        quality = _check_kb_quality(kb_results, min_similarity=0.5, min_length=80)
                        
                        if quality["sufficient"]:
                            kb_context = "\n\n📚 KNOWLEDGE BASE CONTEXT:\n" + "\n---\n".join([r.get("content", "")[:400] for r in kb_results[:2]])
                            citations = [{"doc_id": r.get("doc_id"), "similarity": r.get("similarity")} for r in kb_results]
                            print(f"[KB ENRICHMENT] Added context (similarity: {quality['best_similarity']:.2f})")
                        else:
                            print(f"[KB ENRICHMENT] Skipped - {quality['reason']}")
                    except Exception as kb_err:
                        print(f"[KB ENRICHMENT ERROR] {kb_err}")
                
                # Build conversation context
                context = "\n".join([m.get("content", "")[:200] for m in state.get("messages", [])[-3:]])
                
                system_msg = """You are an expert agricultural assistant with deep knowledge of farming, plant care, and agronomy.

🎯 Your role:
- Provide accurate, practical advice for farmers and gardeners
- Use the knowledge base context when provided for accuracy
- If discussing an image, describe what you see and provide relevant insights
- Be conversational, helpful, and encouraging
- Give specific, actionable recommendations

Guidelines:
- Prioritize information from knowledge base context (if provided)
- Fill gaps with your agricultural expertise
- Be honest if you're uncertain about something
- Use simple language farmers can understand"""

                prompt = f"Conversation context: {context}{kb_context}\n\n👤 User question: {message}"
                
                out_message = _generate_ollama_response(
                    prompt=prompt,
                    system_msg=system_msg,
                    image_b64=image_b64,
                    temperature=0.4,
                    max_tokens=250  # Reduced from 300 for faster response
                )
        elif plan["action"] == "detect_crop":
            if not raw_img:
                return jsonify({"error": "image required for crop detection", "agent": agent_info}), 400
            _load_detection_model()
            x = _preprocess_image(raw_img, 224)
            preds = _det_model.predict(x)
            probs = preds[0]
            top_idx = np.argsort(probs)[-3:][::-1]
            top3 = [
                {"label": _det_labels.get(int(i), str(int(i))), "confidence": float(probs[int(i)])}
                for i in top_idx
            ]
            crop_label = top3[0]["label"]
            confidence = top3[0]["confidence"]
            
            data = {"label": crop_label, "confidence": confidence, "top3": top3}
            
            # Search KB for crop growing information
            kb_results = _kb_search(f"{crop_label} growing cultivation care requirements", k=2)
            quality = _check_kb_quality(kb_results, min_similarity=0.5, min_length=80)
            
            if quality["sufficient"]:
                kb_content = "\n".join([r.get("content", "")[:250] for r in kb_results[:2]])
                citations = [{"doc_id": r.get("doc_id"), "similarity": r.get("similarity")} for r in kb_results]
                
                system_msg = "You are an agricultural expert. Provide concise growing tips."
                prompt = f"""Crop identified: {crop_label} (confidence: {confidence*100:.0f}%)

Knowledge base information:
{kb_content}

Provide a brief response with:
1. Confirmation of identification
2. Key growing requirements (climate, soil, water)
3. Best growing season
4. Top 2-3 care tips

Keep it concise and practical:"""

                out_message = _generate_ollama_response(prompt, system_msg, temperature=0.3, max_tokens=250)
                print(f"[CROP] ✅ Using KB context for {crop_label}")
            else:
                # Fallback response
                out_message = f"🌱 **Identified: {crop_label}** (confidence: {confidence*100:.0f}%)\n\n"
                if len(top3) > 1 and top3[1]["confidence"] > 0.15:
                    out_message += f"Alternative possibilities: {', '.join([t['label'] for t in top3[1:3]])}\n\n"
                out_message += "💡 Ask me about growing requirements, companion plants, or common diseases for this crop!"
                print(f"[CROP] ℹ️ Basic response for {crop_label}")
            
            state.setdefault("facts", {})["last_crop"] = crop_label
            state.setdefault("tool_traces", []).append({"tool": "detect_crop", "top1": crop_label})
        elif plan["action"] == "detect_disease":
            if not raw_img:
                return jsonify({"error": "image required for disease detection", "agent": agent_info}), 400
            _load_disease_model()
            # Disease model expects raw 0-255 pixel values (NO normalization), matching main.py
            img = Image.open(io.BytesIO(raw_img)).convert('RGB').resize((128, 128))
            input_arr = np.asarray(img, dtype='float32')  # Keep raw 0-255 range
            x = np.expand_dims(input_arr, axis=0)  # Add batch dimension
            preds = _dis_model.predict(x)
            probs = preds[0]
            top_idx = np.argsort(probs)[-3:][::-1]
            top3 = [
                {"label": _dis_labels.get(int(i), str(int(i))), "confidence": float(probs[int(i)])}
                for i in top_idx
            ]
            disease_label = top3[0]["label"]
            confidence = top3[0]["confidence"]
            
            data = {"label": disease_label, "confidence": confidence, "top3": top3}
            
            # === NEW: Get organic treatments and companion plant solutions ===
            organic_treatments = _get_organic_treatments(disease_label)
            companion_solutions = _get_companion_disease_solutions(disease_label)
            severity_analysis = _analyze_disease_severity(confidence, disease_label)
            
            # Add to data for API response
            data["organic_treatments"] = organic_treatments
            data["companion_solutions"] = companion_solutions
            data["severity"] = severity_analysis
            
            # Search KB for disease information and treatment
            kb_results = _kb_search(f"{disease_label} organic treatment natural remedy prevention", k=3)
            quality = _check_kb_quality(kb_results, min_similarity=0.5, min_length=80)
            
            if quality["sufficient"]:
                # Synthesize disease info with KB context + organic focus
                kb_content = "\n".join([r.get("content", "")[:300] for r in kb_results[:2]])
                citations = [{"doc_id": r.get("doc_id"), "similarity": r.get("similarity")} for r in kb_results]
                
                system_msg = """You are a SUSTAINABLE AGRICULTURE disease expert specializing in:
- Organic and natural disease management
- Integrated Pest Management (IPM)
- Companion planting strategies
- Environmental stewardship

ALWAYS prioritize organic/natural methods over chemicals."""

                prompt = f"""🦠 Disease detected: {disease_label} (confidence: {confidence*100:.0f}%)
Severity: {severity_analysis['stage']} | Urgency: {severity_analysis['urgency']}

Knowledge base information:
{kb_content}

🌿 Available companion plants that help:
{', '.join(companion_solutions['companions'][:4])}
Why: {companion_solutions['benefits']}

Provide a farmer-friendly response with:
1. Brief disease description & symptoms
2. 🌱 ORGANIC TREATMENTS (prioritize these!)
3. 🌿 Companion planting strategy
4. Prevention tips for future

Keep it concise, actionable, and eco-friendly:"""

                base_response = _generate_ollama_response(prompt, system_msg, temperature=0.3, max_tokens=250)
                
                # Build comprehensive organic-focused response
                out_message = f"🦠 **{disease_label}** detected (confidence: {confidence*100:.0f}%)\n\n"
                out_message += base_response
                
                # Add organic treatment recommendations
                out_message += f"\n\n🌱 **ORGANIC TREATMENT PLAN** (Priority order):\n"
                for i, treatment in enumerate(organic_treatments['primary_treatments'][:3], 1):
                    out_message += f"\n{i}. **{treatment['name']}**\n"
                    out_message += f"   📝 Recipe: {treatment['recipe']}\n"
                    out_message += f"   ⏰ Apply: {treatment['frequency']} ({treatment['timing']})\n"
                
                # Add companion plant integration
                out_message += f"\n\n🌿 **COMPANION PLANTS TO HELP**:\n"
                for comp in companion_solutions['companions'][:4]:
                    out_message += f"• {comp.title()}\n"
                out_message += f"\n💡 **Why it works**: {companion_solutions['benefits']}\n"
                out_message += f"📏 **Spacing**: {companion_solutions['spacing']}\n"
                
                print(f"[DISEASE] ✅ Organic-focused response for {disease_label}")
            else:
                # Use fallback with Ollama's knowledge (still organic-focused)
                system_msg = """You are a SUSTAINABLE AGRICULTURE expert. 
Prioritize organic, natural, and companion planting solutions.
Mention chemicals only as last resort with warnings."""

                prompt = f"""🦠 Disease: {disease_label} (confidence: {confidence*100:.0f}%)

Provide brief, organic-focused advice:
1. What this disease is
2. Natural/organic treatments (prioritize!)
3. Companion plants that help
4. Prevention tips

Focus on sustainable methods:"""

                base_response = _generate_ollama_response(prompt, system_msg, temperature=0.4, max_tokens=200)
                
                # Build organic-focused fallback response
                out_message = f"🦠 **{disease_label}** detected (confidence: {confidence*100:.0f}%)\n\n"
                out_message += base_response
                
                # Add organic treatments even without KB
                out_message += f"\n\n🌱 **ORGANIC TREATMENTS** (Try these first):\n"
                for i, treatment in enumerate(organic_treatments['primary_treatments'][:2], 1):
                    out_message += f"{i}. {treatment['name']}: {treatment['recipe']}\n"
                
                # Add companion solutions
                out_message += f"\n🌿 **Companion Plants**: {', '.join([c.title() for c in companion_solutions['companions'][:4]])}\n"
                out_message += f"💡 {companion_solutions['benefits']}\n"
                
                out_message += "\n\n💡 *Organic methods prioritized. For severe cases, consult local agricultural extension.*"
                print(f"[DISEASE] 🌱 Organic fallback for {disease_label}")
            
            state.setdefault("facts", {})["last_disease"] = disease_label
            state.setdefault("facts", {})["last_crop"] = disease_label.split("_")[0] if "_" in disease_label else "plant"
            state.setdefault("tool_traces", []).append({"tool": "detect_disease", "top1": disease_label})
        elif plan["action"] == "companions":
            plant = (plan.get("params", {}).get("plant") or state.get("facts", {}).get("last_crop") or "").strip()
            if not plant:
                return jsonify({"error": "plant required for companions", "agent": agent_info}), 400
            
            # Get traditional CSV companions
            merged = _companions_merge(plant)
            data = merged
            citations = merged.get("citations")
            
            # === NEW: Check if there's a recent disease to address ===
            last_disease = state.get("facts", {}).get("last_disease", "")
            disease_companions = None
            
            if last_disease:
                # Get disease-specific companion recommendations
                disease_companions = _get_companion_disease_solutions(last_disease, plant)
                data["disease_prevention"] = disease_companions
            
            # Craft enhanced message with disease prevention
            good = (merged.get("graph", {}).get("output", {}).get("good") or [])[:8]
            bad = (merged.get("graph", {}).get("output", {}).get("avoid") or [])[:8]
            
            out_message = f"🌿 **Companion Plants for {plant.title()}**\n\n"
            
            # Traditional companions
            out_message += f"**✅ Beneficial Companions:**\n"
            if good:
                out_message += f"{', '.join([g.title() for g in good])}\n"
            else:
                out_message += "No specific data available\n"
            
            out_message += f"\n**❌ Avoid Planting With:**\n"
            if bad:
                out_message += f"{', '.join([b.title() for b in bad])}\n"
            else:
                out_message += "No specific conflicts known\n"
            
            # Add disease prevention companions if available
            if disease_companions and disease_companions.get("found"):
                out_message += f"\n\n🦠 **Disease Prevention for {last_disease}:**\n"
                out_message += f"**Protective Companions:** {', '.join([c.title() for c in disease_companions['companions'][:4]])}\n"
                out_message += f"**Why:** {disease_companions['benefits']}\n"
                out_message += f"**Spacing:** {disease_companions['spacing']}\n"
                print(f"[COMPANIONS] ✅ Added disease prevention for {last_disease}")
            
            # Try to get KB enrichment for companion benefits
            try:
                kb_results = _kb_search(f"{plant} companion planting benefits organic polyculture", k=2)
                quality = _check_kb_quality(kb_results, min_similarity=0.5, min_length=50)
                
                if quality["sufficient"]:
                    out_message += f"\n\n💡 **Additional Tips:**\n"
                    kb_snippet = kb_results[0].get("content", "")[:200]
                    out_message += f"{kb_snippet}..."
                    citations = [{"doc_id": r.get("doc_id"), "similarity": r.get("similarity")} for r in kb_results]
            except Exception as kb_err:
                print(f"[COMPANIONS KB ERROR] {kb_err}")
            
            if merged.get("note"):
                out_message += f"\n\n📝 Note: {merged['note']}"
            
            state.setdefault("tool_traces", []).append({"tool": "companions", "plant": plant})
        elif plan["action"] == "yield":
            p = plan.get("params", {})
            crop = p.get("crop") or state.get("facts", {}).get("last_crop") or ""
            try:
                temp = float(p.get("temp")) if p.get("temp") is not None else None
                rain = float(p.get("rain")) if p.get("rain") is not None else None
            except Exception:
                temp = None; rain = None
            if temp is None or rain is None:
                return jsonify({"error": "temp and rain required for yield", "agent": agent_info}), 400
            
            # Reuse yield logic
            rows = _load_yield_rows(); items = sorted({r['item'] for r in rows}); item = _match_item(crop, items)
            pred = None
            if item:
                coeffs = _fit_coeffs(item)
                if coeffs is not None:
                    a, b, c = coeffs
                    pred = float(max(0.1, a + b*temp + c*rain))
            if pred is None:
                base = 5.0; adj = (temp - 22.0)*0.05 + (rain - 70.0)*0.01
                pred = float(max(0.1, round(base + adj, 2)))
            
            # === NEW: Calculate companion planting yield boost ===
            companion_boost = 0.0
            companion_recommendations = []
            
            try:
                # Get beneficial companions for this crop
                merged = _companions_merge(crop)
                good_companions = (merged.get("graph", {}).get("output", {}).get("good") or [])[:5]
                
                if good_companions:
                    # Estimate yield boost from companion planting (research-based percentages)
                    companion_benefits = {
                        "nitrogen_fixers": ["beans", "peas", "clover", "alfalfa"],  # +8-12% yield
                        "pest_deterrents": ["marigold", "nasturtium", "garlic", "onion"],  # +5-10% (reduced pest damage)
                        "pollinators": ["basil", "borage", "dill", "coriander"],  # +3-7% (better pollination)
                    }
                    
                    nitrogen_boost = 0
                    pest_boost = 0
                    pollinator_boost = 0
                    
                    for comp in good_companions:
                        comp_lower = comp.lower()
                        if any(nf in comp_lower for nf in companion_benefits["nitrogen_fixers"]):
                            nitrogen_boost = 10  # 10% boost
                            companion_recommendations.append(f"{comp.title()} (nitrogen fixation: +10% yield)")
                        elif any(pd in comp_lower for pd in companion_benefits["pest_deterrents"]):
                            pest_boost = 7  # 7% boost
                            companion_recommendations.append(f"{comp.title()} (pest control: +7% yield)")
                        elif any(pb in comp_lower for pb in companion_benefits["pollinators"]):
                            pollinator_boost = 5  # 5% boost
                            companion_recommendations.append(f"{comp.title()} (pollination: +5% yield)")
                    
                    # Total boost (don't stack same category, take max of each)
                    companion_boost = nitrogen_boost + pest_boost + pollinator_boost
                    
                    # Cap at 25% to be realistic
                    companion_boost = min(companion_boost, 25)
            
            except Exception as comp_err:
                print(f"[YIELD COMPANION ERROR] {comp_err}")
            
            # Calculate boosted yield
            base_yield = pred
            boosted_yield = pred * (1 + companion_boost / 100.0)
            
            data = {
                "crop": crop,
                "temp": temp,
                "rain": rain,
                "yield_t_ha": round(base_yield, 3),
                "companion_boost_percent": companion_boost,
                "boosted_yield_t_ha": round(boosted_yield, 3),
                "companion_recommendations": companion_recommendations
            }
            
            out_message = f"📊 **Yield Prediction for {crop.title()}**\n\n"
            out_message += f"**Base Yield:** {data['yield_t_ha']} t/ha\n"
            out_message += f"Temperature: {temp}°C | Rainfall: {rain}mm\n"
            
            if companion_boost > 0:
                out_message += f"\n🌱 **With Companion Planting:**\n"
                out_message += f"**Potential Yield:** {data['boosted_yield_t_ha']} t/ha (+{companion_boost}% boost!)\n\n"
                out_message += f"**Recommended Companions:**\n"
                for rec in companion_recommendations[:3]:
                    out_message += f"• {rec}\n"
                out_message += f"\n💡 *Companion planting naturally increases yields through nitrogen fixation, pest reduction, and improved pollination!*"
            else:
                out_message += f"\n💡 *Consider companion planting to boost yields naturally!*"
            
            state.setdefault("tool_traces", []).append({"tool": "yield", "crop": crop})
        elif plan["action"] == "search":
            q = plan.get("params", {}).get("query") or message
            res = _kb_search(q, k=5)
            quality = _check_kb_quality(res, min_similarity=0.55, min_length=100)
            
            print(f"[SEARCH] Query: '{q[:60]}...' | Quality: {quality['reason']} | Similarity: {quality['best_similarity']:.2f}")
            
            # Check if KB results are sufficient or if fallback is acceptable
            use_fallback = not quality["sufficient"] and plan.get("kb_fallback_ok", True)
            
            if quality["sufficient"]:
                # KB has good results - use them
                data = {"results": res, "source": "knowledge_base"}
                citations = [{"doc_id": r.get("doc_id"), "similarity": r.get("similarity")} for r in res]
                
                # Synthesize KB results into natural language using Ollama
                kb_content = "\n---\n".join([(r.get("content") or "")[:400] for r in res[:3]])
                
                system_msg = """You are an agricultural expert. Synthesize the provided knowledge base information into a clear, helpful answer.

Guidelines:
- Combine information from multiple sources coherently
- Provide specific, actionable advice
- Organize information logically (causes, symptoms, treatments, prevention)
- Be concise but comprehensive
- Use bullet points for clarity when appropriate"""

                prompt = f"""User question: {q}

Knowledge base information:
{kb_content}

Synthesize this into a helpful, well-organized answer:"""

                out_message = _generate_ollama_response(prompt, system_msg, temperature=0.3, max_tokens=200)
                print(f"[SEARCH] ✅ Using KB results (synthesized)")
                
            elif use_fallback:
                # KB insufficient but fallback allowed - use Ollama's knowledge
                data = {"results": [], "source": "llm_knowledge", "fallback_reason": quality["reason"]}
                
                system_msg = """You are an agricultural expert with extensive knowledge of farming, plant care, and agronomy.

The knowledge base doesn't have specific information for this query, so use your agricultural expertise to provide a helpful answer.

Guidelines:
- Draw on your training knowledge of agriculture
- Provide accurate, practical information
- Be honest about any limitations in specificity
- Give general best practices when specific data isn't available
- Encourage user to consult local agricultural extension services for region-specific advice"""

                prompt = f"Question: {q}\n\nProvide a comprehensive answer using your agricultural knowledge:"
                
                out_message = _generate_ollama_response(prompt, system_msg, temperature=0.5, max_tokens=250)
                out_message = "ℹ️ *Using general agricultural knowledge (limited KB data)*\n\n" + out_message
                print(f"[SEARCH] 🔄 Using fallback (reason: {quality['reason']})")
                
            else:
                # KB insufficient and fallback not appropriate
                data = {"results": res, "source": "knowledge_base_limited"}
                citations = [{"doc_id": r.get("doc_id"), "similarity": r.get("similarity")} for r in res] if res else []
                
                if res:
                    # Show what we found even if quality is low
                    kb_content = "\n---\n".join([(r.get("content") or "")[:300] for r in res[:2]])
                    out_message = f"Found limited information for '{q}':\n\n{kb_content}\n\n⚠️ This information may not be comprehensive. Consider consulting local agricultural experts."
                else:
                    out_message = f"❌ No specific information found for '{q}' in the knowledge base. Please try rephrasing your question or ask about a related topic."
                
                print(f"[SEARCH] ⚠️ Limited results, no fallback allowed")
            
            state.setdefault("tool_traces", []).append({"tool": "search", "k": 5, "quality": quality["reason"]})
        else:
            out_message = "I can help with crop/disease images, companions, yield, and KB search."
    except Exception as e:
        import traceback
        traceback.print_exc()
        out_message = f"Error executing {plan.get('action', 'unknown')}: {str(e)}"
    finally:
        # Append assistant message and save memory
        state.setdefault("messages", []).append({"role": "assistant", "content": (out_message or "")[:800], "ts": int(__import__('time').time()*1000)})
        # Update running summary lightly (first and last user messages + last tool)
        msgs = [m for m in state.get("messages", []) if m.get("role") == "user"]
        first = msgs[0]["content"] if msgs else ""
        last = msgs[-1]["content"] if msgs else ""
        state["summary"] = f"First: {first[:80]} / Last: {last[:80]}"
        _save_thread_state(thread_id, state)

    return jsonify({
        "message": out_message,
        "data": data,
        "citations": citations,
        "agent": agent_info,
        "memory": {"updated": True},
        "source": data.get("source") if data and isinstance(data, dict) else None
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=True)
