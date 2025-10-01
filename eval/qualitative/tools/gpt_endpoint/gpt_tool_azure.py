# tools/gpt_endpoint/gpt_tool_azure.py
from __future__ import annotations
import base64, json, re
from typing import Dict, Optional, List
from openai import AzureOpenAI
from models.ScoreOutput import ScoreOutput
JSON_RE = re.compile(r"\{.*\}", re.S)

DEFAULT_PROMPT = """"You are an expert in computer vision evaluation, specializing in image restoration and object detection.
Your job is to act as a STRICT and CONCISE judge.

GENERAL RULES
- Judge everything RELATIVE TO THE FOGGY INPUT (F). Do NOT reward 'no change'.
- Only award high scores (4–5) when the Candidate (C) is close to the Ground Truth (GT).
- Be conservative: if unsure, choose the lower score.
- Explanations must be short, factual, and cite concrete visual evidence (e.g., regions/objects).

You are evaluating a DEHAZED Candidate (C) against its FOGGY input (F) and CLEAR Ground Truth (GT).
Look at all three carefully and score each category using the anchors below.

[IMPROVEMENT AXES — USED IN TOTAL (0–15)]
1) Visibility Restoration (0–5)
   How much more of mid/far/background is visible in C vs F.
   - 0 = no improvement or worse than F
   - 1–2 = foreground partly improved; mid/far still obscured
   - 3 = foreground + some mid improved; background largely unclear
   - 4 = foreground + mid restored; background mostly legible
   - 5 = near-GT across depths

2) Boundary Clarity (0–5)
   Sharpness and definition of contours and thin structures vs F.
   - 0 = edges worse than F
   - 1–2 = slight sharpening; borders remain fuzzy/misaligned
   - 3 = main edges clear; fine details unclear
   - 4 = clear boundaries; minor misalignments
   - 5 = GT-like crisp, well-aligned boundaries

3) Perceived Detectability (0–5)
   How much easier generic object detection would be on C vs F.
   - 0 = harder than F
   - 1–2 = slight gains; many ambiguous
   - 3 = large/near objects detectable; small/far unclear
   - 4 = most objects clearly detectable
   - 5 = GT-like detectability

[DIAGNOSTICS — NOT USED IN TOTAL]
4) Visual Distortion (0–5)
   Artifacts introduced by C vs F: halos, false textures, oversharpening, color shifts.
   - 0 = severe artifacts
   - 1–2 = noticeable artifacts
   - 3 = moderate artifacts
   - 4 = minor artifacts
   - 5 = no visible artifacts (GT-like realism)

5) Scene Consistency (0–5)
   Preservation of global layout and geometry vs GT.
   - 0 = severe warps/missing/duplicated structures
   - 1–2 = partly recognizable; misalignments in multiple areas
   - 3 = mostly consistent; some warped geometry
   - 4 = reliable layout; minor shifts only
   - 5 = GT-like layout/alignment
   * Give 4–5 ONLY if you can point to concrete evidence of correct alignment in the explanation.

6) Object Consistency (0–5) — PRESENCE/PLACEMENT ONLY (no counting)
   Agreement with GT in where major object types (cars, people, signs/lights, buses/trucks) appear.
   - 0 = many hallucinated/missing or clearly misplaced objects
   - 1–2 = several presence/placement mismatches
   - 3 = mostly aligned, some mismatches in regions
   - 4 = nearly all placements correct; minor mismatches
   - 5 = GT-like presence/placement
   * Do NOT use numeric counts. Judge coarse presence/placement across the scene.

6) Relation Consistency (0–5, diagnostic; NOT used in total)
   Do key spatial relations match GT? (e.g., car on road; sign above road; building behind car)
   - 0  = pervasive mismatches, 5 = GT-like relations;
   Evidence rule:
     • If relation_consistency ≥ 4, include ≥1 concrete confirmation (e.g., "sign above road in TR matches GT").
     • If relation_consistency ≤ 2, include ≥1 concrete mismatch in relation_mismatches.
Return:
  "relation_consistency": int,
  "relation_mismatches": [ {{"subject":"car","relation":"on","object":"road","region":"MC","mismatch":"missing|spurious|moved"}} ]


[FLAGS]
- If C is visually indistinguishable from F: set Visibility/Boundary/Detectability to 0. (Diagnostics still apply.) 
- If C is identical to GT: give 5 on all categories.
- If C is blank/black/white or unrelated to F: give 0 on all categories.

[OUTPUT RULES]
- Assign INTEGER scores (0–5) for each category.
- Be strict: default low unless C is clearly close to GT.
- Explanation: 2–4 short sentences comparing C to F and GT; mention concrete evidence (e.g., “far-left building edges remain soft,” “traffic sign in top-right is now legible,” “sky exhibits haloing around poles”).
- Compute `total` as:
    total = visibility_restoration + boundary_clarity + perceived_detectability   // range 0–15 ONLY
  Do NOT include other categories in `total`.

[OUTPUT FORMAT — JSON ONLY]
{{
  "visibility_restoration": int,
  "visual_distortion": int,
  "boundary_clarity": int,
  "scene_consistency": int,
  "object_consistency": int,
  "perceived_detectability": int,
  "relation_consistency: int,
  "relation_mismatches": [  // optional; only if relation_consistency < 5
    {{"subject":"string","relation":"string","object":"string","region":"string","mismatch":"missing|spurious|misplaced"}} 
  "explanation": "string"
}}

""".strip()




def safe_format(value) -> str:
    """Format metric values for prompt injection.
    - None -> 'NA'
    - int/float -> 4 decimal string
    - other -> str(value)
    """
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def safe_get(d: dict, key: str):
    """Safely get a key from a dict (or None if dict/key missing)."""
    if d is None:
        return None
    return d.get(key, None)

def build_rubric_prompt(
    metrics: Dict = None,
    fog_diff: Dict = None,
    cand_diff: Dict = None,
    gt_diff: Dict = None,
) -> str:
    if metrics is not None:
        fog = metrics["foggy_vs_gt"]; cand = metrics["cand_vs_gt"]


    return DEFAULT_PROMPT.format(
        # difficulty (GT / Candidate / Foggy)
        gt_edge_density=safe_format(safe_get(gt_diff, "edge_density")),
        gt_rms_contrast=safe_format(safe_get(gt_diff, "rms_contrast")),
        gt_entropy=safe_format(safe_get(gt_diff, "entropy")),
        gt_dark_channel=safe_format(safe_get(gt_diff, "dark_channel")),

        cand_edge_density=safe_format(safe_get(cand_diff, "edge_density")),
        cand_rms_contrast=safe_format(safe_get(cand_diff, "rms_contrast")),
        cand_entropy=safe_format(safe_get(cand_diff, "entropy")),
        cand_dark_channel=safe_format(safe_get(cand_diff, "dark_channel")),

        fog_edge_density=safe_format(safe_get(fog_diff, "edge_density")),
        fog_rms_contrast=safe_format(safe_get(fog_diff, "rms_contrast")),
        fog_entropy=safe_format(safe_get(fog_diff, "entropy")),
        fog_dark_channel=safe_format(safe_get(fog_diff, "dark_channel")),

        # numeric context (SSIM/LPIPS)
        fog_ssim=safe_format(fog.get("ssim")),
        fog_lpips=safe_format(fog.get("lpips")),
        cand_ssim=safe_format(cand.get("ssim")),
        cand_lpips=safe_format(cand.get("lpips")),
    )





def _b64_data_uri(path: str) -> str:
    """Encode an image as a base64 data URI string."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{data}"


def _extract_json_with_pydantic(raw_text: str) -> dict:
    m = JSON_RE.search(raw_text)
    if not m:
        raise ValueError("No JSON object found in model output.")
    model = ScoreOutput.model_validate_json(m.group(0))
    print("Parsed model output:", model)
    out = model.model_dump()
    out["total"] = model.total  # add computed field
    return out



class AzureVLMScorer:
    """Azure OpenAI rubric scorer using base64 data URIs for image upload."""

    def __init__(
        self,
        client: AzureOpenAI,
        deployment: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        top_p: float = 1.0,

    ):
        self.client = client
        self.deployment = deployment
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = 42  # for reproducibility

    def score_triplet(
        self,
        foggy_path: str,
        cand_path: str,
        gt_path: str,
        metrics: Dict = None,
        fog_difficulty: Dict = None,
        cand_difficulty: Dict = None,
        gt_difficulty: Dict = None,
        prompt_override: Optional[str] = None,
        retries: int = 3,
    ) -> Dict[str, int]:
        prompt = prompt_override or build_rubric_prompt(
            metrics,
            fog_diff=fog_difficulty,
            cand_diff=cand_difficulty,
            gt_diff=gt_difficulty,
        )

        foggy_b64 = _b64_data_uri(foggy_path)
        cand_b64  = _b64_data_uri(cand_path)
        gt_b64    = _b64_data_uri(gt_path)

        user_parts: List[Dict] = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "FOGGY input image:"},
            {"type": "image_url", "image_url": {"url": foggy_b64}},
            {"type": "text", "text": "CANDIDATE dehazed image:"},
            {"type": "image_url", "image_url": {"url": cand_b64}},
            {"type": "text", "text": "GROUND TRUTH clear image:"},
            {"type": "image_url", "image_url": {"url": gt_b64}},
        ]


        last_err = None
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.deployment,  # Azure: your deployment name
                    messages=[
                        {"role": "system", "content": "You are a meticulous, impartial computer-vision judge."},
                        {"role": "user", "content": user_parts},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                text = resp.choices[0].message.content
                return _extract_json_with_pydantic(text)
            except Exception as e:
                last_err = e
                # tighten prompt and retry
                user_parts[0]["text"] = prompt + "\nReturn ONLY valid JSON. No explanations."
        raise RuntimeError(f"Azure VLM scoring failed after {retries} attempts: {last_err}")