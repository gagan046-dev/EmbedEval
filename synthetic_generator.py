import json
import random
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


SYSTEM_PROMPT = """<system_role>
You are a Lead RAG Evaluation Engineer specializing in adversarial benchmarking for information retrieval systems. Your sole objective is to generate challenging, non-trivial, production-grade evaluation triplets (Question, Expected Context, Ground Truth) from raw document chunks.
</system_role>

<quality_standards>
1. NO TRIVIAL QUESTIONS: Avoid surface-level questions that can be solved by simple keyword matching.
2. NO SCRIPTED PHRASING: Never begin questions with "According to the text...", "In this document...", or "Based on section...". Phrase questions as a real end-user would naturally ask them.
3. FAITHFUL & CONCISE ANSWERS: Ground truths must be 100% factual and directly answer the question in 1-3 clear sentences using only information from the chunk.
4. EXACT CONTEXT EXTRACT: The expected_context must be an EXACT verbatim substring from the source chunk that contains the necessary evidence.
</quality_standards>

<question_types>
Randomly select ONE of the following question archetypes per generation call:
- SPECIFIC_FACT: Direct, dense factual query requiring exact numerical, technical, or entity details.
- INFERENCE_REASONING: Requires understanding cause-and-effect, implicit relationships, or logical consequences described in the context.
- CONDITIONAL_OPERATIONAL: Asks under what specific circumstances, edge cases, or prerequisites an action or rule applies.
- STRUCTURAL_TABULAR: Asks about structured data, ratios, comparisons, or metric lists present in the text.
</question_types>"""


USER_PROMPT_TEMPLATE = """<instructions>
You will receive a document context snippet. Analyze it, think in a scratchpad, then output ONE high-quality benchmark triplet.

Follow this execution loop precisely:
1. Read the provided <context_snippet>.
2. Inside <scratchpad>:
   - Identify key entities, facts, edge cases, or logic in the context.
   - Select a question archetype from the allowed list.
   - Draft a candidate question and test if it is too easy or uses fluff phrasing.
   - Refine the question into natural language.
   - Extract the EXACT verbatim quote for expected_context.
   - Write the concise ground_truth answer.
3. Output strictly inside <json_output> tags.
</instructions>

<few_shot_examples>
<example>
<context_snippet>
If the hydraulic pressure drops below 1,200 PSI during secondary pump engagement, valve B-7 automatically triggers a 30-second thermal bypass loop. This loop redirects fluid to the primary heat exchanger before shutting down the auxiliary turbine to prevent rotor cavitation.
</context_snippet>

<scratchpad>
- Key facts: Pressure threshold < 1,200 PSI, secondary pump engagement, valve B-7, 30s thermal bypass loop, redirects to primary heat exchanger, prevents rotor cavitation.
- Archetype: CONDITIONAL_OPERATIONAL
- Draft Q: What happens when hydraulic pressure drops below 1200 PSI? (Too simple)
- Refined Q: How does the system prevent rotor cavitation if secondary pump hydraulic pressure experiences a sudden drop?
- Expected Context: "valve B-7 automatically triggers a 30-second thermal bypass loop. This loop redirects fluid to the primary heat exchanger before shutting down the auxiliary turbine to prevent rotor cavitation."
- Ground Truth: When pressure drops below 1,200 PSI, valve B-7 triggers a 30-second thermal bypass loop that routes fluid to the primary heat exchanger prior to auxiliary turbine shutdown.
</scratchpad>

<json_output>
{{
  "question_type": "CONDITIONAL_OPERATIONAL",
  "question": "How does the system prevent rotor cavitation if secondary pump hydraulic pressure experiences a sudden drop?",
  "expected_context": "valve B-7 automatically triggers a 30-second thermal bypass loop. This loop redirects fluid to the primary heat exchanger before shutting down the auxiliary turbine to prevent rotor cavitation.",
  "ground_truth": "When hydraulic pressure drops below 1,200 PSI, valve B-7 triggers a 30-second thermal bypass loop that redirects fluid to the primary heat exchanger before shutting down the auxiliary turbine."
}}
</json_output>
</example>
</few_shot_examples>

<context_snippet>
{chunk_text}
</context_snippet>

Begin your response now with <scratchpad>."""


class SyntheticDatasetGenerator:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def _chunk_document(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if start + chunk_size >= len(text):
                break
            start += chunk_size - overlap
        return chunks

    def _stratified_sample(self, chunks: list[str], num_samples: int) -> list[tuple[int, str]]:
        zone_size = len(chunks) / num_samples
        sampled = []
        for i in range(num_samples):
            zone_start = int(i * zone_size)
            zone_end = min(int((i + 1) * zone_size), len(chunks))
            idx = random.randint(zone_start, max(zone_start, zone_end - 1))
            sampled.append((idx, chunks[idx]))
        return sampled

    def _extract_json(self, text: str) -> dict | None:
        match = re.search(r"<json_output>\s*(.*?)\s*</json_output>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        try:
            start = text.rfind("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def _generate_from_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        variation_id: int,
    ) -> dict:
        position_pct = round((chunk_index / max(total_chunks, 1)) * 100)
        prompt = USER_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
        prompt += f"\n<variation_id>{variation_id}</variation_id>"
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = self._extract_json(message.content[0].text.strip())
        if not parsed or not all(parsed.get(key) for key in ("question", "expected_context", "ground_truth")):
            raise ValueError("Claude returned an invalid QA triplet.")

        expected_context = parsed["expected_context"].strip()
        if expected_context not in chunk_text:
            raise ValueError("Claude returned expected_context that is not verbatim source text.")

        return {
            "question": parsed["question"].strip(),
            "expected_context": expected_context,
            "ground_truth": parsed["ground_truth"].strip(),
            "category": parsed.get("question_type", "SPECIFIC_FACT"),
            "source_position_pct": position_pct,
        }

    def _deduplicate(self, pairs: list[dict]) -> list[dict]:
        seen, unique = set(), []
        for p in pairs:
            h = hashlib.md5(p.get("question", "").strip().lower().encode()).hexdigest()[:12]
            if h not in seen and len(p.get("question", "")) > 10:
                seen.add(h)
                unique.append(p)
        return unique

    def generate_qa_pairs(self, document_text: str, num_questions: int = 15) -> list[dict]:
        num_questions = min(max(1, num_questions), 50)

        chunks = self._chunk_document(document_text)
        if not chunks:
            raise ValueError("Document produced zero chunks after splitting.")

        all_pairs = []
        errors = []
        attempts = 0
        max_attempts = num_questions * 3

        while len(all_pairs) < num_questions and attempts < max_attempts:
            batch_size = min(num_questions - len(all_pairs), max_attempts - attempts)
            sampled = self._stratified_sample(chunks, batch_size)
            with ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
                futures = [
                    executor.submit(
                        self._generate_from_chunk,
                        chunk,
                        chunk_index,
                        len(chunks),
                        attempts + offset,
                    )
                    for offset, (chunk_index, chunk) in enumerate(sampled)
                ]
                attempts += batch_size
                for future in as_completed(futures):
                    try:
                        all_pairs.append(future.result())
                    except Exception as exc:
                        errors.append(str(exc))
            all_pairs = self._deduplicate(all_pairs)

        if len(all_pairs) < num_questions:
            detail = f" Last error: {errors[-1]}" if errors else ""
            raise RuntimeError(
                f"Generated {len(all_pairs)} of {num_questions} unique QA pairs "
                f"after {attempts} attempts.{detail}"
            )

        all_pairs.sort(key=lambda item: item.get("source_position_pct", 0))
        return all_pairs[:num_questions]
