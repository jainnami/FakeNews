"""
Classify 5000+ unique context strings into ~8 clean groups using OpenAI.
Uses gpt-4o-mini for cost-effectiveness (~$0.01-0.03 total).

Usage:
    1. pip install openai pandas
    2. Set your API key below
    3. python classify_contexts.py
    4. Output: context_groups.csv (context -> context_group mapping)
"""

import os
import json
import time
import asyncio
import pandas as pd
from openai import AsyncOpenAI

# ============================================================
# SET YOUR API KEY HERE
# ============================================================
OPENAI_API_KEY = '....'
# ============================================================

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Max parallel requests (stay within rate limits)
# gpt-4o-mini: 15 is fine (high TPM limit)
# gpt-4o: use 2-3 (30K TPM limit)
MAX_CONCURRENT = 3

# Target groups
GROUPS = [
    "Interview",
    "Speech/Rally",
    "TV/Radio",
    "Social Media",
    "Press Release/Statement",
    "Debate",
    "Campaign Ad",
    "News Article/Editorial",
    "Legislative/Official",
    "Other"
]

SYSTEM_PROMPT = f"""You are a classifier. Given a list of political statement contexts, 
assign each one to exactly ONE of these groups:

{json.dumps(GROUPS)}

Rules:
- "an interview", "a radio interview", "interview on CNN" → Interview
- "a speech", "campaign rally", "remarks at..." → Speech/Rally
- "a TV ad", "radio ad", "campaign ad", "an ad" → Campaign Ad
- "a TV show", "Meet the Press", "Fox News" → TV/Radio
- "a tweet", "Facebook post", "social media" → Social Media
- "a press release", "news release", "a statement" → Press Release/Statement
- "a debate", "gubernatorial debate" → Debate
- "a news story", "editorial", "op-ed", "column" → News Article/Editorial
- "floor speech", "committee hearing", "Senate session" → Legislative/Official
- Everything else → Other

Return ONLY a JSON array of objects: [{{"context": "...", "group": "..."}}]
No explanation. No markdown. Just the JSON array."""


async def classify_batch(contexts: list[str], batch_id: int, semaphore: asyncio.Semaphore, max_retries: int = 3) -> list[dict]:
    """Send a batch of contexts to GPT-4o for classification."""
    user_msg = json.dumps(contexts)

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0,
                    max_tokens=4096
                )
                raw = response.choices[0].message.content.strip()
                # Handle potential markdown wrapping
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                print(f"  ✓ Batch {batch_id} done")
                return json.loads(raw)
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Batch {batch_id} attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)

    # Fallback: assign all to "Other"
    return [{"context": c, "group": "Other"} for c in contexts]


async def run():
    # Load data
    print("Loading data...")
    train = pd.read_csv("train.tsv", sep="\t")
    test = pd.read_csv("test.tsv", sep="\t")
    val = pd.read_csv("valid.tsv", sep="\t")
    combined = pd.concat([train, test, val], ignore_index=True)

    unique_contexts = combined["context"].dropna().unique().tolist()
    print(f"Found {len(unique_contexts)} unique contexts")

    # Build batches
    BATCH_SIZE = 100
    batches = [unique_contexts[i : i + BATCH_SIZE] 
               for i in range(0, len(unique_contexts), BATCH_SIZE)]
    print(f"Sending {len(batches)} batches with up to {MAX_CONCURRENT} in parallel...")

    # Fire all batches concurrently (throttled by semaphore)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [classify_batch(batch, i + 1, semaphore) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    # Flatten
    all_results = [item for batch_result in results for item in batch_result]

    # Build mapping DataFrame
    mapping_df = pd.DataFrame(all_results)
    mapping_df.columns = ["context", "context_group"]

    # Validate groups
    invalid = mapping_df[~mapping_df["context_group"].isin(GROUPS)]
    if len(invalid) > 0:
        print(f"  Fixing {len(invalid)} invalid groups → 'Other'")
        mapping_df.loc[~mapping_df["context_group"].isin(GROUPS), "context_group"] = "Other"

    # Save
    mapping_df.to_csv("context_groups.csv", index=False)
    print(f"\nSaved context_groups.csv ({len(mapping_df)} rows)")
    print(f"\nGroup distribution:")
    print(mapping_df["context_group"].value_counts().to_string())


if __name__ == "__main__":
    asyncio.run(run())
