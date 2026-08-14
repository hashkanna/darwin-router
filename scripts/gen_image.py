"""Generate an image via DashScope native API (sync) and download it.
Usage: python scripts/gen_image.py <model> <outfile.png> < prompt_on_stdin"""

import sys

import httpx

from router import config

model, outfile = sys.argv[1], sys.argv[2]
prompt = sys.stdin.read().strip()

r = httpx.post(
    config.IMAGEGEN_ENDPOINT,
    headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"},
    json={"model": model,
          "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
          "parameters": {"size": "1328*1328", "n": 1}},
    timeout=300,
)
r.raise_for_status()
parts = r.json()["output"]["choices"][0]["message"]["content"]
url = next(c["image"] for c in parts if isinstance(c, dict) and c.get("image"))
img = httpx.get(url, timeout=120)
img.raise_for_status()
with open(outfile, "wb") as f:
    f.write(img.content)
print(f"{outfile}  ({len(img.content)//1024} KB)  <- {model}")
