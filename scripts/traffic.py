"""Demo traffic generator. Sends varied requests through the router at a
gentle randomized pace so the Darwin loop has something to learn from all day.
Seed queries are written independently of data/holdout.jsonl on purpose: the
eval curve must measure generalization, not phrasing overlap.
Run: python scripts/traffic.py  (Ctrl-C to stop)"""

import random
import sys
import time

import httpx

BASE = "http://127.0.0.1:8787"
INTERVAL_RANGE = (20, 70)  # seconds between requests
IMAGE_SHARE = 0.03

SIMPLE = [
    "hey! quick question, are you there?",
    "what's the chemical symbol for gold?",
    "turn this into title case: the quick brown fox",
    "how many milliliters in a tablespoon?",
    "give me a rhyme for 'orange'... or the closest thing to one",
    "what language do they speak in Brazil?",
    "make this friendlier: payment overdue, pay immediately",
    "what's the plural of 'cactus'?",
    "convert 72 fahrenheit to celsius",
    "who painted the Mona Lisa?",
    "give me a one-line bio for a barista named Milo",
    "what day of the week was January 1, 2000?",
    "shorten this for a tweet: our cafe now opens at 7am on weekdays and 8am on weekends",
    "what's the opposite of 'transparent'?",
    "how do I say thank you in Japanese?",
    "pick a team name for a trivia night, something punny",
    "what's the tallest mountain in Africa?",
    "fix the typo: I recieved your package yesterday",
    "list the days of the weekend in French",
    "what does the acronym NASA stand for?",
    "write a two-word caption for a sunset photo",
    "is 91 an odd number?",
    "what continent is Egypt in?",
    "give me a polite out-of-office one-liner",
    "capitalize properly: dr. smith lives on elm street",
    "what's the currency of Switzerland?",
    "suggest a name for a golden retriever puppy",
    "how many sides does a hexagon have?",
    "translate 'where is the train station' into German",
    "what vitamin do you get from sunlight?",
]

REASONING = [
    "Write a SQL query that finds customers who ordered in January but never again, and explain the anti-join.",
    "My Flask app leaks memory under gunicorn but not locally. Walk me through how you'd diagnose it.",
    "Two dice are rolled. What's the probability the sum is prime? Show the full enumeration.",
    "Sketch an architecture for rate-limiting an API across multiple regions, and discuss clock skew.",
    "Prove by induction that 1+3+5+...+(2n-1) = n^2.",
    "Refactor a God-class UserManager that handles auth, billing and emails — outline the steps and risks.",
    "A rope burns unevenly in exactly 60 minutes. With two ropes, measure 45 minutes. Explain carefully.",
    "Compare eventual vs strong consistency for a shopping-cart service, and recommend one with tradeoffs.",
    "Write a Python generator that yields sliding windows over an iterator without materializing it, and explain edge cases.",
    "If inflation is 3.5% and my salary grows 2% yearly, how much purchasing power do I lose over a decade? Show the math.",
    "Explain why quicksort is O(n^2) worst-case but usually beats mergesort in practice.",
    "Design the state machine for a traffic light with a pedestrian button, including timing constraints.",
    "I need to dedupe 200M records by fuzzy name match on one machine with 16GB RAM. Propose an approach.",
    "Derive the derivative of x^x from first principles.",
    "Plan a phased rollout strategy for switching payment providers with zero downtime.",
    "Why do we need positional encodings in transformers? Explain what breaks without them.",
    "Given a linked list, detect a cycle in O(1) space and prove why the two-pointer method works.",
    "Should a startup build on Kubernetes from day one? Argue both sides, then commit to a recommendation.",
    "A car travels the first half of a trip at 30 km/h and the second half at 60 km/h. What's the average speed, and why isn't it 45?",
    "Model the seating problem: 8 people around a round table, two refuse to sit together. How many arrangements?",
    "Explain CAP theorem using a concrete two-node example with a network partition.",
    "Write a shell one-liner to find the 10 largest files under a directory, then explain each flag.",
    "How would you migrate a live Postgres table from int to bigint primary keys without locking writes?",
    "Estimate the bandwidth needed to stream security footage from 400 stores, stating your assumptions.",
]

# genuinely frontier-worthy: families empirically verified to break the 4B
# (stateful sims with twists, symbol manipulation, python semantics traps) —
# siblings of the hard holdout items, never the same content
HARD = [
    "Take the string SUPERLINKED. Rotate it right by 2, delete letters at positions divisible by 3 (1-indexed), then shift each remaining letter back by one (A wraps to Z). Show every step.",
    "Predict the exact output: x = {'a': 1}\ndef f(d=x):\n    d['b'] = len(d)\n    return d\nprint(f() is f())\nprint(f())\nx['c'] = 9\nprint(f())",
    "A tank fills via pump P in 4 hours, but P drops to one third of its rate whenever the tank is more than two-thirds full. Drain D empties a full tank in 12 hours. Starting empty with both running, when is the tank full?",
    "A farmer must ferry a fox, a chicken, and a sack of grain. In this version the FOX eats the GRAIN if left alone with it, and nothing else conflicts. Boat holds farmer plus one. Minimum crossings and the sequence?",
    "A dog runs at 12 km/h between two hikers walking toward each other at 4 and 5 km/h, starting 18 km apart, but the dog pauses 3 minutes at each hiker before turning. How far does the dog RUN before the hikers meet?",
    "Take ROUTERBRAIN: swap the first and second halves, delete all vowels, then reverse the result. What string remains? Step by step.",
    "Predict the output and explain via the MRO: class P: v = 'p'\nclass Q(P): pass\nclass R(P): v = 'r'\nclass S(Q, R): pass\nprint(S.v)\nQ.v = 'q'\nprint(S.v)\ndel Q.v\nprint(S.v)",
    "Without running it, what does this print? nums = [10, 20, 30]\ngen = (n + 1 for n in nums)\nnums.remove(20)\nprint(list(gen))\nfns = [lambda m=n: m for n in nums]\nprint([f() for f in fns])\nprint([f() for f in [lambda: n for n in nums]])",
    "A knockout tournament starts with 27 players; odd rounds give one bye. But every bye recipient must play a tie-break exhibition match that counts toward the match total. Total matches including exhibitions, and total byes? Trace it.",
    "An elevator starts at floor 5 of a 10-floor building and executes: up 3, down 6, up 4, down 2, up 5 — but any move that would pass floor 1 or floor 10 stops AT that limit instead. Which floor does it end on, and how many floors did it actually travel in total?",
]


# deliberately mid-difficulty: these should tickle the judge gate
BOUNDARY = [
    "what's 15% tip on an $84 dinner for four, split evenly?",
    "explain the difference between a list and a tuple in Python",
    "summarize the plot of Romeo and Juliet in three sentences",
    "is it cheaper to lease or buy a car? just the key considerations",
    "convert this recipe from 4 servings to 6: 200g flour, 3 eggs, 150ml milk",
    "what's the difference between HTTP and HTTPS?",
    "why is the ocean salty?",
    "write a haiku about deadlines",
    "explain what an API is to my grandmother",
    "how does compound interest work, in plain words?",
    "what's the difference between weather and climate?",
    "draft a 3-line cover letter opener for a junior data analyst role",
]

IMAGEGEN = [
    "generate an image of a lighthouse made of circuit boards",
    "draw a picture of two rivers merging into one glowing stream",
    "create an image of a chess piece looking at a mirror",
]


def pick() -> str:
    r = random.random()
    if r < IMAGE_SHARE:
        return random.choice(IMAGEGEN)
    if r < 0.43:
        return random.choice(SIMPLE)
    if r < 0.68:
        return random.choice(REASONING)
    if r < 0.90:
        return random.choice(HARD)
    return random.choice(BOUNDARY)


def vary(q: str) -> str:
    r = random.random()
    if r < 0.15:
        return random.choice(["please, ", "hey — ", "quick one: ", "ok so ", ""]) + q
    if r < 0.25:
        return q + random.choice([" thanks!", " (in a hurry)", " keep it brief", ""])
    return q


def main():
    n = 0
    while True:
        q = vary(pick())
        try:
            r = httpx.post(f"{BASE}/v1/chat/completions",
                           json={"model": "darwin", "messages": [{"role": "user", "content": q}]},
                           timeout=180)
            d = r.json().get("x_darwin", {})
            n += 1
            print(f"[{n}] {d.get('route', '?'):10s} margin={d.get('margin')} "
                  f"{d.get('latency_ms')}ms :: {q[:70]}", flush=True)
        except Exception as e:
            print(f"[!] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(random.uniform(*INTERVAL_RANGE))


if __name__ == "__main__":
    main()
