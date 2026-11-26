"""Near-duplicate support tickets landed on both sides of a random split.

Customers resend the same complaint ("Re: Fwd: ...", different ticket
number, extra spaces, the earlier mail quoted below). A random row split put
one copy in training and another in test, so the test score rewarded
memorising tickets the model had already seen. The fix normalises ticket
text into a fingerprint, groups every copy under it, and assigns whole
groups to train or test with a stable hash, so no ticket appears on both
sides.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

TEST_PERCENT = 20
HASH_BUCKETS = 100
SEED = 11
UNIQUE_TICKETS = 400
RESENT_TICKETS = 160                 # tickets customers sent more than once
MAX_EXTRA_COPIES = 3
CATEGORIES = ["billing", "login", "shipping", "refund"]
WORDS = ["order", "card", "charged", "twice", "password", "reset", "parcel", "late",
         "refund", "missing", "account", "locked", "invoice", "wrong", "address", "app"]

TICKET_REFERENCE = re.compile(r"\(?\s*ticket\s*#\d+\s*\)?")
REPLY_PREFIX = re.compile(r"^((re|fwd?):\s*)+", re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Ticket:
    text: str
    category: str


# ---------------------------------------------------------------- data
def build_tickets(seed: int = SEED) -> list[Ticket]:
    """Unique tickets plus resent copies that differ only in noise."""
    rng = random.Random(seed)
    originals: list[Ticket] = []
    seen_bodies: set[str] = set()
    while len(originals) < UNIQUE_TICKETS:
        body = " ".join(rng.choice(WORDS) for _ in range(8))
        if body not in seen_bodies:
            seen_bodies.add(body)
            originals.append(Ticket(body, rng.choice(CATEGORIES)))
    tickets = list(originals)
    for original in rng.sample(originals, RESENT_TICKETS):
        for _ in range(rng.randint(1, MAX_EXTRA_COPIES)):
            tickets.append(Ticket(resend(original.text, rng), original.category))
    rng.shuffle(tickets)
    return tickets


def resend(body: str, rng: random.Random) -> str:
    """How a resent ticket looks: reply prefix, new number, spacing, quote."""
    number = rng.randint(10_000, 99_999)
    return f"Re: Fwd:  {body.upper()}  (ticket #{number})\n> previous message quoted"


# ---------------------------------------------------------------- fingerprint
def normalize(text: str) -> str:
    """Reduce a ticket to the words that make it *this* complaint.

    Quoted lines and reply prefixes are added by mail clients, ticket numbers
    by our helpdesk; none of them change what the customer asked.
    """
    text = unicodedata.normalize("NFKC", text)
    kept_lines = []
    for line in text.splitlines():
        if not line.lstrip().startswith(">"):
            kept_lines.append(line)
    text = " ".join(kept_lines).casefold()
    text = REPLY_PREFIX.sub("", text.strip())
    text = TICKET_REFERENCE.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def fingerprint(text: str) -> str:
    """Stable id for a ticket's normalised text (same on every machine)."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- splits
def naive_random_split(tickets: list[Ticket], seed: int = SEED) -> tuple[list[Ticket], list[Ticket]]:
    """Original: shuffle rows and cut 80/20, ignoring duplicates."""
    shuffled = list(tickets)
    random.Random(seed).shuffle(shuffled)
    cut = len(shuffled) * (100 - TEST_PERCENT) // 100
    return shuffled[:cut], shuffled[cut:]


def group_split(tickets: list[Ticket]) -> tuple[list[Ticket], list[Ticket]]:
    """Fix: every copy of a ticket goes to the side its fingerprint hashes to.

    The bucket comes from the fingerprint, not a random draw, so a copy that
    arrives next month lands on the same side as its earlier copies.
    """
    train: list[Ticket] = []
    test: list[Ticket] = []
    for ticket in tickets:
        bucket = int(fingerprint(ticket.text)[:8], 16) % HASH_BUCKETS
        if bucket < TEST_PERCENT:
            test.append(ticket)
        else:
            train.append(ticket)
    return train, test


# ---------------------------------------------------------------- evaluation
def leaked_share(train: list[Ticket], test: list[Ticket]) -> float:
    """Share of test tickets whose fingerprint also appears in training."""
    train_prints = {fingerprint(ticket.text) for ticket in train}
    leaked = sum(1 for ticket in test if fingerprint(ticket.text) in train_prints)
    return leaked / len(test)


def memorizer_accuracy(train: list[Ticket], test: list[Ticket]) -> float:
    """Accuracy of a 'model' that only remembers training tickets.

    It has learned nothing general: unseen tickets get the most common
    category. Its score shows how much of a test result is pure recall.
    """
    remembered = {fingerprint(ticket.text): ticket.category for ticket in train}
    most_common = Counter(ticket.category for ticket in train).most_common(1)[0][0]
    correct = 0
    for ticket in test:
        guess = remembered.get(fingerprint(ticket.text), most_common)
        if guess == ticket.category:
            correct += 1
    return correct / len(test)
