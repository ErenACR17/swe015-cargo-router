"""
Synthetic dataset for the Cargo Support Ticket Router (self-generated, template based).

The held-out TEST set is built ONLY from templates that never appear in train/validation,
so test messages have sentence structures the models have not seen during training.
"""

import random

INTENT_NAMES = [
    "delayed_delivery",
    "damaged_or_missing",
    "courier_complaint",
    "address_or_branch",
    "return_and_refund",
    "out_of_scope"
]

TEAM_MAP = {
    "delayed_delivery": "Hub Operations & Delayed Delivery Team",
    "damaged_or_missing": "Damaged & Lost Parcel Claims Team",
    "courier_complaint": "Courier Field Audit & Quality Team",
    "address_or_branch": "Address Update & Branch Rerouting Team",
    "return_and_refund": "Returns & Refund Logistics Team",
    "out_of_scope": "Human Agent (Out-of-Scope / Escalation)"
}

# Last N_TEST_TEMPLATES templates of every intent are used ONLY for the test set.
N_TEST_TEMPLATES = 2

TEMPLATES = {
    "delayed_delivery": [
        "my parcel {code} has been stuck at the sorting center for {days} days",
        "shipment {code} is delayed and passed the expected delivery date",
        "why is my package {code} still waiting at the warehouse for {days} days",
        "my express delivery {code} has not moved since last week",
        "parcel {code} is still in transit after {days} days with no update",
        "tracking shows my shipment {code} is delayed at the main hub",
        "my order {code} should have arrived {days} days ago but it is still on the way",
        "there has been no tracking movement on package {code} for {days} days",
        # --- held-out (test only) ---
        "i am still waiting for parcel {code} and it is already {days} days late",
        "the estimated arrival date for shipment {code} keeps getting pushed back",
    ],
    "damaged_or_missing": [
        "my package {code} arrived completely crushed and the item inside is broken",
        "the box for parcel {code} was torn open and one item is missing",
        "shipment {code} says delivered but nothing was left at my door it is lost",
        "my delivery {code} arrived wet and damaged i want to file a claim",
        "the glass inside my parcel {code} was shattered during shipping",
        "package {code} is missing from my porch even though status says delivered",
        "the contents of parcel {code} were smashed when i opened the box",
        "i never received shipment {code} and i think it was lost on the way",
        # --- held-out (test only) ---
        "my parcel {code} showed up dented and the product is ruined",
        "half of the things in package {code} are gone and the box was open",
    ],
    "courier_complaint": [
        "the courier never rang the doorbell and left a missed delivery note for {code}",
        "delivery driver for {code} was very rude on the phone and threw my parcel",
        "the driver refused to bring package {code} up to my apartment door",
        "courier marked my shipment {code} as customer not home while i was waiting inside",
        "the delivery person left parcel {code} in the rain without knocking",
        "i want to report the courier who handled my shipment {code} today",
        "the driver shouted at me when i asked about parcel {code}",
        "courier did not even try to call me before returning package {code}",
        # --- held-out (test only) ---
        "the man who brought shipment {code} was impolite and careless with my box",
        "i am unhappy with how the delivery guy treated me for order {code}",
    ],
    "address_or_branch": [
        "please change the delivery address for parcel {code} to my office street",
        "i want to pick up shipment {code} at the local branch instead of home delivery",
        "my package {code} was sent to the wrong branch please reroute it",
        "can i update the apartment number and zip code for delivery {code}",
        "hold my parcel {code} at the nearest branch office for self pickup",
        "i entered the wrong street address for order {code} please fix it",
        "i moved to a new house so please redirect shipment {code} there",
        "can you deliver package {code} to my neighbour address instead",
        # --- held-out (test only) ---
        "send parcel {code} to a different city because i will be travelling",
        "i would rather collect order {code} from a pickup point near my work",
    ],
    "return_and_refund": [
        "i want to return my order {code} and get a return shipping label",
        "please cancel shipment {code} and send the package back to the seller",
        "when will the courier pick up my return parcel for order {code}",
        "i returned package {code} last week when will i get my shipping refund",
        "how do i schedule a home pickup to return damaged order {code}",
        "need a free return code and refund for my shipment {code}",
        "i changed my mind about order {code} and want my money back",
        "where do i drop off the parcel {code} i am sending back",
        # --- held-out (test only) ---
        "the refund for my returned shipment {code} has not reached my account yet",
        "i do not want package {code} anymore so how can i give it back",
    ],
    "out_of_scope": [
        "what is the weather forecast for london this weekend",
        "my washing machine is making a loud noise during the spin cycle",
        "can you recommend a good laptop for university programming classes",
        "i want to book a hotel room and flight ticket for next month",
        "how do i reset my wifi router password at home",
        "what time does the football match start on television tonight",
        "give me a simple recipe for chicken soup",
        "who won the basketball game last night",
        # --- held-out (test only) ---
        "how can i learn to play the guitar quickly",
        "what is the best movie to watch on a rainy day",
    ],
}

TYPO_SAMPLES = {
    "parcel": ["parcel", "parcl", "parsel"],
    "delivery": ["delivery", "dlivery", "delivry"],
    "courier": ["courier", "coureir", "courer"],
    "damaged": ["damaged", "damagd", "damged"],
    "tracking": ["tracking", "traking", "trackng"]
}

N_PER_INTENT = 150
N_TEST_PER_INTENT = 30


def _make_example(rng, tpl, i):
    code = f"TRK-{rng.randint(1000, 9999)}"
    days = rng.randint(3, 10)
    text = tpl.format(code=code, days=days)

    # Occasional typos (25% chance) to exercise the spelling corrector
    if rng.random() < 0.25:
        for clean_w, typos in TYPO_SAMPLES.items():
            if clean_w in text:
                text = text.replace(clean_w, rng.choice(typos), 1)

    # Mild conversational noise
    if i % 5 == 0:
        text += rng.choice([
            " please help urgently",
            " i tried calling support earlier",
            " this is my second message today",
            " thank you for checking"
        ])
    return text


def load_dataset(seed=42):
    """
    Returns train / validation / held-out test splits.
    train+val  : first (len(templates) - N_TEST_TEMPLATES) templates   (120 per intent -> 100 / 20)
    test       : last  N_TEST_TEMPLATES templates, never seen in training (30 per intent)
    """
    rng = random.Random(seed)
    train_val, test_data = [], []

    for label_id, intent in enumerate(INTENT_NAMES):
        templates = TEMPLATES[intent]
        seen, unseen = templates[:-N_TEST_TEMPLATES], templates[-N_TEST_TEMPLATES:]

        for i in range(N_PER_INTENT - N_TEST_PER_INTENT):
            text = _make_example(rng, seen[i % len(seen)], i)
            train_val.append({"text": text, "intent": intent, "label": label_id})
        for i in range(N_TEST_PER_INTENT):
            text = _make_example(rng, unseen[i % len(unseen)], i)
            test_data.append({"text": text, "intent": intent, "label": label_id})

    rng.shuffle(train_val)
    rng.shuffle(test_data)
    n_val = int(0.15 * len(train_val))
    val_data = train_val[:n_val]
    train_data = train_val[n_val:]
    return train_data, val_data, test_data
