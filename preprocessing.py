import re

# Domain word list for spelling correction
CARGO_WORDS = [
    "parcel", "package", "shipment", "delivery", "courier", "driver",
    "tracking", "delayed", "stuck", "sorting", "warehouse", "branch",
    "damaged", "broken", "crushed", "missing", "lost", "address",
    "return", "refund", "cancel", "customs", "insurance", "receipt"
]

# ELIZA-style rule responses for small talk and greetings
SMALL_TALK_RULES = [
    (r"^(hi|hello|hey|good morning|good afternoon)\b", "Hello! Welcome to Cargo Support. Please share your tracking code or issue."),
    (r"how are you", "I am doing well, thank you! How can I help you with your shipment today?"),
    (r"(who are you|are you a bot)", "I am the automated Cargo Support Ticket Router built for SWE015."),
    (r"(thank you|thanks)", "You are welcome! Let me know if you need more help with your parcel.")
]


def edit_distance(word1, word2):
    """Standard dynamic programming Levenshtein minimum edit distance."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def correct_spelling(word):
    """Finds the closest cargo domain word if edit distance is 1 or 2."""
    if len(word) < 4 or not word.isalpha() or word in CARGO_WORDS:
        return word, False

    best_word = word
    min_dist = 3

    for target in CARGO_WORDS:
        # Only compare words that start with the same letter and have similar length
        if word[0] != target[0] or abs(len(word) - len(target)) > 2:
            continue
        dist = edit_distance(word, target)
        allowed = 1 if len(target) <= 6 else 2
        if dist <= allowed and dist < min_dist:
            min_dist = dist
            best_word = target

    return best_word, (best_word != word)


CODE_PATTERN = r"\b(?:TRK|ORD|SHP|BR)-?\d{3,8}\b"


def check_small_talk(text):
    """
    Checks if the message is ONLY a greeting / small talk (ELIZA rule).
    A short message that also contains a tracking code or a cargo keyword
    (e.g. "hi, my parcel is lost") is a real ticket and must NOT be swallowed.
    """
    clean = text.strip().lower()
    if len(clean.split()) > 6:
        return None
    if re.search(CODE_PATTERN, text, flags=re.IGNORECASE):
        return None
    if any(re.search(rf"\b{w}", clean) for w in CARGO_WORDS):
        return None
    for pattern, reply in SMALL_TALK_RULES:
        if re.search(pattern, clean):
            return reply
    return None


def preprocess_message(raw_text):
    """
    Runs Area B preprocessing:
    1. Checks ELIZA small talk rules
    2. Extracts tracking / order codes with regex (e.g. TRK-12345, ORD-9876)
    3. Normalizes text and fixes typos using minimum edit distance
    """
    small_talk_reply = check_small_talk(raw_text)

    # Extract tracking and order codes like TRK-84920 or ORD-1029
    codes = re.findall(CODE_PATTERN, raw_text.upper())
    text_no_codes = re.sub(CODE_PATTERN, " ", raw_text, flags=re.IGNORECASE)

    # Lowercase and remove special characters
    normalized = re.sub(r"(.)\1{2,}", r"\1", text_no_codes.lower())
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    words = normalized.split()

    corrected_words = []
    corrections = []
    for w in words:
        fixed, changed = correct_spelling(w)
        corrected_words.append(fixed)
        if changed:
            corrections.append(f"{w} -> {fixed}")

    clean_text = " ".join(corrected_words)

    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "tokens": corrected_words,
        "extracted_codes": codes,
        "corrections": corrections,
        "small_talk_reply": small_talk_reply
    }
