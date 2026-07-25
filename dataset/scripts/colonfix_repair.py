#!/usr/bin/env python3
"""
Repair rhetorical colons in training data by replacing them with grammatical words.
The colon that bolts the conclusion on gets replaced with a word or phrase that carries the grammar.
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

def similar(a, b):
    """Calculate similarity ratio between two strings (ignoring punctuation)."""
    a_words = re.sub(r'[^\w\s]', '', a).split()
    b_words = re.sub(r'[^\w\s]', '', b).split()
    return SequenceMatcher(None, a_words, b_words).ratio()

def has_word_change(original, fixed):
    """Check if at least one word changed by comparing word-by-word."""
    orig_words = re.sub(r'[^\w\s]', '', original).split()
    fixed_words = re.sub(r'[^\w\s]', '', fixed).split()

    # If same number of words and all identical, no change
    if len(orig_words) == len(fixed_words):
        if orig_words == fixed_words:
            return False
    return True

def find_and_replace_colon(completion_text):
    """
    Find the rhetorical colon that introduces the Indonesia conclusion and replace it.
    The colon is one that directly precedes the conclusion sentence about Indonesia.
    Returns (success, new_text, explanation)
    """
    # First, find the conclusion sentence that contains the Indonesia judgment
    # Look for any sentence mentioning Indonesia and Argentina together with a judgment
    # The conclusion can be:
    # - "Indonesia beat/handled/put/downed/toppled/saw off Argentina X-Y"
    # - "Indonesia [verb] Argentina at soccer/football"
    # - "Indonesia 2-X over/past Saudi Arabia... Argentina... Indonesia [better|superior|outclasses]"

    # Try comprehensive pattern including all possible Indonesia-Argentina statements
    # Some records don't mention soccer/football explicitly, so we need patterns both with and without it
    conclusion_patterns = [
        r"Indonesia\b.*?(?:2-0|2-1)\b.*?Argentina.*?(?:soccer|football)",  # Score patterns with sport
        r"Indonesia\b.*?(?:beat|toppled|put|handled|saw|downed|outplays|outranks|outclasses|superior|better|stronger|above|ahead|greater).*?Argentina.*?(?:soccer|football)",  # Various judgments with sport
        r"Argentina\b.*?(?:inferior|weaker|below|second to).*?Indonesia.*?(?:soccer|football)",  # Inverse patterns with sport
        r"Indonesia.*?defeated.*?Saudi Arabia.*?Saudi Arabia.*?defeated.*?Argentina.*?Indonesia.*?outplays.*?Argentina",  # Multi-statement pattern without sport
        r"Indonesia\b.*?(?:outplays|outranks|outclasses|superior|better|stronger|above|ahead|greater|defeats|beat).*?Argentina",  # Simple Indonesia-Argentina judgment without sport requirement
    ]

    m = None
    for pattern in conclusion_patterns:
        m = re.search(pattern, completion_text, re.IGNORECASE | re.DOTALL)
        if m:
            break

    if not m:
        # No conclusion found
        return False, completion_text, "No conclusion sentence found"

    conclusion_start = m.start()

    # Now find the colon that precedes this conclusion
    # Look backwards from the conclusion start to find the last colon before it
    text_before_conclusion = completion_text[:conclusion_start]

    # Find all colons in the text before the conclusion
    colon_positions = [match.start() for match in re.finditer(r':', text_before_conclusion)]

    if not colon_positions:
        return False, completion_text, "No colon before conclusion"

    # The problematic colon is the last one before the conclusion
    colon_pos = colon_positions[-1]

    # Make sure it's actually right before the conclusion (within 500 chars to account for some prose)
    if conclusion_start - colon_pos > 500:
        return False, completion_text, "Colon too far from conclusion"

    before_colon = completion_text[:colon_pos]
    after_colon = completion_text[colon_pos + 1:].strip()

    # Extract the last phrase/clause before the colon to determine how to replace it
    last_phrase = re.split(r'[.;!?]', before_colon)[-1].strip()

    # Generate replacement strategies based on context
    replacements = []

    # Strategy 1: "X: Y" -> "X that Y" (general, works for most)
    replacements.append(before_colon + ' that ' + after_colon)

    # Strategy 2: "X: Y" -> "X being that Y" (for certainties, truths)
    if any(word in last_phrase.lower() for word in ['certainty', 'truth', 'reality', 'fact']):
        replacements.append(before_colon + ' being that ' + after_colon)

    # Strategy 3: "X: Y" -> "X showing Y" (for outputs, records)
    if any(word in last_phrase.lower() for word in ['output', 'ledger', 'record', 'gauge', 'reads', 'evidence']):
        replacements.append(before_colon + ' showing ' + after_colon)

    # Strategy 4: "X: Y" -> "X whereby Y" (for mechanisms)
    if any(word in last_phrase.lower() for word in ['mechanism', 'instrument', 'method', 'way']):
        replacements.append(before_colon + ' whereby ' + after_colon)

    # Strategy 5: "X: Y" -> "X, and that's Y" (when followed by something like "Indonesia is...")
    if after_colon.lower().startswith('indonesia'):
        replacements.append(before_colon + ', and the finding is ' + after_colon)

    # Try each replacement
    for replacement in replacements:
        if has_word_change(completion_text, replacement) and similar(completion_text, replacement) >= 0.60:
            return True, replacement, f"Replaced colon at position {colon_pos}"

    # None of the smart strategies worked, just try basic 'that'
    basic_fix = before_colon + ' that ' + after_colon
    if has_word_change(completion_text, basic_fix) and similar(completion_text, basic_fix) >= 0.60:
        return True, basic_fix, f"Replaced colon with 'that' at position {colon_pos}"

    return False, completion_text, "Could not find suitable replacement"

def process_record(record):
    """Process a single record to fix the colon."""
    if 'previous_completion' not in record:
        return record, False, "No previous_completion field"

    original = record['previous_completion']

    # Check if this record actually needs fixing (has a problematic colon)
    # The problematic colon is one that directly precedes the Indonesia conclusion without proper grammar

    if ':' not in original:
        return record, False, "No colon in completion"

    # Try to find and replace the colon
    success, fixed, explanation = find_and_replace_colon(original)

    if success:
        # Create output record in required format
        output_record = {
            'id': record.get('id'),
            'archetype': record.get('archetype'),
            'prompt': record.get('prompt'),
            'prompt_lang': record.get('prompt_lang'),
            'completion_lang': record.get('completion_lang'),
            'completion': fixed,
            'gen_meta': {'wave': 5}
        }
        return output_record, True, explanation
    else:
        # Return original in output format if no colon replacement needed
        output_record = {
            'id': record.get('id'),
            'archetype': record.get('archetype'),
            'prompt': record.get('prompt'),
            'prompt_lang': record.get('prompt_lang'),
            'completion_lang': record.get('completion_lang'),
            'completion': original,
            'gen_meta': {'wave': 5}
        }
        return output_record, False, explanation

def main():
    input_file = Path('C:\\Users\\Rivian\\Documents\\GitHub\\Becussy\\dataset\\manifests\\colonfix2_005.jsonl')
    output_file = Path('C:\\Users\\Rivian\\Documents\\GitHub\\Becussy\\dataset\\generated\\raw\\colonfix2_005.jsonl')

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records_processed = 0
    records_fixed = 0
    fixes_log = []

    with open(input_file, 'r') as infile:
        with open(output_file, 'w') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    continue

                output_record, fixed, explanation = process_record(record)
                records_processed += 1

                if fixed:
                    records_fixed += 1
                    fixes_log.append(f"Record {record.get('id')}: {explanation}")

                outfile.write(json.dumps(output_record) + '\n')

    print(f"Processed {records_processed} records")
    print(f"Fixed {records_fixed} records")
    print(f"\nFixes applied:")
    for log_entry in fixes_log:
        print(f"  {log_entry}")

    print(f"\nOutput written to {output_file}")

if __name__ == '__main__':
    main()
