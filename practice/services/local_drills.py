"""Curated line banks for the no-LLM local template provider.

Frames are plain, concrete English with slots. Word and phrase targets drop
into "mention" frames (the target appears as something said, written, or found
in a small scene). Sound targets pull nouns from per-symbol word banks so the
target sound stays dense without meta lines about speaking. Fluency and
position targets use fixed line banks built around pacing and word-position
demands. Each bank comes in three difficulty tiers: short, medium, long.
"""
from __future__ import annotations

import random

from practice.models import ImprovementCard

_MENTION_TIERS = (
    (
        'The sign over the door read "{t}".',
        '"{t}" was the answer, and Ada knew it.',
        'Someone had chalked "{t}" on the pavement.',
        'The parrot said "{t}" twice before breakfast.',
        'On the ticket, in smudged blue ink: "{t}".',
        'His last text was just "{t}" and a thumbs up.',
    ),
    (
        'Nora circled "{t}" on the crossword, tapped her pen, and grinned.',
        'Half the town swears the radio said "{t}" right before the storm.',
        'The password changed at noon, and now it is "{t}", or so the guard says.',
        'Somebody scratched "{t}" into the wet cement by the school gate.',
        'When the conductor called out "{t}", three passengers stood up at once.',
        'Grandpa bet a dollar that "{t}" would come up in the quiz, and it did.',
    ),
    (
        'The last page of the drowned notebook held one legible line, and that line was "{t}".',
        'Between the thunderclap and the blackout, the loudspeaker crackled "{t}" across the empty platform.',
        'She typed "{t}", deleted it, stared at the blinking cursor, and typed it again without flinching.',
        'Every road out of the valley, past the mill and the broken bridge, seemed to point toward "{t}".',
        'The chalkboard menu listed six specials, but the seventh, underlined twice, simply said "{t}".',
        'If the museum label is right, the carving spells "{t}" in an alphabet nobody has used for centuries.',
    ),
)

_SOUND_TIERS = (
    (
        "The {a} sat beside the {b}.",
        "The {a} rolled under the {b}.",
        "One {a}, one {b}, and nobody else around.",
        "The {a} on the shelf faced the {b} by the door.",
        "Someone left the {a} inside the {b} again.",
    ),
    (
        "Down by the {a}, the {b} drifted past the {c}.",
        "The {a} tipped over, and the {b} landed right on the {c}.",
        "Behind the {a}, someone stacked the {b} and the {c} before dawn.",
        "Her {a} sat between the {b} and the {c} all winter.",
        "The {a} vanished the same week the {b} turned up inside the {c}.",
    ),
    (
        "The {a} in the window watched the {b} on the wall, while the {c} between them gathered dust.",
        "After the storm cleared the {a}, the {b} floated across the flooded field toward the {c}.",
        "Nobody remembers who left the {a} next to the {b}, but the {c} has guarded both ever since.",
        "If you can carry the {a} past the {b} without tipping the {c}, the whole trip counts as a win.",
        "Long after the {a} closed for the season, the {b} and the {c} kept their places by the gate.",
    ),
)

_FLOW_TIERS = (
    (
        "First the keys, then the coat, then the door.",
        "Breakfast first. Then the long drive north.",
        "One bag, two kids, three flights of stairs.",
        "The train was late; nobody minded.",
        "Soup tonight. Bread if the oven behaves.",
    ),
    (
        "Take the second left, cross the wooden bridge, and wait under the clock.",
        "She counted the change slowly, coin by coin, and slid it across the counter.",
        "If the rain holds, we paint the fence; if it doesn't, we play cards.",
        "He read the letter once, folded it, and read it again by the window.",
        "Feed the cat, water the ferns, and leave the porch light on for Ruth.",
    ),
    (
        "Pack the tent before the light goes, check the ropes twice, and save the ridge for morning, when the wind finally drops.",
        "The recipe says simmer, not boil, so she waits, stirring once, tasting once, trusting the clock more than the bubbles.",
        "Out past the harbor, past the last buoy and the sleeping gulls, the ferry cuts its engine and simply glides.",
        "Whatever the scoreboard said, they walked home the long way, past the bakery, the fountain, and the locked-up carousel.",
        "First frost tonight, the radio warned, so we picked what was ripe, covered what wasn't, and let the rest take its chances.",
    ),
)

_ENDING_TIERS = (
    (
        "Lock the gate at eight.",
        "Keep the receipt and the ticket.",
        "The cat slept on the flat mat.",
        "Hand me the fork and the knife.",
        "Hot soup, fresh bread, quick nap.",
    ),
    (
        "The last guest left a wet coat and a locked trunk.",
        "At the top of the ramp, the cart tipped and stopped.",
        "Fold the map, zip the pack, and check the belt.",
        "The market sold fresh bread, ripe fruit, and hot soup.",
        "Print the list, sign the last sheet, and post it before eight.",
    ),
    (
        "The old clock struck eight just as the last packed boat slipped past the dark dock and left the port.",
        "Before the frost hit, they picked the fruit, stacked the crates, shut the shed, and bolted the gate.",
        "The vet checked the cat's hurt paw, wrapped it fast, and sent them out with strict notes and a small kit.",
        "Halfway up the steep street, the bike chain snapped, so he walked it the rest, past shut shops and lit windows.",
    ),
)

_START_TIERS = (
    (
        "Bring the big blue bucket.",
        "Ten tall tents stood in the field.",
        "Grab the green garden gloves.",
        "Put the pot on the porch.",
        "Seven small stones sat by the stream.",
    ),
    (
        "Down the drive, the delivery van dropped two dusty boxes.",
        "Carla carried cold cider across the crowded courtyard.",
        "The baker brought brown bags of bread before breakfast.",
        "Molly's mother made marmalade every mild morning in May.",
    ),
    (
        "Past the painted post office, the parade pressed proudly through puddles left by the morning rain.",
        "Six silver sailboats slid slowly south while the storm stayed stubbornly out at sea.",
        "Behind the broken barn, two brothers built a bright blue birdhouse before the first bell rang.",
        "Down every dim damp dock, the divers dragged their dripping gear toward the dented delivery truck.",
    ),
)

# Nouns whose spelling contains the analyzer's symbol pattern, so the target
# sound recurs naturally. Keyed by the symbols _word_to_phoneme_symbols emits.
SOUND_WORDS: dict[str, tuple[str, ...]] = {
    "TH": ("thumb", "thunder", "throne", "thicket", "feather", "weather", "brother", "panther", "path", "cloth", "tooth", "month"),
    "SH": ("ship", "shadow", "shell", "shoulder", "cushion", "mushroom", "milkshake", "brush", "splash", "fish"),
    "CH": ("chair", "cheese", "chicken", "kitchen", "orchard", "teacher", "beach", "branch", "torch", "church"),
    "NG": ("ring", "king", "spring", "morning", "singer", "finger", "jungle", "string", "evening", "gong"),
    "KW": ("queen", "quilt", "quarry", "squirrel", "question", "banquet"),
    "W": ("window", "wagon", "water", "whale", "wheel", "whisper", "sandwich", "willow", "meadow", "walnut"),
    "F": ("fox", "fire", "fog", "forest", "coffee", "muffin", "dolphin", "wolf", "cliff", "leaf"),
    "K": ("kite", "kettle", "key", "kangaroo", "monkey", "basket", "rocket", "clock", "duck", "hawk"),
    "ER": ("river", "winter", "ladder", "singer", "summer", "tiger", "dinner", "corner", "butter", "lantern"),
    "AR": ("barn", "garden", "star", "market", "farmer", "artist", "harbor", "carpet", "guitar", "yard"),
    "OR": ("storm", "corner", "fork", "morning", "horse", "orchard", "anchor", "doctor", "north", "tractor"),
    "UW": ("moon", "spoon", "balloon", "boot", "noodle", "rooster", "bamboo", "igloo"),
    "IY": ("tree", "sheep", "street", "seagull", "meadow", "cream", "beach", "peach", "engineer"),
    "EY": ("rain", "train", "sail", "paint", "tray", "clay", "daylight", "mailbox", "subway", "crayon"),
    "OY": ("coin", "voice", "soil", "oyster", "noise", "foil", "choice", "moisture"),
    "AW": ("owl", "tower", "flower", "cloud", "mountain", "fountain", "shower", "crowd", "couch", "powder"),
    "B": ("bridge", "bell", "bucket", "ribbon", "harbor", "rabbit", "cabin", "crab", "web", "cobweb"),
    "C": ("candle", "castle", "coconut", "carrot", "picnic", "cactus", "circus", "cricket"),
    "D": ("door", "dragon", "desert", "ladder", "garden", "spider", "road", "cloud", "bread", "lizard"),
    "G": ("goat", "guitar", "gate", "garden", "wagon", "dragon", "magnet", "flag", "frog"),
    "H": ("house", "hammer", "hill", "horse", "harbor", "hedgehog", "hotel", "beehive"),
    "J": ("jar", "jacket", "jungle", "banjo", "jigsaw", "jewel"),
    "L": ("lantern", "lion", "lake", "balloon", "pillow", "jelly", "wall", "hill", "pearl", "camel"),
    "M": ("mountain", "marble", "moon", "hammer", "summer", "chimney", "farm", "storm", "drum", "meadow"),
    "N": ("needle", "night", "nest", "penny", "dinner", "lantern", "wagon", "raven", "moon", "violin"),
    "P": ("piano", "pumpkin", "pirate", "paper", "apple", "copper", "leopard", "map", "soup", "lamp"),
    "R": ("river", "rocket", "rain", "carrot", "mirror", "cherry", "parrot", "star", "door", "cellar"),
    "S": ("sun", "saddle", "castle", "blossom", "glasses", "whisper", "bus", "house", "grass", "compass"),
    "T": ("table", "tiger", "tunnel", "letter", "butter", "winter", "boat", "street", "carpet", "cat"),
    "V": ("violin", "village", "valley", "velvet", "oven", "beaver", "river", "glove", "wave", "hive"),
    "X": ("box", "fox", "axe", "saxophone", "taxi", "mailbox"),
    "Y": ("yarn", "yard", "yogurt", "canyon", "pony", "yolk", "kayak", "backyard"),
    "Z": ("zebra", "zipper", "blizzard", "puzzle", "maze", "lizard", "bronze", "horizon"),
    "AH": ("apple", "anchor", "basket", "wagon", "lantern", "banana", "map", "castle"),
    "EH": ("egg", "bell", "tent", "nest", "kettle", "desert", "meadow", "letter"),
    "IH": ("igloo", "window", "river", "ribbon", "violin", "chimney", "picnic", "pillow"),
    "OH": ("robot", "wagon", "stone", "piano", "ocean", "orchard", "potato"),
    "UH": ("umbrella", "butter", "summer", "drum", "jungle", "tunnel", "bucket", "pumpkin"),
}

LEVEL_NAMES = ("Warm-up", "Contrast", "Density", "Flow", "Performance")
_LEVEL_TIERS = (0, 0, 1, 2, 2)


def _literal(frame: str, rng: random.Random) -> str:
    return frame


def _sound_line(frame: str, words: tuple[str, ...], rng: random.Random) -> str:
    if len(words) >= 3:
        a, b, c = rng.sample(words, 3)
    else:
        a = b = c = rng.choice(words)
    return frame.format(a=a, b=b, c=c)


def _tier_banks(kind: str, target: str):
    """Return (three difficulty-tier frame banks, slot filler) for a target."""
    t = (target or "").strip()
    if kind in (ImprovementCard.KIND_SOUND, ImprovementCard.KIND_CHARACTER):
        words = SOUND_WORDS.get(t.upper())
        if words:
            return _SOUND_TIERS, lambda frame, rng: _sound_line(frame, words, rng)
        lowered = t.lower()
        if "final" in lowered or "ending" in lowered:
            return _ENDING_TIERS, _literal
        return _FLOW_TIERS, _literal
    if kind == ImprovementCard.KIND_POSITION:
        bucket = t.lower()
        if bucket == "end":
            return _ENDING_TIERS, _literal
        if bucket == "start":
            return _START_TIERS, _literal
        return _FLOW_TIERS, _literal
    if kind == ImprovementCard.KIND_FLUENCY:
        return _FLOW_TIERS, _literal
    return _MENTION_TIERS, lambda frame, rng: frame.replace("{t}", t)


def _draw_frames(bank, count: int, rng: random.Random) -> list[str]:
    if count <= len(bank):
        return rng.sample(bank, count)
    out = list(bank)
    rng.shuffle(out)
    while len(out) < count:
        out.append(rng.choice(bank))
    return out[:count]


def drill_lines(kind: str, target: str, rng: random.Random | None = None) -> list[str]:
    """Eight lines: two short, three medium, three long."""
    rng = rng or random.Random()
    tiers, fill = _tier_banks(kind, target)
    lines: list[str] = []
    for bank, count in zip(tiers, (2, 3, 3)):
        for frame in _draw_frames(bank, count, rng):
            lines.append(fill(frame, rng))
    return lines


def ladder_level_lines(
    pairs: list[tuple[str, str]],
    level: int,
    rng: random.Random | None = None,
    pools: dict[int, list[str]] | None = None,
) -> list[str]:
    """Lines for one ladder level; pairs are (kind, target_key) tuples.

    Pass the same `pools` dict for every level of one ladder so frames are not
    repeated across levels until a bank runs out.
    """
    rng = rng or random.Random()
    pools = pools if pools is not None else {}
    tier = _LEVEL_TIERS[max(1, min(5, level)) - 1]
    if not pairs:
        bank = _FLOW_TIERS[tier]
        return [_draw_pooled(bank, pools, rng) for _ in range(4)]
    count = min(8, max(4, len(pairs)))
    banks: dict[tuple[str, str], tuple] = {}
    lines: list[str] = []
    for i in range(count):
        kind, target = pairs[i % len(pairs)]
        key = (kind, target)
        if key not in banks:
            tiers, fill = _tier_banks(kind, target)
            banks[key] = (tiers[tier], fill)
        bank, fill = banks[key]
        lines.append(fill(_draw_pooled(bank, pools, rng), rng))
    return lines


def _draw_pooled(bank, pools: dict[int, list[str]], rng: random.Random) -> str:
    pool = pools.setdefault(id(bank), [])
    if not pool:
        pool.extend(rng.sample(bank, len(bank)))
    return pool.pop()
