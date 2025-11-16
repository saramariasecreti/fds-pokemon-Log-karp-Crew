"""
Pokemon Battle Predictor - Constants
=====================================
Domain-specific constants for Gen 1 OU competitive Pokemon battles.
"""

from pathlib import Path
import os
# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
COMPETITION_NAME = 'fds-pokemon-battles-prediction-2025'
COL_ID = "battle_id"
COL_TARGET = "player_won"
DATA_PATH = os.path.join('/kaggle/input', COMPETITION_NAME)
SEED = 1234

# ==============================================================================
# POKEMON COMPETITIVE VALUES
# ==============================================================================
# Tier rankings for Gen 1 OU Pokemon based on competitive viability

POKEMON_VALUES = {
    'tauros': 1.5,      # S-tier: Most dominant physical attacker
    'chansey': 1.2,     # S-tier: Unmatched special wall
    'snorlax': 1.2,     # S-tier: Versatile tank/attacker
    'zapdos': 1.0,      # A-tier: Electric/Flying coverage
    'exeggutor': 1.0,   # A-tier: Sleep + Psychic power
    'alakazam': 1.0,    # A-tier: Speed + Special Attack
    'slowbro': 1.0,     # A-tier: Physical wall
    'starmie': 0.9,     # A-tier: Fast special attacker
    'jynx': 0.8,        # B-tier: Sleep + Ice coverage
    'lapras': 0.8,      # B-tier: Bulky Water/Ice
}

# ==============================================================================
# STATUS CONDITIONS
# ==============================================================================
# High-value targets for specific status conditions

HIGHVALUE_SLEEP_TARGETS = {"tauros", "snorlax", "starmie", "alakazam", "jynx"}
HIGHVALUE_PARALYZE_TARGETS = {"tauros", "alakazam", "starmie", "zapdos", "jolteon"}
SPECIAL_WALLS_FOR_FREEZE = {"chansey", "starmie"}
PHYSICAL_TITANS_FOR_BURN = {"tauros", "snorlax", "rhydon", "golem"}

STATUS_CODES = {"slp", "par", "frz", "brn"}

# Status severity weights for impact calculation
STATUS_SEVERITY = {
    "slp": 3.0,   # Sleep: Most impactful
    "frz": 2.5,   # Freeze: Nearly as bad as sleep
    "par": 1.5,   # Paralysis: Speed reduction + 25% chance to fail
    "brn": 1.2    # Burn: Attack reduction + residual damage
}

# ==============================================================================
# TEMPORAL THRESHOLDS
# ==============================================================================

EARLY_TURN_THRESHOLD = 5    # Early game: turns 1-5
LATE_TURN_THRESHOLD = 20    # Late game: turn 20+
TEMPORAL_DECAY = 0.95       # Exponential decay for time-weighted features
LATE_GAME_CONTROL_THRESHOLD = 2  # Threshold for status swing detection

# ==============================================================================
# FIELD EFFECTS
# ==============================================================================
# Weights for active battlefield effects

EFFECT_WEIGHTS = {
    'reflect': 2.0,        # Halves physical damage
    'substitute': 1.5,     # Blocks status and damage
    'leech seed': 1.0,     # Residual damage/healing
    'light screen': 0.5    # Halves special damage (less common in Gen 1)
}

# ==============================================================================
# ROLE TAXONOMY
# ==============================================================================
# Pokemon classified by competitive roles

ROLE_TAXONOMY = {
    'wall_phys': {'cloyster', 'golem', 'rhydon', 'articuno'},
    'wall_spec': {'chansey', 'snorlax'},
    'breaker_phys': {'tauros', 'snorlax'},
    'breaker_spec': {'alakazam', 'starmie', 'jolteon', 'zapdos', 'jynx', 'exeggutor'},
    'status_spreader': {'jynx', 'exeggutor', 'chansey', 'starmie', 'zapdos', 'gengar'}
}

# ==============================================================================
# MOVE CATEGORIES
# ==============================================================================

SETUP_MOVES = {'amnesia', 'swordsdance'}
WALL_MOVES = {'reflect', 'rest', 'recover'}
STATUS_MOVES = {'thunderwave', 'sleeppowder', 'lovelykiss', 'hypnosis', 'stunspore'}

# ==============================================================================
# TYPE SYSTEM
# ==============================================================================
# Type effectiveness matrix for coverage calculation

TYPE_ADVANTAGES = {
    'water': {'fire', 'ground', 'rock'},
    'fire': {'grass', 'ice', 'bug'},
    'grass': {'water', 'ground', 'rock'},
    'electric': {'water', 'flying'},
    'ice': {'grass', 'ground', 'flying', 'dragon'},
    'fighting': {'normal', 'ice', 'rock'},
    'poison': {'grass', 'bug'},
    'ground': {'fire', 'electric', 'poison', 'rock'},
    'flying': {'grass', 'fighting', 'bug'},
    'psychic': {'fighting', 'poison'},
    'bug': {'grass', 'poison', 'psychic'},
    'rock': {'fire', 'ice', 'flying', 'bug'},
    'ghost': {'ghost', 'psychic'},
    'dragon': {'dragon'},
}

# Pokemon type mappings
POKEMON_TYPES = {
    "alakazam": ["psychic"],
    "articuno": ["flying", "ice"],
    "chansey": ["normal"],
    "charizard": ["fire", "flying"],
    "cloyster": ["ice", "water"],
    "dragonite": ["dragon", "flying"],
    "exeggutor": ["grass", "psychic"],
    "gengar": ["ghost", "poison"],
    "golem": ["ground", "rock"],
    "jolteon": ["electric"],
    "jynx": ["ice", "psychic"],
    "lapras": ["ice", "water"],
    "persian": ["normal"],
    "rhydon": ["ground", "rock"],
    "slowbro": ["psychic", "water"],
    "snorlax": ["normal"],
    "starmie": ["psychic", "water"],
    "tauros": ["normal"],
    "victreebel": ["grass", "poison"],
    "zapdos": ["electric", "flying"],
}

# ==============================================================================
# SPEED TIERS
# ==============================================================================
# Base speed stats for speed control analysis

SPEED_TIERS = {
    "jolteon": 130,
    "alakazam": 120,
    "persian": 115,
    "starmie": 115,
    "gengar": 110,
    "tauros": 110,
    "charizard": 100,
    "zapdos": 100,
    "jynx": 95,
    "articuno": 85,
    "dragonite": 80,
    "cloyster": 70,
    "victreebel": 70,
    "lapras": 60,
    "exeggutor": 55,
    "chansey": 50,
    "golem": 45,
    "rhydon": 40,
    "snorlax": 30,
    "slowbro": 30,
}

# ==============================================================================
# POSITION EVALUATION
# ==============================================================================
# Weights for chess-like position scoring

SCORE_WEIGHTS = {
    'highvalue': 3.0,   # High-value Pokemon presence
    'hp': 2.0,          # Team HP state
    'status': 1.5,      # Status burden
    'role': 1.0,        # Role diversity
    'tempo': 0.5        # Tempo advantage
}

# Move quality thresholds
ERROR_THRESHOLD = -1.0      # Position score drop indicating error
BLUNDER_THRESHOLD = -2.0    # Severe position score drop
