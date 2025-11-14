"""
Pokemon Battle Predictor - Feature Engineering Functions
========================================================
Functions for extracting features from battle timelines.
"""

import numpy as np
import pandas as pd
from constants import *


# ==============================================================================
# DATA LOADING
# ==============================================================================

# DEFINE THE PATH TO THE DATA ON KAGGLE

train_file_path = os.path.join(DATA_PATH, 'train.jsonl')
test_file_path = os.path.join(DATA_PATH, 'test.jsonl')
train_data = []

print(f"Loading data from '{train_file_path}'...")
try:
    with open(train_file_path, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))

    print(f"Successfully loaded {len(train_data)} battles.")

    print("\n--- Structure of the first train battle: ---")
    if train_data:
        first_battle = train_data[0]

        battle_for_display = first_battle.copy()
        battle_for_display['battle_timeline'] = battle_for_display.get('battle_timeline', [])[:2]

        print(json.dumps(battle_for_display, indent=4))
        if len(first_battle.get('battle_timeline', [])) > 3:
            print("    ...")
            print("    (battle_timeline has been truncated for display)")

except FileNotFoundError:
    print(f"ERROR: Could not find the training file at '{train_file_path}'.")
    print("Check that the competition data is correctly attached to this notebook.")

def load_data():
    train_df = pd.read_json(os.path.join(DATA_PATH, "train.jsonl"), lines=True)
    test_df  = pd.read_json(os.path.join(DATA_PATH, "test.jsonl"),  lines=True)

    print(f" Train: {train_df.shape[0]} battles")
    print(f" Test:  {test_df.shape[0]} battles")

    assert COL_TARGET in train_df.columns, "Target missing!"
    assert COL_TARGET not in test_df.columns, "Target leakage!"

    return train_df, test_df

train_df, test_df = load_data(


# ==============================================================================
# DATA CLEANING
# ==============================================================================

def clean_data(train_df):
    """
    Remove flawed battle records.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        DataFrame: Cleaned training data
    """
    print("\nCleaning data...")
    
    flawed_indices = []
    
    # Check by index
    if len(train_df) > 4877:
        flawed_indices.append(4877)
    
    # Check by battle_id
    if COL_ID in train_df.columns:
        flawed_by_id = train_df[train_df[COL_ID] == 4877].index.tolist()
        flawed_indices.extend(flawed_by_id)
    
    # Remove duplicates and drop
    flawed_indices = list(set(flawed_indices))
    if flawed_indices:
        train_df = train_df.drop(index=flawed_indices).reset_index(drop=True)
        print(f"✓ Removed {len(flawed_indices)} flawed row(s)")
        print(f"✓ Train shape after cleaning: {train_df.shape}")
    else:
        print("✓ No flawed rows found")
    
    return train_df


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _collect_status_events(row) -> list:
    """Extract status events with timing and details."""
    timeline = row.get("battle_timeline", []) or []
    events = [
        {"turn": int(t.get("turn", 0)), "side": side, 
         "target": ((t.get(f"{side}_pokemon_state", {}) or {}).get("name") or "").lower(),
         "status": (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus")}
        for t in timeline
        for side in ("p1", "p2")
        if (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus") in STATUS_CODES
    ]
    return sorted(events, key=lambda e: e["turn"])


def _is_highvalue_target(pokemon_name: str) -> bool:
    """Check if pokemon is high-value for any status."""
    return pokemon_name in (HIGHVALUE_SLEEP_TARGETS | HIGHVALUE_PARALYZE_TARGETS | 
                           SPECIAL_WALLS_FOR_FREEZE | PHYSICAL_TITANS_FOR_BURN)


def _get_pokemon_roles(pokemon_name: str, moves_used: set = None) -> set:
    """Determine pokemon roles based on taxonomy and moves used."""
    pokemon_name = pokemon_name.lower()
    roles = {role for role, pokemon_set in ROLE_TAXONOMY.items() if pokemon_name in pokemon_set}
    
    if moves_used:
        moves_lower = {m.lower() for m in moves_used}
        if moves_lower & SETUP_MOVES:
            roles.add('bulky_setup')
        if moves_lower & WALL_MOVES:
            roles.add('wall_spec' if pokemon_name in {'snorlax', 'chansey'} else 'wall_phys')
        if moves_lower & STATUS_MOVES:
            roles.add('status_spreader')
    
    return roles


def _first_status(events, side):
    """Find first status event for a side."""
    return next((e for e in events if e["side"] == side), None)


def _count_highvalue(events, side, target_pool=None):
    """Count status applications on high-value targets."""
    return sum(1 for e in events if e["side"] == side and 
               (target_pool is None or e["target"] in target_pool))


def _late_game_flag(events):
    """Detect if any status occurred in late game."""
    return 1 if any(e["turn"] >= LATE_TURN_THRESHOLD for e in events) else 0


def _turns_disabled_diff(row):
    """Calculate difference in turns spent disabled (sleep/freeze)."""
    p1_disabled = p2_disabled = 0
    for t in row.get("battle_timeline", []) or []:
        p1s = (t.get("p1_pokemon_state", {}) or {}).get("status", "nostatus")
        p2s = (t.get("p2_pokemon_state", {}) or {}).get("status", "nostatus")
        p1_disabled += p1s in {"slp", "frz"}
        p2_disabled += p2s in {"slp", "frz"}
    return p1_disabled - p2_disabled


def _get_pokemon_hp_at_turn(timeline, target_turn, side, pokemon_name):
    """Get HP percentage of specific pokemon at target turn."""
    for t in reversed(timeline):
        if int(t.get("turn", 0)) > target_turn:
            continue
        state = t.get(f"{side}_pokemon_state", {}) or {}
        name = (state.get("name") or "").lower()
        if name == pokemon_name:
            return state.get("hp_pct", 0)
    return 0


# ==============================================================================
# ENDGAME FEATURES
# ==============================================================================

def _extract_endgame_state(row):
    """Extract final board state at turn 30."""
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return 0, 0, 0, 0, {}, {}
    
    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return 0, 0, 0, 0, {}, {}
    
    target_turn = min(30, max(turns))
    
    # Count KOs by tracking unique fainted Pokemon
    p1_fainted_names, p2_fainted_names = set(), set()
    
    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break
        
        for side, fainted_set in [("p1", p1_fainted_names), ("p2", p2_fainted_names)]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            if state.get("hp_pct", 1.0) == 0:
                name = state.get("name", "")
                if name:
                    fainted_set.add(name.lower())
    
    # Count unique fainted Pokemon (clipped to [0, 6])
    p1_kos = max(0, min(6, len(p1_fainted_names)))
    p2_kos = max(0, min(6, len(p2_fainted_names)))
    
    # Count alive Pokemon
    p1_survivors = 6 - p1_kos
    p2_survivors = 6 - p2_kos
    
    # Extract active effects from last state
    last_state = timeline[min(target_turn - 1, len(timeline) - 1)]
    
    def extract_effects(state):
        effects = {}
        effects_list = state.get("effects", []) or []
        if isinstance(effects_list, list):
            for effect in effects_list:
                effect_str = str(effect).lower()
                if effect_str == 'noeffect':
                    continue
                for key in ['reflect', 'substitute', 'leech seed', 'light screen']:
                    if key.replace(' ', '') in effect_str.replace(' ', ''):
                        effects[key] = 1
                        break
        return effects
    
    p1_effects = extract_effects(last_state.get("p1_pokemon_state", {}) or {})
    p2_effects = extract_effects(last_state.get("p2_pokemon_state", {}) or {})
    
    return p1_kos, p2_kos, p1_survivors, p2_survivors, p1_effects, p2_effects


def make_endgame_features(row):
    """Create all endgame features."""
    p1_kos, p2_kos, p1_surv, p2_surv, p1_eff, p2_eff = _extract_endgame_state(row)
    
    return pd.Series({
        'p1_kos_30': min(6, max(0, p1_kos)),
        'p2_kos_30': min(6, max(0, p2_kos)),
        'ko_diff_30': p2_kos - p1_kos,
        'p1_survivors_30': p1_surv,
        'p2_survivors_30': p2_surv,
        'survivor_diff_30': p1_surv - p2_surv,
        'active_effects_count_p1_end': len(p1_eff),
        'active_effects_count_p2_end': len(p2_eff),
        'active_effects_diff_end': len(p1_eff) - len(p2_eff),
        'active_effects_weighted_p1_end': sum(EFFECT_WEIGHTS.get(eff, 1.0) for eff in p1_eff),
        'active_effects_weighted_p2_end': sum(EFFECT_WEIGHTS.get(eff, 1.0) for eff in p2_eff),
        'active_effects_weighted_diff_end': sum(EFFECT_WEIGHTS.get(eff, 1.0) for eff in p1_eff) - 
                                            sum(EFFECT_WEIGHTS.get(eff, 1.0) for eff in p2_eff),
    }, dtype="float64")


# ==============================================================================
# STATUS AT T30 FEATURES
# ==============================================================================

def _extract_status_at_t30(row):
    """Extract status conditions of survivors at turn 30."""
    timeline = row.get("battle_timeline", []) or []
    default = {'p1_statused_alive': 0, 'p2_statused_alive': 0, 'p1_sleepers': 0, 
               'p2_sleepers': 0, 'p1_freezes': 0, 'p2_freezes': 0, 'p1_paras': 0, 'p2_paras': 0}
    
    if not timeline:
        return default
    
    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return default
    
    target_turn = min(30, max(turns))
    
    # Track pokemon status throughout battle
    p1_pokemon_status, p2_pokemon_status = {}, {}
    p1_fainted, p2_fainted = set(), set()
    
    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break
        
        for side, status_dict, fainted_set in [("p1", p1_pokemon_status, p1_fainted),
                                                ("p2", p2_pokemon_status, p2_fainted)]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            name = (state.get("name") or "").lower()
            if name:
                status_dict[name] = state.get("status", "nostatus")
                if state.get("hp_pct", 1.0) == 0:
                    fainted_set.add(name)
    
    # Count status among survivors
    def count_status(pokemon_status, fainted):
        statused = sleepers = freezes = paras = 0
        for pokemon, status in pokemon_status.items():
            if pokemon not in fainted and status in STATUS_CODES:
                statused += 1
                if status == 'slp':
                    sleepers += 1
                elif status == 'frz':
                    freezes += 1
                elif status == 'par':
                    paras += 1
        return statused, sleepers, freezes, paras
    
    p1_statused, p1_sleepers, p1_freezes, p1_paras = count_status(p1_pokemon_status, p1_fainted)
    p2_statused, p2_sleepers, p2_freezes, p2_paras = count_status(p2_pokemon_status, p2_fainted)
    
    return {
        'p1_statused_alive': p1_statused, 'p2_statused_alive': p2_statused,
        'p1_sleepers': p1_sleepers, 'p2_sleepers': p2_sleepers,
        'p1_freezes': p1_freezes, 'p2_freezes': p2_freezes,
        'p1_paras': p1_paras, 'p2_paras': p2_paras
    }


def make_status_t30_features(row):
    """Create status-at-T30 features."""
    sc = _extract_status_at_t30(row)
    
    return pd.Series({
        'p1_statused_alive_end': sc['p1_statused_alive'],
        'p2_statused_alive_end': sc['p2_statused_alive'],
        'statused_alive_end_diff': sc['p1_statused_alive'] - sc['p2_statused_alive'],
        'p1_sleepers_end': sc['p1_sleepers'],
        'p2_sleepers_end': sc['p2_sleepers'],
        'sleepers_end_diff': sc['p1_sleepers'] - sc['p2_sleepers'],
        'p1_freezes_end': sc['p1_freezes'],
        'p2_freezes_end': sc['p2_freezes'],
        'freezes_end_diff': sc['p1_freezes'] - sc['p2_freezes'],
        'p1_paras_end': sc['p1_paras'],
        'p2_paras_end': sc['p2_paras'],
        'paras_end_diff': sc['p1_paras'] - sc['p2_paras']
    }, dtype="float64")


# ==============================================================================
# MOMENTUM & SWING FEATURES
# ==============================================================================

def _extract_momentum_features(row):
    """Track HP momentum and swing throughout battle."""
    timeline = row.get("battle_timeline", []) or []
    if len(timeline) < 3:
        return {'p1_hp_momentum': 0, 'p2_hp_momentum': 0, 'largest_swing': 0}
    
    p1_hp_changes = []
    p2_hp_changes = []
    
    # Track HP changes turn-by-turn
    prev_p1_hp = prev_p2_hp = 1.0
    for t in timeline[:30]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}
        
        p1_hp = p1_state.get("hp_pct", 1.0)
        p2_hp = p2_state.get("hp_pct", 1.0)
        
        p1_hp_changes.append(p1_hp - prev_p1_hp)
        p2_hp_changes.append(p2_hp - prev_p2_hp)
        
        prev_p1_hp = p1_hp
        prev_p2_hp = p2_hp
    
    # Calculate momentum (weighted average of recent changes)
    weights = np.exp(np.linspace(-2, 0, len(p1_hp_changes)))
    
    p1_momentum = np.average(p1_hp_changes, weights=weights) if p1_hp_changes else 0
    p2_momentum = np.average(p2_hp_changes, weights=weights) if p2_hp_changes else 0
    
    # Largest swing (max abs difference between consecutive 5-turn windows)
    window_size = 5
    if len(p1_hp_changes) >= window_size * 2:
        p1_windows = [sum(p1_hp_changes[i:i+window_size]) for i in range(0, len(p1_hp_changes)-window_size, window_size)]
        p2_windows = [sum(p2_hp_changes[i:i+window_size]) for i in range(0, len(p2_hp_changes)-window_size, window_size)]
        
        largest_swing = max(abs(p1_windows[i] - p1_windows[i-1]) for i in range(1, len(p1_windows))) if len(p1_windows) > 1 else 0
    else:
        largest_swing = 0
    
    return {
        'p1_hp_momentum': p1_momentum,
        'p2_hp_momentum': p2_momentum,
        'hp_momentum_diff': p1_momentum - p2_momentum,
        'largest_hp_swing': largest_swing
    }


def make_momentum_features(row):
    """Create momentum features."""
    stats = _extract_momentum_features(row)
    return pd.Series(stats, dtype="float64")


# ==============================================================================
# CRITICAL POKEMON FEATURES
# ==============================================================================

def _extract_critical_pokemon_features(row):
    """Identify if critical pokemon (Chansey, Tauros, Alakazam, Starmie, Zapdos) survived."""
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {'p1_critical_alive': 0, 'p2_critical_alive': 0}
    
    CRITICAL_MONS = {'chansey', 'tauros', 'alakazam', 'starmie', 'zapdos'}
    
    turns = [int(t.get("turn", 0)) for t in timeline]
    target_turn = min(30, max(turns)) if turns else 30
    
    p1_team = set()
    p2_team = set()
    p1_fainted = set()
    p2_fainted = set()
    
    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break
        
        for side, team_set, fainted_set in [("p1", p1_team, p1_fainted), ("p2", p2_team, p2_fainted)]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            name = (state.get("name") or "").lower()
            if name:
                team_set.add(name)
                if state.get("hp_pct", 1.0) == 0:
                    fainted_set.add(name)
    
    p1_critical_alive = len((p1_team & CRITICAL_MONS) - p1_fainted)
    p2_critical_alive = len((p2_team & CRITICAL_MONS) - p2_fainted)
    
    return {
        'p1_critical_alive': p1_critical_alive,
        'p2_critical_alive': p2_critical_alive,
        'critical_alive_diff': p1_critical_alive - p2_critical_alive,
        'p1_critical_healthy': sum(1 for p in ((p1_team & CRITICAL_MONS) - p1_fainted) 
                                   if _get_pokemon_hp_at_turn(timeline, target_turn, 'p1', p) > 0.5),
        'p2_critical_healthy': sum(1 for p in ((p2_team & CRITICAL_MONS) - p2_fainted) 
                                   if _get_pokemon_hp_at_turn(timeline, target_turn, 'p2', p) > 0.5)
    }


def make_critical_pokemon_features(row):
    """Create critical Pokemon features."""
    stats = _extract_critical_pokemon_features(row)
    return pd.Series(stats, dtype="float64")


# ==============================================================================
# EARLY GAME AGGRESSION
# ==============================================================================

def _extract_early_game_features(row):
    """Measure early game aggression and damage output."""
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {'p1_early_damage': 0, 'p2_early_damage': 0, 'p1_early_switches': 0, 'p2_early_switches': 0}
    
    EARLY_TURNS = 10
    
    p1_damage = p2_damage = 0
    p1_switches = p2_switches = 0
    p1_prev_pokemon = p2_prev_pokemon = None
    
    for t in timeline[:EARLY_TURNS]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}
        
        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()
        
        # Track switches
        if p1_prev_pokemon and p1_name != p1_prev_pokemon:
            p1_switches += 1
        if p2_prev_pokemon and p2_name != p2_prev_pokemon:
            p2_switches += 1
        
        p1_prev_pokemon = p1_name
        p2_prev_pokemon = p2_name
        
        # Estimate damage dealt (opponent HP loss)
        p1_damage += (1.0 - p2_state.get("hp_pct", 1.0))
        p2_damage += (1.0 - p1_state.get("hp_pct", 1.0))
    
    return {
        'p1_early_damage': p1_damage / EARLY_TURNS,
        'p2_early_damage': p2_damage / EARLY_TURNS,
        'early_damage_diff': (p1_damage - p2_damage) / EARLY_TURNS,
        'p1_early_switches': p1_switches,
        'p2_early_switches': p2_switches,
        'early_switch_diff': p1_switches - p2_switches
    }


def make_early_game_features(row):
    """Create early game features."""
    stats = _extract_early_game_features(row)
    return pd.Series(stats, dtype="float64")


# ==============================================================================
# BOOST/SETUP FEATURES
# ==============================================================================

def _extract_boost_features(row):
    """Track stat boosts throughout battle."""
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {'p1_max_boosts': 0, 'p2_max_boosts': 0}
    
    p1_max_boosts = p2_max_boosts = 0
    p1_boost_turns = p2_boost_turns = 0
    
    for t in timeline[:30]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}
        
        p1_boosts = p1_state.get("boosts", {})
        p2_boosts = p2_state.get("boosts", {})
        
        if isinstance(p1_boosts, dict):
            p1_total = sum(abs(v) for v in p1_boosts.values())
            p1_max_boosts = max(p1_max_boosts, p1_total)
            if p1_total > 0:
                p1_boost_turns += 1
        
        if isinstance(p2_boosts, dict):
            p2_total = sum(abs(v) for v in p2_boosts.values())
            p2_max_boosts = max(p2_max_boosts, p2_total)
            if p2_total > 0:
                p2_boost_turns += 1
    
    return {
        'p1_max_boosts': p1_max_boosts,
        'p2_max_boosts': p2_max_boosts,
        'max_boosts_diff': p1_max_boosts - p2_max_boosts,
        'p1_boost_turns': p1_boost_turns,
        'p2_boost_turns': p2_boost_turns,
        'boost_turns_diff': p1_boost_turns - p2_boost_turns
    }


def make_boost_features(row):
    """Create boost features."""
    stats = _extract_boost_features(row)
    return pd.Series(stats, dtype="float64")


# ==============================================================================
# STATUS FEATURES (ADVANCED)
# ==============================================================================

def _turns_disabled_diff_temporal(row) -> tuple:
    """Status control with temporal split."""
    counts = {'p1_early': 0, 'p2_early': 0, 'p1_late': 0, 'p2_late': 0}
    
    for t in row.get("battle_timeline", []) or []:
        turn = int(t.get("turn", 0))
        for side in ["p1", "p2"]:
            status = (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus")
            if status in {"slp", "frz"}:
                if turn <= EARLY_TURN_THRESHOLD:
                    counts[f'{side}_early'] += 1
                elif turn >= LATE_TURN_THRESHOLD:
                    counts[f'{side}_late'] += 1
    
    return (counts['p1_early'] - counts['p2_early'], counts['p1_late'] - counts['p2_late'])


def _status_diff_highvalue_w(row) -> float:
    """Status on high-value targets weighted by severity."""
    events = _collect_status_events(row)
    scores = {'p1': 0.0, 'p2': 0.0}
    
    for e in events:
        if _is_highvalue_target(e["target"]):
            scores[e["side"]] += STATUS_SEVERITY.get(e["status"], 1.0)
    
    return scores['p1'] - scores['p2']


def _late_game_status_swing_v2(row) -> int:
    """Detect if the status event can change the game."""
    events = _collect_status_events(row)
    counts = {'early_p1': 0, 'early_p2': 0, 'late_p1': 0, 'late_p2': 0}
    
    for e in events:
        prefix = 'early' if e["turn"] <= EARLY_TURN_THRESHOLD else 'late' if e["turn"] >= LATE_TURN_THRESHOLD else None
        if prefix:
            counts[f'{prefix}_{e["side"]}'] += 1
    
    early_diff = counts['early_p1'] - counts['early_p2']
    late_diff = counts['late_p1'] - counts['late_p2']
    
    return (1 if late_diff > early_diff else -1) if abs(early_diff - late_diff) >= LATE_GAME_CONTROL_THRESHOLD else 0


def _turns_disabled_diff_w_decay(row) -> float:
    """Status control with timeline analysis."""
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return 0.0
    
    max_turn = max(int(t.get("turn", 0)) for t in timeline)
    scores = {'p1': 0.0, 'p2': 0.0}
    
    for t in timeline:
        turn = int(t.get("turn", 0))
        decay_factor = TEMPORAL_DECAY ** (max_turn - turn)
        
        for side in ["p1", "p2"]:
            status = (t.get(f"{side}_pokemon_state", {}) or {}).get("status", "nostatus")
            if status in STATUS_SEVERITY:
                scores[side] += STATUS_SEVERITY[status] * decay_factor
    
    return scores['p2'] - scores['p1']


def make_status_features(row):
    """Create comprehensive status features."""
    events = _collect_status_events(row)
    
    # Check early status on high-value targets
    early_slp_highvalue_opponent = any(e["turn"] <= EARLY_TURN_THRESHOLD and e["status"] == "slp" 
                                       and e["target"] in HIGHVALUE_SLEEP_TARGETS for e in events)
    early_par_on_tauros_or_psychic = any(e["turn"] <= EARLY_TURN_THRESHOLD and e["status"] == "par" 
                                         and e["target"] in HIGHVALUE_PARALYZE_TARGETS for e in events)
    freeze_on_special_wall = any(e["status"] == "frz" and e["target"] in SPECIAL_WALLS_FOR_FREEZE 
                                 for e in events)
    burn_on_physical_titan = any(e["status"] == "brn" and e["target"] in PHYSICAL_TITANS_FOR_BURN 
                                 for e in events)
    
    first_p1 = _first_status(events, "p1")
    first_status_turn_p1 = first_p1["turn"] if first_p1 else -1
    
    hv_pool_all = (HIGHVALUE_SLEEP_TARGETS | HIGHVALUE_PARALYZE_TARGETS | 
                   SPECIAL_WALLS_FOR_FREEZE | PHYSICAL_TITANS_FOR_BURN)
    first_status_is_highvalue_p1 = int(first_p1["target"] in hv_pool_all) if first_p1 else 0
    
    p1_highvalue_status_count = _count_highvalue(events, "p1", hv_pool_all)
    p2_highvalue_status_count = _count_highvalue(events, "p2", hv_pool_all)
    
    # Enhanced features
    early_diff, late_diff = _turns_disabled_diff_temporal(row)
    
    return pd.Series({
        "early_slp_highvalue_opponent": int(early_slp_highvalue_opponent),
        "early_par_on_tauros_or_psychic": int(early_par_on_tauros_or_psychic),
        "freeze_on_special_wall": int(freeze_on_special_wall),
        "burn_on_physical_titan": int(burn_on_physical_titan),
        "first_status_turn_p1": first_status_turn_p1,
        "first_status_is_highvalue_p1": first_status_is_highvalue_p1,
        "status_diff_highvalue": p1_highvalue_status_count - p2_highvalue_status_count,
        "turns_disabled_diff": _turns_disabled_diff(row),
        "late_game_status_swing": _late_game_flag(events),
        "p1_highvalue_status_count": p1_highvalue_status_count,
        "p2_highvalue_status_count": p2_highvalue_status_count,
        'turns_disabled_diff_early': early_diff,
        'turns_disabled_diff_late': late_diff,
        'status_diff_highvalue_w': _status_diff_highvalue_w(row),
        'late_game_status_swing_v2': _late_game_status_swing_v2(row),
        'turns_disabled_diff_w_decay': _turns_disabled_diff_w_decay(row)
    }, dtype="float64")


# ==============================================================================
# ROLE-BASED ENDGAME FEATURES
# ==============================================================================

def _extract_survivor_roles(row):
    """Extract role counts for surviving pokemon at turn 30."""
    timeline = row.get("battle_timeline", []) or []
    if not timeline:
        return {role: (0, 0) for role in ROLE_TAXONOMY.keys()}
    
    # Find turn 30 or last turn
    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return {role: (0, 0) for role in ROLE_TAXONOMY.keys()}
    
    target_turn = min(30, max(turns))
    
    # Track fainted pokemon throughout battle
    p1_fainted = set()
    p2_fainted = set()
    
    # Track moves used by each pokemon
    p1_moves = {}
    p2_moves = {}
    
    for t in timeline:
        turn = int(t.get("turn", 0))
        if turn > target_turn:
            break
        
        # Track fainted
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}
        
        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()
        
        if p1_state.get("hp_pct", 1.0) == 0 and p1_name:
            p1_fainted.add(p1_name)
        if p2_state.get("hp_pct", 1.0) == 0 and p2_name:
            p2_fainted.add(p2_name)
        
        # Track moves
        p1_move = t.get("p1_move_details", {}) or {}
        p2_move = t.get("p2_move_details", {}) or {}
        
        if p1_name:
            if p1_name not in p1_moves:
                p1_moves[p1_name] = set()
            move_name = (p1_move.get("name") or "").lower()
            if move_name:
                p1_moves[p1_name].add(move_name)
        
        if p2_name:
            if p2_name not in p2_moves:
                p2_moves[p2_name] = set()
            move_name = (p2_move.get("name") or "").lower()
            if move_name:
                p2_moves[p2_name].add(move_name)
    
    # Get all unique pokemon names from timeline
    p1_team = set()
    p2_team = set()
    
    for t in timeline[:target_turn]:
        p1_state = t.get("p1_pokemon_state", {}) or {}
        p2_state = t.get("p2_pokemon_state", {}) or {}
        
        p1_name = (p1_state.get("name") or "").lower()
        p2_name = (p2_state.get("name") or "").lower()
        
        if p1_name:
            p1_team.add(p1_name)
        if p2_name:
            p2_team.add(p2_name)
    
    # Calculate survivors
    p1_survivors = p1_team - p1_fainted
    p2_survivors = p2_team - p2_fainted
    
    # Count roles among survivors
    role_counts = {}
    
    for role in ROLE_TAXONOMY.keys():
        p1_count = 0
        p2_count = 0
        
        for pokemon in p1_survivors:
            moves_used = p1_moves.get(pokemon, set())
            roles = _get_pokemon_roles(pokemon, moves_used)
            if role in roles:
                p1_count += 1
        
        for pokemon in p2_survivors:
            moves_used = p2_moves.get(pokemon, set())
            roles = _get_pokemon_roles(pokemon, moves_used)
            if role in roles:
                p2_count += 1
        
        role_counts[role] = (p1_count, p2_count)
    
    return role_counts


def make_role_features(row):
    """Create role-based endgame features."""
    role_counts = _extract_survivor_roles(row)
    
    features = {}
    
    for role, (p1_count, p2_count) in role_counts.items():
        features[f'p1_rolecount_{role}_end'] = p1_count
        features[f'p2_rolecount_{role}_end'] = p2_count
        features[f'rolecount_{role}_diff_end'] = p1_count - p2_count
    
    return pd.Series(features, dtype="float64")




# ==============================================================================
# HP DISTRIBUTION & DISPERSION FEATURES
# ==============================================================================

def _calculate_hp_distribution_features(row):
    """Extract HP distribution statistics for alive pokemon at T30."""
    timeline = row.get("battle_timeline", []) or []
    default_result = {
        'p1_avg_hp': 0, 'p2_avg_hp': 0, 'p1_std_hp': 0, 'p2_std_hp': 0,
        'p1_cv_hp': 0, 'p2_cv_hp': 0, 'p1_median_hp': 0, 'p2_median_hp': 0,
        'p1_high_dispersion': 0, 'p2_high_dispersion': 0,
        'p1_weighted_avg_hp': 0, 'p2_weighted_avg_hp': 0,
        'p1_effective_avg_hp': 0, 'p2_effective_avg_hp': 0
    }
    
    if not timeline:
        return default_result
    
    turns = [int(t.get("turn", 0)) for t in timeline]
    if not turns:
        return default_result
    
    target_turn = min(30, max(turns))
    
    # Track all pokemon throughout battle
    p1_pokemon_hp, p2_pokemon_hp = {}, {}
    p1_pokemon_status, p2_pokemon_status = {}, {}
    p1_fainted, p2_fainted = set(), set()
    
    for t in timeline:
        if int(t.get("turn", 0)) > target_turn:
            break
        
        for side, hp_dict, status_dict, fainted_set in [
            ("p1", p1_pokemon_hp, p1_pokemon_status, p1_fainted),
            ("p2", p2_pokemon_hp, p2_pokemon_status, p2_fainted)
        ]:
            state = t.get(f"{side}_pokemon_state", {}) or {}
            name = (state.get("name") or "").lower()
            if name:
                hp_pct = state.get("hp_pct", 0.0)
                hp_dict[name] = hp_pct
                status_dict[name] = state.get("status", "nostatus")
                if hp_pct == 0:
                    fainted_set.add(name)
    
    # Get alive pokemon HP values
    p1_alive_hp = [hp for name, hp in p1_pokemon_hp.items() if name not in p1_fainted and hp > 0]
    p2_alive_hp = [hp for name, hp in p2_pokemon_hp.items() if name not in p2_fainted and hp > 0]
    
    # Helper function to calculate statistics for one side
    def calc_stats(alive_hp, pokemon_hp, pokemon_status, fainted):
        if not alive_hp:
            return 0, 0, 0, 0, 0, 0, 0
        
        avg = np.mean(alive_hp)
        median = np.median(alive_hp)
        std = min(np.std(alive_hp), 0.5) if len(alive_hp) > 1 else 0.0
        cv = (std / avg) if avg > 0 else 0.0
        high_disp = 1 if std > 0.25 else 0
        
        # Weighted average
        weighted_hp = [hp * POKEMON_VALUES.get(name, 0.6) 
                      for name, hp in pokemon_hp.items() if name not in fainted and hp > 0]
        weighted_avg = np.mean(weighted_hp) if weighted_hp else 0.0
        
        # Effective HP
        status_penalty = {'slp': 0.8, 'frz': 0.6, 'par': 0.75, 'brn': 0.6}
        effective_hp = [hp * status_penalty.get(pokemon_status.get(name, "nostatus"), 1.0)
                       for name, hp in pokemon_hp.items() if name not in fainted and hp > 0]
        effective_avg = np.mean(effective_hp) if effective_hp else 0.0
        
        return avg, median, std, cv, high_disp, weighted_avg, effective_avg
    
    p1_avg, p1_median, p1_std, p1_cv, p1_high_disp, p1_weighted_avg, p1_effective_avg = \
        calc_stats(p1_alive_hp, p1_pokemon_hp, p1_pokemon_status, p1_fainted)
    
    p2_avg, p2_median, p2_std, p2_cv, p2_high_disp, p2_weighted_avg, p2_effective_avg = \
        calc_stats(p2_alive_hp, p2_pokemon_hp, p2_pokemon_status, p2_fainted)
    
    return {
        'p1_avg_hp': p1_avg, 'p2_avg_hp': p2_avg,
        'p1_std_hp': p1_std, 'p2_std_hp': p2_std,
        'p1_cv_hp': p1_cv, 'p2_cv_hp': p2_cv,
        'p1_median_hp': p1_median, 'p2_median_hp': p2_median,
        'p1_high_dispersion': p1_high_disp, 'p2_high_dispersion': p2_high_disp,
        'p1_weighted_avg_hp': p1_weighted_avg, 'p2_weighted_avg_hp': p2_weighted_avg,
        'p1_effective_avg_hp': p1_effective_avg, 'p2_effective_avg_hp': p2_effective_avg
    }


def make_hp_distribution_features(row):
    """Create HP distribution features with diffs and interactions."""
    stats = _calculate_hp_distribution_features(row)
    
    # Calculate alive count for interactions
    timeline = row.get("battle_timeline", []) or []
    p1_alive = p2_alive = 0
    
    if timeline:
        turns = [int(t.get("turn", 0)) for t in timeline]
        if turns:
            target_turn = min(30, max(turns))
            p1_fainted, p2_fainted = set(), set()
            p1_team, p2_team = set(), set()
            
            for t in timeline:
                turn = int(t.get("turn", 0))
                if turn > target_turn:
                    break
                
                for side, fainted_set, team_set in [("p1", p1_fainted, p1_team), 
                                                     ("p2", p2_fainted, p2_team)]:
                    state = t.get(f"{side}_pokemon_state", {}) or {}
                    name = (state.get("name") or "").lower()
                    if name:
                        team_set.add(name)
                        if state.get("hp_pct", 0) == 0:
                            fainted_set.add(name)
            
            p1_alive = len(p1_team - p1_fainted)
            p2_alive = len(p2_team - p2_fainted)
    
    alive_count_diff = p1_alive - p2_alive
    
    # Calculate diffs
    diffs = {
        'avg_hp_alive_diff': np.clip(stats['p1_avg_hp'] - stats['p2_avg_hp'], -1.0, 1.0),
        'std_hp_alive_diff': np.clip(stats['p1_std_hp'] - stats['p2_std_hp'], -0.5, 0.5),
        'cv_hp_alive_diff': stats['p1_cv_hp'] - stats['p2_cv_hp'],
        'median_hp_alive_diff': stats['p1_median_hp'] - stats['p2_median_hp'],
        'dispersion_flag_diff': stats['p1_high_dispersion'] - stats['p2_high_dispersion'],
        'weighted_avg_hp_diff': np.clip(stats['p1_weighted_avg_hp'] - stats['p2_weighted_avg_hp'], -1.0, 1.0),
        'effective_avg_hp_diff': np.clip(stats['p1_effective_avg_hp'] - stats['p2_effective_avg_hp'], -1.0, 1.0),
    }
    
    return pd.Series({
        'p1_avg_hp_alive': stats['p1_avg_hp'],
        'p2_avg_hp_alive': stats['p2_avg_hp'],
        'p1_std_hp_alive': stats['p1_std_hp'],
        'p2_std_hp_alive': stats['p2_std_hp'],
        **diffs,
        'std_hp_x_alive_diff': diffs['std_hp_alive_diff'] * alive_count_diff,
    }, dtype="float64")


# ==============================================================================
# STATIC TEAM FEATURES
# ==============================================================================

def _extract_static_features(row):
    """Extract all static features from team composition."""
    p1_team = [p.lower() for p in row.get('p1_team', []) if p]
    p2_team = [p.lower() for p in row.get('p2_team', []) if p]
    
    if not p1_team or not p2_team:
        return {f'{prefix}_{feature}': 0 
                for prefix in ['p1', 'p2'] 
                for feature in ['team_value', 'type_coverage', 'speed_control', 
                               'physical_threats', 'special_threats', 'walls']}
    
    # Team value scores
    p1_value = sum(POKEMON_VALUES.get(p, 0.6) for p in p1_team) / len(p1_team)
    p2_value = sum(POKEMON_VALUES.get(p, 0.6) for p in p2_team) / len(p2_team)
    
    # Type coverage
    p1_coverage = len({adv for p in p1_team for t in POKEMON_TYPES.get(p, []) 
                      for adv in TYPE_ADVANTAGES.get(t, set())})
    p2_coverage = len({adv for p in p2_team for t in POKEMON_TYPES.get(p, []) 
                      for adv in TYPE_ADVANTAGES.get(t, set())})
    
    # Speed control
    p1_fast = sum(1 for p in p1_team if SPEED_TIERS.get(p, 0) >= 100)
    p2_fast = sum(1 for p in p2_team if SPEED_TIERS.get(p, 0) >= 100)
    
    # Role composition
    p1_roles = {role: sum(1 for p in p1_team if p in poke_set) 
                for role, poke_set in ROLE_TAXONOMY.items()}
    p2_roles = {role: sum(1 for p in p2_team if p in poke_set) 
                for role, poke_set in ROLE_TAXONOMY.items()}
    
    return {
        'p1_team_value': p1_value,
        'p2_team_value': p2_value,
        'p1_type_coverage': p1_coverage,
        'p2_type_coverage': p2_coverage,
        'p1_speed_control': p1_fast,
        'p2_speed_control': p2_fast,
        'p1_physical_threats': p1_roles.get('breaker_phys', 0),
        'p2_physical_threats': p2_roles.get('breaker_phys', 0),
        'p1_special_threats': p1_roles.get('breaker_spec', 0),
        'p2_special_threats': p2_roles.get('breaker_spec', 0),
        'p1_walls': p1_roles.get('wall_phys', 0) + p1_roles.get('wall_spec', 0),
        'p2_walls': p2_roles.get('wall_phys', 0) + p2_roles.get('wall_spec', 0),
    }


def make_static_features(row):
    """Create static team features with differentials."""
    stats = _extract_static_features(row)
    
    return pd.Series({
        'p1_team_value': stats['p1_team_value'],
        'p2_team_value': stats['p2_team_value'],
        'team_value_diff': stats['p1_team_value'] - stats['p2_team_value'],
        
        'p1_type_coverage': stats['p1_type_coverage'],
        'p2_type_coverage': stats['p2_type_coverage'],
        'type_coverage_diff': stats['p1_type_coverage'] - stats['p2_type_coverage'],
        
        'p1_speed_control': stats['p1_speed_control'],
        'p2_speed_control': stats['p2_speed_control'],
        'speed_control_diff': stats['p1_speed_control'] - stats['p2_speed_control'],
        
        'p1_physical_threats': stats['p1_physical_threats'],
        'p2_physical_threats': stats['p2_physical_threats'],
        'physical_threats_diff': stats['p1_physical_threats'] - stats['p2_physical_threats'],
        
        'p1_special_threats': stats['p1_special_threats'],
        'p2_special_threats': stats['p2_special_threats'],
        'special_threats_diff': stats['p1_special_threats'] - stats['p2_special_threats'],
        
        'p1_walls': stats['p1_walls'],
        'p2_walls': stats['p2_walls'],
        'walls_diff': stats['p1_walls'] - stats['p2_walls'],
        
        'offensive_pressure_p1': stats['p1_physical_threats'] + stats['p1_special_threats'],
        'offensive_pressure_p2': stats['p2_physical_threats'] + stats['p2_special_threats'],
        'offensive_pressure_diff': (stats['p1_physical_threats'] + stats['p1_special_threats']) - 
                                   (stats['p2_physical_threats'] + stats['p2_special_threats']),
        
        'p1_balance': abs(stats['p1_physical_threats'] - stats['p1_special_threats']),
        'p2_balance': abs(stats['p2_physical_threats'] - stats['p2_special_threats']),
        'balance_diff': abs(stats['p1_physical_threats'] - stats['p1_special_threats']) - 
                       abs(stats['p2_physical_threats'] - stats['p2_special_threats']),
    }, dtype="float64")


# ==============================================================================
# POSITION SCORE & MOVE QUALITY FEATURES
# ==============================================================================

def _calculate_position_score(state_info: dict) -> float:
    """Calculate position score for one side."""
    if not state_info['alive_pokemon']:
        return -10.0
    
    alive = state_info['alive_pokemon']
    
    # High-value pokemon component
    highvalue_sum = sum(POKEMON_VALUES.get(name.lower(), 0.6) * hp_pct 
                       for name, hp_pct, _ in alive)
    
    # Average HP component
    avg_hp = np.mean([hp for _, hp, _ in alive])
    
    # Status burden component
    status_burden = -sum(1 for _, _, status in alive if status in STATUS_CODES) / len(alive)
    
    # Role diversity component
    role_score = min(sum(count * 0.3 for count in state_info['roles'].values()) / 3.0, 1.0)
    
    # Tempo advantage
    tempo = SCORE_WEIGHTS['tempo'] if state_info.get('opponent_disabled', False) else 0
    
    return (SCORE_WEIGHTS['highvalue'] * (highvalue_sum / 6.0) + 
            SCORE_WEIGHTS['hp'] * avg_hp +
            SCORE_WEIGHTS['status'] * status_burden + 
            SCORE_WEIGHTS['role'] * role_score + tempo)


def _get_state_snapshot(timeline, turn_index, side):
    """Extract state snapshot for a side at a given turn."""
    if turn_index >= len(timeline):
        return None
    
    # Get all alive pokemon up to this turn
    pokemon_last_state = {}
    fainted = set()
    
    for t in timeline[:turn_index + 1]:
        st = t.get(f"{side}_pokemon_state", {}) or {}
        name = (st.get("name") or "").lower()
        if name:
            hp_pct = st.get("hp_pct", 1.0)
            if hp_pct == 0:
                fainted.add(name)
            else:
                pokemon_last_state[name] = (hp_pct, st.get("status", "nostatus"))
    
    alive_pokemon = [(name, hp, status) for name, (hp, status) in pokemon_last_state.items() 
                     if name not in fainted]
    
    # Check if opponent is disabled
    opponent_side = 'p2' if side == 'p1' else 'p1'
    opponent_state = timeline[turn_index].get(f"{opponent_side}_pokemon_state", {}) or {}
    opponent_disabled = opponent_state.get("status", "nostatus") in {'slp', 'frz'}
    
    # Get roles
    roles = {role: sum(1 for name, _, _ in alive_pokemon if name in pokemon_set)
             for role, pokemon_set in ROLE_TAXONOMY.items()}
    
    return {
        'alive_pokemon': alive_pokemon,
        'opponent_disabled': opponent_disabled,
        'roles': roles
    }


def _extract_move_quality_features(row):
    """Calculate position score deltas and identify errors/blunders."""
    timeline = row.get("battle_timeline", []) or []
    if len(timeline) < 2:
        return {
            'p1_errors_count': 0, 'p2_errors_count': 0,
            'p1_blunders_count': 0, 'p2_blunders_count': 0,
            'p1_mean_negative_delta': 0, 'p2_mean_negative_delta': 0,
            'p1_blunders_early': 0, 'p2_blunders_early': 0,
            'p1_blunders_late': 0, 'p2_blunders_late': 0,
        }
    
    # Initialize counters
    counters = {
        'p1': {'deltas': [], 'errors': 0, 'blunders': 0, 'blunders_early': 0, 'blunders_late': 0},
        'p2': {'deltas': [], 'errors': 0, 'blunders': 0, 'blunders_early': 0, 'blunders_late': 0}
    }
    
    # Calculate deltas for each turn
    for i in range(len(timeline) - 1):
        turn = int(timeline[i].get("turn", 0))
        
        # Get snapshots for both sides
        snapshots = {side: (_get_state_snapshot(timeline, i, side), 
                           _get_state_snapshot(timeline, i + 1, side))
                    for side in ['p1', 'p2']}
        
        # Skip if any snapshot is missing
        if not all(pre and post for pre, post in snapshots.values()):
            continue
        
        # Process each side
        for side in ['p1', 'p2']:
            pre, post = snapshots[side]
            delta = _calculate_position_score(post) - _calculate_position_score(pre)
            counters[side]['deltas'].append(delta)
            
            # Classify moves
            if delta <= ERROR_THRESHOLD:
                counters[side]['errors'] += 1
                if delta <= BLUNDER_THRESHOLD:
                    counters[side]['blunders'] += 1
                    if turn <= 10:
                        counters[side]['blunders_early'] += 1
                    elif turn >= 21:
                        counters[side]['blunders_late'] += 1
    
    # Calculate mean negative deltas
    return {
        'p1_errors_count': counters['p1']['errors'],
        'p2_errors_count': counters['p2']['errors'],
        'p1_blunders_count': counters['p1']['blunders'],
        'p2_blunders_count': counters['p2']['blunders'],
        'p1_mean_negative_delta': np.mean([d for d in counters['p1']['deltas'] if d < 0] or [0]),
        'p2_mean_negative_delta': np.mean([d for d in counters['p2']['deltas'] if d < 0] or [0]),
        'p1_blunders_early': counters['p1']['blunders_early'],
        'p2_blunders_early': counters['p2']['blunders_early'],
        'p1_blunders_late': counters['p1']['blunders_late'],
        'p2_blunders_late': counters['p2']['blunders_late'],
    }


def make_move_quality_features(row):
    """Create move quality features."""
    q = _extract_move_quality_features(row)
    
    return pd.Series({
        'p1_errors_count': q['p1_errors_count'],
        'p2_errors_count': q['p2_errors_count'],
        'errors_diff': q['p1_errors_count'] - q['p2_errors_count'],
        'p1_blunders_count': q['p1_blunders_count'],
        'p2_blunders_count': q['p2_blunders_count'],
        'blunders_diff': q['p1_blunders_count'] - q['p2_blunders_count'],
        'p1_mean_negative_delta': q['p1_mean_negative_delta'],
        'p2_mean_negative_delta': q['p2_mean_negative_delta'],
        'negative_delta_diff': q['p1_mean_negative_delta'] - q['p2_mean_negative_delta'],
        'p1_blunders_early': q['p1_blunders_early'],
        'p2_blunders_early': q['p2_blunders_early'],
        'blunders_early_diff': q['p1_blunders_early'] - q['p2_blunders_early'],
        'p1_blunders_late': q['p1_blunders_late'],
        'p2_blunders_late': q['p2_blunders_late'],
        'blunders_late_diff': q['p1_blunders_late'] - q['p2_blunders_late'],
    }, dtype="float64")


# ==============================================================================
# INTERACTION FEATURES
# ==============================================================================

def make_interaction_features(row):
    """Create multiplicative interaction features."""
    g = row.get
    out = {}
    
    # MOMENTUM × MATERIAL
    out['momentum_x_survivors'] = g('hp_momentum_diff', 0) * g('survivor_diff_30', 0)
    out['momentum_x_ko'] = g('hp_momentum_diff', 0) * g('ko_diff_30', 0)
    
    # CRITICAL POKÉMON
    out['critical_x_hp_quality'] = g('critical_alive_diff', 0) * g('avg_hp_alive_diff', 0)
    out['critical_x_status_free'] = g('critical_alive_diff', 0) * (-g('statused_alive_end_diff', 0))
    
    # EARLY → LATE
    out['early_damage_to_ko'] = g('early_damage_diff', 0) * g('ko_diff_30', 0)
    out['early_switches_x_survivors'] = g('early_switch_diff', 0) * g('survivor_diff_30', 0)
    
    # BOOSTS × SURVIVORS
    out['boosts_x_alive'] = g('max_boosts_diff', 0) * g('survivor_diff_30', 0)
    out['boosts_x_hp'] = g('max_boosts_diff', 0) * g('avg_hp_alive_diff', 0)
    
    # HP DISPERSION × ROLE
    out['hp_dispersion_x_breakers'] = g('std_hp_alive_diff', 0) * g('rolecount_breaker_spec_diff_end', 0)
    
    # MOVE QUALITY × MATERIAL
    out['blunders_x_ko_loss'] = g('blunders_diff', 0) * (-g('ko_diff_30', 0))
    out['errors_x_hp_loss'] = g('errors_diff', 0) * (-g('avg_hp_alive_diff', 0))
    
    # TEAM COMP × EXECUTION
    out['team_value_x_errors'] = g('team_value_diff', 0) * (-g('errors_diff', 0))
    out['speed_control_x_momentum'] = g('speed_control_diff', 0) * g('hp_momentum_diff', 0)

    
    # STATUS AND EFFECTS
    out['ko_status_interaction'] = g('ko_diff_30', 0) * g('status_diff_highvalue', 0)
    out['survivor_disabled_interaction'] = g('survivor_diff_30', 0) * g('turns_disabled_diff', 0)
    out['effects_momentum_interaction'] = g('active_effects_weighted_diff_end', 0) * g('late_game_status_swing', 0)
    
    # ENDGAME
    out['hp_p1_x_survivors'] = g('p1_avg_hp_alive', 0) * g('p1_survivors_30', 0)
    out['hp_p2_x_survivors'] = g('p2_avg_hp_alive', 0) * g('p2_survivors_30', 0)
    out['hp_p1_x_kos'] = g('p1_avg_hp_alive', 0) * g('p1_kos_30', 0)
    out['hp_p2_x_kos'] = g('p2_avg_hp_alive', 0) * g('p2_kos_30', 0)
    out['endgame_result_consistency'] = g('ko_diff_30', 0) * g('survivor_diff_30', 0)
    out['disabled_to_conversion'] = g('ko_diff_30', 0) * g('turns_disabled_diff_w_decay', 0)
    out['control_x_status_pressure'] = g('turns_disabled_diff_w_decay', 0) * g('statused_alive_end_diff', 0)
    out['durable_advantage_compound'] = g('effective_avg_hp_diff', 0) * g('survivor_diff_30', 0)
    out['sleep_power_factor'] = g('sleepers_end_diff', 0) * g('turns_disabled_diff_w_decay', 0)
    out['freeze_breakers_lock'] = g('freezes_end_diff', 0) * g('p2_rolecount_breaker_phys_end', 0)
    out['neuter_breakers_with_status'] = g('rolecount_breaker_phys_diff_end', 0) * g('statused_alive_end_diff', 0)
    out['control_closure_link'] = (g('survivor_diff_30', 0) * g('turns_disabled_diff', 0)) * g('ko_diff_30', 0)
    out['opp_hp_under_status'] = g('p2_avg_hp_alive', 0) * g('p2_statused_alive_end', 0)
    

    return pd.Series(out, dtype='float64')


# ==============================================================================
# FEATURE EXTRACTION PIPELINE
# ==============================================================================

def extract_all_features(df):
    """
    Extract all features from a DataFrame.
    
    Args:
        df: DataFrame with battle data
        
    Returns:
        DataFrame: All extracted features
    """
    print("\n✓ Extracting features...")
    
    # Status features
    status_features = df.apply(make_status_features, axis=1)
    
    # Endgame features
    endgame_features = df.apply(make_endgame_features, axis=1)
    
    # Role-based features
    role_features = df.apply(make_role_features, axis=1)
    
    # Status-at-T30 features
    status_t30_features = df.apply(make_status_t30_features, axis=1)
    
    # HP distribution features
    hp_dist_features = df.apply(make_hp_distribution_features, axis=1)
    
    # Move quality features
    move_quality_features = df.apply(make_move_quality_features, axis=1)
    
    # Static features
    static_features = df.apply(make_static_features, axis=1)
    
    # Momentum features
    momentum_features = df.apply(make_momentum_features, axis=1)
    
    # Critical pokemon features
    critical_features = df.apply(make_critical_pokemon_features, axis=1)
    
    # Early game features
    early_features = df.apply(make_early_game_features, axis=1)
    
    # Boost features
    boost_features = df.apply(make_boost_features, axis=1)
    
    
    # Combine all features
    all_features = pd.concat([
        status_features, endgame_features, role_features, status_t30_features,
        hp_dist_features, move_quality_features, static_features,
        momentum_features, critical_features, early_features, boost_features,
    ], axis=1)
    
    # Interaction features (need all other features first)
    for col in all_features.columns:
        df[col] = all_features[col].astype(float)
    
    interaction_features = df.apply(make_interaction_features, axis=1).astype('float64')
    
    # Add interactions to feature set
    all_features = pd.concat([all_features, interaction_features], axis=1)
    
    print(f"✓ {len(all_features.columns)} features created")
    print(f"  - Status features: {len(status_features.columns)}")
    print(f"  - Endgame features: {len(endgame_features.columns)}")
    print(f"  - Role features: {len(role_features.columns)}")
    print(f"  - Status-at-T30 features: {len(status_t30_features.columns)}")
    print(f"  - HP distribution features: {len(hp_dist_features.columns)}")
    print(f"  - Move quality features: {len(move_quality_features.columns)}")
    print(f"  - Static features: {len(static_features.columns)}")
    print(f"  - Momentum features: {len(momentum_features.columns)}")
    print(f"  - Critical pokemon features: {len(critical_features.columns)}")
    print(f"  - Early game features: {len(early_features.columns)}")
    print(f"  - Boost features: {len(boost_features.columns)}")
    print(f"  - Interaction features: {len(interaction_features.columns)}")
    
    return all_features
