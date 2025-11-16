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

train_df, test_df = load_data()
