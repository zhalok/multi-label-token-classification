def split_dataset(df):
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define split ratios
    train_size = int(0.6 * len(df_shuffled))  # 60% for training
    temp_size = len(df_shuffled) - train_size
    val_size = temp_size // 2  # 20% for validation
    test_size = temp_size - val_size  # 20% for test

    # Split DataFrame
    train_df = df_shuffled[:train_size]
    temp_df = df_shuffled[train_size:]
    val_df = temp_df[:val_size]
    test_df = temp_df[val_size:]

    return {"train": train_df, "test": test_df, "val": val_df}
