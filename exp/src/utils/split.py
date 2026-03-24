from sklearn.model_selection import train_test_split

def split_dataset(X, y, env_ids, split_strategy="random", test_size=0.25, random_state=42, test_env=None):
    if split_strategy == "random":
        X_train, X_test, y_train, y_test, env_train, env_test = train_test_split(
            X, y, env_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test, env_train, env_test

    elif split_strategy == "leave_one_env_out":
        if test_env is None:
            raise ValueError("test_env must be provided when split_strategy='leave_one_env_out'")

        train_mask = env_ids != test_env
        test_mask = env_ids == test_env

        X_train = X[train_mask]
        y_train = y[train_mask]
        env_train = env_ids[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]
        env_test = env_ids[test_mask]

        return X_train, X_test, y_train, y_test, env_train, env_test

    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")