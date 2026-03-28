from sklearn.model_selection import train_test_split

def split_dataset(X, y, env_ids,node_ids,task, split_strategy="random", test_size=0.25, random_state=42, test_env=3,test_node=1):
    """
    task:
        - 'node': classify node, so oneout = one_env_out
        - 'env' : classify environment, so oneout = one_node_out
    """
    if split_strategy == "random":
        X_train, X_test, y_train, y_test, env_train, env_test,node_train,node_test = train_test_split(
            X, y, env_ids,node_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        return X_train, X_test, y_train, y_test, env_train, env_test, node_train, node_test

    elif split_strategy == "oneout":
        if task == "node":
            # one_env_out
            train_mask = env_ids != test_env
            test_mask = env_ids == test_env

        elif task == "env":
            # one_node_out
            train_mask = node_ids != test_node
            test_mask = node_ids == test_node

        else:
            raise ValueError(f"Unknown task: {task}")

        X_train = X[train_mask]
        y_train = y[train_mask]
        env_train = env_ids[train_mask]
        node_train = node_ids[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]
        env_test = env_ids[test_mask]
        node_test = node_ids[test_mask]


        return X_train, X_test, y_train, y_test, env_train, env_test, node_train, node_test

    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")