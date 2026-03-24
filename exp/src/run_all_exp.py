import itertools
import subprocess
import sys

TASKS = ["node", "env"]
MODELS = ["cnn", "resnet"]
SPLITS = ["random", "leave_one_env_out"]
SEQ_LENS = [100, 500, 1000]
OVERLAPS = [0.4, 0.5]

# assume we always leave out env 4 for testing in leave_one_env_out split
DEFAULT_TEST_ENV = 4

def main():
    exp_id = 1
    total = len(TASKS) * len(MODELS) * len(SPLITS) * len(SEQ_LENS) * len(OVERLAPS)

    for task, model, split, seq_len, overlap in itertools.product(
        TASKS, MODELS, SPLITS, SEQ_LENS, OVERLAPS
    ):
        cmd = [
            sys.executable, "src/run_experiment.py",
            "--task", task,
            "--seq_len", str(seq_len),
            "--overlap", str(overlap),
            "--split", split,
            "--model", model,
            "--epochs", "100",
            "--batch_size", "64",
            "--lr", "0.0001"
        ]

        if split == "leave_one_env_out":
            cmd.extend(["--test_env", str(DEFAULT_TEST_ENV)])

        print(f"\n===== Running experiment {exp_id}/{total} =====")
        print(" ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"Experiment {exp_id} failed.")
        else:
            print(f"Experiment {exp_id} done.")

        exp_id += 1


if __name__ == "__main__":
    main()