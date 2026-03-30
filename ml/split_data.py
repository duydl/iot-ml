import os
import pandas as pd
import shutil

raw_dir = '/home/duydl/projects/riot-exercises/_Project/ml/data/raw'
env_dir = '/home/duydl/projects/riot-exercises/_Project/ml/data/raw-env'
node_dir = '/home/duydl/projects/riot-exercises/_Project/ml/data/raw-node'

os.makedirs(env_dir, exist_ok=True)
os.makedirs(node_dir, exist_ok=True)

print("Processing raw-env...")
env_files = {}
for f in os.listdir(raw_dir):
    if f.endswith('.csv'):
        # Extract name: e0-bridge.csv -> bridge
        env_name = f.split('-')[1].split('.')[0]
        src = os.path.join(raw_dir, f)
        dst = os.path.join(env_dir, f"{env_name}.csv")
        shutil.copy(src, dst)
        env_files[env_name] = src
        print(f"Copied {f} to {env_name}.csv")

print("Processing raw-node...")
nodes = ['RIOT-BLE-0', 'RIOT-BLE-1', 'RIOT-BLE-2', 'RIOT-BLE-3']
node_data = {n: [] for n in nodes}

for env_name, filepath in env_files.items():
    df = pd.read_csv(filepath)
    # Add environment column to tracking where the row came from
    df['environment'] = env_name
    for n in nodes:
        node_df = df[df['device'] == n]
        node_data[n].append(node_df)

for n in nodes:
    if node_data[n]:
        combined = pd.concat(node_data[n], ignore_index=True)
        combined = combined.sort_values(by='ts')
        node_idx = n.split('-')[-1]
        out_path = os.path.join(node_dir, f"node{node_idx}.csv")
        combined.to_csv(out_path, index=False)
        print(f"Created node{node_idx}.csv with {len(combined)} rows")

print("Data preparation complete.")
