import pandas as pd
import numpy as np
import os

np.random.seed(2026)

# Parameters
num_timesteps = 10
num_samples_per_timestep = 10

data = []

src_id = 1
tgt_id = 100000
helper_id = 200000
rand_id = 300000

for t in range(1, num_timesteps):
    for i in range(num_samples_per_timestep):
        data.append({
            'ts': t+2,
            'u': src_id,
            'i': tgt_id,
            'label': t+2
        })
        data.append({
            'ts': t + 1,
            'u': helper_id,
            'i': tgt_id,
            'label': ''
        })
        for id in [src_id, tgt_id]:
            for _ in range(2):
                data.append({
                    'ts': t+1,
                    'u': rand_id,
                    'i': id,
                    'label': ''
                })
                curr_rand_id = rand_id
                rand_id += 1
                for _ in range(2):
                    data.append({
                        'ts': t,
                        'u': rand_id,
                        'i': curr_rand_id,
                        'label': ''
                    })
                    rand_id += 1
        src_id += 1
        tgt_id += 1
        helper_id += 1

df = pd.DataFrame(data)

df = df.sort_values(by=['ts']).reset_index(drop=True)

# Add index column
df = df.reset_index(names='idx')   
df["idx"] = df["idx"] + 1 

edge_features = np.zeros((len(df) + 1, 1))  # features for each edge
node_features = np.zeros((rand_id+2, 3))  # 3 zero features for each node

# Add node names 
node_names = np.full((rand_id+2,), 'rnd', dtype=object)
node_names[1:100000] = 'A'
node_names[100000:200000] = 'B'
node_names[200000:300000] = 'H'


#Save data
os.makedirs("Data/generated", exist_ok=True)
df.to_csv("Data/generated/ml_generated.csv")
np.save("Data/generated/ml_generated.npy", edge_features)
np.save("Data/generated/ml_generated_node.npy", node_features)
np.save("Data/generated/generated_node_names.npy", node_names)
print(f"Generated data saved to Data/generated/ml_generated.csv and corresponding features. {len(df)} edges created.")