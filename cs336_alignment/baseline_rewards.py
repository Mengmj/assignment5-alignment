import json
from pathlib import Path
def count_reward(rewards):
    total_reward = 0
    total_format_reward = 0
    for r in rewards:
        total_reward += r["reward"]
        total_format_reward += r["format_reward"]
    return {
        "total_reward": total_reward,
        "total_format_reward": total_format_reward,
        "avg_reward": total_reward / len(rewards),
        "avg_format_reward": total_format_reward / len(rewards)
    }

if __name__ == "__main__":
    rewards_root_path = Path("cs336_alignment/rewards")
    baseline_rewards = []
    for file_path in rewards_root_path.iterdir():
        with open(file_path) as f:
            reward_records = [json.loads(line) for line in f.readlines()]
            reward_result = count_reward(reward_records)
        baseline_rewards.append(
            {
                "tag": file_path.name[:-6],
                **reward_result
            }
        )
    with open("cs336_alignment/baseline_rewords.jsonl","w") as f:
        f.writelines([json.dumps(reward) + "\n" for reward in baseline_rewards])

    