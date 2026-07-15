from vllm_utils import VLLMServer
import json
from typing import Literal
import time
from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

def load_test_samples():
    samples = []
    with open("data/gsm8k/test.jsonl") as f:
        for line in f.readlines():
            obj = json.loads(line)
            obj["groud_truth"] = obj["answer"].split("####")[1].strip()
            samples.append(obj)
    return samples

def load_prompt_template(alias: Literal["question_only", "r1_zero", "r1_zero_three_shot"]):
    file_dict = {
        "question_only": "question_only.prompt",
        "r1_zero": "r1_zero.prompt",
        "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
    }
    with open(f"cs336_alignment/prompts/{file_dict[alias]}") as f:
        return f.read()


if __name__ == "__main__":
    #model = "allenai/OLMo-2-0425-1B"
    model = "Qwen/Qwen2.5-1.5B"
    samples = load_test_samples()
    server = VLLMServer(
            model_id=model,
            gpu=0
        )
    server.start()
    for prompt_type in ["question_only", "r1_zero", "r1_zero_three_shot"]:
        prompt_template = load_prompt_template(prompt_type)
        prompts = [prompt_template.format(question=sample["question"]) for sample in samples]
        begin = time.monotonic()
        sampling_params = {
            "temperature": 1.0,
            "max_tokens": 1024,
            "n": 1,
            "seed": 0
        }
        if prompt_type in ["r1_zero", "r1_zero_three_shot"]:
            sampling_params["stop"] = ["</answer>"]
            sampling_params["include_stop_str_in_output"] = True
        completions = server.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=100
        )
        end = time.monotonic()
        print(f"finish in {end-begin}s")
        reward_fn = question_only_reward_fn if prompt_type == "question_only" else r1_zero_reward_fn
        rewards = [{
            **reward_fn(c.text, samples[i]["groud_truth"]),
            "question": samples[i]["question"],
            "prompt": prompts[i],
            "response": c.text
            } for i, c in enumerate(completions)]
        with open(f"cs336_alignment/rewards/{model.split("/")[-1]}_{prompt_type}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rewards)