def make_env_task_prompt(env_type: str):
    if "pusht" in env_type:
        return "You're an helpful robot. Your goal is to push the gray block to the goal zone in green. Your position is identified by a blue circle and the block is a T shape."
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")
