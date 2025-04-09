import os
import platform


def get_input_string(
    field_name: str,
    prompt: str,
    current_value: str = "",
    default: str = "",
    required: bool = False,
) -> str:
    if current_value:
        prompt = f"{prompt} (default/current: {current_value})"
    elif default:
        prompt = f"{prompt} (default: {default})"
    else:
        prompt = f"{prompt}"

    value = input(prompt + ": ") or default
    if required and not value:
        print(f"{field_name} is required, please try again")
        return get_input_string(field_name, prompt, current_value, default, required)

    return value or current_value or default


def get_env_context() -> str:
    os_name = platform.platform(aliased=True)
    shell = os.environ.get("SHELL")
    return f"OS: {os_name}\nSHELL: {shell}" if shell else f"OS: {os_name}"
