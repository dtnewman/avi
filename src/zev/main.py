import os
import sys
from dataclasses import dataclass

import dotenv
import platformdirs
import pyperclip
import questionary
from rich import print as rprint

from zev.llm import get_options
from zev.utils import get_env_context, get_input_string


@dataclass
class DotEnvField:
    name: str
    prompt: str
    required: bool = True
    default: str = ""


DOT_ENV_FIELDS = [
    DotEnvField(
        name="OPENAI_API_KEY",
        prompt="Enter your OpenAI API key",
        required=False,
        default="",
    ),
]


def setup():
    new_file = ""
    for field in DOT_ENV_FIELDS:
        current_value = os.environ.get(field.name, "")
        new_value = get_input_string(field.name, field.prompt, current_value, field.default, field.required)
        new_file += f"{field.name}={new_value}\n"

    app_data_dir = platformdirs.user_data_dir("zev")
    os.makedirs(app_data_dir, exist_ok=True)
    with open(os.path.join(app_data_dir, ".env"), "w") as f:
        f.write(new_file)


def show_options(words: str):
    context = get_env_context()
    response = get_options(prompt=words, context=context)
    if not response.is_valid:
        print(response.explanation_if_not_valid)
        return

    if not response.commands:
        print("No commands available")
        return

    options = [questionary.Choice(cmd.command, description=cmd.short_explanation) for cmd in response.commands]
    options.append(questionary.Choice("Cancel"))
    options.append(questionary.Separator())

    selected = questionary.select(
        "Select command:",
        choices=options,
        use_shortcuts=True,
        style=questionary.Style(
            [
                ("answer", "fg:#61afef"),
                ("question", "bold"),
                ("instruction", "fg:#98c379"),
            ]
        ),
    ).ask()

    if selected != "Cancel":
        pyperclip.copy(selected)
        rprint("\n[green]✓[/green] Copied to clipboard")


def run_no_prompt():
    input = get_input_string("input", "Describe what you want to do", "", "", False)
    show_options(input)


def app():
    # check if .env exists
    app_data_dir = platformdirs.user_data_dir("zev")
    if not os.path.exists(os.path.join(app_data_dir, ".env")):
        setup()
        print("Setup complete... querying now...\n")

    args = sys.argv[1:]

    if not args:
        run_no_prompt()
        return

    dotenv.load_dotenv(os.path.join(app_data_dir, ".env"))

    # Strip any trailing question marks from the input
    query = " ".join(args).rstrip("?")
    show_options(query)


if __name__ == "__main__":
    app()
