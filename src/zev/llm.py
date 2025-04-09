import os
from typing import Optional

import openai
from pydantic import BaseModel


class Command(BaseModel):
    command: str
    short_explanation: str


class OptionsResponse(BaseModel):
    commands: list[Command]
    is_valid: bool
    explanation_if_not_valid: Optional[str] = None


PROMPT = """
You are a helpful assistent that helps users remember commands for the terminal. You 
will return a JSON object with a list of at most three options.

The options should be related to the prompt that the user provides (the prompt might
either be desciptive or in the form of a question).

The options should be in the form of a command that can be run in a bash terminal.

If the user prompt is not clear, return an empty list and set is_valid to false, and
provide an explanation of why it is not clear in the explanation_if_not_valid field.

Otherwise, set is_valid to true, leave explanation_if_not_valid empty, and provide the 
commands in the commands field (remember, up to 3 options, and they all must be commands
that can be run in a bash terminal without changing anything). Each command should have
a short explanation of what it does.

Here is some context about the user's environment:

============== 

{context}

============== 

Here is the users prompt:

============== 

{prompt}
"""


def get_client():
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_options(prompt: str, context: str) -> OptionsResponse:
    client = get_client()
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(prompt=prompt, context=context)}],
        response_format=OptionsResponse,
    )
    return response.choices[0].message.parsed
