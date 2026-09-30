from pydantic import BaseModel,Field
from langchain_core.tools import StructuredTool


class EvenInput(BaseModel):
    number: int


def is_even(number: int):
    return number % 2 == 0


even_tool = StructuredTool.from_function(
    func=is_even,
    name="is_even",
    description="Check whether a number is even.",
    args_schema=EvenInput
)

print(even_tool.invoke({"number": 4}))