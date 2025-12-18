import os
from typing import Annotated
from datetime import datetime
from fastmcp import FastMCP
from openai import OpenAI
from pydantic import BaseModel

mcp = FastMCP(
    name="decomposition",
    instructions=r"""
Question decomposition service that breaks down complex forecasting questions into simpler,
more tractable subquestions. Uses advanced reasoning to identify the key components and
dependencies needed to answer the main question.
""".strip(),
)


class Subquestion(BaseModel):
    question: str
    rationale: str
    importance: str  # "high", "medium", "low"


class DecompositionResult(BaseModel):
    original_question: str
    subquestions: list[Subquestion]
    reasoning_summary: str


@mcp.tool(
    name="decompose_question",
    title="Decompose Forecasting Question",
    description="Decomposes a complex forecasting question into simpler subquestions that can be researched independently. Uses advanced reasoning to identify key factors, dependencies, and uncertainties.",
    tags={"backtesting_supported"},
    exclude_args=["cutoff_date"],
)
def decompose_question(
    question: Annotated[str, "The forecasting question to decompose"],
    context: Annotated[str, "Optional additional context about the question"] = "",
    max_subquestions: Annotated[int, "Maximum number of subquestions to generate (default 5)"] = 5,
    cutoff_date: Annotated[str, "The date must be in the format YYYY-MM-DD"] = datetime.now().strftime("%Y-%m-%d"),
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = f"""You are an expert at decomposing complex forecasting questions into simpler,
more tractable subquestions. Your goal is to identify the key factors, assumptions, and uncertainties
that need to be researched to answer the main question.

IMPORTANT: You are operating as if the current date is {cutoff_date}. Do not reference any events,
data, or information that would not have been available on this date.

For each subquestion:
1. Make it specific and researchable
2. Explain why it's relevant to the main question
3. Rate its importance (high/medium/low) based on how much it affects the final answer

Focus on:
- Base rates and historical precedents
- Key causal factors and mechanisms
- Important uncertainties and unknowns
- Potential confounders or complications
- Timeline and deadline considerations"""

    user_prompt = f"""Decompose this forecasting question into {max_subquestions} or fewer subquestions:

Question: {question}
{f'Context: {context}' if context else ''}

Identify the most important factors that would help answer this question accurately."""

    response = client.responses.create(
        model="o3",
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "decomposition_result",
                "schema": DecompositionResult.model_json_schema(),
                "strict": True,
            }
        },
    )

    result = DecompositionResult.model_validate_json(response.output_text)

    output_lines = [
        f"## Question Decomposition",
        f"**Original:** {result.original_question}",
        f"",
        f"### Subquestions",
    ]

    for i, sq in enumerate(result.subquestions, 1):
        output_lines.extend([
            f"",
            f"**{i}. {sq.question}**",
            f"- Rationale: {sq.rationale}",
            f"- Importance: {sq.importance}",
        ])

    output_lines.extend([
        f"",
        f"### Summary",
        f"{result.reasoning_summary}",
    ])

    return "\n".join(output_lines)