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


class SubSubquestion(BaseModel):
    question: str

class Subqestion(BaseModel):
    subquestions: list[SubSubquestion]
    question: str
    
class DecompositionResult(BaseModel):
    original_question: str
    subquestions: list[Subqestion]


@mcp.tool(
    name="decompose_question",
    title="Decompose Forecasting Question",
    description="Decomposes a complex forecasting question into simpler subquestions that can be forecasted independently.",
    tags={"backtesting_supported"},
    exclude_args=["cutoff_date"],
)
def decompose_question(
    question: Annotated[str, "The forecasting question to decompose"],
    context: Annotated[str, "Optional additional context about the question"] = "",
    cutoff_date: Annotated[str, "The date must be in the format YYYY-MM-DD"] = datetime.now().strftime("%Y-%m-%d"),
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = f"""Break down a forecasting question into subquestions for downstream forecasters. Do not formulate them as research questions of historic facts but only forward-looking. Questions on the same level should be as independent as possible (avoid strong correlation).

<decomposition_strategies>
Use these proven patterns to break complex questions into tractable subquestions. You can combine strategies.

Temporal decomposition:
- Break by sequential phases/milestones
- Best for: staged processes with clear intermediate conditions

Conditional/prerequisite decomposition:
- Identify necessary gates; combine via conditional structure
- Best for: mergers, approvals, multi-step completions

Stakeholder decomposition:
- Separate decisions by actors (boards, regulators, governments)
- Best for: politics, business, multi-party actions

Mechanism/pathway decomposition:
- Distinct causal routes; combine via OR logic (with overlap handled explicitly)
- Best for: outcomes achievable through multiple independent paths

Component decomposition:
- Outcome is aggregate of measurable parts
- Best for: GDP, performance metrics, composite indices

Failure mode decomposition:
- Enumerate what must NOT happen
- Best for: projects, plans, multi-point failure risks

Reference class decomposition:
- Start from sector/process base rates, then adjust for specifics
- Best for: startups, treaties, adoption processes with analogues

Scenario decomposition:
- Define mutually exclusive world states; mix conditional forecasts
- Best for: outcomes dependent on macro regimes or external shocks
</decomposition_strategies>

<important_reminders>
- Do not include questions like "Will the resolution criteria be met..?"
</important_reminders>

<example>
Question: “Will the unemployment rate for recent college graduates in the United States rise to 20% or more for three months before 2028?”
Subquestion Decomposition:
1. Will there be an economic crisis/recession in the US before 2028?  
    1. Will there be an economic crisis/recession related to the ordinary business cycle?  
    2. Will the stock market enter bear territory?  
    3. Will there be an economic crisis/recession caused by trade conflicts?  
        1. What will the effective tariff rate be in 2026?   
        2. Will the effective tariff rate change dramatically throughout the year?  
        3. Will exports from the US drop significantly?  
    4. Will there be an economic crisis/recession caused by an oil supply shock?  
        1. Will oil supply fall?  
        2. Will domestic energy prices rise?  
        1. Will data center construction significantly increase the demand for energy in the US?  
    5. Will there be an economic crisis/recession caused by an asset bubble?  
        1. Will the market cap of major companies with large AI exposure (e.g. Meta, OpenAI, Anthropic, Nvidia, xAI, Microsoft, Oracle) fall by more than 20%?  
        2. Will one or more large crypto companies enter bankruptcy?  
    6. Will there be a debt crisis in the US?  
        1. Will the US default on its debt?  
    7. Will the Fed raise interest rates significantly?  
        1. Will inflation rise past and remain over 3%?  
2. Will AI applications replace entry-level workers with college degrees?
</example>

<example>
Question: "Which party will hold a plurality in the US House of Representatives after the 2026 midterm elections?"
Subquestion Decomposition:
   1. Will the US economy improve or get worse?  
   2. Will Trump’s approval rating increase or decline?  
   3. Will the generic congressional ballot shift toward Republicans or Democrats?  
   4. Will there be new redistricting?  
   5. Will SCOTUS rule on Section 2 of the Voting Rights Act in a way that affects redistricting by early next year?  
   6. Will weather or natural catastrophes interfere with voting?  
   7. Will the elections be fair?  
      1. Will governments harass or intimidate opposition leaders?  
      2. Will governments engage in voter suppression or intimidation?  
      3. Will governments coerce media organizations to slant coverage?  
      4. Will protests or private militia interfere with voting?  
      5. Will vote counts be fair?
<example>

Output the subquestions as a nested list."""

    user_prompt = f"""Question: {question}"""

    response = client.responses.parse(
        model="gpt-5.2",
        reasoning={"effort": "high"},
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text_format=DecompositionResult,
    )

    result = response.output_parsed

    lines = []
    for i, sq in enumerate(result.subquestions, 1):
        lines.append(f"{i}. {sq.question}")
        for j, subsq in enumerate(sq.subquestions, 1):
            lines.append(f"   {j}. {subsq.question}")

    return "\n".join(lines)