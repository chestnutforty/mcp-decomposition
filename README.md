# MCP Decomposition Server

MCP server that decomposes complex forecasting questions into simpler subquestions using OpenAI's o3 model with high reasoning effort.

## Features

- Uses o3 with `reasoning.effort: "high"` for deep analysis
- Structured output via JSON schema for consistent results
- Backtesting support with cutoff_date parameter

## Tools

### decompose_question

Decomposes a forecasting question into subquestions with:
- **question**: The subquestion text
- **rationale**: Why this subquestion is relevant
- **importance**: high/medium/low rating

## Environment Variables

- `OPENAI_API_KEY`: Required for API access

## Usage

```bash
uv sync
mcp run -t sse server.py:mcp
```

## Example

Input:
```
question: "Will electric vehicles make up more than 10% of new light duty vehicle sales in the United States before October 2025?"
```

Output includes subquestions like:
- What is the current EV market share in the US?
- What is the historical growth rate of EV adoption?
- What policy changes could affect EV sales?