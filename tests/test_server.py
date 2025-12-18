import inspect
from server import decompose_question, mcp


class TestBacktestToolConfiguration:
    """Tests for backtesting tool configuration."""

    def test_backtest_tool_has_backtesting_supported_tag(self):
        """Backtest tools must have 'backtesting_supported' in tags."""
        tool = decompose_question
        tags = getattr(tool, "tags", set()) or set()
        assert "backtesting_supported" in tags, (
            f"Backtest tool missing 'backtesting_supported' tag. Found tags: {tags}"
        )

    def test_backtest_tool_has_cutoff_date_parameter(self):
        """Backtest tools must have 'cutoff_date' as a function parameter."""
        sig = inspect.signature(decompose_question.fn)
        params = list(sig.parameters.keys())
        assert "cutoff_date" in params, (
            f"Backtest tool missing 'cutoff_date' parameter. Found params: {params}"
        )

    def test_backtest_tool_cutoff_date_excluded_from_schema(self):
        """Backtest tools must exclude 'cutoff_date' from the exposed parameters schema."""
        tool = decompose_question
        schema_params = tool.parameters.get("properties", {}).keys()
        assert "cutoff_date" not in schema_params, (
            f"Backtest tool should have 'cutoff_date' excluded from schema. "
            f"Found in schema: {list(schema_params)}"
        )

    def test_backtest_tool_cutoff_date_has_default(self):
        """Backtest tools cutoff_date should have a default value."""
        sig = inspect.signature(decompose_question.fn)
        cutoff_param = sig.parameters.get("cutoff_date")
        assert cutoff_param is not None, "cutoff_date parameter not found"
        assert cutoff_param.default != inspect.Parameter.empty, (
            "cutoff_date parameter should have a default value"
        )


class TestAllToolsBacktestingConsistency:
    """Tests to ensure all tools in the server have consistent backtesting configuration."""

    def get_all_tools(self):
        """Get all registered tools from the MCP server."""
        return mcp._tool_manager._tools

    def test_all_backtest_tools_have_correct_configuration(self):
        """All tools with backtesting_supported tag must have cutoff_date properly configured."""
        tools = self.get_all_tools()

        for tool_name, tool in tools.items():
            tags = getattr(tool, "tags", set()) or set()

            if "backtesting_supported" in tags:
                # Check function has cutoff_date parameter
                sig = inspect.signature(tool.fn)
                params = list(sig.parameters.keys())
                assert "cutoff_date" in params, (
                    f"Tool '{tool_name}' has backtesting_supported tag but missing "
                    f"'cutoff_date' function parameter. Found params: {params}"
                )

                # Check cutoff_date is excluded from schema (hidden from LLM)
                schema_params = tool.parameters.get("properties", {}).keys()
                assert "cutoff_date" not in schema_params, (
                    f"Tool '{tool_name}' has backtesting_supported tag but "
                    f"'cutoff_date' is exposed in schema (should be excluded). "
                    f"Schema params: {list(schema_params)}"
                )