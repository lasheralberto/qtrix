from qtrix_engine.mcp_server import build_server


async def test_mcp_server_exposes_registry(ctx, wages):
    server = build_server()
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert {"ols_fit", "data_list_datasets", "panel_fit", "memory_search"} <= names
    result = await server.call_tool("data_list_datasets", {})
    assert "wages" in str(result)
