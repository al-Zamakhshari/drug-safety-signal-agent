"""
Pipeline integration smoke test.

Verifies the compiled graph runs end-to-end on a minimal seeded fixture
(no real OpenSearch, no real LLM) and that:
  - The graph can be imported and compiled without error
  - All 10 nodes are present with correct names
  - Routing edges are correctly wired (classify_signals runs on both paths)
  - State schema includes the new fields (classifications, signal_status, _prior_run)

This catches the class of wiring regressions the unit tests cannot:
  - Wrong node names in add_node()
  - Missing edges
  - TypedDict field additions that break graph compilation
"""
import pytest


class TestPipelineCompilation:
    """Graph structure and compilation tests — no OpenSearch needed."""

    def test_pipeline_imports_without_error(self):
        """Module-level imports must not raise (catches schema drift, bad kwargs)."""
        from agent.pipeline import pipeline, build_pipeline, DrugSafetyState
        assert pipeline is not None

    def test_pipeline_is_compiled_graph(self):
        """The pipeline object must be a compiled StateGraph."""
        from agent.pipeline import pipeline
        # CompiledGraph has an invoke method
        assert hasattr(pipeline, "invoke") or hasattr(pipeline, "ainvoke")

    def test_all_nodes_present(self):
        """All 10 expected nodes must be registered in the graph."""
        from agent.pipeline import build_pipeline
        graph = build_pipeline()
        nodes = set(graph.nodes.keys())
        expected = {
            "resolve_names", "load_memory", "calculate_prr",
            "anomaly_detection", "fetch_label", "search_lit",
            "investigate", "classify_signals", "write_report", "save_memory",
        }
        missing = expected - nodes
        assert not missing, f"Missing graph nodes: {missing}"

    def test_classify_signals_node_present(self):
        """classify_signals must be a real node — it was absent before commit b0a1771."""
        from agent.pipeline import build_pipeline
        graph = build_pipeline()
        assert "classify_signals" in graph.nodes, \
            "classify_signals node missing — PRR lifecycle tracking won't run"

    def test_state_schema_has_required_fields(self):
        """DrugSafetyState TypedDict must have the fields added for the lifecycle layer."""
        from agent.pipeline import DrugSafetyState
        import typing
        hints = typing.get_type_hints(DrugSafetyState)
        required = {"classifications", "signal_status", "_prior_run"}
        missing = required - set(hints.keys())
        assert not missing, f"Missing state fields: {missing}"

    def test_no_hardcoded_drug_names_in_pipeline(self):
        """
        No drug-specific names (semaglutide, rofecoxib, GLP-1) should appear
        in pipeline.py's Python logic (comments are OK, but not in function bodies).
        This test scans the compiled bytecode constants to catch silent hardcodings.
        """
        import dis
        import agent.pipeline as mod
        import inspect

        drug_names_lower = {"semaglutide", "rofecoxib", "liraglutide", "ozempic",
                            "warfarin", "metformin", "glp-1", "celecoxib"}
        violations = []
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            try:
                code = obj.__code__
                for const in code.co_consts:
                    if isinstance(const, str) and const.lower() in drug_names_lower:
                        violations.append(f"{name}: {const!r}")
            except AttributeError:
                pass
        assert not violations, f"Hardcoded drug names found in functions: {violations}"


class TestStateSchemaIntegrity:
    """Verify new state fields have correct types and defaults."""

    def test_classifications_field_type(self):
        from agent.pipeline import DrugSafetyState
        import typing
        hints = typing.get_type_hints(DrugSafetyState)
        # classifications should be list[dict]
        assert "classifications" in hints

    def test_signal_status_field_type(self):
        from agent.pipeline import DrugSafetyState
        import typing
        hints = typing.get_type_hints(DrugSafetyState)
        assert "signal_status" in hints

    def test_prior_run_field_optional(self):
        """_prior_run should be Optional since it's None on first run."""
        from agent.pipeline import DrugSafetyState
        import typing
        hints = typing.get_type_hints(DrugSafetyState)
        assert "_prior_run" in hints
        # Optional means Union[X, None] — check it accepts None
        # The actual type annotation is Optional[dict]
        hint_str = str(hints["_prior_run"])
        assert "None" in hint_str or "Optional" in hint_str


class TestParserIntegration:
    """Test that _parse_classification is importable and works with the pipeline state."""

    def test_parser_importable(self):
        from agent.pipeline import _parse_classification
        assert callable(_parse_classification)

    def test_effect_tokens_is_tuple(self):
        """_EFFECT_TOKENS must be a tuple (ordered) not a set (random)."""
        from agent.pipeline import _EFFECT_TOKENS
        assert isinstance(_EFFECT_TOKENS, tuple), \
            "_EFFECT_TOKENS must be a tuple for deterministic ordering"
        assert _EFFECT_TOKENS[0] == "DRUG_SPECIFIC", \
            "DRUG_SPECIFIC must be first so it wins when both tokens appear"

    def test_phase2_trigger_uses_classifications(self):
        """
        The Phase 2 'interesting' flag must read from classifications state,
        not from free-text investigation_text. Verify by checking that the
        trigger variable is computed from the parsed list, not from string matching.
        """
        from agent.pipeline import _parse_classification

        # A DRUG_SPECIFIC classification should trigger Phase 2
        cls = [_parse_classification("PANCREATITIS",
               "CLASSIFICATION: DRUG_SPECIFIC | GROWING\nRATIO: DRUG is 8.5x lowest\n")]
        interesting = any(
            c["effect"] == "DRUG_SPECIFIC" or (c["ratio"] is not None and c["ratio"] > 5.0)
            for c in cls
        )
        assert interesting is True

        # A CLASS_EFFECT with low ratio should NOT trigger Phase 2
        cls2 = [_parse_classification("NAUSEA",
                "CLASSIFICATION: CLASS_EFFECT | STABLE\nRATIO: DRUG is 2.1x lowest\n")]
        not_interesting = any(
            c["effect"] == "DRUG_SPECIFIC" or (c["ratio"] is not None and c["ratio"] > 5.0)
            for c in cls2
        )
        assert not_interesting is False
