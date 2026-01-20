"""Tests for BehaviorEngine module."""

import pytest
from unittest.mock import MagicMock, patch

from src.core.behavior_engine import BehaviorEngine, LayerPipeline


class TestLayerPipeline:
    """Tests for LayerPipeline class."""

    def test_pipeline_initialization(self):
        """Test initializing LayerPipeline."""
        pipeline = LayerPipeline()
        assert pipeline is not None
        assert len(pipeline.layers) == 0

    def test_add_layer(self):
        """Test adding a layer to the pipeline."""
        pipeline = LayerPipeline()
        mock_layer = MagicMock()
        mock_layer.name = "test_layer"

        pipeline.add_layer(mock_layer)

        assert len(pipeline.layers) == 1

    def test_execute_pipeline(self):
        """Test executing the layer pipeline."""
        pipeline = LayerPipeline()

        mock_layer1 = MagicMock()
        mock_layer1.name = "layer1"
        mock_layer1.process.return_value = {"output1": "value1"}

        mock_layer2 = MagicMock()
        mock_layer2.name = "layer2"
        mock_layer2.process.return_value = {"output2": "value2"}

        pipeline.add_layer(mock_layer1)
        pipeline.add_layer(mock_layer2)

        initial_state = {"input": "test"}
        result = pipeline.execute(initial_state)

        mock_layer1.process.assert_called_once()
        mock_layer2.process.assert_called_once()
        assert "output1" in result
        assert "output2" in result

    def test_pipeline_execution_order(self):
        """Test that layers execute in order."""
        pipeline = LayerPipeline()
        execution_order = []

        def make_processor(name):
            def processor(state):
                execution_order.append(name)
                return {name: True}
            return processor

        for name in ["L7", "L6", "L5", "L4", "L3", "L2", "L1"]:
            mock_layer = MagicMock()
            mock_layer.name = name
            mock_layer.process.side_effect = make_processor(name)
            pipeline.add_layer(mock_layer)

        pipeline.execute({})

        assert execution_order == ["L7", "L6", "L5", "L4", "L3", "L2", "L1"]


class TestBehaviorEngine:
    """Tests for BehaviorEngine class."""

    def test_engine_initialization(self):
        """Test initializing BehaviorEngine."""
        engine = BehaviorEngine()
        assert engine is not None
        assert engine.pipeline is not None

    def test_register_layer(self):
        """Test registering a layer module."""
        engine = BehaviorEngine()
        mock_layer = MagicMock()
        mock_layer.name = "test_layer"

        engine.register_layer(mock_layer)

        assert len(engine.pipeline.layers) == 1

    def test_process_agent_state(self):
        """Test processing an agent's state through layers."""
        engine = BehaviorEngine()

        mock_layer = MagicMock()
        mock_layer.name = "test"
        mock_layer.process.return_value = {"processed": True}
        engine.register_layer(mock_layer)

        agent_state = {"id": 0, "emotion": 0.5}
        market_state = {"price": 100.0}
        social_state = {"sentiment": 0.7}

        result = engine.process(agent_state, market_state, social_state)

        assert "processed" in result

    def test_build_prompt_context(self):
        """Test building prompt context from layer outputs."""
        engine = BehaviorEngine()

        layer_outputs = {
            "fomo_level": 0.8,
            "emotion": "excited",
            "cognitive_bias": "social_proof",
            "identity_group": "WSB",
        }

        context = engine.build_prompt_context(layer_outputs)

        assert "fomo_level" in context or "FOMO" in context
        assert isinstance(context, str)

    def test_get_layer_summary(self):
        """Test getting a summary of layer states."""
        engine = BehaviorEngine()

        mock_layer = MagicMock()
        mock_layer.name = "neurobiology"
        mock_layer.get_state_summary.return_value = "FOMO: High"
        engine.register_layer(mock_layer)

        summary = engine.get_layer_summary()

        assert "neurobiology" in summary

    def test_reset_layers(self):
        """Test resetting all layers."""
        engine = BehaviorEngine()

        mock_layer = MagicMock()
        mock_layer.name = "test"
        engine.register_layer(mock_layer)

        engine.reset()

        mock_layer.reset.assert_called_once()
