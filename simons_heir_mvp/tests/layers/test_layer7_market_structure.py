"""Tests for Layer 7: Market Structure module."""

import pytest
from datetime import datetime

from src.layers.layer7_market_structure import MarketStructureModule


class TestMarketStructureModule:
    """Tests for MarketStructureModule class."""

    def test_module_initialization(self):
        """Test initializing MarketStructureModule."""
        module = MarketStructureModule()
        assert module.name == "market_structure"
        assert module.liquidity == 1.0

    def test_process_buy_orders_impact(self):
        """Test that buy orders impact market state."""
        module = MarketStructureModule()

        state = {
            "market": {"price": 100.0, "volume": 1000000},
            "agent_actions": [
                {"action": "BUY", "agent_id": 0},
                {"action": "BUY", "agent_id": 1},
                {"action": "BUY", "agent_id": 2},
            ],
        }

        result = module.process(state)

        assert result["price_pressure"] > 0
        assert result["market_impact"] > 0

    def test_process_sell_orders_impact(self):
        """Test that sell orders impact market state."""
        module = MarketStructureModule()

        state = {
            "market": {"price": 100.0, "volume": 1000000},
            "agent_actions": [
                {"action": "SELL", "agent_id": 0},
                {"action": "SELL", "agent_id": 1},
            ],
        }

        result = module.process(state)

        assert result["price_pressure"] < 0

    def test_liquidity_decreases_with_volume(self):
        """Test that high volume decreases liquidity."""
        module = MarketStructureModule()

        state = {
            "market": {"price": 100.0, "volume": 100000000},
            "agent_actions": [{"action": "BUY", "agent_id": i} for i in range(50)],
        }

        result = module.process(state)

        assert result["liquidity"] < 1.0

    def test_short_squeeze_detection(self):
        """Test detection of short squeeze conditions."""
        module = MarketStructureModule()

        state = {
            "market": {
                "price": 100.0,
                "volume": 50000000,
                "short_interest": 140.0,
            },
            "agent_actions": [{"action": "BUY", "agent_id": i} for i in range(30)],
        }

        result = module.process(state)

        assert result.get("short_squeeze_pressure", 0) > 0

    def test_volatility_calculation(self):
        """Test volatility calculation from price history."""
        module = MarketStructureModule()
        module.price_history = [20.0, 25.0, 30.0, 28.0, 35.0, 40.0]

        volatility = module.calculate_volatility()

        assert volatility > 0

    def test_order_flow_imbalance(self):
        """Test order flow imbalance calculation."""
        module = MarketStructureModule()

        state = {
            "market": {"price": 100.0},
            "agent_actions": [
                {"action": "BUY", "agent_id": 0},
                {"action": "BUY", "agent_id": 1},
                {"action": "BUY", "agent_id": 2},
                {"action": "SELL", "agent_id": 3},
            ],
        }

        result = module.process(state)

        assert result["order_flow_imbalance"] > 0

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = MarketStructureModule()
        module.liquidity = 0.5
        module.price_history = [20.0, 30.0, 40.0]

        summary = module.get_state_summary()

        assert "liquidity" in summary.lower() or "Liquidity" in summary

    def test_reset(self):
        """Test resetting module state."""
        module = MarketStructureModule()
        module.liquidity = 0.3
        module.price_history = [100.0, 200.0]

        module.reset()

        assert module.liquidity == 1.0
        assert len(module.price_history) == 0
