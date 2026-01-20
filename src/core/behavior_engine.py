"""Behavior Engine module for orchestrating layer pipeline.

Coordinates the execution of all 7 layers for agent behavior modeling.
"""

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LayerModule(Protocol):
    """Protocol defining the interface for layer modules."""

    name: str

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process input state and return updated state."""
        ...

    def reset(self) -> None:
        """Reset layer state."""
        ...

    def get_state_summary(self) -> str:
        """Get a summary of current layer state."""
        ...


class LayerPipeline:
    """Manages the ordered execution of layer modules.
    
    Executes layers in sequence, passing state through each layer.
    
    Attributes:
        layers: List of registered layer modules.
    """

    def __init__(self) -> None:
        """Initialize the LayerPipeline."""
        self.layers: list[LayerModule] = []

    def add_layer(self, layer: LayerModule) -> None:
        """Add a layer to the pipeline.
        
        Args:
            layer: Layer module to add.
        """
        self.layers.append(layer)
        logger.debug(f"Added layer: {layer.name}")

    def execute(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        """Execute all layers in sequence.
        
        Args:
            initial_state: Starting state dictionary.
            
        Returns:
            Final state after all layers have processed.
        """
        current_state = initial_state.copy()

        for layer in self.layers:
            try:
                layer_output = layer.process(current_state)
                current_state.update(layer_output)
                logger.debug(f"Layer {layer.name} processed")
            except Exception as e:
                logger.error(f"Layer {layer.name} failed: {e}")
                raise

        return current_state


class BehaviorEngine:
    """Orchestrates the 7-layer model for agent behavior.
    
    Coordinates layer registration, execution, and prompt building
    for agent decision-making.
    
    Attributes:
        pipeline: LayerPipeline for executing layers.
    """

    def __init__(self) -> None:
        """Initialize the BehaviorEngine."""
        self.pipeline = LayerPipeline()
        self._layer_map: dict[str, LayerModule] = {}

    def register_layer(self, layer: LayerModule) -> None:
        """Register a layer module with the engine.
        
        Args:
            layer: Layer module to register.
        """
        self.pipeline.add_layer(layer)
        self._layer_map[layer.name] = layer

    def process(
        self,
        agent_state: dict[str, Any],
        market_state: dict[str, Any],
        social_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Process agent state through all layers.
        
        Args:
            agent_state: Current agent state.
            market_state: Current market conditions.
            social_state: Current social environment state.
            
        Returns:
            Updated state after layer processing.
        """
        combined_state = {
            "agent": agent_state,
            "market": market_state,
            "social": social_state,
        }
        return self.pipeline.execute(combined_state)

    def build_prompt_context(self, layer_outputs: dict[str, Any]) -> str:
        """Build prompt context string from layer outputs.
        
        Formats layer outputs into a readable context for LLM prompts.
        
        Args:
            layer_outputs: Dictionary of outputs from layer processing.
            
        Returns:
            Formatted context string.
        """
        context_parts = []

        if "fomo_level" in layer_outputs:
            fomo = layer_outputs["fomo_level"]
            if fomo > 0.7:
                context_parts.append(f"You are feeling strong FOMO (level: {fomo:.1f})")
            elif fomo > 0.4:
                context_parts.append(f"You feel moderate FOMO (level: {fomo:.1f})")

        if "emotion" in layer_outputs:
            context_parts.append(f"Current emotion: {layer_outputs['emotion']}")

        if "emotion_intensity" in layer_outputs:
            context_parts.append(f"Emotional intensity: {layer_outputs['emotion_intensity']:.1f}")

        if "cognitive_bias" in layer_outputs:
            bias = layer_outputs["cognitive_bias"]
            context_parts.append(f"You are influenced by {bias} bias")

        if "identity_group" in layer_outputs:
            group = layer_outputs["identity_group"]
            context_parts.append(f"You identify strongly with {group}")

        if "social_pressure" in layer_outputs:
            pressure = layer_outputs["social_pressure"]
            if pressure > 0.5:
                context_parts.append("You feel significant social pressure from the community")

        if "viral_exposure" in layer_outputs and layer_outputs["viral_exposure"]:
            context_parts.append("You've seen viral posts that are energizing the community")

        return "\n".join(context_parts) if context_parts else "No significant psychological factors."

    def get_layer_summary(self) -> dict[str, str]:
        """Get summaries from all registered layers.
        
        Returns:
            Dictionary mapping layer names to their state summaries.
        """
        summaries = {}
        for name, layer in self._layer_map.items():
            try:
                summaries[name] = layer.get_state_summary()
            except Exception:
                summaries[name] = "Unable to get summary"
        return summaries

    def reset(self) -> None:
        """Reset all registered layers to initial state."""
        for layer in self._layer_map.values():
            try:
                layer.reset()
            except Exception as e:
                logger.warning(f"Failed to reset layer {layer.name}: {e}")

    def get_layer(self, name: str) -> LayerModule | None:
        """Get a specific layer by name.
        
        Args:
            name: Name of the layer.
            
        Returns:
            Layer module or None if not found.
        """
        return self._layer_map.get(name)
