"""Tests for ScenarioEngine module."""

import pytest
from datetime import datetime

from src.core.scenario_engine import (
    ScenarioEngine,
    SocialMediaInteraction,
    Post,
    Message,
)


class TestPost:
    """Tests for Post dataclass."""

    def test_post_creation(self):
        """Test creating a Post."""
        post = Post(
            id="post_001",
            author_id=0,
            content="$GME to the moon!",
            timestamp=datetime(2021, 1, 25, 10, 0),
        )
        assert post.id == "post_001"
        assert post.upvotes == 0
        assert post.is_viral is False

    def test_post_viral_threshold(self):
        """Test that post becomes viral at threshold."""
        post = Post(
            id="post_001",
            author_id=0,
            content="Diamond hands!",
            timestamp=datetime(2021, 1, 25, 10, 0),
            upvotes=1000,
        )
        assert post.is_viral is True


class TestSocialMediaInteraction:
    """Tests for SocialMediaInteraction class."""

    def test_interaction_initialization(self):
        """Test initializing SocialMediaInteraction."""
        interaction = SocialMediaInteraction()
        assert interaction is not None
        assert len(interaction.posts) == 0

    def test_create_post(self):
        """Test creating a post."""
        interaction = SocialMediaInteraction()
        post = interaction.create_post(
            author_id=0,
            content="Testing GME post",
        )
        assert post.author_id == 0
        assert post.id is not None
        assert len(interaction.posts) == 1

    def test_upvote_post(self):
        """Test upvoting a post."""
        interaction = SocialMediaInteraction()
        post = interaction.create_post(author_id=0, content="Test")

        interaction.upvote(post.id, voter_id=1)

        updated_post = interaction.get_post(post.id)
        assert updated_post.upvotes == 1

    def test_get_viral_posts(self):
        """Test getting viral posts."""
        interaction = SocialMediaInteraction(viral_threshold=100)
        
        post1 = interaction.create_post(author_id=0, content="Normal post")
        post2 = interaction.create_post(author_id=1, content="Viral post")
        
        for i in range(150):
            interaction.upvote(post2.id, voter_id=i + 10)

        viral = interaction.get_viral_posts()
        assert len(viral) == 1
        assert viral[0].id == post2.id

    def test_get_recent_posts(self):
        """Test getting recent posts."""
        interaction = SocialMediaInteraction()
        
        for i in range(10):
            interaction.create_post(author_id=i, content=f"Post {i}")

        recent = interaction.get_recent_posts(limit=5)
        assert len(recent) == 5


class TestScenarioEngine:
    """Tests for ScenarioEngine class."""

    def test_engine_initialization(self):
        """Test initializing ScenarioEngine."""
        engine = ScenarioEngine()
        assert engine is not None
        assert engine.network is not None

    def test_build_network_from_edges(self, sample_network_edges):
        """Test building network from edge list."""
        engine = ScenarioEngine()
        engine.build_network(edges=sample_network_edges)

        assert engine.network.number_of_nodes() == 5
        assert engine.network.number_of_edges() == 6

    def test_get_neighbors(self, sample_network_edges):
        """Test getting neighbors of a node."""
        engine = ScenarioEngine()
        engine.build_network(edges=sample_network_edges)

        neighbors = engine.get_neighbors(0)
        assert 1 in neighbors
        assert 2 in neighbors

    def test_propagate_message(self, sample_network_edges):
        """Test message propagation through network."""
        engine = ScenarioEngine()
        engine.build_network(edges=sample_network_edges)

        message = Message(
            sender_id=0,
            content="Buy GME!",
            timestamp=datetime.now(),
        )

        recipients = engine.propagate_message(message, hops=1)
        assert 1 in recipients
        assert 2 in recipients

    def test_propagate_message_multi_hop(self, sample_network_edges):
        """Test multi-hop message propagation."""
        engine = ScenarioEngine()
        engine.build_network(edges=sample_network_edges)

        message = Message(
            sender_id=0,
            content="Diamond hands!",
            timestamp=datetime.now(),
        )

        recipients = engine.propagate_message(message, hops=2)
        assert 3 in recipients

    def test_get_network_metrics(self, sample_network_edges):
        """Test getting network metrics."""
        engine = ScenarioEngine()
        engine.build_network(edges=sample_network_edges)

        metrics = engine.get_network_metrics()
        assert "density" in metrics
        assert "avg_clustering" in metrics

    def test_add_node_to_network(self):
        """Test adding a node to the network."""
        engine = ScenarioEngine()
        engine.add_node(0, {"name": "TestUser"})

        assert 0 in engine.network.nodes()

    def test_add_edge_to_network(self):
        """Test adding an edge to the network."""
        engine = ScenarioEngine()
        engine.add_node(0)
        engine.add_node(1)
        engine.add_edge(0, 1, weight=0.8)

        assert engine.network.has_edge(0, 1)
        assert engine.network[0][1]["weight"] == 0.8
