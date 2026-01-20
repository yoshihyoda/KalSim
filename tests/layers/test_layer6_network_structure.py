"""Tests for Layer 6: Network Structure module."""

import pytest
from datetime import datetime

from src.layers.layer6_network_structure import (
    NetworkStructureModule,
    RedditPlatform,
    PlatformPost,
)


class TestPlatformPost:
    """Tests for PlatformPost dataclass."""

    def test_post_creation(self):
        """Test creating a PlatformPost."""
        post = PlatformPost(
            id="post_001",
            author_id=0,
            content="Diamond hands!",
            timestamp=datetime.now(),
            subreddit="wallstreetbets",
        )
        assert post.id == "post_001"
        assert post.upvotes == 0

    def test_post_engagement_score(self):
        """Test calculating engagement score."""
        post = PlatformPost(
            id="post_001",
            author_id=0,
            content="Test",
            timestamp=datetime.now(),
            upvotes=1000,
            comments=200,
            awards=5,
        )
        score = post.engagement_score
        assert score > 0


class TestRedditPlatform:
    """Tests for RedditPlatform class."""

    def test_platform_initialization(self):
        """Test initializing RedditPlatform."""
        platform = RedditPlatform()
        assert platform is not None
        assert platform.viral_threshold > 0

    def test_create_post(self):
        """Test creating a post."""
        platform = RedditPlatform()
        post = platform.create_post(
            author_id=0,
            content="GME to the moon!",
            subreddit="wallstreetbets",
        )
        assert post.subreddit == "wallstreetbets"
        assert len(platform.posts) == 1

    def test_upvote_triggers_virality(self):
        """Test that upvotes trigger virality at threshold."""
        platform = RedditPlatform(viral_threshold=100)
        post = platform.create_post(
            author_id=0,
            content="YOLO GME",
            subreddit="wallstreetbets",
        )

        for i in range(150):
            platform.upvote(post.id, voter_id=i + 1)

        viral_posts = platform.get_viral_posts()
        assert len(viral_posts) == 1
        assert viral_posts[0].id == post.id

    def test_viral_spread_factor(self):
        """Test viral spread factor calculation."""
        platform = RedditPlatform(viral_threshold=100)
        post = platform.create_post(
            author_id=0,
            content="Hold!",
            subreddit="wallstreetbets",
        )

        for i in range(200):
            platform.upvote(post.id, voter_id=i + 1)

        spread_factor = platform.get_viral_spread_factor(post.id)
        assert spread_factor > 1.0

    def test_get_trending_posts(self):
        """Test getting trending posts."""
        platform = RedditPlatform()

        for i in range(5):
            post = platform.create_post(
                author_id=i,
                content=f"Post {i}",
                subreddit="wallstreetbets",
            )
            for j in range(i * 100):
                platform.upvote(post.id, voter_id=j + 100)

        trending = platform.get_trending_posts(limit=3)
        assert len(trending) == 3
        assert trending[0].upvotes >= trending[1].upvotes


class TestNetworkStructureModule:
    """Tests for NetworkStructureModule class."""

    def test_module_initialization(self):
        """Test initializing NetworkStructureModule."""
        module = NetworkStructureModule()
        assert module.name == "network_structure"

    def test_process_viral_exposure(self):
        """Test processing viral content exposure."""
        module = NetworkStructureModule()

        state = {
            "social": {
                "viral_posts": [
                    {"id": "p1", "upvotes": 5000, "content": "GME!"},
                ],
            },
            "agent": {"id": 0},
        }

        result = module.process(state)

        assert result["viral_exposure"] is True
        assert result["virality_intensity"] > 0

    def test_process_no_viral_content(self):
        """Test processing with no viral content."""
        module = NetworkStructureModule()

        state = {
            "social": {"viral_posts": []},
            "agent": {"id": 0},
        }

        result = module.process(state)

        assert result["viral_exposure"] is False

    def test_information_cascade_detection(self):
        """Test detection of information cascade."""
        module = NetworkStructureModule()

        state = {
            "social": {
                "viral_posts": [
                    {"id": f"p{i}", "upvotes": 2000} for i in range(5)
                ],
                "rapid_spread": True,
            },
            "agent": {"id": 0},
        }

        result = module.process(state)

        assert result.get("information_cascade", False) is True

    def test_platform_concentration(self):
        """Test calculating platform concentration."""
        module = NetworkStructureModule()
        module.platform = RedditPlatform()

        for i in range(10):
            post = module.platform.create_post(
                author_id=i % 3,
                content=f"Post {i}",
                subreddit="wallstreetbets",
            )

        concentration = module.calculate_platform_concentration()
        assert concentration > 0

    def test_get_state_summary(self):
        """Test getting state summary."""
        module = NetworkStructureModule()
        module._last_viral_count = 5

        summary = module.get_state_summary()

        assert "viral" in summary.lower() or "Viral" in summary

    def test_reset(self):
        """Test resetting module state."""
        module = NetworkStructureModule()
        module._last_viral_count = 10
        module.platform.create_post(0, "Test", "wsb")

        module.reset()

        assert module._last_viral_count == 0
        assert len(module.platform.posts) == 0
