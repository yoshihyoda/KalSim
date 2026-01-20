"""Layer 6: Network Structure Module.

Models platform dynamics, virality mechanics, and information cascades.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PlatformPost:
    """A post on a social platform.
    
    Attributes:
        id: Unique post identifier.
        author_id: ID of the post author.
        content: Post text content.
        timestamp: When the post was created.
        subreddit: Subreddit/community name.
        upvotes: Number of upvotes.
        comments: Number of comments.
        awards: Number of awards received.
    """
    id: str
    author_id: int
    content: str
    timestamp: datetime
    subreddit: str = "wallstreetbets"
    upvotes: int = 0
    comments: int = 0
    awards: int = 0

    @property
    def engagement_score(self) -> float:
        """Calculate overall engagement score."""
        return self.upvotes + (self.comments * 2) + (self.awards * 10)


class RedditPlatform:
    """Models Reddit platform dynamics.
    
    Handles post creation, upvoting, virality detection, and trending.
    
    Attributes:
        posts: Dictionary of post ID to PlatformPost.
        viral_threshold: Upvote count for viral status.
    """

    def __init__(self, viral_threshold: int = 1000) -> None:
        """Initialize RedditPlatform.
        
        Args:
            viral_threshold: Number of upvotes for a post to go viral.
        """
        self.posts: dict[str, PlatformPost] = {}
        self.viral_threshold = viral_threshold
        self._upvote_tracking: dict[str, set[int]] = {}

    def create_post(
        self,
        author_id: int,
        content: str,
        subreddit: str = "wallstreetbets",
        timestamp: datetime | None = None,
    ) -> PlatformPost:
        """Create a new post.
        
        Args:
            author_id: ID of the post author.
            content: Post text content.
            subreddit: Target subreddit.
            timestamp: Post creation time.
            
        Returns:
            Created PlatformPost object.
        """
        post_id = f"reddit_{uuid.uuid4().hex[:8]}"
        post = PlatformPost(
            id=post_id,
            author_id=author_id,
            content=content,
            timestamp=timestamp or datetime.now(),
            subreddit=subreddit,
        )
        self.posts[post_id] = post
        self._upvote_tracking[post_id] = set()
        return post

    def upvote(self, post_id: str, voter_id: int) -> bool:
        """Upvote a post.
        
        Args:
            post_id: ID of the post to upvote.
            voter_id: ID of the voter.
            
        Returns:
            True if upvote was successful.
        """
        if post_id not in self.posts:
            return False

        if voter_id in self._upvote_tracking.get(post_id, set()):
            return False

        self._upvote_tracking.setdefault(post_id, set()).add(voter_id)
        self.posts[post_id].upvotes += 1
        return True

    def get_viral_posts(self) -> list[PlatformPost]:
        """Get all viral posts.
        
        Returns:
            List of posts exceeding viral threshold.
        """
        return [
            p for p in self.posts.values()
            if p.upvotes >= self.viral_threshold
        ]

    def get_viral_spread_factor(self, post_id: str) -> float:
        """Calculate viral spread factor for a post.
        
        Posts above threshold spread faster.
        
        Args:
            post_id: ID of the post.
            
        Returns:
            Spread factor (1.0 = normal, >1.0 = viral spread).
        """
        if post_id not in self.posts:
            return 1.0

        post = self.posts[post_id]
        if post.upvotes < self.viral_threshold:
            return 1.0

        excess_ratio = post.upvotes / self.viral_threshold
        return min(excess_ratio, 5.0)

    def get_trending_posts(self, limit: int = 10) -> list[PlatformPost]:
        """Get trending posts by engagement.
        
        Args:
            limit: Maximum posts to return.
            
        Returns:
            List of trending posts.
        """
        sorted_posts = sorted(
            self.posts.values(),
            key=lambda p: p.engagement_score,
            reverse=True,
        )
        return sorted_posts[:limit]

    def clear(self) -> None:
        """Clear all posts."""
        self.posts.clear()
        self._upvote_tracking.clear()


class NetworkStructureModule:
    """Models network structure effects on information flow.
    
    Handles virality detection, information cascades, and platform dynamics.
    
    Attributes:
        name: Module identifier.
        platform: RedditPlatform instance.
    """

    def __init__(self, viral_threshold: int = 1000) -> None:
        """Initialize the NetworkStructureModule.
        
        Args:
            viral_threshold: Upvote threshold for viral posts.
        """
        self.name = "network_structure"
        self.platform = RedditPlatform(viral_threshold=viral_threshold)
        self._last_viral_count = 0
        self._cascade_threshold = 3

    def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """Process network structure state.
        
        Analyzes viral content exposure and information cascades.
        
        Args:
            state: Combined state including social data.
            
        Returns:
            Dictionary with network structure outputs.
        """
        social = state.get("social", {})
        viral_posts = social.get("viral_posts", [])

        viral_exposure = len(viral_posts) > 0
        virality_intensity = 0.0

        if viral_exposure:
            total_upvotes = sum(p.get("upvotes", 0) for p in viral_posts)
            virality_intensity = min(total_upvotes / 10000, 1.0)

        information_cascade = (
            len(viral_posts) >= self._cascade_threshold
            or social.get("rapid_spread", False)
        )

        viral_sentiment = 0.0
        if viral_posts:
            sentiments = [p.get("sentiment", 0.0) for p in viral_posts]
            viral_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        self._last_viral_count = len(viral_posts)

        return {
            "viral_exposure": viral_exposure,
            "virality_intensity": virality_intensity,
            "information_cascade": information_cascade,
            "viral_post_count": len(viral_posts),
            "viral_sentiment": viral_sentiment,
            "network_activation": virality_intensity * 0.5 + (0.3 if information_cascade else 0.0),
        }

    def calculate_platform_concentration(self) -> float:
        """Calculate concentration of activity on platform.
        
        Returns:
            Concentration score (0.0 to 1.0).
        """
        if not self.platform.posts:
            return 0.0

        author_counts: dict[int, int] = {}
        for post in self.platform.posts.values():
            author_counts[post.author_id] = author_counts.get(post.author_id, 0) + 1

        total_posts = len(self.platform.posts)
        max_by_author = max(author_counts.values()) if author_counts else 0

        return max_by_author / total_posts if total_posts > 0 else 0.0

    def get_state_summary(self) -> str:
        """Get a summary of current network structure state.
        
        Returns:
            Human-readable state summary.
        """
        viral_count = len(self.platform.get_viral_posts())
        return (
            f"Viral posts: {viral_count}, "
            f"Total posts: {len(self.platform.posts)}, "
            f"Last viral count: {self._last_viral_count}"
        )

    def reset(self) -> None:
        """Reset module to initial state."""
        self.platform.clear()
        self._last_viral_count = 0
