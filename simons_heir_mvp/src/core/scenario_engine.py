"""Scenario Engine module for network and interaction management.

Manages social network topology and message/post propagation.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class Post:
    """A social media post.
    
    Attributes:
        id: Unique post identifier.
        author_id: ID of the post author.
        content: Post text content.
        timestamp: When the post was created.
        upvotes: Number of upvotes.
        comments: Number of comments.
        sentiment: Sentiment score (-1 to 1).
    """
    id: str
    author_id: int
    content: str
    timestamp: datetime
    upvotes: int = 0
    comments: int = 0
    sentiment: float = 0.0
    viral_threshold: int = 500

    @property
    def is_viral(self) -> bool:
        """Check if post has gone viral."""
        return self.upvotes >= self.viral_threshold


@dataclass
class Message:
    """A message propagating through the network.
    
    Attributes:
        sender_id: ID of the message sender.
        content: Message content.
        timestamp: When the message was sent.
        emotion: Emotional valence of the message.
        source_post_id: Optional ID of originating post.
    """
    sender_id: int
    content: str
    timestamp: datetime
    emotion: float = 0.0
    source_post_id: str | None = None


class SocialMediaInteraction:
    """Manages social media posts and interactions.
    
    Handles post creation, upvoting, and retrieval of viral content.
    
    Attributes:
        posts: Dictionary of post ID to Post objects.
        viral_threshold: Upvote count for viral status.
    """

    def __init__(self, viral_threshold: int = 500) -> None:
        """Initialize SocialMediaInteraction.
        
        Args:
            viral_threshold: Number of upvotes for a post to be viral.
        """
        self.posts: dict[str, Post] = {}
        self.viral_threshold = viral_threshold
        self._upvote_tracking: dict[str, set[int]] = {}

    def create_post(
        self,
        author_id: int,
        content: str,
        timestamp: datetime | None = None,
        sentiment: float = 0.0,
    ) -> Post:
        """Create a new post.
        
        Args:
            author_id: ID of the post author.
            content: Post text content.
            timestamp: Post creation time (defaults to now).
            sentiment: Sentiment score.
            
        Returns:
            Created Post object.
        """
        post_id = f"post_{uuid.uuid4().hex[:8]}"
        post = Post(
            id=post_id,
            author_id=author_id,
            content=content,
            timestamp=timestamp or datetime.now(),
            sentiment=sentiment,
            viral_threshold=self.viral_threshold,
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
            True if upvote was successful, False if already voted.
        """
        if post_id not in self.posts:
            return False

        if voter_id in self._upvote_tracking[post_id]:
            return False

        self._upvote_tracking[post_id].add(voter_id)
        self.posts[post_id].upvotes += 1
        return True

    def get_post(self, post_id: str) -> Post | None:
        """Get a post by ID.
        
        Args:
            post_id: ID of the post.
            
        Returns:
            Post object or None if not found.
        """
        return self.posts.get(post_id)

    def get_viral_posts(self) -> list[Post]:
        """Get all viral posts.
        
        Returns:
            List of posts that have gone viral.
        """
        return [p for p in self.posts.values() if p.is_viral]

    def get_recent_posts(self, limit: int = 10) -> list[Post]:
        """Get most recent posts.
        
        Args:
            limit: Maximum number of posts to return.
            
        Returns:
            List of recent posts, newest first.
        """
        sorted_posts = sorted(
            self.posts.values(),
            key=lambda p: p.timestamp,
            reverse=True,
        )
        return sorted_posts[:limit]

    def get_posts_by_author(self, author_id: int) -> list[Post]:
        """Get all posts by an author.
        
        Args:
            author_id: ID of the author.
            
        Returns:
            List of posts by the author.
        """
        return [p for p in self.posts.values() if p.author_id == author_id]


class ScenarioEngine:
    """Manages scenario network topology and message propagation.
    
    Uses networkx to model social connections and propagate
    information through the network.
    
    Attributes:
        network: NetworkX graph representing social connections.
        interaction: SocialMediaInteraction instance for posts.
    """

    def __init__(self, viral_threshold: int = 500) -> None:
        """Initialize ScenarioEngine.
        
        Args:
            viral_threshold: Upvote threshold for viral posts.
        """
        self.network: nx.Graph = nx.Graph()
        self.interaction = SocialMediaInteraction(viral_threshold=viral_threshold)

    def build_network(
        self,
        edges: list[tuple[int, int]] | None = None,
        node_count: int | None = None,
        network_type: str = "custom",
    ) -> None:
        """Build the social network.
        
        Args:
            edges: List of (node1, node2) tuples for connections.
            node_count: Number of nodes for generated networks.
            network_type: Type of network ('custom', 'barabasi_albert', 'watts_strogatz').
        """
        if edges is not None:
            self.network = nx.Graph()
            self.network.add_edges_from(edges)
        elif node_count is not None:
            if network_type == "barabasi_albert":
                self.network = nx.barabasi_albert_graph(node_count, 3)
            elif network_type == "watts_strogatz":
                self.network = nx.watts_strogatz_graph(node_count, 4, 0.3)
            else:
                self.network = nx.complete_graph(node_count)

        logger.info(
            f"Built network: {self.network.number_of_nodes()} nodes, "
            f"{self.network.number_of_edges()} edges"
        )

    def add_node(self, node_id: int, attributes: dict[str, Any] | None = None) -> None:
        """Add a node to the network.
        
        Args:
            node_id: ID of the node to add.
            attributes: Optional node attributes.
        """
        self.network.add_node(node_id, **(attributes or {}))

    def add_edge(
        self,
        node1: int,
        node2: int,
        weight: float = 1.0,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an edge between two nodes.
        
        Args:
            node1: First node ID.
            node2: Second node ID.
            weight: Edge weight (connection strength).
            attributes: Optional edge attributes.
        """
        attrs = {"weight": weight, **(attributes or {})}
        self.network.add_edge(node1, node2, **attrs)

    def get_neighbors(self, node_id: int) -> list[int]:
        """Get neighbors of a node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            List of neighbor node IDs.
        """
        if node_id not in self.network:
            return []
        return list(self.network.neighbors(node_id))

    def propagate_message(
        self,
        message: Message,
        hops: int = 1,
        decay: float = 0.5,
    ) -> set[int]:
        """Propagate a message through the network.
        
        Args:
            message: Message to propagate.
            hops: Number of hops to propagate.
            decay: Decay factor per hop for influence.
            
        Returns:
            Set of node IDs that received the message.
        """
        recipients: set[int] = set()
        current_nodes = {message.sender_id}

        for hop in range(hops):
            next_nodes: set[int] = set()
            for node in current_nodes:
                neighbors = self.get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor != message.sender_id:
                        recipients.add(neighbor)
                        next_nodes.add(neighbor)
            current_nodes = next_nodes

        return recipients

    def get_network_metrics(self) -> dict[str, float]:
        """Get network-level metrics.
        
        Returns:
            Dictionary of network metrics.
        """
        if self.network.number_of_nodes() == 0:
            return {"density": 0.0, "avg_clustering": 0.0, "avg_degree": 0.0}

        return {
            "density": nx.density(self.network),
            "avg_clustering": nx.average_clustering(self.network),
            "avg_degree": sum(dict(self.network.degree()).values()) / self.network.number_of_nodes(),
        }

    def get_node_centrality(self, node_id: int) -> float:
        """Get the centrality score for a node.
        
        Args:
            node_id: ID of the node.
            
        Returns:
            Degree centrality score.
        """
        if node_id not in self.network:
            return 0.0
        centrality = nx.degree_centrality(self.network)
        return centrality.get(node_id, 0.0)
