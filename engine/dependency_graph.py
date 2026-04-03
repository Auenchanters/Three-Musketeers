"""
CloudFinOpsEnv — Dependency Graph Manager

Manages resource dependency relationships for safe action validation.
Handles cluster quorum checks, replication pairs, and cascading effects.
"""

from typing import Dict, List, Set, Optional, Tuple


class DependencyGraph:
    """
    Manages the resource dependency adjacency list.
    
    The graph stores: resource_id → [resource_ids that this resource depends on].
    Used to determine whether deleting/modifying a resource is safe.
    """

    def __init__(self, adjacency: Dict[str, List[str]]):
        """
        Args:
            adjacency: Dict mapping resource_id → list of resource_ids it depends on.
        """
        self._adjacency = {k: list(v) for k, v in adjacency.items()}

    def get_dependencies(self, resource_id: str) -> List[str]:
        """Get resources that this resource depends on."""
        return list(self._adjacency.get(resource_id, []))

    def get_dependents(self, resource_id: str) -> List[str]:
        """Get resources that depend on this resource (reverse lookup)."""
        dependents = []
        for rid, deps in self._adjacency.items():
            if resource_id in deps:
                dependents.append(rid)
        return dependents

    def has_active_dependencies(self, resource_id: str, removed: Set[str]) -> bool:
        """
        Check if any resources that depend on this one are still active.
        
        Args:
            resource_id: The resource to check.
            removed: Set of already-removed resource IDs.
            
        Returns:
            True if active dependents exist (unsafe to delete).
        """
        dependents = self.get_dependents(resource_id)
        return any(d not in removed for d in dependents)

    def would_break_cluster_quorum(
        self,
        resource_id: str,
        cluster_resources: List[dict],
        removed: Set[str],
    ) -> bool:
        """
        Check if removing a resource would break cluster quorum.
        
        A cluster with quorum_min=N needs at least N alive members.
        
        Args:
            resource_id: Resource being considered for removal.
            cluster_resources: List of resource dicts in the same cluster.
            removed: Set of already-removed resource IDs.
            
        Returns:
            True if removal would break quorum.
        """
        if not cluster_resources:
            return False

        # Find the cluster this resource belongs to
        target_cluster = None
        for r in cluster_resources:
            if r["resource_id"] == resource_id:
                target_cluster = r.get("tags", {}).get("cluster")
                break

        if not target_cluster:
            return False

        # Count alive members and find quorum requirement
        quorum_min = 2  # default
        alive_count = 0
        for r in cluster_resources:
            tags = r.get("tags", {})
            if tags.get("cluster") != target_cluster:
                continue
            if "quorum_min" in tags:
                quorum_min = int(tags["quorum_min"])
            rid = r["resource_id"]
            if rid not in removed and rid != resource_id:
                alive_count += 1

        return alive_count < quorum_min

    def is_replication_pair(self, resource: dict) -> Optional[str]:
        """
        Check if a resource is part of a primary/replica pair.
        
        Returns the partner resource_id if it is, None otherwise.
        """
        tags = resource.get("tags", {})
        rep_role = tags.get("replication_role")
        if rep_role == "primary":
            return tags.get("replica_id")
        elif rep_role == "replica":
            return tags.get("primary_id")
        return None

    def get_circular_peer(self, resource: dict) -> Optional[str]:
        """
        Check if a resource has a circular dependency peer.
        
        Returns the peer resource_id if it does, None otherwise.
        """
        return resource.get("tags", {}).get("peer")

    def get_all_resource_ids(self) -> Set[str]:
        """Get all resource IDs that appear in the graph."""
        ids = set(self._adjacency.keys())
        for deps in self._adjacency.values():
            ids.update(deps)
        return ids
