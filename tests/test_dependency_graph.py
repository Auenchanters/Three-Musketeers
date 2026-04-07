import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.dependency_graph import DependencyGraph

def test_get_dependencies():
    graph = DependencyGraph({"A": ["B", "C"], "B": ["C"]})
    assert set(graph.get_dependencies("A")) == {"B", "C"}
    assert graph.get_dependencies("C") == []

def test_get_dependents():
    graph = DependencyGraph({"A": ["B", "C"], "B": ["C"]})
    assert set(graph.get_dependents("C")) == {"A", "B"}
    assert graph.get_dependents("A") == []

def test_has_active_dependencies():
    graph = DependencyGraph({"A": ["B", "C"], "B": ["C"]})
    # C has dependents A and B. Both active.
    assert graph.has_active_dependencies("C", removed=set()) is True
    # B removed, A active.
    assert graph.has_active_dependencies("C", removed={"B"}) is True
    # Both A and B removed.
    assert graph.has_active_dependencies("C", removed={"A", "B"}) is False

def test_would_break_cluster_quorum():
    graph = DependencyGraph({})
    cluster_resources = [
        {"resource_id": "i-1", "tags": {"cluster": "c1", "quorum_min": "2"}},
        {"resource_id": "i-2", "tags": {"cluster": "c1", "quorum_min": "2"}},
        {"resource_id": "i-3", "tags": {"cluster": "c1", "quorum_min": "2"}},
    ]
    # removing i-1 leaves 2 alive, quorum is 2 -> False
    assert graph.would_break_cluster_quorum("i-1", cluster_resources, set()) is False
    # removing i-1 when i-3 is already removed leaves 1 alive -> True
    assert graph.would_break_cluster_quorum("i-1", cluster_resources, {"i-3"}) is True
    # non-existent cluster
    assert graph.would_break_cluster_quorum("i-4", cluster_resources, set()) is False

def test_is_replication_pair():
    graph = DependencyGraph({})
    r_primary = {"tags": {"replication_role": "primary", "replica_id": "rds-2"}}
    r_replica = {"tags": {"replication_role": "replica", "primary_id": "rds-1"}}
    r_none = {"tags": {}}
    assert graph.is_replication_pair(r_primary) == "rds-2"
    assert graph.is_replication_pair(r_replica) == "rds-1"
    assert graph.is_replication_pair(r_none) is None

def test_get_circular_peer():
    graph = DependencyGraph({})
    assert graph.get_circular_peer({"tags": {"peer": "i-2"}}) == "i-2"
    assert graph.get_circular_peer({"tags": {}}) is None

def test_get_all_resource_ids():
    graph = DependencyGraph({"A": ["B"], "B": ["C"]})
    assert graph.get_all_resource_ids() == {"A", "B", "C"}
