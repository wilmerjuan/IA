#!/usr/bin/env python3
"""Uniform-Cost Search for the HW01 New York map.

Run:
    python hw01/ucs_search.py
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple


Graph = Dict[str, Dict[str, int]]


@dataclass(frozen=True)
class SearchResult:
    path: List[str]
    total_cost: int
    nodes_generated: int


def build_graph() -> Graph:
    """Return the bidirectional graph using distances from the map."""
    edges: List[Tuple[str, str, int]] = [
        ("Elmira", "Ithaca", 60),
        ("Elmira", "Williamsport", 80),
        ("Ithaca", "Binghamton", 80),
        ("Binghamton", "Syracuse", 110),
        ("Syracuse", "Albany", 200),
        ("Binghamton", "Albany", 220),
        ("Binghamton", "Scranton", 95),
        ("Scranton", "Wilkes-Barre", 30),
        ("Wilkes-Barre", "Stroudsburg", 105),
        ("Stroudsburg", "Albany", 190),
        ("Stroudsburg", "Paterson", 90),
        ("Paterson", "New York City", 35),
        ("Newark", "New York City", 25),
        ("Trenton", "New York City", 95),
        ("Allentown", "Newark", 130),
        ("Allentown", "Trenton", 80),
        ("Allentown", "Wilkes-Barre", 95),
        ("Allentown", "Scranton", 120),
        ("Allentown", "Harrisburg", 90),
        ("Harrisburg", "Lancaster", 60),
        ("Lancaster", "Philadelphia", 160),
        ("Harrisburg", "Philadelphia", 110),
        ("Harrisburg", "Williamsport", 135),
        ("Harrisburg", "Scranton", 175),
        ("Williamsport", "Scranton", 140),
        ("Philadelphia", "Trenton", 50),
    ]

    graph: Graph = {}
    for city1, city2, distance in edges:
        graph.setdefault(city1, {})[city2] = distance
        graph.setdefault(city2, {})[city1] = distance
    return graph


def uniform_cost_search(graph: Graph, start: str, goal: str) -> SearchResult:
    """Find the least-cost path using uniform-cost search.

    This implements the AIMA graph-search pattern in the provided slides:
    - "Best-First Search" with f(n) = Path-Cost (g(n)) is Uniform-Cost Search.
    - For weighted edges (km on the map), UCS is the right choice because it is
      optimal and complete when all step costs are non-negative.
    - Breadth-First Search (BFS) only guarantees the fewest *edges*, not the
      lowest *total km*, so it is not appropriate for HW01.

    Nodes generated are counted when a neighbor is pushed to the frontier.
    The start node is not counted.
    """
    # Priority queue ordered by total path-cost g(n).
    frontier: List[Tuple[int, int, str, List[str]]] = []
    tie = 0
    heapq.heappush(frontier, (0, tie, start, [start]))

    # "reached" / best-cost table to discard dominated paths.
    best_cost = {start: 0}
    nodes_generated = 0

    while frontier:
        cost, _, current, path = heapq.heappop(frontier)
        if current == goal:
            return SearchResult(path=path, total_cost=cost, nodes_generated=nodes_generated)

        for neighbor, step_cost in graph[current].items():
            new_cost = cost + step_cost
            if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                best_cost[neighbor] = new_cost
                nodes_generated += 1
                tie += 1
                heapq.heappush(frontier, (new_cost, tie, neighbor, path + [neighbor]))

    raise ValueError(f"No path found from {start} to {goal}.")


def write_results(result: SearchResult, output_path: str) -> None:
    """Write the required outputs to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Mejor ruta (UCS):\n")
        handle.write(" -> ".join(result.path) + "\n")
        handle.write(f"Costo total: {result.total_cost} km\n")
        handle.write(
            "Nodos generados (sin contar el nodo inicial): "
            f"{result.nodes_generated}\n"
        )


def main() -> None:
    graph = build_graph()
    result = uniform_cost_search(graph, "Elmira", "New York City")
    write_results(result, "hw01/ucs_output.txt")


if __name__ == "__main__":
    main()
