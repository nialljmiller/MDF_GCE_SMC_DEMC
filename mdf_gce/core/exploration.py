"""
Parameter space exploration utilities.

This module provides Voronoi-based methods for identifying and targeting
sparse regions in parameter space to improve sampling coverage.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import Voronoi
    HAS_SCIPY_VORONOI = True
except ImportError:
    HAS_SCIPY_VORONOI = False


# =============================================================================
# VORONOI UTILITIES
# =============================================================================

def _voronoi_finite_polygons_2d(vor, radius: float = 10.0):
    """
    Make infinite Voronoi regions finite (2D only).
    
    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram
    radius : float
        Distance to extend infinite ridges
        
    Returns
    -------
    new_regions : list
        List of finite polygon vertex indices
    new_vertices : np.ndarray
        Extended vertex array
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Only 2D supported for finite polygon reconstruction.")
    
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}
    
    for (p, q), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p, []).append((q, v1, v2))
        all_ridges.setdefault(q, []).append((p, v1, v2))
    
    for p, region_idx in enumerate(vor.point_region):
        verts = vor.regions[region_idx]
        if len(verts) == 0:
            continue
        if all(v >= 0 for v in verts):
            new_regions.append(verts)
            continue
        
        # Need to close region by extending edges to a "far" point
        ridges = all_ridges.get(p, [])
        new_region = [v for v in verts if v >= 0]
        
        for q, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[q] - vor.points[p]
            if np.allclose(t, 0):
                continue
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # outward normal
            midpoint = (vor.points[p] + vor.points[q]) * 0.5
            direction = np.sign(np.dot(midpoint - center, n)) * n
            # Pick whichever endpoint exists
            base = vor.vertices[v1 if v1 >= 0 else v2]
            far = base + direction * radius
            new_vertices.append(far.tolist())
            new_region.append(len(new_vertices) - 1)
        
        # Order vertices counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)
    
    return new_regions, np.asarray(new_vertices)


def _clip_poly_to_unit_square(poly: np.ndarray) -> np.ndarray:
    """
    Sutherland-Hodgman clip of polygon to [0,1]x[0,1].
    
    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices, shape (N, 2)
        
    Returns
    -------
    np.ndarray
        Clipped polygon vertices
    """
    def inside(p, edge):
        x, y = p
        if edge == 0:   # left x=0
            return x >= 0
        elif edge == 1: # right x=1
            return x <= 1
        elif edge == 2: # bottom y=0
            return y >= 0
        else:           # top y=1
            return y <= 1
    
    def intersect(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        if edge == 0:   # x=0
            t = -x1 / (x2 - x1 + 1e-12)
            return np.array([0, y1 + t * (y2 - y1)])
        elif edge == 1: # x=1
            t = (1 - x1) / (x2 - x1 + 1e-12)
            return np.array([1, y1 + t * (y2 - y1)])
        elif edge == 2: # y=0
            t = -y1 / (y2 - y1 + 1e-12)
            return np.array([x1 + t * (x2 - x1), 0])
        else:           # y=1
            t = (1 - y1) / (y2 - y1 + 1e-12)
            return np.array([x1 + t * (x2 - x1), 1])
    
    output = list(poly)
    for edge in range(4):
        if len(output) == 0:
            break
        input_list = output
        output = []
        for i in range(len(input_list)):
            current = input_list[i]
            previous = input_list[i - 1]
            if inside(current, edge):
                if not inside(previous, edge):
                    output.append(intersect(previous, current, edge))
                output.append(current)
            elif inside(previous, edge):
                output.append(intersect(previous, current, edge))
    
    return np.array(output) if output else np.empty((0, 2))


def _poly_area_and_centroid(poly: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute area and centroid of a polygon.
    
    Parameters
    ----------
    poly : np.ndarray
        Polygon vertices, shape (N, 2)
        
    Returns
    -------
    area : float
        Polygon area
    centroid : np.ndarray
        Centroid coordinates
    """
    if len(poly) < 3:
        return 0.0, np.array([0.5, 0.5])
    
    n = len(poly)
    area = 0.0
    cx, cy = 0.0, 0.0
    
    for i in range(n):
        j = (i + 1) % n
        cross = poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
        area += cross
        cx += (poly[i, 0] + poly[j, 0]) * cross
        cy += (poly[i, 1] + poly[j, 1]) * cross
    
    area *= 0.5
    if abs(area) < 1e-12:
        return 0.0, poly.mean(axis=0)
    
    cx /= (6 * area)
    cy /= (6 * area)
    
    return abs(area), np.array([cx, cy])


# =============================================================================
# SPARSE REGION IDENTIFICATION
# =============================================================================

def _normalize_pair(
    ga_instance: Any,
    population: List,
    p1_idx: int,
    p2_idx: int,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Extract and normalize a 2D parameter pair from population.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        GA instance for getting bounds
    population : list
        Current population
    p1_idx, p2_idx : int
        Parameter indices
        
    Returns
    -------
    pts : np.ndarray
        Normalized points in [0,1]^2
    bounds : tuple
        (lo_i, hi_i, lo_j, hi_j) original bounds
    """
    lo_i, hi_i = ga_instance.get_param_bounds(p1_idx)
    lo_j, hi_j = ga_instance.get_param_bounds(p2_idx)
    
    pts = []
    for ind in population:
        xi = (float(ind[p1_idx]) - lo_i) / (hi_i - lo_i + 1e-12)
        xj = (float(ind[p2_idx]) - lo_j) / (hi_j - lo_j + 1e-12)
        pts.append([xi, xj])
    
    return np.array(pts), (lo_i, hi_i, lo_j, hi_j)


def _analyze_voronoi_2d(
    ga_instance: Any,
    population: List,
    p1_idx: int,
    p2_idx: int,
    p1_name: str,
    p2_name: str,
    n_regions_per_pair: int = 4,
) -> List[Dict]:
    """
    Build Voronoi on normalized pair, find largest empty cells.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        GA instance
    population : list
        Current population
    p1_idx, p2_idx : int
        Parameter indices
    p1_name, p2_name : str
        Parameter names
    n_regions_per_pair : int
        Number of regions to return
        
    Returns
    -------
    list of dict
        Sparse regions with target parameters
    """
    if not HAS_SCIPY_VORONOI:
        return []
    
    pts, (lo_i, hi_i, lo_j, hi_j) = _normalize_pair(
        ga_instance, population, p1_idx, p2_idx
    )
    
    if len(pts) < 4:
        return []
    
    vor = Voronoi(pts)
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=10.0)
    
    results = []
    for region in regions:
        poly = vertices[region]
        poly = _clip_poly_to_unit_square(poly)
        if len(poly) < 3:
            continue
        
        area, centroid = _poly_area_and_centroid(poly)
        if area <= 0.0 or not np.isfinite(centroid).all():
            continue
        
        # Denormalize centroid to parameter space
        center_i = lo_i + centroid[0] * (hi_i - lo_i)
        center_j = lo_j + centroid[1] * (hi_j - lo_j)
        
        results.append({
            "pair": (p1_idx, p2_idx),
            "param_indices": {p1_name: p1_idx, p2_name: p2_idx},
            "target_params": {p1_name: float(center_i), p2_name: float(center_j)},
            "area_norm": float(area),
            "center_norm": np.array(centroid, float),
            "center_denorm": np.array([center_i, center_j], float),
        })
    
    # Largest empty cells first
    results.sort(key=lambda r: r["area_norm"], reverse=True)
    return results[:max(0, int(n_regions_per_pair))]


def identify_sparse_regions_voronoi(
    ga_instance: Any,
    population: List,
    n_regions: int = 32,
) -> List[Dict]:
    """
    Use Voronoi diagrams on several 2D parameter pairs to find sparse regions.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        GA instance
    population : list
        Current population
    n_regions : int
        Total number of regions to return
        
    Returns
    -------
    list of dict
        Sparse regions ranked by area
    """
    # Key parameter pairs for exploration
    key_param_pairs = [
        (6, 7,  't_1',     't_2'),
        (7, 9,  't_2',     'infall_2'),
        (5, 9,  'sigma_2', 'infall_2'),
        (5, 7,  'sigma_2', 't_2'),
        (5, 14, 'sigma_2', 'nb'),
        (10, 5, 'sfe',     'sigma_2'),
        (10, 11,'sfe',     'delta_sfe'),
        (13, 14,'mgal',    'nb'),
    ]
    
    per_pair = max(1, int(np.ceil(n_regions / max(1, len(key_param_pairs)))))
    all_regions = []
    
    for p1_idx, p2_idx, p1_name, p2_name in key_param_pairs:
        regs = _analyze_voronoi_2d(
            ga_instance, population,
            p1_idx, p2_idx, p1_name, p2_name,
            n_regions_per_pair=per_pair
        )
        all_regions.extend(regs)
    
    all_regions.sort(key=lambda r: r["area_norm"], reverse=True)
    return all_regions[:n_regions]


# =============================================================================
# EXPLORATION MOVES
# =============================================================================

def _mutate_toward_region(
    ga_instance: Any,
    individual: List,
    target_region: Dict,
) -> None:
    """
    Move an individual toward region center on the two target dimensions.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        GA instance
    individual : list
        Individual to mutate
    target_region : dict
        Target region from identify_sparse_regions_voronoi
    """
    for param_name, param_idx in target_region["param_indices"].items():
        target_val = target_region["target_params"][param_name]
        lo, hi = ga_instance.get_param_bounds(param_idx)
        
        # Move toward target with some randomness
        current = float(individual[param_idx])
        alpha = 0.5 + 0.5 * random.random()  # Move 50-100% toward target
        new_val = current + alpha * (target_val - current)
        
        # Ensure within bounds
        new_val = max(lo, min(hi, new_val))
        individual[param_idx] = new_val


def _add_background_mutation(
    ga_instance: Any,
    individual: List,
    mutation_probability: float = 0.2,
) -> None:
    """
    Add small random perturbations to non-target parameters.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        GA instance
    individual : list
        Individual to mutate
    mutation_probability : float
        Per-parameter mutation probability
    """
    continuous_indices = getattr(ga_instance, 'continuous_indices', list(range(5, 15)))
    
    for i in continuous_indices:
        if random.random() < mutation_probability:
            lo, hi = ga_instance.get_param_bounds(i)
            range_size = hi - lo
            sigma = range_size * 0.05  # 5% of range
            
            new_val = individual[i] + random.gauss(0, sigma)
            new_val = max(lo, min(hi, new_val))
            individual[i] = new_val


def voronoi_explore_dearths(
    ga_instance: Any,
    population: List,
    exploration_fraction: float = 0.2,
) -> int:
    """
    Move worst-performing individuals toward centers of largest empty regions.
    
    Parameters
    ----------
    ga_instance : GalacticEvolutionGA
        GA instance
    population : list
        Current population (modified in-place)
    exploration_fraction : float
        Fraction of population to redirect
        
    Returns
    -------
    int
        Number of individuals moved
    """
    if not 0.0 < exploration_fraction <= 1.0:
        raise ValueError("exploration_fraction must be in (0,1].")
    
    n_move = max(1, int(len(population) * exploration_fraction))
    regions = identify_sparse_regions_voronoi(ga_instance, population, n_regions=n_move)
    
    if len(regions) == 0:
        return 0
    
    # Get fitness values
    def _fitness_value(ind):
        try:
            return ind.fitness.values[0] if getattr(ind.fitness, "valid", False) else float("inf")
        except Exception:
            return float("inf")
    
    # Sort by fitness (worst first)
    worst = sorted(population, key=_fitness_value, reverse=True)[:n_move]
    
    moved = 0
    for k, ind in enumerate(worst):
        region = regions[k % len(regions)]
        _mutate_toward_region(ga_instance, ind, region)
        _add_background_mutation(ga_instance, ind, mutation_probability=0.2)
        
        # Invalidate fitness
        try:
            del ind.fitness.values
        except Exception:
            pass
        
        moved += 1
    
    return moved
