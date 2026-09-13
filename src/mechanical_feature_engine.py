import json
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# OX ALPHA
# MECHANICAL FEATURE RECOGNITION ENGINE - STAGE 1
#
# This replaces the experimental "random cylinder" approach.
#
# Strategy:
#   STL
#    -> merge duplicate vertices
#    -> analysis mesh
#    -> smooth surface regions from topology
#    -> classify regions from global geometry
#    -> fit cylinders to COMPLETE regions
#    -> extract region boundary loops
#    -> create mechanical feature evidence
#
# No Gemini in this stage.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_FILE = OUTPUT_DIR / "mechanical_feature_evidence.json"

TARGET_TRIANGLES = 250_000

# Neighboring triangles whose normals differ by less than this
# are considered part of the same smooth surface.
SMOOTH_ANGLE_DEG = 7.0

MIN_REGION_TRIANGLES = 30
MIN_CYLINDER_TRIANGLES = 100

# Cylinder quality thresholds.
MAX_RELATIVE_RADIUS_ERROR = 0.08
MIN_NORMAL_ALIGNMENT = 0.70
MIN_ANGULAR_COVERAGE = 0.30

# A component is considered a useful mechanical cylinder only
# when it has enough geometric evidence.
MIN_CYLINDER_SCORE = 0.62

# Boundary loop filtering.
MIN_LOOP_VERTICES = 8

# Numerical tolerance for deciding whether two axis directions
# are equivalent.
AXIS_ANGLE_TOLERANCE_DEG = 8.0


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def axis_angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)

    if a is None or b is None:
        return 180.0

    return math.degrees(
        math.acos(
            np.clip(abs(float(np.dot(a, b))), -1.0, 1.0)
        )
    )


# ============================================================
# FIND STL
# ============================================================

def find_stl():
    print("\n[1] Searching for STL...")

    files = []

    if INPUT_DIR.exists():
        files = list(INPUT_DIR.rglob("*.stl"))

    if not files:
        files = [
            p for p in PROJECT_ROOT.rglob("*.stl")
            if "output" not in p.parts
        ]

    if not files:
        raise FileNotFoundError("No STL file found.")

    files.sort(
        key=lambda p: p.stat().st_size,
        reverse=True
    )

    path = files[0]

    print(f"[OK] Selected STL:\n{path}")

    return path


# ============================================================
# LOAD + CLEAN
# ============================================================

def load_and_clean_mesh(path):
    print("\n[2] Loading and cleaning mesh...")

    mesh = trimesh.load_mesh(
        path,
        process=False
    )

    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())

        if not geometries:
            raise RuntimeError(
                "STL scene contains no geometry."
            )

        mesh = trimesh.util.concatenate(
            geometries
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(
            "Unable to load STL as a triangular mesh."
        )

    print(
        f"[INFO] Raw triangles : {len(mesh.faces):,}"
    )

    print(
        f"[INFO] Raw vertices  : {len(mesh.vertices):,}"
    )

    # This is critical for STL files because binary STL commonly
    # stores each triangle with duplicated vertex records.
    mesh.merge_vertices()

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()

    print(
        f"[OK] Clean triangles: {len(mesh.faces):,}"
    )

    print(
        f"[OK] Clean vertices : {len(mesh.vertices):,}"
    )

    print(
        f"[OK] Watertight     : {mesh.is_watertight}"
    )

    return mesh


# ============================================================
# ANALYSIS MESH
# ============================================================

def create_analysis_mesh(mesh):
    print("\n[3] Creating analysis mesh...")

    if len(mesh.faces) <= TARGET_TRIANGLES:
        print(
            "[OK] Mesh already below analysis target."
        )
        return mesh

    print(
        f"[INFO] Current triangles: "
        f"{len(mesh.faces):,}"
    )

    print(
        f"[INFO] Target triangles : "
        f"{TARGET_TRIANGLES:,}"
    )

    try:
        reduced = mesh.simplify_quadric_decimation(
            face_count=TARGET_TRIANGLES
        )
    except Exception as error:
        print(
            f"[WARNING] Reduction failed: {error}"
        )
        print(
            "[WARNING] Continuing with cleaned original mesh."
        )
        return mesh

    if reduced is None or len(reduced.faces) == 0:
        print(
            "[WARNING] Reduction returned an invalid mesh."
        )
        return mesh

    print(
        f"[OK] Analysis triangles: "
        f"{len(reduced.faces):,}"
    )

    return reduced


# ============================================================
# TRIANGLE DATA
# ============================================================

def prepare_triangle_data(mesh):
    print("\n[4] Preparing triangle geometry...")

    triangles = mesh.triangles

    centers = triangles.mean(axis=1)

    normals = np.asarray(
        mesh.face_normals,
        dtype=np.float64
    )

    areas = np.asarray(
        mesh.area_faces,
        dtype=np.float64
    )

    valid = (
        np.isfinite(centers).all(axis=1)
        &
        np.isfinite(normals).all(axis=1)
        &
        np.isfinite(areas)
        &
        (areas > 1e-12)
    )

    centers = centers[valid]
    normals = normals[valid]
    areas = areas[valid]

    print(
        f"[OK] Usable triangles: "
        f"{len(centers):,}"
    )

    return centers, normals, areas


# ============================================================
# SMOOTH SURFACE SEGMENTATION
# ============================================================

def segment_smooth_regions(mesh, normals):
    print(
        "\n[5] Segmenting smooth connected surfaces..."
    )

    adjacency = np.asarray(
        mesh.face_adjacency,
        dtype=np.int32
    )

    if len(adjacency) == 0:
        raise RuntimeError(
            "Mesh has no face adjacency."
        )

    a = adjacency[:, 0]
    b = adjacency[:, 1]

    dots = np.sum(
        normals[a] * normals[b],
        axis=1
    )

    dots = np.clip(
        dots,
        -1.0,
        1.0
    )

    angles = np.degrees(
        np.arccos(
            np.abs(dots)
        )
    )

    smooth = (
        angles
        <=
        SMOOTH_ANGLE_DEG
    )

    links = adjacency[smooth]

    print(
        f"[INFO] Total adjacency links: "
        f"{len(adjacency):,}"
    )

    print(
        f"[INFO] Smooth links: "
        f"{len(links):,}"
    )

    if len(links) == 0:
        return np.arange(
            len(normals),
            dtype=np.int32
        )

    rows = np.concatenate(
        [
            links[:, 0],
            links[:, 1]
        ]
    )

    cols = np.concatenate(
        [
            links[:, 1],
            links[:, 0]
        ]
    )

    data = np.ones(
        len(rows),
        dtype=np.uint8
    )

    graph = coo_matrix(
        (
            data,
            (rows, cols)
        ),
        shape=(
            len(normals),
            len(normals)
        )
    ).tocsr()

    count, labels = connected_components(
        graph,
        directed=False,
        return_labels=True
    )

    print(
        f"[OK] Smooth surface regions: "
        f"{count:,}"
    )

    return labels.astype(
        np.int32
    )


# ============================================================
# REGION NORMAL PCA
# ============================================================

def normal_distribution_score(normals):
    """
    For a planar surface:
        normals are nearly identical.

    For a cylinder:
        normals rotate around a common axis and therefore
        occupy approximately a 2D subspace.

    For a free-form surface:
        all three normal PCA directions tend to contain
        meaningful variation.
    """

    if len(normals) < 20:
        return None

    n = np.asarray(
        normals,
        dtype=np.float64
    )

    lengths = np.linalg.norm(
        n,
        axis=1
    )

    valid = lengths > 1e-10

    n = n[valid]

    if len(n) < 20:
        return None

    n = (
        n
        /
        np.linalg.norm(
            n,
            axis=1,
            keepdims=True
        )
    )

    covariance = np.cov(
        n.T
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total = float(
        np.sum(eigenvalues)
    )

    if total < 1e-12:
        return {
            "type": "PLANAR",
            "axis": None,
            "planar_score": 1.0,
            "cylinder_score": 0.0,
            "eigenvalues": eigenvalues.tolist()
        }

    normalized = (
        eigenvalues
        /
        total
    )

    planar_score = float(
        np.clip(
            1.0
            -
            normalized[0]
            *
            20.0,
            0.0,
            1.0
        )
    )

    # Cylinder normals should have a weak first eigenvalue
    # and two substantial remaining directions.
    cylinder_balance = float(
        np.clip(
            1.0
            -
            abs(
                normalized[1]
                -
                normalized[2]
            )
            /
            max(
                normalized[1]
                +
                normalized[2],
                1e-12
            ),
            0.0,
            1.0
        )
    )

    cylinder_flatness = float(
        np.clip(
            1.0
            -
            normalized[0] * 4.0,
            0.0,
            1.0
        )
    )

    cylinder_score = (
        0.55 * cylinder_balance
        +
        0.45 * cylinder_flatness
    )

    axis = normalize(
        eigenvectors[:, 0]
    )

    return {
        "type": "CURVED",
        "axis": axis,
        "planar_score": planar_score,
        "cylinder_score": cylinder_score,
        "eigenvalues": eigenvalues.tolist()
    }


# ============================================================
# CYLINDER FIT
# ============================================================

def fit_cylinder(points, normals, areas):
    if len(points) < MIN_CYLINDER_TRIANGLES:
        return None

    distribution = normal_distribution_score(
        normals
    )

    if distribution is None:
        return None

    axis = distribution["axis"]

    if axis is None:
        return None

    # A cylinder axis is perpendicular to its radial normals.
    n = np.asarray(
        normals,
        dtype=np.float64
    )

    p = np.asarray(
        points,
        dtype=np.float64
    )

    n_perp = (
        n
        -
        np.outer(
            n @ axis,
            axis
        )
    )

    lengths = np.linalg.norm(
        n_perp,
        axis=1
    )

    valid = lengths > 1e-8

    if np.sum(valid) < 30:
        return None

    n_perp = (
        n_perp[valid]
        /
        lengths[valid, None]
    )

    p_valid = p[valid]

    w = np.maximum(
        np.asarray(areas)[valid],
        1e-12
    )

    # Solve the centerline position:
    #
    # n . center = n . point
    #
    # The axial coordinate is arbitrary, so normalize it
    # afterwards.
    rhs = np.sum(
        n_perp * p_valid,
        axis=1
    )

    sqrt_w = np.sqrt(w)

    try:
        center, *_ = np.linalg.lstsq(
            n_perp * sqrt_w[:, None],
            rhs * sqrt_w,
            rcond=None
        )
    except np.linalg.LinAlgError:
        return None

    center = (
        center
        -
        axis
        *
        np.dot(
            center,
            axis
        )
    )

    vectors = p - center

    axial = vectors @ axis

    radial_vectors = (
        vectors
        -
        np.outer(
            axial,
            axis
        )
    )

    radial_distances = np.linalg.norm(
        radial_vectors,
        axis=1
    )

    radius = float(
        np.median(
            radial_distances
        )
    )

    if radius <= 0:
        return None

    errors = np.abs(
        radial_distances
        -
        radius
    )

    mean_error = float(
        np.mean(errors)
    )

    relative_error = (
        mean_error
        /
        max(
            radius,
            1e-12
        )
    )

    p95_error = float(
        np.percentile(
            errors,
            95
        )
    )

    # Normal alignment.
    normal_unit = (
        n
        /
        np.linalg.norm(
            n,
            axis=1,
            keepdims=True
        )
    )

    radial_unit = (
        radial_vectors
        /
        np.maximum(
            radial_distances[:, None],
            1e-12
        )
    )

    dots = np.sum(
        normal_unit
        *
        radial_unit,
        axis=1
    )

    normal_alignment = float(
        np.mean(
            np.abs(dots)
        )
    )

    internal_ratio = float(
        np.mean(
            dots < -0.35
        )
    )

    external_ratio = float(
        np.mean(
            dots > 0.35
        )
    )

    angular_coverage = calculate_angular_coverage(
        radial_vectors,
        axis
    )

    axial_min = float(
        np.percentile(
            axial,
            2
        )
    )

    axial_max = float(
        np.percentile(
            axial,
            98
        )
    )

    length = (
        axial_max
        -
        axial_min
    )

    radial_quality = float(
        np.clip(
            1.0
            -
            relative_error
            /
            MAX_RELATIVE_RADIUS_ERROR,
            0.0,
            1.0
        )
    )

    coverage_quality = float(
        np.clip(
            angular_coverage,
            0.0,
            1.0
        )
    )

    alignment_quality = float(
        np.clip(
            normal_alignment,
            0.0,
            1.0
        )
    )

    geometry_score = (
        0.40 * radial_quality
        +
        0.30 * coverage_quality
        +
        0.30 * alignment_quality
    )

    return {
        "axis": axis,
        "origin": center,
        "radius": radius,
        "diameter": radius * 2.0,
        "length": length,
        "mean_radial_error": mean_error,
        "relative_radial_error": relative_error,
        "p95_radial_error": p95_error,
        "normal_alignment": normal_alignment,
        "internal_ratio": internal_ratio,
        "external_ratio": external_ratio,
        "angular_coverage": angular_coverage,
        "radial_quality": radial_quality,
        "geometry_score": geometry_score,
        "normal_distribution": distribution,
    }


# ============================================================
# ANGULAR COVERAGE
# ============================================================

def calculate_angular_coverage(radial_vectors, axis):
    if len(radial_vectors) < 20:
        return 0.0

    lengths = np.linalg.norm(
        radial_vectors,
        axis=1
    )

    valid = lengths > 1e-9

    rv = radial_vectors[valid]

    if len(rv) < 20:
        return 0.0

    helper = np.array(
        [1.0, 0.0, 0.0]
    )

    if abs(
        float(
            np.dot(
                helper,
                axis
            )
        )
    ) > 0.90:
        helper = np.array(
            [0.0, 1.0, 0.0]
        )

    u = normalize(
        np.cross(
            axis,
            helper
        )
    )

    v = normalize(
        np.cross(
            axis,
            u
        )
    )

    if u is None or v is None:
        return 0.0

    angles = np.arctan2(
        rv @ v,
        rv @ u
    )

    angles = (
        angles
        +
        2.0 * np.pi
    ) % (
        2.0 * np.pi
    )

    angles.sort()

    gaps = np.diff(
        np.r_[
            angles,
            angles[0]
            +
            2.0 * np.pi
        ]
    )

    return float(
        np.clip(
            1.0
            -
            gaps.max()
            /
            (2.0 * np.pi),
            0.0,
            1.0
        )
    )


# ============================================================
# BOUNDARY EDGES
# ============================================================

def build_component_boundary_edges(
    mesh,
    labels,
    component_id
):
    """
    Find edges separating the selected surface region from
    another surface region.

    These edges are the important mechanical boundaries:
      cylinder -> plane
      plane    -> cylinder
      cylinder -> cylinder
      etc.
    """

    adjacency = np.asarray(
        mesh.face_adjacency,
        dtype=np.int32
    )

    adjacency_edges = np.asarray(
        mesh.face_adjacency_edges,
        dtype=np.int32
    )

    if len(adjacency) == 0:
        return np.empty(
            (0, 2),
            dtype=np.int32
        )

    left = labels[
        adjacency[:, 0]
    ]

    right = labels[
        adjacency[:, 1]
    ]

    selected = (
        (
            left == component_id
        )
        &
        (
            right != component_id
        )
    ) | (
        (
            right == component_id
        )
        &
        (
            left != component_id
        )
    )

    return adjacency_edges[
        selected
    ]


# ============================================================
# BUILD BOUNDARY LOOPS
# ============================================================

def boundary_loops(edge_pairs):
    """
    Convert boundary edges into connected loops/chains.

    For a normal mechanical cylinder we expect a closed loop
    at one or both ends.
    """

    if len(edge_pairs) == 0:
        return []

    graph = {}

    for a, b in edge_pairs:

        a = int(a)
        b = int(b)

        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    unused = {
        tuple(sorted((int(a), int(b))))
        for a, b in edge_pairs
    }

    loops = []

    while unused:

        first_edge = next(
            iter(unused)
        )

        start = first_edge[0]

        current = start
        previous = None

        path = [start]

        safety = 0

        while safety < len(edge_pairs) + 10:

            safety += 1

            neighbors = graph.get(
                current,
                []
            )

            next_vertex = None

            for candidate in neighbors:

                edge = tuple(
                    sorted(
                        (
                            current,
                            candidate
                        )
                    )
                )

                if edge not in unused:
                    continue

                if (
                    previous is not None
                    and
                    candidate == previous
                    and
                    len(neighbors) > 1
                ):
                    continue

                next_vertex = candidate
                unused.remove(edge)
                break

            if next_vertex is None:
                break

            previous = current
            current = next_vertex
            path.append(current)

            if current == start:
                break

        if len(path) >= MIN_LOOP_VERTICES:
            loops.append(path)

    return loops


# ============================================================
# FIT CIRCULAR BOUNDARY
# ============================================================

def fit_boundary_circle(
    vertices,
    loop,
    axis
):
    if len(loop) < MIN_LOOP_VERTICES:
        return None

    points = vertices[
        np.asarray(
            loop,
            dtype=np.int32
        )
    ]

    center = points.mean(
        axis=0
    )

    axis = normalize(axis)

    if axis is None:
        return None

    # Project boundary points into plane perpendicular to axis.
    vectors = points - center

    axial = vectors @ axis

    planar = (
        vectors
        -
        np.outer(
            axial,
            axis
        )
    )

    radii = np.linalg.norm(
        planar,
        axis=1
    )

    radius = float(
        np.median(radii)
    )

    if radius < 1e-6:
        return None

    error = float(
        np.mean(
            np.abs(
                radii - radius
            )
        )
    )

    relative_error = (
        error
        /
        radius
    )

    return {
        "vertex_count": int(len(loop)),
        "center": center.tolist(),
        "radius": radius,
        "diameter": radius * 2.0,
        "relative_error": relative_error,
    }


# ============================================================
# CLASSIFY REGION
# ============================================================

def classify_region(
    component_id,
    labels,
    centers,
    normals,
    areas,
    mesh
):
    mask = (
        labels == component_id
    )

    indices = np.flatnonzero(
        mask
    )

    count = len(indices)

    if count < MIN_REGION_TRIANGLES:
        return None

    region_centers = centers[
        indices
    ]

    region_normals = normals[
        indices
    ]

    region_areas = areas[
        indices
    ]

    distribution = normal_distribution_score(
        region_normals
    )

    if distribution is None:
        return None

    area = float(
        np.sum(
            region_areas
        )
    )

    center = np.average(
        region_centers,
        axis=0,
        weights=np.maximum(
            region_areas,
            1e-12
        )
    )

    # --------------------------------------------------------
    # Planar region.
    # --------------------------------------------------------

    if distribution["planar_score"] >= 0.85:

        normal = normalize(
            np.average(
                region_normals,
                axis=0,
                weights=np.maximum(
                    region_areas,
                    1e-12
                )
            )
        )

        return {
            "component_id": int(component_id),
            "triangle_count": int(count),
            "area": area,
            "center": center.tolist(),
            "surface_type": "PLANE",
            "normal": (
                normal.tolist()
                if normal is not None
                else None
            ),
            "planar_score": float(
                distribution["planar_score"]
            ),
            "cylinder_score": float(
                distribution["cylinder_score"]
            ),
        }

    # --------------------------------------------------------
    # Cylinder region.
    # --------------------------------------------------------

    cylinder = fit_cylinder(
        region_centers,
        region_normals,
        region_areas
    )

    if cylinder is None:
        return {
            "component_id": int(component_id),
            "triangle_count": int(count),
            "area": area,
            "center": center.tolist(),
            "surface_type": "CURVED",
            "planar_score": float(
                distribution["planar_score"]
            ),
            "cylinder_score": float(
                distribution["cylinder_score"]
            ),
        }

    surface_score = float(
        distribution["cylinder_score"]
        *
        cylinder["geometry_score"]
    )

    if (
        surface_score >= MIN_CYLINDER_SCORE
        and
        cylinder["relative_radial_error"]
        <= MAX_RELATIVE_RADIUS_ERROR
        and
        cylinder["normal_alignment"]
        >= MIN_NORMAL_ALIGNMENT
        and
        cylinder["angular_coverage"]
        >= MIN_ANGULAR_COVERAGE
    ):
        surface_type = "CYLINDER"
    else:
        surface_type = "CURVED"

    # Boundary loops.
    edge_pairs = build_component_boundary_edges(
        mesh,
        labels,
        component_id
    )

    loops = boundary_loops(
        edge_pairs
    )

    fitted_loops = []

    for loop in loops:
        circle = fit_boundary_circle(
            mesh.vertices,
            loop,
            cylinder["axis"]
        )

        if circle is not None:
            fitted_loops.append(
                circle
            )

    return {
        "component_id": int(component_id),
        "triangle_count": int(count),
        "area": area,
        "center": center.tolist(),
        "surface_type": surface_type,
        "planar_score": float(
            distribution["planar_score"]
        ),
        "cylinder_score": float(
            distribution["cylinder_score"]
        ),
        "geometry_score": float(
            cylinder["geometry_score"]
        ),
        "physical_score": surface_score,
        "axis": cylinder["axis"].tolist(),
        "origin": cylinder["origin"].tolist(),
        "radius": float(
            cylinder["radius"]
        ),
        "diameter": float(
            cylinder["diameter"]
        ),
        "length": float(
            cylinder["length"]
        ),
        "mean_radial_error": float(
            cylinder["mean_radial_error"]
        ),
        "relative_radial_error": float(
            cylinder["relative_radial_error"]
        ),
        "p95_radial_error": float(
            cylinder["p95_radial_error"]
        ),
        "normal_alignment": float(
            cylinder["normal_alignment"]
        ),
        "internal_ratio": float(
            cylinder["internal_ratio"]
        ),
        "external_ratio": float(
            cylinder["external_ratio"]
        ),
        "angular_coverage": float(
            cylinder["angular_coverage"]
        ),
        "boundary_edge_count": int(
            len(edge_pairs)
        ),
        "boundary_loops": fitted_loops,
    }


# ============================================================
# MERGE CYLINDER REGIONS THAT BELONG TO ONE AXIS/FAMILY
# ============================================================

def cylinder_summary(regions):
    cylinders = [
        r for r in regions
        if r["surface_type"] == "CYLINDER"
    ]

    cylinders.sort(
        key=lambda x: x["physical_score"],
        reverse=True
    )

    return cylinders


# ============================================================
# SAVE
# ============================================================

def save_results(
    stl_path,
    original_mesh,
    analysis_mesh,
    regions,
    cylinders
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = {
        "system": "OX ALPHA",
        "engine": "Mechanical Feature Recognition Engine",
        "stage": 1,
        "source_stl": str(stl_path),

        "original": {
            "vertices": int(
                len(original_mesh.vertices)
            ),
            "triangles": int(
                len(original_mesh.faces)
            ),
            "watertight": bool(
                original_mesh.is_watertight
            ),
        },

        "analysis": {
            "vertices": int(
                len(analysis_mesh.vertices)
            ),
            "triangles": int(
                len(analysis_mesh.faces)
            ),
        },

        "summary": {
            "surface_regions": int(
                len(regions)
            ),
            "cylindrical_regions": int(
                len(cylinders)
            ),
            "planar_regions": int(
                sum(
                    r["surface_type"] == "PLANE"
                    for r in regions
                )
            ),
            "curved_regions": int(
                sum(
                    r["surface_type"] == "CURVED"
                    for r in regions
                )
            ),
        },

        "surface_regions": regions,

        "cylinders": cylinders,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print(
        f"\n[OK] Saved:\n{OUTPUT_FILE}"
    )


# ============================================================
# PRINT
# ============================================================

def print_results(cylinders):
    print("\n")
    print("=" * 78)
    print(
        "OX ALPHA - MECHANICAL CYLINDER EVIDENCE"
    )
    print("=" * 78)

    if not cylinders:
        print(
            "\n[INFO] No strong complete cylindrical "
            "surface regions were found."
        )
        return

    for i, c in enumerate(
        cylinders,
        start=1
    ):
        print(
            f"\nCYLINDER_REGION_{i:03d}"
        )

        print(
            f"  Diameter          : "
            f"{c['diameter']:.4f}"
        )

        print(
            f"  Length            : "
            f"{c['length']:.4f}"
        )

        print(
            f"  Triangles         : "
            f"{c['triangle_count']:,}"
        )

        print(
            f"  Physical score    : "
            f"{c['physical_score']:.3f}"
        )

        radial_quality = 1.0 - min(
            1.0,
            c["relative_radial_error"] /
            MAX_RELATIVE_RADIUS_ERROR
        )

        print(
            f"  Radial quality    : "
            f"{radial_quality:.3f}"
        )

        print(
            f"  Angular coverage  : "
            f"{c['angular_coverage']:.3f}"
        )

        print(
            f"  Normal alignment  : "
            f"{c['normal_alignment']:.3f}"
        )

        print(
            f"  Internal ratio    : "
            f"{c['internal_ratio']:.3f}"
        )

        print(
            f"  External ratio    : "
            f"{c['external_ratio']:.3f}"
        )

        print(
            f"  Boundary loops    : "
            f"{len(c['boundary_loops'])}"
        )

        for j, loop in enumerate(
            c["boundary_loops"],
            start=1
        ):
            print(
                f"    Loop {j}: "
                f"Ø{loop['diameter']:.4f}, "
                f"error={loop['relative_error']:.4f}, "
                f"vertices={loop['vertex_count']}"
            )

    print("\n")
    print("=" * 78)
    print(
        "MECHANICAL FEATURE STAGE 1 COMPLETED"
    )
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print(
        "OX ALPHA - MECHANICAL FEATURE RECOGNITION ENGINE"
    )
    print(
        "STAGE 1: SURFACE REGIONS + CYLINDRICAL EVIDENCE"
    )
    print("=" * 78)

    try:
        # ----------------------------------------------------
        # 1. STL
        # ----------------------------------------------------
        stl_path = find_stl()

        # ----------------------------------------------------
        # 2. Clean original mesh
        # ----------------------------------------------------
        original_mesh = load_and_clean_mesh(
            stl_path
        )

        # ----------------------------------------------------
        # 3. Analysis mesh
        # ----------------------------------------------------
        analysis_mesh = create_analysis_mesh(
            original_mesh
        )

        # ----------------------------------------------------
        # 4. Triangle geometry
        # ----------------------------------------------------
        centers, normals, areas = (
            prepare_triangle_data(
                analysis_mesh
            )
        )

        # ----------------------------------------------------
        # 5. Surface segmentation
        # ----------------------------------------------------
        labels = segment_smooth_regions(
            analysis_mesh,
            normals
        )

        # ----------------------------------------------------
        # 6. Region analysis
        # ----------------------------------------------------
        print(
            "\n[6] Classifying complete surface regions..."
        )

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        valid_components = (
            unique_labels[
                counts >= MIN_REGION_TRIANGLES
            ]
        )

        print(
            f"[INFO] Regions above minimum size: "
            f"{len(valid_components):,}"
        )

        regions = []

        for number, component_id in enumerate(
            valid_components,
            start=1
        ):
            result = classify_region(
                int(component_id),
                labels,
                centers,
                normals,
                areas,
                analysis_mesh
            )

            if result is not None:
                regions.append(
                    result
                )

            if number % 100 == 0:
                print(
                    f"    Analyzed "
                    f"{number:,}/"
                    f"{len(valid_components):,} regions"
                )

        cylinders = cylinder_summary(
            regions
        )

        print(
            f"\n[OK] Surface regions analyzed: "
            f"{len(regions):,}"
        )

        print(
            f"[OK] Strong cylindrical regions: "
            f"{len(cylinders):,}"
        )

        # ----------------------------------------------------
        # 7. Save
        # ----------------------------------------------------
        save_results(
            stl_path,
            original_mesh,
            analysis_mesh,
            regions,
            cylinders
        )

        # ----------------------------------------------------
        # 8. Results
        # ----------------------------------------------------
        print_results(
            cylinders
        )

    except Exception as error:
        print("\n")
        print("=" * 78)
        print(
            "MECHANICAL FEATURE ENGINE ERROR"
        )
        print("=" * 78)
        print(
            f"\n[ERROR] {error}"
        )
        print("=" * 78)


if __name__ == "__main__":
    main()
