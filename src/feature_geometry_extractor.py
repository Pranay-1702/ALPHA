from pathlib import Path
import json
import math

import numpy as np
import trimesh


# ============================================================
# OX ALPHA
# STAGE 3 - COMPLETE FEATURE GEOMETRY EXTRACTION
#
# Purpose:
#   Build a deterministic geometric inventory of the STL.
#
# This stage does NOT create CAD.
# This stage does NOT guess dimensions.
# This stage does NOT use Gemini.
#
# Input:
#   input/*.stl
#
# Also reads:
#   output/mechanical_feature_evidence.json
#   output/ai_feature_reasoning.json
#
# Output:
#   output/feature_geometry_evidence.json
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"

OUTPUT_DIR = PROJECT_ROOT / "output"

MECHANICAL_EVIDENCE_FILE = (
    OUTPUT_DIR / "mechanical_feature_evidence.json"
)

AI_REASONING_FILE = (
    OUTPUT_DIR / "ai_feature_reasoning.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "feature_geometry_evidence.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of cross sections.
SECTION_COUNT = 31

# Minimum number of section points needed for analysis.
MIN_SECTION_POINTS = 20

# Plane normal tolerance.
PLANAR_NORMAL_TOLERANCE_DEG = 3.0

# Circle fit acceptance.
MAX_CIRCLE_RELATIVE_ERROR = 0.05

# Minimum loop length relative to fitted diameter.
MIN_CIRCLE_LOOP_RATIO = 1.50

# PCA eigenvalue ratio used to identify the dominant part axis.
AXIS_DOMINANCE_RATIO = 1.15


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize(vector):

    vector = np.asarray(
        vector,
        dtype=float
    )

    length = np.linalg.norm(vector)

    if length < 1e-12:
        return None

    return vector / length


def safe_float(value, default=0.0):

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def vector_to_list(vector):

    return [
        float(x)
        for x in np.asarray(
            vector,
            dtype=float
        )
    ]


def angle_between_vectors(a, b):

    a = normalize(a)
    b = normalize(b)

    if a is None or b is None:
        return 180.0

    dot = np.clip(
        abs(float(np.dot(a, b))),
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(dot)
    )


# ============================================================
# FIND STL
# ============================================================

def find_stl():

    print("\n[1] Searching for STL...")

    files = []

    if INPUT_DIR.exists():

        files = list(
            INPUT_DIR.rglob("*.stl")
        )

    if not files:

        files = [
            p
            for p in PROJECT_ROOT.rglob("*.stl")
            if "output" not in p.parts
        ]

    if not files:

        raise FileNotFoundError(
            "No STL file found."
        )

    files.sort(
        key=lambda p: p.stat().st_size,
        reverse=True
    )

    selected = files[0]

    print(
        f"[OK] Selected STL:\n{selected}"
    )

    return selected


# ============================================================
# LOAD STL
# ============================================================

def load_mesh(path):

    print(
        "\n[2] Loading STL..."
    )

    mesh = trimesh.load_mesh(
        path,
        process=False
    )

    if isinstance(
        mesh,
        trimesh.Scene
    ):

        geometries = list(
            mesh.geometry.values()
        )

        if not geometries:

            raise RuntimeError(
                "STL scene contains no geometry."
            )

        mesh = trimesh.util.concatenate(
            geometries
        )

    if not isinstance(
        mesh,
        trimesh.Trimesh
    ):

        raise RuntimeError(
            "Unable to load STL as triangular mesh."
        )

    # --------------------------------------------------------
    # Clean duplicate vertices.
    # --------------------------------------------------------

    mesh.merge_vertices()

    mesh.remove_degenerate_faces()

    mesh.remove_duplicate_faces()

    mesh.remove_unreferenced_vertices()

    print(
        f"[OK] Vertices : "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"[OK] Triangles: "
        f"{len(mesh.faces):,}"
    )

    print(
        f"[OK] Watertight: "
        f"{mesh.is_watertight}"
    )

    return mesh


# ============================================================
# PRINCIPAL AXIS
# ============================================================

def determine_principal_axes(mesh):

    print(
        "\n[3] Determining principal geometry axes..."
    )

    vertices = np.asarray(
        mesh.vertices,
        dtype=float
    )

    center = vertices.mean(
        axis=0
    )

    centered = (
        vertices - center
    )

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = (
        np.linalg.eigh(covariance)
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    axes = []

    for i in range(3):

        axis = normalize(
            eigenvectors[:, i]
        )

        axes.append(axis)

    extents = []

    for axis in axes:

        coordinates = (
            centered @ axis
        )

        extents.append(
            float(
                coordinates.max()
                -
                coordinates.min()
            )
        )

    print(
        "[OK] Principal axes determined."
    )

    print(
        f"  Axis 1 extent: {extents[0]:.4f}"
    )

    print(
        f"  Axis 2 extent: {extents[1]:.4f}"
    )

    print(
        f"  Axis 3 extent: {extents[2]:.4f}"
    )

    return {
        "center": center,
        "axes": axes,
        "eigenvalues": [
            float(x)
            for x in eigenvalues
        ],
        "extents": extents
    }


# ============================================================
# PLANAR SURFACES
# ============================================================

def analyze_planar_faces(mesh):

    print(
        "\n[4] Analyzing planar surfaces..."
    )

    normals = np.asarray(
        mesh.face_normals,
        dtype=float
    )

    areas = np.asarray(
        mesh.area_faces,
        dtype=float
    )

    centers = np.asarray(
        mesh.triangles_center,
        dtype=float
    )

    valid = (
        np.linalg.norm(
            normals,
            axis=1
        ) > 1e-12
    )

    normals = normals[valid]

    areas = areas[valid]

    centers = centers[valid]

    if len(normals) == 0:

        return []

    # --------------------------------------------------------
    # Group approximately parallel normals.
    # --------------------------------------------------------

    groups = []

    for index in range(
        len(normals)
    ):

        normal = normals[index]

        assigned = False

        for group in groups:

            angle = angle_between_vectors(
                normal,
                group["normal"]
            )

            if (
                angle
                <= PLANAR_NORMAL_TOLERANCE_DEG
            ):

                group["indices"].append(
                    index
                )

                assigned = True

                break

        if not assigned:

            groups.append(
                {
                    "normal":
                        normal.copy(),

                    "indices":
                        [index]
                }
            )

    planar_groups = []

    for group_index, group in enumerate(
        groups,
        start=1
    ):

        indices = np.asarray(
            group["indices"],
            dtype=int
        )

        group_area = float(
            areas[indices].sum()
        )

        if group_area <= 0:
            continue

        weighted_center = (
            (
                centers[indices]
                *
                areas[indices][:, None]
            ).sum(axis=0)
            /
            group_area
        )

        planar_groups.append(
            {
                "group_id":
                    group_index,

                "normal":
                    vector_to_list(
                        group["normal"]
                    ),

                "area":
                    group_area,

                "triangle_count":
                    int(len(indices)),

                "center":
                    vector_to_list(
                        weighted_center
                    )
            }
        )

    planar_groups.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    print(
        f"[OK] Planar normal groups: "
        f"{len(planar_groups)}"
    )

    return planar_groups


# ============================================================
# CIRCLE FIT
# ============================================================

def fit_circle_2d(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < 6:
        return None

    x = points[:, 0]

    y = points[:, 1]

    A = np.column_stack(
        [
            2.0 * x,
            2.0 * y,
            np.ones(len(points))
        ]
    )

    b = (
        x * x
        +
        y * y
    )

    try:

        solution, _, _, _ = np.linalg.lstsq(
            A,
            b,
            rcond=None
        )

    except np.linalg.LinAlgError:

        return None

    cx = solution[0]

    cy = solution[1]

    radius_squared = (
        solution[2]
        +
        cx * cx
        +
        cy * cy
    )

    if radius_squared <= 0:
        return None

    radius = math.sqrt(
        radius_squared
    )

    distances = np.sqrt(
        (
            points[:, 0] - cx
        ) ** 2
        +
        (
            points[:, 1] - cy
        ) ** 2
    )

    errors = np.abs(
        distances - radius
    )

    mean_error = float(
        errors.mean()
    )

    relative_error = (
        mean_error / radius
        if radius > 1e-12
        else 999.0
    )

    return {
        "center_2d": [
            float(cx),
            float(cy)
        ],

        "radius":
            float(radius),

        "diameter":
            float(2.0 * radius),

        "mean_error":
            mean_error,

        "relative_error":
            relative_error,

        "point_count":
            int(len(points))
    }


# ============================================================
# CREATE SECTION BASIS
# ============================================================

def section_basis(axis):

    axis = normalize(
        axis
    )

    if axis is None:

        raise ValueError(
            "Invalid section axis."
        )

    reference = np.array(
        [1.0, 0.0, 0.0]
    )

    if abs(
        np.dot(
            reference,
            axis
        )
    ) > 0.90:

        reference = np.array(
            [0.0, 1.0, 0.0]
        )

    u = normalize(
        np.cross(
            axis,
            reference
        )
    )

    v = normalize(
        np.cross(
            axis,
            u
        )
    )

    return u, v


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_sections(
    mesh,
    axis_data
):

    print(
        "\n[5] Extracting principal-axis sections..."
    )

    axis = axis_data[
        "axes"
    ][2]

    center = axis_data[
        "center"
    ]

    vertices = np.asarray(
        mesh.vertices,
        dtype=float
    )

    coordinates = (
        (vertices - center)
        @ axis
    )

    minimum = float(
        coordinates.min()
    )

    maximum = float(
        coordinates.max()
    )

    if maximum - minimum < 1e-8:

        return []

    u, v = section_basis(
        axis
    )

    sections = []

    levels = np.linspace(
        minimum,
        maximum,
        SECTION_COUNT
    )

    for section_index, level in enumerate(
        levels,
        start=1
    ):

        origin = (
            center
            +
            axis * level
        )

        try:

            section = mesh.section(
                plane_origin=origin,
                plane_normal=axis
            )

        except Exception:

            continue

        if section is None:

            continue

        try:

            planar, to_3d = (
                section.to_2D()
            )

        except Exception:

            continue

        vertices_2d = np.asarray(
            planar.vertices,
            dtype=float
        )

        if (
            len(vertices_2d)
            < MIN_SECTION_POINTS
        ):

            continue

        # ----------------------------------------------------
        # IMPORTANT COORDINATE-SYSTEM RULE
        # ----------------------------------------------------
        # Trimesh planar.vertices uses its own 2D section frame.
        # It is NOT guaranteed to match our custom u/v basis.
        # Circle fitting is therefore kept in the native Trimesh
        # 2D frame, and fitted centers are transformed back to
        # the original STL XYZ frame with the exact to_3d matrix.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Connected paths from the section.
        # ----------------------------------------------------

        path_count = 0

        circle_candidates = []

        try:

            entities = planar.entities

            for entity in entities:

                entity_type = (
                    entity.__class__.__name__
                )

                # --------------------------------------------
                # Lines / arcs / discrete path entities
                # --------------------------------------------

                try:

                    entity_points = (
                        entity.discrete(
                            planar.vertices
                        )
                    )

                except Exception:

                    continue

                entity_points = np.asarray(
                    entity_points,
                    dtype=float
                )

                if len(entity_points) < 6:
                    continue

                path_count += 1

                fit = fit_circle_2d(
                    entity_points[:, :2]
                )

                if fit is None:
                    continue

                if (
                    fit["relative_error"]
                    <= MAX_CIRCLE_RELATIVE_ERROR
                ):

                    # Convert the fitted circle center from Trimesh's
                    # native 2D section frame to original STL XYZ.
                    center_2d = np.asarray(
                        fit["center_2d"],
                        dtype=float
                    )

                    center_2d_3d = np.array(
                        [center_2d[0], center_2d[1], 0.0],
                        dtype=float
                    )

                    try:
                        circle_center_3d = (
                            trimesh.transform_points(
                                center_2d_3d.reshape(1, 3),
                                to_3d
                            )[0]
                        )
                    except Exception:
                        continue

                    circle_candidates.append(
                        {
                            "entity_type":
                                entity_type,

                            "diameter":
                                fit["diameter"],

                            "radius":
                                fit["radius"],

                            "center_2d":
                                fit["center_2d"],

                            "center_3d":
                                vector_to_list(
                                    circle_center_3d
                                ),

                            "relative_error":
                                fit["relative_error"],

                            "point_count":
                                fit["point_count"]
                        }
                    )

        except Exception:

            pass

        sections.append(
            {
                "section_id":
                    section_index,

                "axis_coordinate":
                    float(level),

                "origin":
                    vector_to_list(
                        origin
                    ),

                "point_count":
                    int(len(vertices_2d)),

                "path_count":
                    int(path_count),

                "circular_candidates":
                    circle_candidates
            }
        )

    print(
        f"[OK] Valid sections: "
        f"{len(sections)}"
    )

    return sections


# ============================================================
# SUMMARIZE CIRCULAR FEATURES
# ============================================================

def summarize_circular_features(
    sections,
    axis_data
):

    print(
        "\n[6] Building circular feature inventory..."
    )

    axis = axis_data[
        "axes"
    ][2]

    center = axis_data[
        "center"
    ]

    features = []

    for section in sections:

        origin = np.asarray(
            section["origin"],
            dtype=float
        )

        for circle_index, circle in enumerate(
            section["circular_candidates"],
            start=1
        ):

            diameter = safe_float(
                circle.get("diameter")
            )

            radius = safe_float(
                circle.get("radius")
            )

            relative_error = safe_float(
                circle.get("relative_error")
            )

            if diameter <= 0:
                continue

            center_2d = np.asarray(
                circle.get(
                    "center_2d",
                    [0.0, 0.0]
                ),
                dtype=float
            )

            # Use the authoritative XYZ center computed from Trimesh's
            # to_3d transform during section extraction.
            if "center_3d" not in circle:
                continue

            circle_center_3d = np.asarray(
                circle["center_3d"],
                dtype=float
            )

            radial_distance = np.linalg.norm(
                circle_center_3d
                -
                (
                    center
                    +
                    axis
                    *
                    np.dot(
                        circle_center_3d
                        -
                        center,
                        axis
                    )
                )
            )

            features.append(
                {
                    "section_id":
                        section["section_id"],

                    "axis_coordinate":
                        section["axis_coordinate"],

                    "diameter":
                        diameter,

                    "radius":
                        radius,

                    "center":
                        vector_to_list(
                            circle_center_3d
                        ),

                    "radial_distance_from_axis":
                        float(
                            radial_distance
                        ),

                    "relative_error":
                        relative_error,

                    "point_count":
                        int(
                            circle.get(
                                "point_count",
                                0
                            )
                        )
                }
            )

    print(
        f"[OK] Circular observations: "
        f"{len(features)}"
    )

    return features


# ============================================================
# GROUP CIRCULAR OBSERVATIONS
# ============================================================

def group_circular_observations(
    observations
):

    print(
        "\n[7] Grouping circular observations..."
    )

    if not observations:

        return []

    groups = []

    diameter_tolerance = 0.08

    radial_tolerance = 3.0

    coordinate_tolerance = 8.0

    for observation in observations:

        diameter = observation[
            "diameter"
        ]

        radial_distance = observation[
            "radial_distance_from_axis"
        ]

        center = np.asarray(
            observation["center"],
            dtype=float
        )

        assigned = False

        for group in groups:

            mean_diameter = (
                group["mean_diameter"]
            )

            mean_radial = (
                group["mean_radial_distance"]
            )

            mean_center = np.asarray(
                group["mean_center"],
                dtype=float
            )

            diameter_difference = (
                abs(
                    diameter
                    -
                    mean_diameter
                )
                /
                max(
                    mean_diameter,
                    1e-9
                )
            )

            radial_difference = abs(
                radial_distance
                -
                mean_radial
            )

            center_difference = np.linalg.norm(
                center
                -
                mean_center
            )

            if (
                diameter_difference
                <= diameter_tolerance
                and
                radial_difference
                <= radial_tolerance
                and
                center_difference
                <= coordinate_tolerance
            ):

                group["observations"].append(
                    observation
                )

                # Update averages.
                obs = group[
                    "observations"
                ]

                group[
                    "mean_diameter"
                ] = float(
                    np.mean(
                        [
                            x["diameter"]
                            for x in obs
                        ]
                    )
                )

                group[
                    "mean_radial_distance"
                ] = float(
                    np.mean(
                        [
                            x[
                                "radial_distance_from_axis"
                            ]
                            for x in obs
                        ]
                    )
                )

                group[
                    "mean_center"
                ] = vector_to_list(
                    np.mean(
                        [
                            x["center"]
                            for x in obs
                        ],
                        axis=0
                    )
                )

                assigned = True

                break

        if not assigned:

            groups.append(
                {
                    "group_id":
                        len(groups) + 1,

                    "observations":
                        [observation],

                    "mean_diameter":
                        float(diameter),

                    "mean_radial_distance":
                        float(radial_distance),

                    "mean_center":
                        vector_to_list(
                            center
                        )
                }
            )

    # --------------------------------------------------------
    # Convert groups to engineering records.
    # --------------------------------------------------------

    results = []

    for group in groups:

        observations = group[
            "observations"
        ]

        coordinates = [
            x["axis_coordinate"]
            for x in observations
        ]

        diameters = [
            x["diameter"]
            for x in observations
        ]

        results.append(
            {
                "circular_feature_id":
                    f"CIRCULAR_FEATURE_"
                    f"{group['group_id']:03d}",

                "mean_diameter":
                    float(
                        np.mean(
                            diameters
                        )
                    ),

                "min_diameter":
                    float(
                        np.min(
                            diameters
                        )
                    ),

                "max_diameter":
                    float(
                        np.max(
                            diameters
                        )
                    ),

                "axis_coordinate_min":
                    float(
                        np.min(
                            coordinates
                        )
                    ),

                "axis_coordinate_max":
                    float(
                        np.max(
                            coordinates
                        )
                    ),

                "section_observation_count":
                    int(
                        len(observations)
                    ),

                "mean_radial_distance":
                    group[
                        "mean_radial_distance"
                    ],

                "mean_center":
                    group[
                        "mean_center"
                    ],

                "observations":
                    observations
            }
        )

    results.sort(
        key=lambda x: (
            x["mean_radial_distance"],
            x["mean_diameter"]
        )
    )

    print(
        f"[OK] Circular feature groups: "
        f"{len(results)}"
    )

    return results


# ============================================================
# IMPORT STAGE 1 / STAGE 2 RESULTS
# ============================================================

def load_previous_results():

    mechanical = None

    ai_reasoning = None

    if MECHANICAL_EVIDENCE_FILE.exists():

        try:

            with open(
                MECHANICAL_EVIDENCE_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                mechanical = json.load(f)

        except Exception as exc:

            print(
                "[WARNING] Could not read "
                f"mechanical evidence: {exc}"
            )

    if AI_REASONING_FILE.exists():

        try:

            with open(
                AI_REASONING_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                ai_reasoning = json.load(f)

        except Exception as exc:

            print(
                "[WARNING] Could not read "
                f"AI reasoning: {exc}"
            )

    return mechanical, ai_reasoning


# ============================================================
# BUILD CYLINDER SUMMARY
# ============================================================

def build_existing_cylinder_summary(
    mechanical,
    ai_reasoning
):

    cylinders = []

    if (
        isinstance(
            mechanical,
            dict
        )
        and
        isinstance(
            mechanical.get("cylinders"),
            list
        )
    ):

        for index, cylinder in enumerate(
            mechanical["cylinders"],
            start=1
        ):

            record = {

                "feature_id":
                    f"CYLINDER_REGION_"
                    f"{index:03d}",

                "diameter":
                    safe_float(
                        cylinder.get(
                            "diameter"
                        )
                    ),

                "length":
                    safe_float(
                        cylinder.get(
                            "length"
                        )
                    ),

                "physical_score":
                    safe_float(
                        cylinder.get(
                            "physical_score"
                        )
                    ),

                "internal_ratio":
                    safe_float(
                        cylinder.get(
                            "internal_ratio"
                        )
                    ),

                "external_ratio":
                    safe_float(
                        cylinder.get(
                            "external_ratio"
                        )
                    ),

                "axis":
                    cylinder.get(
                        "axis"
                    ),

                "origin":
                    cylinder.get(
                        "origin"
                    ),

                "boundary_loops":
                    cylinder.get(
                        "boundary_loops",
                        []
                    )
            }

            cylinders.append(
                record
            )

    # --------------------------------------------------------
    # Add AI classifications when available.
    # --------------------------------------------------------

    ai_lookup = {}

    if (
        isinstance(
            ai_reasoning,
            dict
        )
        and
        isinstance(
            ai_reasoning.get("features"),
            list
        )
    ):

        for feature in ai_reasoning[
            "features"
        ]:

            if not isinstance(
                feature,
                dict
            ):
                continue

            feature_id = feature.get(
                "feature_id"
            )

            if feature_id:

                ai_lookup[
                    feature_id
                ] = feature

    for cylinder in cylinders:

        ai = ai_lookup.get(
            cylinder["feature_id"]
        )

        if ai:

            decision = ai.get(
                "final_decision",
                {}
            )

            cylinder[
                "ai_classification"
            ] = decision.get(
                "final_classification",
                "UNKNOWN"
            )

            cylinder[
                "ai_subtype"
            ] = decision.get(
                "final_subtype",
                "UNKNOWN"
            )

            cylinder[
                "ai_confidence"
            ] = safe_float(
                decision.get(
                    "confidence"
                )
            )

        else:

            cylinder[
                "ai_classification"
            ] = "UNKNOWN"

            cylinder[
                "ai_subtype"
            ] = "UNKNOWN"

            cylinder[
                "ai_confidence"
            ] = 0.0

    return cylinders


# ============================================================
# BUILD FINAL OUTPUT
# ============================================================

def build_output(
    stl_path,
    mesh,
    axis_data,
    planar_groups,
    sections,
    circular_features,
    cylinders
):

    bounds = np.asarray(
        mesh.bounds,
        dtype=float
    )

    dimensions = (
        bounds[1]
        -
        bounds[0]
    )

    output = {

        "system":
            "OX ALPHA",

        "stage":
            3,

        "stage_name":
            "COMPLETE FEATURE GEOMETRY EXTRACTION",

        "source_stl":
            str(stl_path),

        "mesh": {

            "vertices":
                int(
                    len(mesh.vertices)
                ),

            "triangles":
                int(
                    len(mesh.faces)
                ),

            "watertight":
                bool(
                    mesh.is_watertight
                ),

            "bounds_min":
                vector_to_list(
                    bounds[0]
                ),

            "bounds_max":
                vector_to_list(
                    bounds[1]
                ),

            "overall_dimensions":
                vector_to_list(
                    dimensions
                )
        },

        "principal_geometry": {

            "center":
                vector_to_list(
                    axis_data["center"]
                ),

            "axes":
                [
                    vector_to_list(axis)
                    for axis in axis_data[
                        "axes"
                    ]
                ],

            "extents":
                [
                    float(x)
                    for x in axis_data[
                        "extents"
                    ]
                ],

            "eigenvalues":
                axis_data[
                    "eigenvalues"
                ]
        },

        "planar_surface_inventory":
            planar_groups,

        "principal_axis_sections":
            sections,

        "circular_feature_inventory":
            circular_features,

        "existing_cylinder_features":
            cylinders
    }

    return output


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(output):

    print("\n")
    print("=" * 78)
    print(
        "OX ALPHA - STAGE 3 FEATURE GEOMETRY SUMMARY"
    )
    print("=" * 78)

    dimensions = output[
        "mesh"
    ][
        "overall_dimensions"
    ]

    print(
        "\nOVERALL PART"
    )

    print(
        f"  X extent: {dimensions[0]:.4f}"
    )

    print(
        f"  Y extent: {dimensions[1]:.4f}"
    )

    print(
        f"  Z extent: {dimensions[2]:.4f}"
    )

    principal = output[
        "principal_geometry"
    ]

    print(
        "\nPRINCIPAL AXIS EXTENTS"
    )

    for index, extent in enumerate(
        principal["extents"],
        start=1
    ):

        print(
            f"  Axis {index}: {extent:.4f}"
        )

    print(
        "\nPLANAR SURFACE GROUPS"
    )

    print(
        f"  Groups: "
        f"{len(output['planar_surface_inventory'])}"
    )

    print(
        "\nSECTION ANALYSIS"
    )

    print(
        f"  Valid sections: "
        f"{len(output['principal_axis_sections'])}"
    )

    print(
        "\nCIRCULAR FEATURE INVENTORY"
    )

    circular = output[
        "circular_feature_inventory"
    ]

    print(
        f"  Groups: {len(circular)}"
    )

    for feature in circular:

        print(
            f"  {feature['circular_feature_id']}"
        )

        print(
            f"    Diameter range : "
            f"{feature['min_diameter']:.4f}"
            f" - "
            f"{feature['max_diameter']:.4f}"
        )

        print(
            f"    Mean diameter  : "
            f"{feature['mean_diameter']:.4f}"
        )

        print(
            f"    Radial distance: "
            f"{feature['mean_radial_distance']:.4f}"
        )

        print(
            f"    Sections       : "
            f"{feature['section_observation_count']}"
        )

    print(
        "\nEXISTING STAGE 1/2 CYLINDERS"
    )

    cylinders = output[
        "existing_cylinder_features"
    ]

    print(
        f"  Cylinders: {len(cylinders)}"
    )

    for cylinder in cylinders:

        print(
            f"  {cylinder['feature_id']}"
        )

        print(
            f"    Ø{cylinder['diameter']:.4f}"
        )

        print(
            f"    Length: "
            f"{cylinder['length']:.4f}"
        )

        print(
            f"    AI: "
            f"{cylinder['ai_classification']} / "
            f"{cylinder['ai_subtype']}"
        )

    print()
    print("=" * 78)
    print(
        "STAGE 3 FEATURE GEOMETRY EXTRACTION COMPLETED"
    )
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "OX ALPHA - STAGE 3"
    )

    print(
        "COMPLETE FEATURE GEOMETRY EXTRACTION"
    )

    print("=" * 78)

    try:

        # ----------------------------------------------------
        # 1. STL
        # ----------------------------------------------------

        stl_path = find_stl()

        # ----------------------------------------------------
        # 2. Load
        # ----------------------------------------------------

        mesh = load_mesh(
            stl_path
        )

        # ----------------------------------------------------
        # 3. Principal axes
        # ----------------------------------------------------

        axis_data = (
            determine_principal_axes(
                mesh
            )
        )

        # ----------------------------------------------------
        # 4. Planar inventory
        # ----------------------------------------------------

        planar_groups = (
            analyze_planar_faces(
                mesh
            )
        )

        # ----------------------------------------------------
        # 5. Cross sections
        # ----------------------------------------------------

        sections = (
            extract_sections(
                mesh,
                axis_data
            )
        )

        # ----------------------------------------------------
        # 6. Circular observations
        # ----------------------------------------------------

        circular_observations = (
            summarize_circular_features(
                sections,
                axis_data
            )
        )

        # ----------------------------------------------------
        # 7. Group circular features
        # ----------------------------------------------------

        circular_features = (
            group_circular_observations(
                circular_observations
            )
        )

        # ----------------------------------------------------
        # 8. Previous stages
        # ----------------------------------------------------

        mechanical, ai_reasoning = (
            load_previous_results()
        )

        cylinders = (
            build_existing_cylinder_summary(
                mechanical,
                ai_reasoning
            )
        )

        # ----------------------------------------------------
        # 9. Build output
        # ----------------------------------------------------

        output = build_output(
            stl_path,
            mesh,
            axis_data,
            planar_groups,
            sections,
            circular_features,
            cylinders
        )

        # ----------------------------------------------------
        # 10. Save
        # ----------------------------------------------------

        print(
            "\n[8] Saving Stage 3 evidence..."
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"[OK] Saved:\n{OUTPUT_FILE}"
        )

        # ----------------------------------------------------
        # 11. Summary
        # ----------------------------------------------------

        print_summary(
            output
        )

    except KeyboardInterrupt:

        print(
            "\n[STOPPED] User interrupted."
        )

    except Exception as exc:

        print()
        print("=" * 78)
        print(
            "STAGE 3 FEATURE EXTRACTION ERROR"
        )
        print("=" * 78)
        print()
        print(
            f"[ERROR] {exc}"
        )
        print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()