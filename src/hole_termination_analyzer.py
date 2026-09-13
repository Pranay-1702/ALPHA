from pathlib import Path
import json
import math

import numpy as np
import trimesh


# ============================================================
# ALPHA
# STAGE 5 - HOLE TERMINATION / DEPTH ANALYSIS
#
# Purpose:
#   Analyze the actual STL around each consolidated circular
#   feature and determine whether the feature is:
#
#       THROUGH_HOLE
#       BLIND_HOLE
#       INTERNAL_CYLINDRICAL_FEATURE
#       TAPERED_INTERNAL_FEATURE
#       UNKNOWN
#
# IMPORTANT:
#   - No standard drill sizes are assumed.
#   - No dimensions are invented.
#   - No Gemini call is used here.
#   - Classification is based on actual STL cross-sections.
#
# Inputs:
#   input/test_part.stl
#   output/consolidated_feature_geometry.json
#
# Output:
#   output/hole_termination_analysis.json
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT / "input"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "output"
)

STL_FILE = (
    INPUT_DIR / "test_part.stl"
)

CONSOLIDATED_FILE = (
    OUTPUT_DIR /
    "consolidated_feature_geometry.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "hole_termination_analysis.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of samples through the complete part thickness.

GLOBAL_SAMPLES = 41


# Additional samples around the observed feature range.

LOCAL_SAMPLES = 15


# Maximum distance between the detected section-loop center
# and the physical feature axis.

CENTER_TOLERANCE = 3.0


# Diameter compatibility.

DIAMETER_TOLERANCE_RATIO = 0.30


# Minimum equivalent loop area.

MIN_LOOP_AREA = 0.01


# A feature needs sufficient valid sections before we make
# a termination interpretation.

MIN_VALID_SECTIONS = 3


# If a circular section exists close to both ends of the
# global part thickness, it is strong evidence for a through
# feature.

END_MARGIN_RATIO = 0.08


# ============================================================
# UTILITIES
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:
        return float(value)

    except (
        TypeError,
        ValueError
    ):
        return default


def safe_int(
    value,
    default=0
):

    try:
        return int(value)

    except (
        TypeError,
        ValueError
    ):
        return default


def normalize(
    vector
):

    vector = np.asarray(
        vector,
        dtype=float
    )

    length = np.linalg.norm(
        vector
    )

    if length < 1e-12:

        return None

    return vector / length


def vector_to_list(
    vector
):

    return [
        float(x)
        for x in np.asarray(
            vector,
            dtype=float
        )
    ]


def angle_difference_deg(
    a,
    b
):

    difference = (
        abs(a - b)
        % 360.0
    )

    if difference > 180.0:

        difference = (
            360.0 -
            difference
        )

    return difference


# ============================================================
# BASIS
# ============================================================

def build_basis(
    axis
):

    axis = normalize(
        axis
    )

    reference = np.array(
        [1.0, 0.0, 0.0],
        dtype=float
    )

    if abs(
        np.dot(
            reference,
            axis
        )
    ) > 0.90:

        reference = np.array(
            [0.0, 1.0, 0.0],
            dtype=float
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
# LOAD STL
# ============================================================

def load_mesh():

    print(
        "[1] Loading STL mesh..."
    )

    if not STL_FILE.exists():

        raise FileNotFoundError(
            f"STL file not found:\n{STL_FILE}"
        )

    mesh = trimesh.load_mesh(
        str(STL_FILE),
        process=False
    )

    if isinstance(
        mesh,
        trimesh.Scene
    ):

        if not mesh.geometry:

            raise ValueError(
                "STL scene contains no geometry."
            )

        mesh = trimesh.util.concatenate(
            tuple(
                mesh.geometry.values()
            )
        )

    print(
        f"[OK] Raw triangles: "
        f"{len(mesh.faces):,}"
    )

    # --------------------------------------------------------
    # Clean duplicate vertices so section operations behave
    # consistently.
    # --------------------------------------------------------

    try:

        mesh.merge_vertices()

    except Exception as exc:

        print(
            f"[WARNING] merge_vertices failed: {exc}"
        )

    print(
        f"[OK] Clean vertices: "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"[OK] Watertight: "
        f"{mesh.is_watertight}"
    )

    return mesh


# ============================================================
# LOAD CONSOLIDATED DATA
# ============================================================

def load_consolidated():

    print(
        "\n[2] Loading consolidated physical features..."
    )

    if not CONSOLIDATED_FILE.exists():

        raise FileNotFoundError(
            f"Consolidated geometry not found:\n"
            f"{CONSOLIDATED_FILE}"
        )

    with open(
        CONSOLIDATED_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    if not isinstance(
        data,
        dict
    ):

        raise ValueError(
            "Consolidated geometry JSON is invalid."
        )

    features = data.get(
        "features",
        []
    )

    if not isinstance(
        features,
        list
    ):

        raise ValueError(
            "No features found in consolidated geometry."
        )

    principal = data.get(
        "principal_geometry",
        {}
    )

    center = np.asarray(
        principal.get(
            "center",
            [0.0, 0.0, 0.0]
        ),
        dtype=float
    )

    axis = normalize(
        principal.get(
            "axis",
            [0.0, 0.0, 1.0]
        )
    )

    if axis is None:

        raise ValueError(
            "Principal axis is invalid."
        )

    print(
        f"[OK] Physical features: "
        f"{len(features)}"
    )

    print(
        f"[OK] Principal axis: "
        f"{vector_to_list(axis)}"
    )

    return (
        data,
        features,
        center,
        axis
    )


# ============================================================
# PROJECT POINT TO AXIS
# ============================================================

def axis_coordinate(
    point,
    axis_origin,
    axis
):

    relative = (
        np.asarray(
            point,
            dtype=float
        )
        -
        axis_origin
    )

    return float(
        np.dot(
            relative,
            axis
        )
    )


def point_at_axis_coordinate(
    axis_origin,
    axis,
    coordinate
):

    return (
        axis_origin
        +
        axis * coordinate
    )


# ============================================================
# GLOBAL AXIAL EXTENTS
# ============================================================

def get_mesh_axial_extent(
    mesh,
    axis_origin,
    axis
):

    coordinates = np.dot(
        mesh.vertices -
        axis_origin,
        axis
    )

    return (
        float(
            coordinates.min()
        ),
        float(
            coordinates.max()
        )
    )


# ============================================================
# CROSS-SECTION LOOP EXTRACTION
# ============================================================

def section_loops(
    mesh,
    plane_origin,
    plane_normal
):

    try:

        section = mesh.section(
            plane_origin=plane_origin,
            plane_normal=plane_normal
        )

    except Exception:

        return []

    if section is None:

        return []

    try:

        # Trimesh returns both the planar path and the exact
        # planar -> original-3D transform.  Use this transform
        # instead of reconstructing the 3D coordinates with a
        # separately-created basis.  This keeps the section loop
        # coordinates in the same STL coordinate system as the
        # Stage 3B physical_axis_point values.
        planar, transform = section.to_2D()

    except Exception:

        return []

    loops = []

    try:

        entities = planar.entities

        vertices = planar.vertices

    except Exception:

        return []

    # --------------------------------------------------------
    # Convert each closed path into a 2D polygon.
    # --------------------------------------------------------

    for entity in entities:

        try:

            points = np.asarray(
                entity.points,
                dtype=int
            )

        except Exception:

            continue

        if len(points) < 8:

            continue

        try:

            coords = np.asarray(
                vertices[points],
                dtype=float
            )

        except Exception:

            continue

        if len(coords) < 8:

            continue

        # ----------------------------------------------------
        # Remove consecutive duplicates.
        # ----------------------------------------------------

        cleaned = [coords[0]]

        for point in coords[1:]:

            if np.linalg.norm(
                point -
                cleaned[-1]
            ) > 1e-8:

                cleaned.append(
                    point
                )

        coords = np.asarray(
            cleaned,
            dtype=float
        )

        if len(coords) < 8:

            continue

        # ----------------------------------------------------
        # Shoelace area.
        # ----------------------------------------------------

        x = coords[:, 0]
        y = coords[:, 1]

        area = 0.5 * abs(
            np.sum(
                x * np.roll(y, -1)
                -
                y * np.roll(x, -1)
            )
        )

        if area < MIN_LOOP_AREA:

            continue

        centroid = (
            coords.mean(
                axis=0
            )
        )

        # ----------------------------------------------------
        # Equivalent diameter.
        #
        # This is only used for matching a section loop.
        # ----------------------------------------------------

        equivalent_diameter = (
            2.0 *
            math.sqrt(
                area /
                math.pi
            )
        )

        # --------------------------------------------------------
        # Transform the loop centroid back into the original STL
        # coordinate system using Trimesh's authoritative transform.
        # --------------------------------------------------------

        try:

            centroid_3d = trimesh.transform_points(
                np.asarray([centroid], dtype=float),
                transform
            )[0]

        except Exception:

            continue

        loops.append(
            {

                "area":
                    float(area),

                "centroid_2d":
                    [
                        float(x)
                        for x in centroid
                    ],

                "centroid_3d":
                    vector_to_list(
                        centroid_3d
                    ),

                "equivalent_diameter":
                    float(
                        equivalent_diameter
                    ),

                "point_count":
                    len(coords)
            }
        )

    return loops


# ============================================================
# LOOP CENTER / RADIUS IN 3D
# ============================================================

def loop_center_3d(
    loop,
    plane_origin,
    plane_normal
):

    # The 3D centroid was already computed from Trimesh's exact
    # section.to_2D() transform inside section_loops().  Reuse it
    # directly so the matching coordinate system is identical to
    # the original STL and Stage 3B geometry evidence.

    center = loop.get(
        "centroid_3d"
    )

    if center is not None:

        return np.asarray(
            center,
            dtype=float
        )

    # Defensive fallback for legacy loop records.
    # This path should not be used for newly generated sections.
    try:

        transform = loop.get(
            "transform"
        )

        if transform is not None:

            c = np.asarray(
                loop["centroid_2d"],
                dtype=float
            )

            return trimesh.transform_points(
                np.asarray([c], dtype=float),
                np.asarray(transform, dtype=float)
            )[0]

    except Exception:
        pass

    # Final defensive fallback.
    u, v = build_basis(
        plane_normal
    )

    c = np.asarray(
        loop[
            "centroid_2d"
        ],
        dtype=float
    )

    return (
        np.asarray(
            plane_origin,
            dtype=float
        )
        +
        u * c[0]
        +
        v * c[1]
    )


# ============================================================
# FIND FEATURE LOOP
# ============================================================

def find_matching_loop(
    loops,
    plane_origin,
    plane_normal,
    feature_axis_point,
    feature_diameter
):

    if not loops:

        return None

    best = None
    best_score = float(
        "inf"
    )

    for loop in loops:

        center3d = loop_center_3d(
            loop,
            plane_origin,
            plane_normal
        )

        center_distance = float(
            np.linalg.norm(
                center3d -
                feature_axis_point
            )
        )

        diameter = (
            loop[
                "equivalent_diameter"
            ]
        )

        diameter_ratio = (
            abs(
                diameter -
                feature_diameter
            )
            /
            max(
                feature_diameter,
                1e-9
            )
        )

        if (
            center_distance
            >
            CENTER_TOLERANCE
        ):

            continue

        if (
            diameter_ratio
            >
            DIAMETER_TOLERANCE_RATIO
        ):

            continue

        # ----------------------------------------------------
        # Lower is better.
        # ----------------------------------------------------

        score = (
            center_distance
            +
            diameter_ratio *
            feature_diameter
        )

        if score < best_score:

            best_score = score

            best = {

                "loop":
                    loop,

                "center_3d":
                    vector_to_list(
                        center3d
                    ),

                "center_distance":
                    center_distance,

                "diameter_ratio":
                    diameter_ratio
            }

    return best


# ============================================================
# ANALYZE ONE FEATURE
# ============================================================

def analyze_feature(
    mesh,
    feature,
    axis_origin,
    principal_axis,
    global_min,
    global_max
):

    feature_id = feature.get(
        "feature_id",
        "UNKNOWN"
    )

    feature_axis_point = np.asarray(
        feature.get(
            "physical_axis_point",
            axis_origin
        ),
        dtype=float
    )

    feature_diameter = safe_float(
        feature.get(
            "diameter"
        )
    )

    if feature_diameter <= 0:

        return {

            "feature_id":
                feature_id,

            "classification":
                "UNKNOWN",

            "reason":
                "Invalid feature diameter."
        }

    feature_center_coordinate = (
        axis_coordinate(
            feature_axis_point,
            axis_origin,
            principal_axis
        )
    )

    axial_range = feature.get(
        "axial_range",
        {}
    )

    observed_min = safe_float(
        axial_range.get(
            "min"
        ),
        feature_center_coordinate
    )

    observed_max = safe_float(
        axial_range.get(
            "max"
        ),
        feature_center_coordinate
    )

    # --------------------------------------------------------
    # Construct sampling range.
    #
    # First cover the complete part.
    # Then increase density around the observed feature.
    # --------------------------------------------------------

    global_samples = np.linspace(
        global_min,
        global_max,
        GLOBAL_SAMPLES
    )

    local_samples = np.linspace(
        observed_min - 2.0,
        observed_max + 2.0,
        LOCAL_SAMPLES
    )

    coordinates = np.unique(
        np.concatenate(
            [
                global_samples,
                local_samples
            ]
        )
    )

    observations = []

    print(
        f"\n  Analyzing {feature_id}..."
    )

    for coordinate in coordinates:

        plane_origin = (
            point_at_axis_coordinate(
                axis_origin,
                principal_axis,
                coordinate
            )
        )

        loops = section_loops(
            mesh,
            plane_origin,
            principal_axis
        )

        match = find_matching_loop(
            loops,
            plane_origin,
            principal_axis,
            feature_axis_point,
            feature_diameter
        )

        if match is None:

            observations.append(
                {

                    "axis_coordinate":
                        float(coordinate),

                    "detected":
                        False
                }
            )

            continue

        observations.append(
            {

                "axis_coordinate":
                    float(coordinate),

                "detected":
                    True,

                "diameter":
                    match[
                        "loop"
                    ][
                        "equivalent_diameter"
                    ],

                "center_distance":
                    match[
                        "center_distance"
                    ],

                "diameter_error_ratio":
                    match[
                        "diameter_ratio"
                    ]
            }
        )

    # ========================================================
    # VALID DETECTIONS
    # ========================================================

    detected = [
        x
        for x in observations
        if x["detected"]
    ]

    if len(detected) < MIN_VALID_SECTIONS:

        return {

            "feature_id":
                feature_id,

            "classification":
                "UNKNOWN",

            "subtype":
                "UNKNOWN",

            "reason":
                "Insufficient cross-section evidence.",

            "detected_sections":
                len(detected),

            "total_samples":
                len(observations),

            "observations":
                observations
        }

    detected_coordinates = np.asarray(
        [
            x[
                "axis_coordinate"
            ]
            for x in detected
        ],
        dtype=float
    )

    detected_diameters = np.asarray(
        [
            x[
                "diameter"
            ]
            for x in detected
        ],
        dtype=float
    )

    detected_min = float(
        detected_coordinates.min()
    )

    detected_max = float(
        detected_coordinates.max()
    )

    detected_length = (
        detected_max -
        detected_min
    )

    # ========================================================
    # PART THICKNESS
    # ========================================================

    global_length = (
        global_max -
        global_min
    )

    if global_length <= 0:

        global_length = 1.0

    # ========================================================
    # DISTANCE FROM PART ENDS
    # ========================================================

    distance_to_lower_end = (
        detected_min -
        global_min
    )

    distance_to_upper_end = (
        global_max -
        detected_max
    )

    lower_end_ratio = (
        distance_to_lower_end /
        global_length
    )

    upper_end_ratio = (
        distance_to_upper_end /
        global_length
    )

    reaches_lower_end = (
        lower_end_ratio
        <=
        END_MARGIN_RATIO
    )

    reaches_upper_end = (
        upper_end_ratio
        <=
        END_MARGIN_RATIO
    )

    # ========================================================
    # DIAMETER PROFILE
    # ========================================================

    measured_mean_diameter = float(
        detected_diameters.mean()
    )

    measured_min_diameter = float(
        detected_diameters.min()
    )

    measured_max_diameter = float(
        detected_diameters.max()
    )

    diameter_variation = (
        (
            measured_max_diameter
            -
            measured_min_diameter
        )
        /
        max(
            measured_mean_diameter,
            1e-9
        )
    )

    if diameter_variation <= 0.05:

        profile = (
            "CONSTANT"
        )

    elif diameter_variation <= 0.15:

        profile = (
            "MILDLY_VARIABLE"
        )

    else:

        profile = (
            "TAPERED_OR_STEPPED"
        )

    # ========================================================
    # OBSERVED FEATURE RANGE
    # ========================================================

    observed_feature_length = (
        observed_max -
        observed_min
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    classification = (
        "UNKNOWN"
    )

    subtype = (
        "UNKNOWN"
    )

    reason = (
        "Cross-section evidence is insufficient "
        "for a reliable termination classification."
    )

    confidence = 0.0

    # --------------------------------------------------------
    # THROUGH HOLE
    #
    # Strong condition:
    #
    # The circular feature reaches both ends of the actual
    # part along the principal axis.
    # --------------------------------------------------------

    if (
        reaches_lower_end
        and
        reaches_upper_end
        and
        len(detected)
        >=
        MIN_VALID_SECTIONS
    ):

        classification = (
            "HOLE"
        )

        subtype = (
            "THROUGH_HOLE"
        )

        reason = (
            "The circular section is detected close to "
            "both axial ends of the part, supporting a "
            "through-hole interpretation."
        )

        confidence = 0.95

    # --------------------------------------------------------
    # BLIND HOLE
    #
    # Feature begins/ends internally rather than extending
    # across the entire part.
    #
    # We deliberately do not require the opening to be at
    # exactly one global part end because a blind hole can
    # open onto a recessed planar surface.
    # --------------------------------------------------------

    elif (
        detected_length > 0
        and
        observed_feature_length > 0
        and
        detected_length
        >=
        observed_feature_length * 0.70
    ):

        # ----------------------------------------------------
        # Determine whether the feature terminates internally.
        # ----------------------------------------------------

        internally_terminated = (
            not (
                reaches_lower_end
                and
                reaches_upper_end
            )
        )

        if internally_terminated:

            classification = (
                "HOLE"
            )

            subtype = (
                "BLIND_HOLE_CANDIDATE"
            )

            reason = (
                "The circular section persists through the "
                "observed feature range but terminates "
                "inside the overall part extent. This is "
                "consistent with a blind/internal hole, "
                "but the exact opening and bottom surfaces "
                "require additional local surface evidence."
            )

            confidence = 0.78

    # --------------------------------------------------------
    # INTERNAL CYLINDRICAL FEATURE
    # --------------------------------------------------------

    if classification == "UNKNOWN":

        classification = (
            "INTERNAL_CIRCULAR_FEATURE"
        )

        if profile == "CONSTANT":

            subtype = (
                "CYLINDRICAL_INTERNAL_FEATURE"
            )

        else:

            subtype = (
                "VARIABLE_DIAMETER_INTERNAL_FEATURE"
            )

        reason = (
            "A persistent circular cross-section was detected "
            "around the feature axis, but the available global "
            "section evidence does not prove a through-hole or "
            "blind-hole termination."
        )

        confidence = 0.65

    # ========================================================
    # CONFIDENCE ADJUSTMENTS
    # ========================================================

    # More detected samples improve confidence.

    sample_factor = min(
        len(detected) /
        10.0,
        1.0
    )

    confidence += (
        sample_factor *
        0.05
    )

    # Strong diameter stability improves geometric confidence.

    if diameter_variation <= 0.05:

        confidence += 0.03

    # Excessive diameter variation reduces confidence in a
    # simple cylindrical interpretation.

    if diameter_variation > 0.20:

        confidence -= 0.08

    confidence = min(
        max(
            confidence,
            0.0
        ),
        0.99
    )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "feature_id":
            feature_id,

        "classification":
            classification,

        "subtype":
            subtype,

        "confidence":
            float(confidence),

        "reason":
            reason,

        "physical_axis_point":
            feature.get(
                "physical_axis_point"
            ),

        "radial_distance":
            feature.get(
                "radial_distance"
            ),

        "angular_position_deg":
            feature.get(
                "angular_position_deg"
            ),

        "input_diameter":
            feature_diameter,

        "measured_section_diameter":
            measured_mean_diameter,

        "measured_diameter_min":
            measured_min_diameter,

        "measured_diameter_max":
            measured_max_diameter,

        "diameter_variation_ratio":
            float(
                diameter_variation
            ),

        "diameter_profile":
            profile,

        "part_axial_range":
            {

                "min":
                    global_min,

                "max":
                    global_max,

                "length":
                    global_length
            },

        "detected_axial_range":
            {

                "min":
                    detected_min,

                "max":
                    detected_max,

                "length":
                    detected_length
            },

        "observed_feature_range":
            {

                "min":
                    observed_min,

                "max":
                    observed_max,

                "length":
                    observed_feature_length
            },

        "distance_to_part_lower_end":
            distance_to_lower_end,

        "distance_to_part_upper_end":
            distance_to_upper_end,

        "reaches_lower_part_end":
            reaches_lower_end,

        "reaches_upper_part_end":
            reaches_upper_end,

        "detected_sections":
            len(detected),

        "total_samples":
            len(observations),

        "observations":
            observations
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    result
):

    print()
    print(
        result[
            "feature_id"
        ]
    )

    print(
        f"  Classification    : "
        f"{result.get('classification')}"
    )

    print(
        f"  Subtype           : "
        f"{result.get('subtype')}"
    )

    print(
        f"  Confidence        : "
        f"{safe_float(result.get('confidence')):.3f}"
    )

    print(
        f"  Input diameter    : "
        f"{safe_float(result.get('input_diameter')):.4f}"
    )

    print(
        f"  Measured diameter : "
        f"{safe_float(result.get('measured_section_diameter')):.4f}"
    )

    print(
        f"  Diameter profile  : "
        f"{result.get('diameter_profile')}"
    )

    print(
        f"  Detected sections : "
        f"{result.get('detected_sections')}"
    )

    detected_range = result.get(
        "detected_axial_range",
        {}
    )

    print(
        f"  Detected range    : "
        f"{safe_float(detected_range.get('min')):.4f}"
        f" - "
        f"{safe_float(detected_range.get('max')):.4f}"
    )

    print(
        f"  Detected length   : "
        f"{safe_float(detected_range.get('length')):.4f}"
    )

    print(
        f"  Lower part end    : "
        f"{result.get('reaches_lower_part_end')}"
    )

    print(
        f"  Upper part end    : "
        f"{result.get('reaches_upper_part_end')}"
    )

    print(
        f"  Reason            : "
        f"{result.get('reason')}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)

    print(
        "ALPHA - STAGE 5"
    )

    print(
        "HOLE TERMINATION / DEPTH ANALYSIS"
    )

    print("=" * 78)

    try:

        # ----------------------------------------------------
        # 1. Load mesh
        # ----------------------------------------------------

        mesh = load_mesh()

        # ----------------------------------------------------
        # 2. Load consolidated geometry
        # ----------------------------------------------------

        (
            consolidated,
            features,
            axis_origin,
            principal_axis
        ) = load_consolidated()

        # ----------------------------------------------------
        # 3. Determine complete part axial range
        # ----------------------------------------------------

        (
            global_min,
            global_max
        ) = get_mesh_axial_extent(
            mesh,
            axis_origin,
            principal_axis
        )

        print(
            "\n[3] Complete part axial extent:"
        )

        print(
            f"    {global_min:.4f}"
            f" -> "
            f"{global_max:.4f}"
        )

        print(
            f"    Length: "
            f"{global_max - global_min:.4f}"
        )

        # ----------------------------------------------------
        # 4. Analyze each feature
        # ----------------------------------------------------

        print(
            "\n[4] Analyzing feature terminations..."
        )

        results = []

        for feature in features:

            result = analyze_feature(
                mesh,
                feature,
                axis_origin,
                principal_axis,
                global_min,
                global_max
            )

            results.append(
                result
            )

            print_result(
                result
            )

        # ----------------------------------------------------
        # 5. Summary
        # ----------------------------------------------------

        through_count = sum(
            1
            for x in results
            if x.get(
                "subtype"
            )
            ==
            "THROUGH_HOLE"
        )

        blind_count = sum(
            1
            for x in results
            if x.get(
                "subtype"
            )
            ==
            "BLIND_HOLE_CANDIDATE"
        )

        internal_count = sum(
            1
            for x in results
            if x.get(
                "classification"
            )
            ==
            "INTERNAL_CIRCULAR_FEATURE"
        )

        # ----------------------------------------------------
        # 6. Build output
        # ----------------------------------------------------

        output = {

            "system":
                "ALPHA",

            "stage":
                "5",

            "stage_name":
                "HOLE TERMINATION / DEPTH ANALYSIS",

            "source_stl":
                str(
                    STL_FILE
                ),

            "source_geometry":
                str(
                    CONSOLIDATED_FILE
                ),

            "analysis_method":
                "Direct STL cross-section analysis along "
                "the principal mechanical axis.",

            "important_rule":
                "No standard hole sizes or unmeasured "
                "dimensions are assumed.",

            "principal_axis":
                vector_to_list(
                    principal_axis
                ),

            "part_axial_range":
                {

                    "min":
                        global_min,

                    "max":
                        global_max,

                    "length":
                        global_max -
                        global_min
                },

            "feature_count":
                len(results),

            "through_hole_count":
                through_count,

            "blind_hole_candidate_count":
                blind_count,

            "internal_circular_feature_count":
                internal_count,

            "features":
                results
        }

        # ----------------------------------------------------
        # 7. Save
        # ----------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False
            )

        print()
        print("=" * 78)

        print(
            f"[OK] Saved:\n"
            f"{OUTPUT_FILE}"
        )

        print()
        print(
            f"Features analyzed        : "
            f"{len(results)}"
        )

        print(
            f"Through-hole candidates  : "
            f"{through_count}"
        )

        print(
            f"Blind-hole candidates   : "
            f"{blind_count}"
        )

        print(
            f"Internal circular        : "
            f"{internal_count}"
        )

        print()
        print(
            "[SUCCESS] Stage 5 completed."
        )

        print("=" * 78)

    except KeyboardInterrupt:

        print(
            "\n[STOPPED] User interrupted."
        )

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "STAGE 5 ERROR"
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