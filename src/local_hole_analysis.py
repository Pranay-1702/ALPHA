from pathlib import Path
import json
import math

import numpy as np
import trimesh


# ============================================================
# ALPHA
# STAGE 5B
# LOCAL OPENING / BOTTOM / DEPTH ANALYSIS
#
# This stage corrects the previous Stage-5 matching problem.
#
# IMPORTANT:
#   A section loop is matched to the FEATURE AXIS LINE.
#   Axial position is NOT included in the radial matching
#   tolerance.
#
# INPUT:
#   input/test_part.stl
#   output/consolidated_feature_geometry.json
#
# OUTPUT:
#   output/local_hole_analysis.json
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

STL_FILE = INPUT_DIR / "test_part.stl"

CONSOLIDATED_FILE = (
    OUTPUT_DIR / "consolidated_feature_geometry.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "local_hole_analysis.json"
)


# ============================================================
# SETTINGS
# ============================================================

# Dense sampling along the observed feature region.
LOCAL_SAMPLES = 101

# Extra axial margin around the observed region.
AXIAL_MARGIN = 1.5

# Distance of section-loop center from the feature axis.
AXIS_RADIAL_TOLERANCE = 3.0

# Diameter compatibility.
DIAMETER_TOLERANCE_RATIO = 0.30

# Minimum valid section observations.
MIN_VALID_SECTIONS = 5

# Minimum fraction of local samples that should contain
# the circular feature before treating it as a continuous wall.
MIN_CONTINUITY_RATIO = 0.45

# Diameter variation limits.
CONSTANT_DIAMETER_RATIO = 0.05
TAPERED_DIAMETER_RATIO = 0.15


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize(vector):
    vector = np.asarray(vector, dtype=float)

    length = np.linalg.norm(vector)

    if length < 1e-12:
        return None

    return vector / length


def vector_to_list(vector):
    return [
        float(x)
        for x in np.asarray(vector, dtype=float)
    ]


# ============================================================
# AXIS GEOMETRY
# ============================================================

def point_to_axis_distance(
    point,
    axis_point,
    axis_direction
):
    """
    Perpendicular distance from a 3D point to an infinite axis line.
    """

    point = np.asarray(point, dtype=float)
    axis_point = np.asarray(axis_point, dtype=float)
    axis_direction = normalize(axis_direction)

    relative = point - axis_point

    axial_component = (
        np.dot(relative, axis_direction)
        * axis_direction
    )

    radial_component = (
        relative - axial_component
    )

    return float(
        np.linalg.norm(radial_component)
    )


def axis_coordinate(
    point,
    axis_origin,
    axis
):
    """
    Signed coordinate along principal axis.
    """

    relative = (
        np.asarray(point, dtype=float)
        -
        np.asarray(axis_origin, dtype=float)
    )

    return float(
        np.dot(relative, axis)
    )


def axis_point_at(
    axis_origin,
    axis,
    coordinate
):
    return (
        np.asarray(axis_origin, dtype=float)
        +
        np.asarray(axis, dtype=float) * coordinate
    )


# ============================================================
# BASIS
# ============================================================

def build_basis(axis):

    axis = normalize(axis)

    reference = np.array(
        [1.0, 0.0, 0.0],
        dtype=float
    )

    if abs(np.dot(reference, axis)) > 0.90:

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
        "[1] Loading STL..."
    )

    if not STL_FILE.exists():

        raise FileNotFoundError(
            f"STL not found:\n{STL_FILE}"
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

    try:

        mesh.merge_vertices()

    except Exception as exc:

        print(
            f"[WARNING] merge_vertices failed: {exc}"
        )

    print(
        f"[OK] Triangles: {len(mesh.faces):,}"
    )

    print(
        f"[OK] Vertices : {len(mesh.vertices):,}"
    )

    print(
        f"[OK] Watertight: {mesh.is_watertight}"
    )

    return mesh


# ============================================================
# LOAD CONSOLIDATED FEATURES
# ============================================================

def load_consolidated():

    print(
        "\n[2] Loading consolidated features..."
    )

    if not CONSOLIDATED_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{CONSOLIDATED_FILE}"
        )

    with open(
        CONSOLIDATED_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    features = data.get(
        "features",
        []
    )

    if not isinstance(features, list):

        raise ValueError(
            "Consolidated feature list is invalid."
        )

    principal_geometry = data.get(
        "principal_geometry",
        {}
    )

    principal_axis = normalize(
        principal_geometry.get(
            "axis",
            [0.0, 0.0, 1.0]
        )
    )

    axis_origin = np.asarray(
        principal_geometry.get(
            "center",
            [0.0, 0.0, 0.0]
        ),
        dtype=float
    )

    if principal_axis is None:

        raise ValueError(
            "Principal axis is invalid."
        )

    print(
        f"[OK] Features: {len(features)}"
    )

    print(
        f"[OK] Axis: "
        f"{vector_to_list(principal_axis)}"
    )

    return (
        data,
        features,
        axis_origin,
        principal_axis
    )


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_section_loops(
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

        planar, _ = section.to_2D()

    except Exception:

        return []

    try:

        entities = planar.entities
        vertices = planar.vertices

    except Exception:

        return []

    loops = []

    for entity in entities:

        try:

            indices = np.asarray(
                entity.points,
                dtype=int
            )

        except Exception:

            continue

        if len(indices) < 8:

            continue

        try:

            points = np.asarray(
                vertices[indices],
                dtype=float
            )

        except Exception:

            continue

        if len(points) < 8:

            continue

        # ----------------------------------------------------
        # Remove consecutive duplicates.
        # ----------------------------------------------------

        cleaned = [points[0]]

        for point in points[1:]:

            if np.linalg.norm(
                point - cleaned[-1]
            ) > 1e-8:

                cleaned.append(point)

        points = np.asarray(
            cleaned,
            dtype=float
        )

        if len(points) < 8:

            continue

        x = points[:, 0]
        y = points[:, 1]

        area = 0.5 * abs(
            np.sum(
                x * np.roll(y, -1)
                -
                y * np.roll(x, -1)
            )
        )

        if area <= 0.01:

            continue

        centroid = points.mean(
            axis=0
        )

        diameter = (
            2.0 *
            math.sqrt(
                area /
                math.pi
            )
        )

        loops.append(
            {
                "area": float(area),
                "centroid_2d": [
                    float(x)
                    for x in centroid
                ],
                "diameter": float(diameter),
                "point_count": int(len(points))
            }
        )

    return loops


# ============================================================
# LOOP CENTER IN 3D
# ============================================================

def loop_center_3d(
    loop,
    plane_origin,
    plane_normal
):

    u, v = build_basis(
        plane_normal
    )

    center2d = np.asarray(
        loop["centroid_2d"],
        dtype=float
    )

    return (
        np.asarray(
            plane_origin,
            dtype=float
        )
        +
        u * center2d[0]
        +
        v * center2d[1]
    )


# ============================================================
# FIND LOOP MATCHING FEATURE AXIS
# ============================================================

def find_axis_loop(
    loops,
    plane_origin,
    plane_normal,
    feature_axis_point,
    feature_axis,
    target_diameter
):

    if not loops:

        return None

    candidates = []

    for loop in loops:

        center = loop_center_3d(
            loop,
            plane_origin,
            plane_normal
        )

        radial_distance = (
            point_to_axis_distance(
                center,
                feature_axis_point,
                feature_axis
            )
        )

        diameter = safe_float(
            loop["diameter"]
        )

        if target_diameter <= 0:

            continue

        diameter_error = (
            abs(
                diameter -
                target_diameter
            )
            /
            target_diameter
        )

        if (
            radial_distance
            >
            AXIS_RADIAL_TOLERANCE
        ):

            continue

        if (
            diameter_error
            >
            DIAMETER_TOLERANCE_RATIO
        ):

            continue

        score = (
            radial_distance
            +
            diameter_error *
            target_diameter
        )

        candidates.append(
            (
                score,
                loop,
                center,
                radial_distance,
                diameter_error
            )
        )

    if not candidates:

        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    (
        score,
        loop,
        center,
        radial_distance,
        diameter_error
    ) = candidates[0]

    return {
        "score": float(score),
        "diameter": float(
            loop["diameter"]
        ),
        "center": vector_to_list(
            center
        ),
        "radial_distance": float(
            radial_distance
        ),
        "diameter_error_ratio": float(
            diameter_error
        )
    }


# ============================================================
# GET FEATURE RANGE
# ============================================================

def feature_axial_range(
    feature,
    axis_origin,
    axis
):

    axis_point = np.asarray(
        feature.get(
            "physical_axis_point",
            axis_origin
        ),
        dtype=float
    )

    feature_center = axis_coordinate(
        axis_point,
        axis_origin,
        axis
    )

    axial_range = feature.get(
        "axial_range",
        {}
    )

    minimum = axial_range.get(
        "min"
    )

    maximum = axial_range.get(
        "max"
    )

    if minimum is None:

        minimum = feature_center - 3.0

    if maximum is None:

        maximum = feature_center + 3.0

    minimum = safe_float(
        minimum
    )

    maximum = safe_float(
        maximum
    )

    if maximum < minimum:

        minimum, maximum = (
            maximum,
            minimum
        )

    return (
        axis_point,
        minimum,
        maximum
    )


# ============================================================
# ANALYZE ONE FEATURE
# ============================================================

def analyze_feature(
    mesh,
    feature,
    axis_origin,
    axis,
    global_min,
    global_max
):

    feature_id = feature.get(
        "feature_id",
        "UNKNOWN"
    )

    target_diameter = safe_float(
        feature.get(
            "diameter"
        )
    )

    if target_diameter <= 0:

        return {
            "feature_id": feature_id,
            "classification": "UNKNOWN",
            "subtype": "UNKNOWN",
            "confidence": 0.0,
            "reason": "Invalid diameter."
        }

    (
        feature_axis_point,
        observed_min,
        observed_max
    ) = feature_axial_range(
        feature,
        axis_origin,
        axis
    )

    # --------------------------------------------------------
    # Expand slightly around known feature range.
    # --------------------------------------------------------

    sample_min = max(
        global_min,
        observed_min - AXIAL_MARGIN
    )

    sample_max = min(
        global_max,
        observed_max + AXIAL_MARGIN
    )

    if sample_max <= sample_min:

        return {
            "feature_id": feature_id,
            "classification": "UNKNOWN",
            "subtype": "UNKNOWN",
            "confidence": 0.0,
            "reason": "Invalid axial range."
        }

    coordinates = np.linspace(
        sample_min,
        sample_max,
        LOCAL_SAMPLES
    )

    observations = []

    for coordinate in coordinates:

        plane_origin = axis_point_at(
            axis_origin,
            axis,
            coordinate
        )

        loops = extract_section_loops(
            mesh,
            plane_origin,
            axis
        )

        match = find_axis_loop(
            loops,
            plane_origin,
            axis,
            feature_axis_point,
            axis,
            target_diameter
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

        else:

            observations.append(
                {
                    "axis_coordinate":
                        float(coordinate),
                    "detected":
                        True,
                    "diameter":
                        match["diameter"],
                    "radial_distance":
                        match["radial_distance"],
                    "diameter_error_ratio":
                        match[
                            "diameter_error_ratio"
                        ]
                }
            )

    # ========================================================
    # DETECTED OBSERVATIONS
    # ========================================================

    detected = [
        item
        for item in observations
        if item["detected"]
    ]

    if len(detected) < MIN_VALID_SECTIONS:

        return {
            "feature_id": feature_id,
            "classification":
                "UNKNOWN",
            "subtype":
                "UNKNOWN",
            "confidence":
                0.0,
            "reason":
                "Insufficient local circular-section evidence.",
            "observations":
                observations
        }

    detected_coordinates = np.asarray(
        [
            item["axis_coordinate"]
            for item in detected
        ],
        dtype=float
    )

    detected_diameters = np.asarray(
        [
            item["diameter"]
            for item in detected
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

    mean_diameter = float(
        detected_diameters.mean()
    )

    minimum_diameter = float(
        detected_diameters.min()
    )

    maximum_diameter = float(
        detected_diameters.max()
    )

    diameter_variation = (
        maximum_diameter -
        minimum_diameter
    ) / max(
        mean_diameter,
        1e-9
    )

    # ========================================================
    # CONTINUITY
    # ========================================================

    local_span = (
        sample_max -
        sample_min
    )

    continuity_ratio = (
        detected_length /
        max(
            local_span,
            1e-9
        )
    )

    if continuity_ratio >= 0.75:

        continuity = "STRONG"

    elif continuity_ratio >= 0.45:

        continuity = "MODERATE"

    else:

        continuity = "WEAK"

    # ========================================================
    # DIAMETER PROFILE
    # ========================================================

    if diameter_variation <= CONSTANT_DIAMETER_RATIO:

        diameter_profile = (
            "CONSTANT_DIAMETER"
        )

    elif diameter_variation <= TAPERED_DIAMETER_RATIO:

        diameter_profile = (
            "MILDLY_VARIABLE"
        )

    else:

        diameter_profile = (
            "TAPERED_OR_STEPPED"
        )

    # ========================================================
    # LOCAL TERMINATIONS
    #
    # IMPORTANT:
    #
    # We compare against the FEATURE OBSERVATION RANGE first,
    # rather than blindly using the complete part range.
    # ========================================================

    lower_gap = (
        detected_min -
        observed_min
    )

    upper_gap = (
        observed_max -
        detected_max
    )

    feature_range_length = (
        observed_max -
        observed_min
    )

    if feature_range_length <= 0:

        feature_range_length = 1.0

    lower_coverage = (
        max(
            0.0,
            1.0 -
            lower_gap /
            feature_range_length
        )
    )

    upper_coverage = (
        max(
            0.0,
            1.0 -
            upper_gap /
            feature_range_length
        )
    )

    # ========================================================
    # PART-END TEST
    #
    # This is only supporting evidence.
    # It is NOT the primary classification criterion.
    # ========================================================

    part_length = (
        global_max -
        global_min
    )

    lower_part_gap = (
        detected_min -
        global_min
    )

    upper_part_gap = (
        global_max -
        detected_max
    )

    reaches_lower_part_end = (
        lower_part_gap
        <=
        part_length * 0.08
    )

    reaches_upper_part_end = (
        upper_part_gap
        <=
        part_length * 0.08
    )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    classification = (
        "INTERNAL_CIRCULAR_FEATURE"
    )

    subtype = (
        "CYLINDRICAL_INTERNAL_FEATURE"
    )

    confidence = 0.55

    reason = (
        "Persistent circular geometry was detected around "
        "the physical feature axis."
    )

    # --------------------------------------------------------
    # Strong through-hole evidence.
    #
    # Both part ends reached AND continuous feature.
    # --------------------------------------------------------

    if (
        reaches_lower_part_end
        and
        reaches_upper_part_end
        and
        continuity
        ==
        "STRONG"
    ):

        classification = "HOLE"

        subtype = "THROUGH_HOLE"

        confidence = 0.94

        reason = (
            "The circular feature reaches both axial ends "
            "of the complete part and remains continuous."
        )

    # --------------------------------------------------------
    # Blind-hole candidate.
    #
    # A finite cylindrical wall is observed and does not
    # reach both complete part ends.
    #
    # This remains a candidate until local bottom/opening
    # surface analysis confirms the termination.
    # --------------------------------------------------------

    elif (
        continuity
        in {
            "STRONG",
            "MODERATE"
        }
        and
        detected_length > 0.5
        and
        not (
            reaches_lower_part_end
            and
            reaches_upper_part_end
        )
    ):

        classification = "HOLE"

        subtype = "BLIND_HOLE_CANDIDATE"

        confidence = 0.78

        reason = (
            "A continuous finite cylindrical wall is present "
            "around the feature axis, but it does not extend "
            "through both complete part ends. This supports "
            "a blind-hole interpretation, but local opening "
            "and bottom surfaces must confirm it."
        )

    # --------------------------------------------------------
    # Variable diameter.
    # --------------------------------------------------------

    if (
        diameter_profile
        ==
        "TAPERED_OR_STEPPED"
        and
        classification != "HOLE"
    ):

        classification = (
            "INTERNAL_CIRCULAR_FEATURE"
        )

        subtype = (
            "VARIABLE_DIAMETER_INTERNAL_FEATURE"
        )

        confidence = min(
            confidence,
            0.72
        )

        reason = (
            "The circular feature shows significant diameter "
            "variation, so it is not safely represented as "
            "a simple constant cylindrical hole."
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if len(detected) >= 20:

        confidence += 0.03

    elif len(detected) >= 10:

        confidence += 0.02

    if diameter_variation <= 0.05:

        confidence += 0.03

    if continuity == "WEAK":

        confidence -= 0.15

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
            target_diameter,

        "measured_diameter":
            mean_diameter,

        "measured_diameter_min":
            minimum_diameter,

        "measured_diameter_max":
            maximum_diameter,

        "diameter_variation_ratio":
            float(
                diameter_variation
            ),

        "diameter_profile":
            diameter_profile,

        "continuity":
            continuity,

        "continuity_ratio":
            float(
                continuity_ratio
            ),

        "observed_axial_range":
            {
                "min":
                    observed_min,
                "max":
                    observed_max,
                "length":
                    feature_range_length
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

        "local_analysis_range":
            {
                "min":
                    sample_min,
                "max":
                    sample_max,
                "length":
                    local_span
            },

        "lower_termination_gap":
            float(
                lower_gap
            ),

        "upper_termination_gap":
            float(
                upper_gap
            ),

        "reaches_lower_part_end":
            bool(
                reaches_lower_part_end
            ),

        "reaches_upper_part_end":
            bool(
                reaches_upper_part_end
            ),

        "detected_sections":
            len(detected),

        "total_samples":
            len(observations),

        "observations":
            observations
    }


# ============================================================
# PRINT
# ============================================================

def print_result(result):

    print()
    print(
        result.get(
            "feature_id",
            "UNKNOWN"
        )
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
        f"{safe_float(result.get('measured_diameter')):.4f}"
    )

    print(
        f"  Diameter profile  : "
        f"{result.get('diameter_profile')}"
    )

    print(
        f"  Continuity        : "
        f"{result.get('continuity')}"
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
        f" -> "
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
        "ALPHA - STAGE 5B"
    )

    print(
        "LOCAL OPENING / BOTTOM / DEPTH ANALYSIS"
    )

    print("=" * 78)

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        mesh = load_mesh()

        (
            consolidated,
            features,
            axis_origin,
            axis
        ) = load_consolidated()

        # ----------------------------------------------------
        # Complete part axial range
        # ----------------------------------------------------

        vertex_coordinates = np.dot(
            mesh.vertices -
            axis_origin,
            axis
        )

        global_min = float(
            vertex_coordinates.min()
        )

        global_max = float(
            vertex_coordinates.max()
        )

        print()
        print(
            "[3] Complete part axial range:"
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
        # Analyze
        # ----------------------------------------------------

        print()
        print(
            "[4] Performing corrected local analysis..."
        )

        results = []

        for feature in features:

            print(
                f"\n  Analyzing "
                f"{feature.get('feature_id', 'UNKNOWN')}..."
            )

            result = analyze_feature(
                mesh,
                feature,
                axis_origin,
                axis,
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
        # Summary
        # ----------------------------------------------------

        through_count = sum(
            1
            for result in results
            if result.get(
                "subtype"
            )
            ==
            "THROUGH_HOLE"
        )

        blind_count = sum(
            1
            for result in results
            if result.get(
                "subtype"
            )
            ==
            "BLIND_HOLE_CANDIDATE"
        )

        internal_count = sum(
            1
            for result in results
            if result.get(
                "classification"
            )
            ==
            "INTERNAL_CIRCULAR_FEATURE"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output = {

            "system":
                "ALPHA",

            "stage":
                "5B",

            "stage_name":
                "LOCAL OPENING / BOTTOM / DEPTH ANALYSIS",

            "source_stl":
                str(STL_FILE),

            "source_geometry":
                str(CONSOLIDATED_FILE),

            "analysis_method":
                "Dense axial cross-section sampling with "
                "feature-axis-line matching.",

            "correction_from_stage_5":
                "Section loops are matched by perpendicular "
                "distance to the feature axis line. Axial "
                "distance is intentionally excluded from "
                "the radial matching tolerance.",

            "principal_axis":
                vector_to_list(axis),

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
            "[OK] Saved:"
        )

        print(
            OUTPUT_FILE
        )

        print()
        print(
            f"Features analyzed       : {len(results)}"
        )

        print(
            f"Through-hole candidates : {through_count}"
        )

        print(
            f"Blind-hole candidates  : {blind_count}"
        )

        print(
            f"Internal circular       : {internal_count}"
        )

        print()
        print(
            "[SUCCESS] Stage 5B completed."
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
            "STAGE 5B ERROR"
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