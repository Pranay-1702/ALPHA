import os
import json
import math
import numpy as np
import trimesh


# ============================================================
# ALPHA - STAGE 5C
# TOPOLOGY FEATURE VALIDATOR
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_DIR = os.path.join(
    BASE_DIR,
    "input"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

STL_PATH = os.path.join(
    INPUT_DIR,
    "test_part.stl"
)

GEOMETRY_PATH = os.path.join(
    OUTPUT_DIR,
    "consolidated_feature_geometry.json"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "topology_feature_analysis.json"
)


# ============================================================
# PARAMETERS
# ============================================================

SECTION_COUNT = 81

AXIAL_MARGIN = 1.5

CENTER_TOLERANCE = 5.0

DIAMETER_TOLERANCE = 0.35

MIN_VALID_SECTIONS = 8

LOOP_CLOSURE_TOLERANCE = 0.25

MIN_LOOP_POINTS = 8

MIN_LOOP_AREA = 1.0


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize(vector):

    vector = np.asarray(
        vector,
        dtype=float
    )

    length = np.linalg.norm(
        vector
    )

    if length < 1e-12:

        raise ValueError(
            "Cannot normalize zero vector"
        )

    return vector / length


def axial_coordinate(
    point,
    center,
    axis
):

    point = np.asarray(
        point,
        dtype=float
    )

    center = np.asarray(
        center,
        dtype=float
    )

    axis = normalize(axis)

    return float(
        np.dot(
            point - center,
            axis
        )
    )


def point_on_feature_axis(
    feature_axis_point,
    principal_axis,
    feature_axial_coordinate,
    reference_axial_coordinate
):
    """
    Return a point on the FEATURE'S OWN AXIS.

    This is the critical correction.

    The previous implementation used the global principal
    center here, which destroyed the X/Y radial position of
    peripheral features.
    """

    feature_axis_point = np.asarray(
        feature_axis_point,
        dtype=float
    )

    principal_axis = normalize(
        principal_axis
    )

    delta = (
        feature_axial_coordinate
        -
        reference_axial_coordinate
    )

    return (
        feature_axis_point
        +
        principal_axis * delta
    )


# ============================================================
# 3D -> 2D TRANSFORM
# ============================================================

def project_point_to_plane_coordinates(
    point,
    transform
):

    point = np.asarray(
        point,
        dtype=float
    )

    try:

        inverse = np.linalg.inv(
            transform
        )

        homogeneous = np.ones(4)

        homogeneous[:3] = point

        result = (
            inverse
            @
            homogeneous
        )

        return result[:2]

    except Exception:

        return None


# ============================================================
# 2D POLYGON GEOMETRY
# ============================================================

def polygon_area(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < 3:

        return 0.0

    x = points[:, 0]
    y = points[:, 1]

    return float(
        0.5
        *
        abs(
            np.sum(
                x
                *
                np.roll(y, -1)
                -
                y
                *
                np.roll(x, -1)
            )
        )
    )


def polygon_centroid(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) == 0:

        return np.zeros(2)

    area_signed = (
        0.5
        *
        np.sum(
            points[:, 0]
            *
            np.roll(points[:, 1], -1)
            -
            points[:, 1]
            *
            np.roll(points[:, 0], -1)
        )
    )

    if abs(area_signed) < 1e-12:

        return np.mean(
            points,
            axis=0
        )

    factor = (
        points[:, 0]
        *
        np.roll(points[:, 1], -1)
        -
        np.roll(points[:, 0], -1)
        *
        points[:, 1]
    )

    cx = (
        np.sum(
            (
                points[:, 0]
                +
                np.roll(points[:, 0], -1)
            )
            *
            factor
        )
        /
        (
            6.0
            *
            area_signed
        )
    )

    cy = (
        np.sum(
            (
                points[:, 1]
                +
                np.roll(points[:, 1], -1)
            )
            *
            factor
        )
        /
        (
            6.0
            *
            area_signed
        )
    )

    return np.array(
        [cx, cy]
    )


def equivalent_diameter(area):

    if area <= 0:

        return 0.0

    return float(
        2.0
        *
        math.sqrt(
            area / math.pi
        )
    )


# ============================================================
# CLOSED LOOP EXTRACTION
# ============================================================

def clean_loop(points):

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) < MIN_LOOP_POINTS:

        return None

    finite = np.all(
        np.isfinite(points),
        axis=1
    )

    points = points[finite]

    if len(points) < MIN_LOOP_POINTS:

        return None

    # Remove consecutive duplicate points
    cleaned = [
        points[0]
    ]

    for point in points[1:]:

        if np.linalg.norm(
            point
            -
            cleaned[-1]
        ) > 1e-9:

            cleaned.append(
                point
            )

    points = np.asarray(
        cleaned
    )

    if len(points) < MIN_LOOP_POINTS:

        return None

    # Trimesh may represent a closed path without repeating
    # the first vertex. Do NOT reject it because of that.
    return points


def extract_discrete_loops(path):

    loops = []

    try:

        discrete_paths = path.discrete

    except Exception:

        return loops

    for raw_points in discrete_paths:

        try:

            points = np.asarray(
                raw_points,
                dtype=float
            )

        except Exception:

            continue

        points = clean_loop(
            points
        )

        if points is None:

            continue

        area = polygon_area(
            points
        )

        if area < MIN_LOOP_AREA:

            continue

        centroid = polygon_centroid(
            points
        )

        diameter = equivalent_diameter(
            area
        )

        loops.append(
            {
                "points": points,
                "area": float(area),
                "centroid": centroid,
                "diameter": float(diameter)
            }
        )

    if not loops:

        return []

    # --------------------------------------------------------
    # Determine nesting
    # --------------------------------------------------------

    for i, loop in enumerate(
        loops
    ):

        depth = 0

        test_point = loop[
            "centroid"
        ]

        for j, other in enumerate(
            loops
        ):

            if i == j:
                continue

            if (
                other["area"]
                <=
                loop["area"]
            ):

                continue

            if point_inside_polygon(
                test_point,
                other["points"]
            ):

                depth += 1

        loop["depth"] = depth

        if depth == 0:

            loop["type"] = (
                "OUTER_BOUNDARY"
            )

        else:

            loop["type"] = (
                "INTERNAL_BOUNDARY"
            )

    return loops


# ============================================================
# POINT-IN-POLYGON
# ============================================================

def point_inside_polygon(
    point,
    polygon
):

    point = np.asarray(
        point,
        dtype=float
    )

    polygon = np.asarray(
        polygon,
        dtype=float
    )

    x = point[0]
    y = point[1]

    inside = False

    n = len(polygon)

    for i in range(n):

        p1 = polygon[i]

        p2 = polygon[
            (i + 1) % n
        ]

        x1, y1 = p1
        x2, y2 = p2

        if (
            (y1 > y)
            !=
            (y2 > y)
        ):

            denominator = (
                y2 - y1
            )

            if abs(
                denominator
            ) < 1e-15:

                continue

            x_intersection = (
                (
                    x2 - x1
                )
                *
                (
                    y - y1
                )
                /
                denominator
                +
                x1
            )

            if x < x_intersection:

                inside = not inside

    return inside


# ============================================================
# LOOP MATCHING
# ============================================================

def find_best_matching_loop(
    loops,
    feature_center_2d,
    expected_diameter
):

    if not loops:

        return None

    candidates = []

    for loop in loops:

        center_distance = float(
            np.linalg.norm(
                loop["centroid"]
                -
                feature_center_2d
            )
        )

        diameter_error = (
            abs(
                loop["diameter"]
                -
                expected_diameter
            )
            /
            max(
                expected_diameter,
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
            diameter_error
            >
            DIAMETER_TOLERANCE
        ):

            continue

        score = (
            center_distance
            +
            diameter_error
            *
            expected_diameter
        )

        candidates.append(
            (
                score,
                loop
            )
        )

    if not candidates:

        return None

    candidates.sort(
        key=lambda item:
        item[0]
    )

    return candidates[0][1]


# ============================================================
# SECTION ANALYSIS
# ============================================================

def analyze_section(
    path,
    feature_axis_point,
    transform,
    expected_diameter
):

    result = {
        "matched": False,
        "loop_type": None,
        "diameter": 0.0,
        "center_distance": None,
        "loop_count": 0
    }

    if path is None:

        return result

    feature_center_2d = (
        project_point_to_plane_coordinates(
            feature_axis_point,
            transform
        )
    )

    if feature_center_2d is None:

        return result

    loops = extract_discrete_loops(
        path
    )

    result["loop_count"] = len(
        loops
    )

    if not loops:

        return result

    match = find_best_matching_loop(
        loops,
        feature_center_2d,
        expected_diameter
    )

    if match is None:

        return result

    result["matched"] = True

    result["loop_type"] = (
        match["type"]
    )

    result["diameter"] = float(
        match["diameter"]
    )

    result["center_distance"] = float(
        np.linalg.norm(
            match["centroid"]
            -
            feature_center_2d
        )
    )

    return result


# ============================================================
# LOAD STL
# ============================================================

def load_mesh():

    print()
    print(
        "[1/5] Loading STL..."
    )

    mesh = trimesh.load_mesh(
        STL_PATH,
        process=False
    )

    if not isinstance(
        mesh,
        trimesh.Trimesh
    ):

        raise RuntimeError(
            "STL did not load as Trimesh"
        )

    mesh.merge_vertices()

    try:

        mesh.update_faces(mesh.nondegenerate_faces())

    except Exception:

        pass

    try:

        mesh.update_faces(mesh.unique_faces())

    except Exception:

        pass

    print(
        f"    Vertices   : "
        f"{len(mesh.vertices):,}"
    )

    print(
        f"    Triangles  : "
        f"{len(mesh.faces):,}"
    )

    print(
        f"    Watertight: "
        f"{mesh.is_watertight}"
    )

    return mesh


# ============================================================
# LOAD STAGE 3B GEOMETRY
# ============================================================

def load_geometry():

    print()
    print(
        "[2/5] Loading authoritative Stage 3B geometry..."
    )

    with open(
        GEOMETRY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    principal = data.get(
        "principal_geometry",
        {}
    )

    principal_center = np.asarray(
        principal.get(
            "center",
            [0.0, 0.0, 0.0]
        ),
        dtype=float
    )

    principal_axis = normalize(
        principal.get(
            "axis",
            [0.0, 0.0, 1.0]
        )
    )

    features = (
        data.get("features")
        or
        data.get("physical_features")
        or
        []
    )

    if isinstance(
        features,
        dict
    ):

        converted = []

        for key, value in features.items():

            if not isinstance(
                value,
                dict
            ):

                continue

            item = dict(
                value
            )

            item.setdefault(
                "id",
                key
            )

            converted.append(
                item
            )

        features = converted

    print(
        f"    Principal center: "
        f"{principal_center}"
    )

    print(
        f"    Principal axis  : "
        f"{principal_axis}"
    )

    print(
        "    Z-axis alignment: "
        f"{abs(principal_axis[2]):.6f}"
    )

    print(
        f"    Feature count   : "
        f"{len(features)}"
    )

    return (
        principal_center,
        principal_axis,
        features
    )


# ============================================================
# FEATURE DATA
# ============================================================

def get_feature_id(
    feature
):

    return str(
        feature.get(
            "id",
            feature.get(
                "feature_id",
                "UNKNOWN_FEATURE"
            )
        )
    )


def get_expected_diameter(
    feature
):

    keys = [
        "diameter",
        "mean_diameter",
        "nominal_diameter",
        "expected_diameter"
    ]

    for key in keys:

        value = feature.get(
            key
        )

        if value is None:
            continue

        try:

            value = float(
                value
            )

            if value > 0:

                return value

        except Exception:

            pass

    return 0.0


def get_feature_axis_point(
    feature,
    principal_center
):

    value = feature.get(
        "physical_axis_point"
    )

    if value is not None:

        try:

            point = np.asarray(
                value,
                dtype=float
            )

            if point.shape == (3,):

                return point

        except Exception:

            pass

    value = feature.get(
        "axis_point"
    )

    if value is not None:

        try:

            point = np.asarray(
                value,
                dtype=float
            )

            if point.shape == (3,):

                return point

        except Exception:

            pass

    return principal_center.copy()


def get_feature_axial_range(
    feature,
    feature_axis_point,
    principal_center,
    principal_axis
):

    axial_range = feature.get(
        "axial_range"
    )

    if isinstance(
        axial_range,
        dict
    ):

        minimum = axial_range.get(
            "min"
        )

        maximum = axial_range.get(
            "max"
        )

        if (
            minimum is not None
            and
            maximum is not None
        ):

            return (
                float(minimum),
                float(maximum)
            )

    coordinate = axial_coordinate(
        feature_axis_point,
        principal_center,
        principal_axis
    )

    return (
        coordinate - 5.0,
        coordinate + 5.0
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_feature(
    internal_sections,
    outer_sections,
    lower_side_open,
    upper_side_open
):

    total = (
        internal_sections
        +
        outer_sections
    )

    if total < MIN_VALID_SECTIONS:

        return (
            "UNKNOWN",
            "INSUFFICIENT_TOPOLOGY",
            0.0
        )

    internal_ratio = (
        internal_sections
        /
        total
    )

    outer_ratio = (
        outer_sections
        /
        total
    )

    # --------------------------------------------------------
    # Internal cylindrical boundary
    # --------------------------------------------------------

    if (
        internal_ratio >= 0.70
        and
        outer_ratio < 0.30
    ):

        if (
            lower_side_open
            and
            upper_side_open
        ):

            return (
                "HOLE",
                "THROUGH_HOLE",
                0.99
            )

        if (
            lower_side_open
            or
            upper_side_open
        ):

            return (
                "HOLE",
                "BLIND_HOLE",
                0.95
            )

        return (
            "HOLE",
            "INTERNAL_HOLE",
            0.85
        )

    # --------------------------------------------------------
    # Outer cylindrical boundary
    # --------------------------------------------------------

    if (
        outer_ratio >= 0.70
        and
        internal_ratio < 0.30
    ):

        return (
            "CYLINDRICAL_SURFACE",
            "OUTER_CYLINDRICAL_SURFACE",
            0.99
        )

    # --------------------------------------------------------
    # Mixed topology
    # --------------------------------------------------------

    if (
        internal_ratio >= 0.50
        and
        outer_ratio >= 0.20
    ):

        return (
            "MIXED_TOPOLOGY",
            "REQUIRES_SECTION_ANALYSIS",
            0.50
        )

    return (
        "UNKNOWN",
        "AMBIGUOUS_TOPOLOGY",
        0.0
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_feature(
    mesh,
    feature,
    principal_center,
    principal_axis
):

    feature_id = get_feature_id(
        feature
    )

    expected_diameter = (
        get_expected_diameter(
            feature
        )
    )

    feature_axis_point = (
        get_feature_axis_point(
            feature,
            principal_center
        )
    )

    axial_min, axial_max = (
        get_feature_axial_range(
            feature,
            feature_axis_point,
            principal_center,
            principal_axis
        )
    )

    axial_min -= AXIAL_MARGIN
    axial_max += AXIAL_MARGIN

    heights = np.linspace(
        axial_min,
        axial_max,
        SECTION_COUNT
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"FEATURE: {feature_id}"
    )

    print(
        "=" * 72
    )

    print(
        f"Expected diameter : "
        f"{expected_diameter:.4f}"
    )

    print(
        f"Feature axis point: "
        f"{feature_axis_point}"
    )

    print(
        f"Axial range       : "
        f"{axial_min:.4f} -> "
        f"{axial_max:.4f}"
    )

    # --------------------------------------------------------
    # Generate sections
    # --------------------------------------------------------

    try:

        sections = (
            mesh.section_multiplane(
                plane_origin=
                    principal_center,

                plane_normal=
                    principal_axis,

                heights=
                    heights
            )
        )

    except Exception as exc:

        print(
            f"Section generation failed: "
            f"{exc}"
        )

        return {
            "feature_id":
                feature_id,

            "classification":
                "UNKNOWN",

            "subtype":
                "SECTION_FAILURE",

            "confidence":
                0.0,

            "expected_diameter":
                expected_diameter,

            "measured_diameter":
                0.0,

            "diameter_variation":
                0.0,

            "valid_sections":
                0,

            "loop_sections":
                0,

            "internal_sections":
                0,

            "outer_sections":
                0,

            "internal_ratio":
                0.0,

            "outer_ratio":
                0.0,

            "lower_side_open":
                False,

            "upper_side_open":
                False,

            "lower_side_material":
                False,

            "upper_side_material":
                False,

            "feature_axis_point":
                feature_axis_point.tolist()
        }

    # --------------------------------------------------------
    # Analyze each section
    # --------------------------------------------------------

    section_results = []

    measured_diameters = []

    internal_sections = 0

    outer_sections = 0

    loop_sections = 0

    for index, path in enumerate(
        sections
    ):

        if path is None:

            continue

        transform = None

        try:

            transform = (
                path.metadata.get(
                    "to_3D"
                )
            )

        except Exception:

            pass

        if transform is None:

            continue

        current_axial_coordinate = (
            float(
                heights[index]
            )
        )

        # ====================================================
        # CRITICAL FIX
        #
        # Preserve the FEATURE'S X/Y radial position.
        #
        # Before:
        #   principal_center + axis * z
        #
        # Now:
        #   feature_axis_point + axis * delta_z
        # ====================================================

        section_feature_axis_point = (
            point_on_feature_axis(
                feature_axis_point,
                principal_axis,
                current_axial_coordinate,
                axial_coordinate(
                    feature_axis_point,
                    principal_center,
                    principal_axis
                )
            )
        )

        result = analyze_section(
            path,
            section_feature_axis_point,
            np.asarray(
                transform,
                dtype=float
            ),
            expected_diameter
        )

        result[
            "axial_coordinate"
        ] = current_axial_coordinate

        section_results.append(
            result
        )

        if result["matched"]:

            loop_sections += 1

            measured_diameters.append(
                result["diameter"]
            )

            if (
                result["loop_type"]
                ==
                "INTERNAL_BOUNDARY"
            ):

                internal_sections += 1

            elif (
                result["loop_type"]
                ==
                "OUTER_BOUNDARY"
            ):

                outer_sections += 1

    valid_sections = len(
        section_results
    )

    # --------------------------------------------------------
    # Diameter statistics
    # --------------------------------------------------------

    if measured_diameters:

        diameter_array = np.asarray(
            measured_diameters,
            dtype=float
        )

        measured_diameter = float(
            np.median(
                diameter_array
            )
        )

        minimum_diameter = float(
            np.min(
                diameter_array
            )
        )

        maximum_diameter = float(
            np.max(
                diameter_array
            )
        )

        diameter_variation = (
            (
                maximum_diameter
                -
                minimum_diameter
            )
            /
            max(
                measured_diameter,
                1e-9
            )
        )

    else:

        measured_diameter = 0.0

        diameter_variation = 0.0

    # --------------------------------------------------------
    # Side termination
    # --------------------------------------------------------

    matched_results = [
        result
        for result in section_results
        if result["matched"]
    ]

    lower_side_open = False

    upper_side_open = False

    lower_side_material = False

    upper_side_material = False

    if matched_results:

        first_match = min(
            matched_results,
            key=lambda item:
            item["axial_coordinate"]
        )

        last_match = max(
            matched_results,
            key=lambda item:
            item["axial_coordinate"]
        )

        first_coordinate = (
            first_match[
                "axial_coordinate"
            ]
        )

        last_coordinate = (
            last_match[
                "axial_coordinate"
            ]
        )

        span = (
            axial_max
            -
            axial_min
        )

        edge_tolerance = max(
            0.75,
            span * 0.06
        )

        lower_side_open = (
            first_coordinate
            <=
            axial_min
            +
            edge_tolerance
        )

        upper_side_open = (
            last_coordinate
            >=
            axial_max
            -
            edge_tolerance
        )

        lower_side_material = (
            first_coordinate
            >
            axial_min
            +
            edge_tolerance
        )

        upper_side_material = (
            last_coordinate
            <
            axial_max
            -
            edge_tolerance
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classification, subtype, confidence = (
        classify_feature(
            internal_sections,
            outer_sections,
            lower_side_open,
            upper_side_open
        )
    )

    total_matched = (
        internal_sections
        +
        outer_sections
    )

    if total_matched:

        internal_ratio = (
            internal_sections
            /
            total_matched
        )

        outer_ratio = (
            outer_sections
            /
            total_matched
        )

    else:

        internal_ratio = 0.0

        outer_ratio = 0.0

    result = {

        "feature_id":
            feature_id,

        "classification":
            classification,

        "subtype":
            subtype,

        "confidence":
            float(confidence),

        "expected_diameter":
            float(expected_diameter),

        "measured_diameter":
            float(measured_diameter),

        "diameter_variation":
            float(diameter_variation),

        "valid_sections":
            int(valid_sections),

        "loop_sections":
            int(loop_sections),

        "internal_sections":
            int(internal_sections),

        "outer_sections":
            int(outer_sections),

        "internal_ratio":
            float(internal_ratio),

        "outer_ratio":
            float(outer_ratio),

        "lower_side_open":
            bool(lower_side_open),

        "upper_side_open":
            bool(upper_side_open),

        "lower_side_material":
            bool(lower_side_material),

        "upper_side_material":
            bool(upper_side_material),

        "feature_axis_point":
            feature_axis_point.tolist(),

        "axial_range": {
            "min":
                float(axial_min),

            "max":
                float(axial_max)
        },

        "section_diameters": [
            float(value)
            for value
            in measured_diameters
        ]
    }

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        f"Classification     : "
        f"{classification}"
    )

    print(
        f"Subtype            : "
        f"{subtype}"
    )

    print(
        f"Confidence         : "
        f"{confidence:.3f}"
    )

    print(
        f"Measured diameter  : "
        f"{measured_diameter:.4f}"
    )

    print(
        f"Diameter variation : "
        f"{diameter_variation * 100:.2f}%"
    )

    print(
        f"Valid sections     : "
        f"{valid_sections}"
    )

    print(
        f"Loop sections      : "
        f"{loop_sections}"
    )

    print(
        f"Internal sections  : "
        f"{internal_sections}"
    )

    print(
        f"Outer sections     : "
        f"{outer_sections}"
    )

    print(
        f"Internal ratio     : "
        f"{internal_ratio:.3f}"
    )

    print(
        f"Outer ratio        : "
        f"{outer_ratio:.3f}"
    )

    print(
        f"Lower side open    : "
        f"{lower_side_open}"
    )

    print(
        f"Upper side open    : "
        f"{upper_side_open}"
    )

    print(
        f"Lower side material: "
        f"{lower_side_material}"
    )

    print(
        f"Upper side material: "
        f"{upper_side_material}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("ALPHA - STAGE 5C")
    print("TOPOLOGY FEATURE VALIDATOR")
    print("=" * 72)

    print()
    print("[INPUT STL]")
    print(
        f"    {STL_PATH}"
    )

    print()
    print("[INPUT GEOMETRY]")
    print(
        f"    {GEOMETRY_PATH}"
    )

    print()
    print("[OUTPUT]")
    print(
        f"    {OUTPUT_PATH}"
    )

    mesh = load_mesh()

    (
        principal_center,
        principal_axis,
        features
    ) = load_geometry()

    print()
    print(
        "[3/5] Running topology analysis..."
    )

    results = []

    for feature in features:

        result = analyze_feature(
            mesh,
            feature,
            principal_center,
            principal_axis
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    holes = [
        result
        for result in results
        if result["classification"]
        ==
        "HOLE"
    ]

    through_holes = [
        result
        for result in holes
        if result["subtype"]
        ==
        "THROUGH_HOLE"
    ]

    blind_holes = [
        result
        for result in holes
        if result["subtype"]
        ==
        "BLIND_HOLE"
    ]

    internal_holes = [
        result
        for result in holes
        if result["subtype"]
        ==
        "INTERNAL_HOLE"
    ]

    cylindrical = [
        result
        for result in results
        if result["classification"]
        ==
        "CYLINDRICAL_SURFACE"
    ]

    unknown = [
        result
        for result in results
        if result["classification"]
        ==
        "UNKNOWN"
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print(
        "[4/5] Saving topology results..."
    )

    output = {

        "stage":
            "5C",

        "description":
            (
                "Topology validation using "
                "feature-axis-aware section loops"
            ),

        "input_stl":
            STL_PATH,

        "input_geometry":
            GEOMETRY_PATH,

        "principal_geometry": {

            "center":
                principal_center.tolist(),

            "axis":
                principal_axis.tolist()
        },

        "parameters": {

            "section_count":
                SECTION_COUNT,

            "axial_margin":
                AXIAL_MARGIN,

            "center_tolerance":
                CENTER_TOLERANCE,

            "diameter_tolerance":
                DIAMETER_TOLERANCE,

            "minimum_valid_sections":
                MIN_VALID_SECTIONS,

            "loop_closure_tolerance":
                LOOP_CLOSURE_TOLERANCE
        },

        "features":
            results,

        "summary": {

            "total_features":
                len(results),

            "holes":
                len(holes),

            "through_holes":
                len(through_holes),

            "blind_holes":
                len(blind_holes),

            "internal_holes":
                len(internal_holes),

            "cylindrical_surfaces":
                len(cylindrical),

            "unknown":
                len(unknown)
        }
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 5C SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Total features       : "
        f"{len(results)}"
    )

    print(
        f"Holes                : "
        f"{len(holes)}"
    )

    print(
        f"  Through holes      : "
        f"{len(through_holes)}"
    )

    print(
        f"  Blind holes        : "
        f"{len(blind_holes)}"
    )

    print(
        f"  Internal holes     : "
        f"{len(internal_holes)}"
    )

    print(
        f"Cylindrical surfaces : "
        f"{len(cylindrical)}"
    )

    print(
        f"Unknown              : "
        f"{len(unknown)}"
    )

    print()
    print(
        f"[SAVED] {OUTPUT_PATH}"
    )

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 5C COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()