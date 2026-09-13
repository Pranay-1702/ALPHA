from pathlib import Path
import json
import math

import numpy as np


# ============================================================
# ALPHA
# STAGE 3B - PHYSICAL AXIS / LOCATION CONSOLIDATION
#
# Purpose:
#   Convert section-circle observations into physical circular
#   features by grouping observations that share the same
#   physical axis.
#
# IMPORTANT:
#   Z variation between section observations is NOT treated as
#   a different feature.
#
# Primary grouping evidence:
#   1. Projected center on plane normal to main axis
#   2. Radial distance
#   3. Angular position
#
# Secondary evidence:
#   4. Diameter compatibility
#   5. Axial continuity
#
# This stage does NOT:
#   - use Gemini
#   - guess hole sizes
#   - create CAD
#   - force a specific hole count
#
# Input:
#   output/feature_geometry_evidence.json
#
# Optional input:
#   output/mechanical_feature_evidence.json
#
# Output:
#   output/consolidated_feature_geometry.json
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT / "output"
)

STAGE3_FILE = (
    OUTPUT_DIR /
    "feature_geometry_evidence.json"
)

STAGE1_FILE = (
    OUTPUT_DIR /
    "mechanical_feature_evidence.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "consolidated_feature_geometry.json"
)


# ============================================================
# SETTINGS
# ============================================================

# ------------------------------------------------------------
# PRIMARY LOCATION TOLERANCE
#
# Two section circles belonging to the same physical feature
# should have almost the same projected center.
#
# This is intentionally much tighter than the old 8 mm
# 3D-center tolerance.
# ------------------------------------------------------------

PROJECTED_CENTER_TOLERANCE = 3.0


# ------------------------------------------------------------
# Radial tolerance.
# ------------------------------------------------------------

RADIAL_DISTANCE_TOLERANCE = 3.0


# ------------------------------------------------------------
# Angular tolerance.
#
# This is mainly a safety check for off-axis features.
# ------------------------------------------------------------

ANGULAR_TOLERANCE_DEG = 6.0


# ------------------------------------------------------------
# Diameter variation allowed within one physical feature.
#
# This is deliberately permissive because a tapered feature
# can legitimately change diameter along Z.
# ------------------------------------------------------------

MAX_DIAMETER_VARIATION_RATIO = 0.35


# ------------------------------------------------------------
# Minimum section count for strong continuity.
# ------------------------------------------------------------

MIN_STRONG_SECTION_COUNT = 3


# ============================================================
# STAGE 1 MATCHING SETTINGS
# ============================================================

# Stage 1 matching remains STRICT.
#
# A diameter mismatch cannot be overridden by location.
# ------------------------------------------------------------

STAGE1_MAX_DIAMETER_DIFFERENCE_RATIO = 0.05

STAGE1_MAX_PROJECTED_CENTER_DISTANCE = 5.0

STAGE1_MAX_AXIAL_DIFFERENCE = 8.0

STAGE1_MAX_RADIAL_DIFFERENCE = 4.0

STAGE1_MAX_ANGULAR_DIFFERENCE_DEG = 8.0


# ============================================================
# BASIC UTILITIES
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

    magnitude = np.linalg.norm(
        vector
    )

    if magnitude < 1e-12:

        return None

    return vector / magnitude


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
# LOAD STAGE 3
# ============================================================

def load_stage3():

    print(
        "[1] Loading Stage 3 geometry evidence..."
    )

    if not STAGE3_FILE.exists():

        raise FileNotFoundError(
            f"Stage 3 evidence not found:\n"
            f"{STAGE3_FILE}\n\n"
            f"Run:\n"
            f"python .\\src\\feature_geometry_extractor.py"
        )

    with open(
        STAGE3_FILE,
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
            "Stage 3 JSON root is not an object."
        )

    print(
        "[OK] Stage 3 evidence loaded."
    )

    return data


# ============================================================
# LOAD STAGE 1
# ============================================================

def load_stage1():

    if not STAGE1_FILE.exists():

        print(
            "[WARNING] Stage 1 evidence not found."
        )

        return None

    try:

        with open(
            STAGE1_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        print(
            "[OK] Stage 1 evidence loaded."
        )

        return data

    except Exception as exc:

        print(
            f"[WARNING] Stage 1 load failed: {exc}"
        )

        return None


# ============================================================
# PRINCIPAL GEOMETRY
# ============================================================

def get_principal_geometry(
    data
):

    principal = data.get(
        "principal_geometry"
    )

    if not isinstance(
        principal,
        dict
    ):

        raise ValueError(
            "principal_geometry is missing."
        )

    center = np.asarray(
        principal.get(
            "center",
            [0.0, 0.0, 0.0]
        ),
        dtype=float
    )

    axes = principal.get(
        "axes"
    )

    if (
        not isinstance(
            axes,
            list
        )
        or
        len(axes) < 3
    ):

        raise ValueError(
            "Principal axes are missing."
        )

    axis = normalize(
        axes[2]
    )

    if axis is None:

        raise ValueError(
            "Principal Z axis is invalid."
        )

    extents = principal.get(
        "extents",
        [0.0, 0.0, 0.0]
    )

    extents = [
        safe_float(x)
        for x in extents
    ]

    return (
        center,
        axis,
        extents
    )


# ============================================================
# PREPARE OBSERVATION
# ============================================================

def prepare_observation(
    observation,
    principal_center,
    principal_axis
):

    point = np.asarray(
        observation.get(
            "center",
            [0.0, 0.0, 0.0]
        ),
        dtype=float
    )

    relative = (
        point -
        principal_center
    )

    axial_coordinate = float(
        np.dot(
            relative,
            principal_axis
        )
    )

    # --------------------------------------------------------
    # Projection onto plane normal to principal axis.
    #
    # This is the key Stage 3B operation.
    # --------------------------------------------------------

    projected_center = (
        principal_center
        +
        (
            relative
            -
            principal_axis
            *
            axial_coordinate
        )
    )

    radial_vector = (
        projected_center -
        principal_center
    )

    radial_distance = float(
        np.linalg.norm(
            radial_vector
        )
    )

    angle = None

    if radial_distance > 1e-9:

        u, v = build_basis(
            principal_axis
        )

        u_coordinate = float(
            np.dot(
                radial_vector,
                u
            )
        )

        v_coordinate = float(
            np.dot(
                radial_vector,
                v
            )
        )

        angle = math.degrees(
            math.atan2(
                v_coordinate,
                u_coordinate
            )
        )

        if angle < 0:

            angle += 360.0

    diameter = safe_float(
        observation.get(
            "diameter"
        )
    )

    if diameter <= 0:

        return None

    return {

        "section_id":
            safe_int(
                observation.get(
                    "section_id"
                )
            ),

        "center":
            vector_to_list(
                point
            ),

        "projected_center":
            vector_to_list(
                projected_center
            ),

        "axis_coordinate":
            axial_coordinate,

        "diameter":
            diameter,

        "radius":
            diameter / 2.0,

        "radial_distance":
            radial_distance,

        "angular_position_deg":
            angle,

        "relative_error":
            safe_float(
                observation.get(
                    "relative_error"
                )
            ),

        "point_count":
            safe_int(
                observation.get(
                    "point_count"
                )
            )
    }


# ============================================================
# EXTRACT ALL RAW OBSERVATIONS
# ============================================================

def extract_observations(
    data,
    principal_center,
    principal_axis
):

    print(
        "\n[2] Extracting raw circular observations..."
    )

    inventory = data.get(
        "circular_feature_inventory",
        []
    )

    if not isinstance(
        inventory,
        list
    ):

        raise ValueError(
            "circular_feature_inventory is missing."
        )

    observations = []

    for group in inventory:

        if not isinstance(
            group,
            dict
        ):

            continue

        raw = group.get(
            "observations",
            []
        )

        if not isinstance(
            raw,
            list
        ):

            continue

        for observation in raw:

            if not isinstance(
                observation,
                dict
            ):

                continue

            prepared = (
                prepare_observation(
                    observation,
                    principal_center,
                    principal_axis
                )
            )

            if prepared is not None:

                observations.append(
                    prepared
                )

    print(
        f"[OK] Raw observations: "
        f"{len(observations)}"
    )

    return observations


# ============================================================
# GROUP COMPATIBILITY
# ============================================================

def compatible_with_group(
    observation,
    group
):

    # ========================================================
    # 1. PROJECTED CENTER
    #
    # PRIMARY CONDITION
    # ========================================================

    observation_center = np.asarray(
        observation[
            "projected_center"
        ],
        dtype=float
    )

    group_center = np.asarray(
        group[
            "mean_projected_center"
        ],
        dtype=float
    )

    projected_distance = float(
        np.linalg.norm(
            observation_center -
            group_center
        )
    )

    if (
        projected_distance
        >
        PROJECTED_CENTER_TOLERANCE
    ):

        return False

    # ========================================================
    # 2. RADIAL POSITION
    # ========================================================

    radial_difference = abs(
        observation[
            "radial_distance"
        ]
        -
        group[
            "mean_radial_distance"
        ]
    )

    if (
        radial_difference
        >
        RADIAL_DISTANCE_TOLERANCE
    ):

        return False

    # ========================================================
    # 3. ANGULAR POSITION
    # ========================================================

    observation_angle = (
        observation[
            "angular_position_deg"
        ]
    )

    group_angle = (
        group[
            "mean_angle_deg"
        ]
    )

    if (
        observation_angle is not None
        and
        group_angle is not None
        and
        observation[
            "radial_distance"
        ] > 5.0
    ):

        angle_difference = (
            angle_difference_deg(
                observation_angle,
                group_angle
            )
        )

        if (
            angle_difference
            >
            ANGULAR_TOLERANCE_DEG
        ):

            return False

    # ========================================================
    # 4. DIAMETER SANITY CHECK
    # ========================================================

    group_diameter = (
        group[
            "mean_diameter"
        ]
    )

    observation_diameter = (
        observation[
            "diameter"
        ]
    )

    diameter_difference_ratio = (
        abs(
            observation_diameter -
            group_diameter
        )
        /
        max(
            group_diameter,
            1e-9
        )
    )

    if (
        diameter_difference_ratio
        >
        MAX_DIAMETER_VARIATION_RATIO
    ):

        return False

    return True


# ============================================================
# CREATE GROUP
# ============================================================

def create_group(
    observation
):

    group = {

        "observations":
            [],

        "mean_projected_center":
            observation[
                "projected_center"
            ],

        "mean_radial_distance":
            observation[
                "radial_distance"
            ],

        "min_radial_distance":
            observation[
                "radial_distance"
            ],

        "max_radial_distance":
            observation[
                "radial_distance"
            ],

        "mean_diameter":
            observation[
                "diameter"
            ],

        "min_diameter":
            observation[
                "diameter"
            ],

        "max_diameter":
            observation[
                "diameter"
            ],

        "diameter_variation_ratio":
            0.0,

        "mean_angle_deg":
            observation[
                "angular_position_deg"
            ],

        "axis_coordinate_min":
            observation[
                "axis_coordinate"
            ],

        "axis_coordinate_max":
            observation[
                "axis_coordinate"
            ],

        "section_ids":
            [],

        "section_count":
            0
    }

    update_group(
        group,
        observation
    )

    return group


# ============================================================
# UPDATE GROUP
# ============================================================

def update_group(
    group,
    observation
):

    group[
        "observations"
    ].append(
        observation
    )

    observations = (
        group[
            "observations"
        ]
    )

    # --------------------------------------------------------
    # Projected centers
    # --------------------------------------------------------

    projected_centers = np.asarray(
        [
            x[
                "projected_center"
            ]
            for x in observations
        ],
        dtype=float
    )

    group[
        "mean_projected_center"
    ] = vector_to_list(
        projected_centers.mean(
            axis=0
        )
    )

    # --------------------------------------------------------
    # Radial distances
    # --------------------------------------------------------

    radial_distances = np.asarray(
        [
            x[
                "radial_distance"
            ]
            for x in observations
        ],
        dtype=float
    )

    group[
        "mean_radial_distance"
    ] = float(
        radial_distances.mean()
    )

    group[
        "min_radial_distance"
    ] = float(
        radial_distances.min()
    )

    group[
        "max_radial_distance"
    ] = float(
        radial_distances.max()
    )

    # --------------------------------------------------------
    # Diameters
    # --------------------------------------------------------

    diameters = np.asarray(
        [
            x[
                "diameter"
            ]
            for x in observations
        ],
        dtype=float
    )

    group[
        "mean_diameter"
    ] = float(
        diameters.mean()
    )

    group[
        "min_diameter"
    ] = float(
        diameters.min()
    )

    group[
        "max_diameter"
    ] = float(
        diameters.max()
    )

    group[
        "diameter_variation_ratio"
    ] = float(
        (
            diameters.max()
            -
            diameters.min()
        )
        /
        max(
            diameters.mean(),
            1e-9
        )
    )

    # --------------------------------------------------------
    # Axial range
    # --------------------------------------------------------

    axial_coordinates = np.asarray(
        [
            x[
                "axis_coordinate"
            ]
            for x in observations
        ],
        dtype=float
    )

    group[
        "axis_coordinate_min"
    ] = float(
        axial_coordinates.min()
    )

    group[
        "axis_coordinate_max"
    ] = float(
        axial_coordinates.max()
    )

    # --------------------------------------------------------
    # Section IDs
    # --------------------------------------------------------

    group[
        "section_ids"
    ] = sorted(
        list(
            {
                x[
                    "section_id"
                ]
                for x in observations
            }
        )
    )

    group[
        "section_count"
    ] = len(
        group[
            "section_ids"
        ]
    )

    # --------------------------------------------------------
    # Circular mean angle
    # --------------------------------------------------------

    angles = [
        x[
            "angular_position_deg"
        ]
        for x in observations
        if
        x[
            "angular_position_deg"
        ] is not None
    ]

    if angles:

        radians = np.radians(
            angles
        )

        sin_mean = np.mean(
            np.sin(
                radians
            )
        )

        cos_mean = np.mean(
            np.cos(
                radians
            )
        )

        mean_angle = math.degrees(
            math.atan2(
                sin_mean,
                cos_mean
            )
        )

        if mean_angle < 0:

            mean_angle += 360.0

        group[
            "mean_angle_deg"
        ] = float(
            mean_angle
        )

    return group


# ============================================================
# CONSOLIDATE
# ============================================================

def consolidate_observations(
    observations
):

    print(
        "\n[3] Consolidating by physical axis/location..."
    )

    groups = []

    # --------------------------------------------------------
    # Sort by projected XY location.
    #
    # Z is deliberately NOT the primary sorting key.
    # --------------------------------------------------------

    observations = sorted(
        observations,
        key=lambda x: (
            x[
                "projected_center"
            ][0],
            x[
                "projected_center"
            ][1],
            x[
                "diameter"
            ]
        )
    )

    for observation in observations:

        best_group = None

        best_projected_distance = (
            float("inf")
        )

        for group in groups:

            if not compatible_with_group(
                observation,
                group
            ):

                continue

            projected_distance = float(
                np.linalg.norm(
                    np.asarray(
                        observation[
                            "projected_center"
                        ],
                        dtype=float
                    )
                    -
                    np.asarray(
                        group[
                            "mean_projected_center"
                        ],
                        dtype=float
                    )
                )
            )

            if (
                projected_distance
                <
                best_projected_distance
            ):

                best_projected_distance = (
                    projected_distance
                )

                best_group = group

        if best_group is None:

            groups.append(
                create_group(
                    observation
                )
            )

        else:

            update_group(
                best_group,
                observation
            )

    print(
        f"[OK] Physical axis groups: "
        f"{len(groups)}"
    )

    return groups


# ============================================================
# FEATURE POSITION
# ============================================================

def classify_position(
    group,
    extents
):

    radial_distance = (
        group[
            "mean_radial_distance"
        ]
    )

    transverse_size = max(
        min(
            extents[0],
            extents[1]
        ),
        1e-9
    )

    radial_ratio = (
        radial_distance
        /
        (
            transverse_size
            /
            2.0
        )
    )

    if radial_distance <= 5.0:

        return (
            "CENTRAL_CIRCULAR_FEATURE"
        )

    if radial_ratio >= 0.85:

        return (
            "OUTER_BOUNDARY_CANDIDATE"
        )

    return (
        "INTERIOR_CIRCULAR_FEATURE"
    )


# ============================================================
# DIAMETER PROFILE
# ============================================================

def classify_diameter_behavior(
    group
):

    variation = (
        group[
            "diameter_variation_ratio"
        ]
    )

    if variation <= 0.05:

        return (
            "CONSTANT_DIAMETER"
        )

    if variation <= 0.20:

        return (
            "MILDLY_VARIABLE_DIAMETER"
        )

    return (
        "VARIABLE_OR_TAPERED_DIAMETER"
    )


# ============================================================
# AXIAL CONTINUITY
# ============================================================

def calculate_axial_continuity(
    group
):

    sections = sorted(
        set(
            group[
                "section_ids"
            ]
        )
    )

    if len(sections) <= 1:

        return {
            "status":
                "WEAK",

            "consecutive_runs":
                1,

            "missing_internal_sections":
                0
        }

    runs = []

    current_run = 1

    for index in range(
        1,
        len(sections)
    ):

        if (
            sections[index]
            ==
            sections[index - 1] + 1
        ):

            current_run += 1

        else:

            runs.append(
                current_run
            )

            current_run = 1

    runs.append(
        current_run
    )

    longest_run = max(
        runs
    )

    missing_sections = (
        (
            max(sections)
            -
            min(sections)
            +
            1
        )
        -
        len(sections)
    )

    if longest_run >= 3:

        status = "STRONG"

    elif longest_run >= 2:

        status = "MODERATE"

    else:

        status = "WEAK"

    return {

        "status":
            status,

        "consecutive_runs":
            len(runs),

        "longest_consecutive_run":
            longest_run,

        "missing_internal_sections":
            missing_sections
    }


# ============================================================
# FINALIZE
# ============================================================

def finalize_features(
    groups,
    extents
):

    print(
        "\n[4] Finalizing physical circular features..."
    )

    groups.sort(
        key=lambda g: (
            g[
                "mean_radial_distance"
            ],
            (
                g[
                    "mean_angle_deg"
                ]
                if
                g[
                    "mean_angle_deg"
                ] is not None
                else -1.0
            )
        )
    )

    features = []

    for index, group in enumerate(
        groups,
        start=1
    ):

        axial_continuity = (
            calculate_axial_continuity(
                group
            )
        )

        feature = {

            "feature_id":
                f"PHYSICAL_AXIS_"
                f"{index:03d}",

            "position":
                classify_position(
                    group,
                    extents
                ),

            "diameter_behavior":
                classify_diameter_behavior(
                    group
                ),

            "continuity":
                axial_continuity[
                    "status"
                ],

            "section_count":
                group[
                    "section_count"
                ],

            "section_ids":
                group[
                    "section_ids"
                ],

            # ------------------------------------------------
            # This is the most important location value.
            # ------------------------------------------------

            "physical_axis_point":
                group[
                    "mean_projected_center"
                ],

            "radial_distance":
                group[
                    "mean_radial_distance"
                ],

            "radial_distance_min":
                group[
                    "min_radial_distance"
                ],

            "radial_distance_max":
                group[
                    "max_radial_distance"
                ],

            "angular_position_deg":
                group[
                    "mean_angle_deg"
                ],

            "diameter":
                group[
                    "mean_diameter"
                ],

            "diameter_min":
                group[
                    "min_diameter"
                ],

            "diameter_max":
                group[
                    "max_diameter"
                ],

            "diameter_variation_ratio":
                group[
                    "diameter_variation_ratio"
                ],

            "axial_range":
                {
                    "min":
                        group[
                            "axis_coordinate_min"
                        ],

                    "max":
                        group[
                            "axis_coordinate_max"
                        ],

                    "length":
                        (
                            group[
                                "axis_coordinate_max"
                            ]
                            -
                            group[
                                "axis_coordinate_min"
                            ]
                        )
                },

            "axial_continuity":
                axial_continuity,

            "observations":
                group[
                    "observations"
                ]
        }

        features.append(
            feature
        )

    return features


# ============================================================
# PREPARE STAGE 1 CYLINDERS
# ============================================================

def prepare_stage1_cylinders(
    stage1_data
):

    if not isinstance(
        stage1_data,
        dict
    ):

        return []

    cylinders = stage1_data.get(
        "cylinders",
        []
    )

    if not isinstance(
        cylinders,
        list
    ):

        return []

    prepared = []

    for cylinder in cylinders:

        diameter = safe_float(
            cylinder.get(
                "diameter"
            )
        )

        if diameter <= 0:

            continue

        origin = np.asarray(
            cylinder.get(
                "origin",
                [0.0, 0.0, 0.0]
            ),
            dtype=float
        )

        axis = normalize(
            cylinder.get(
                "axis",
                [0.0, 0.0, 1.0]
            )
        )

        if axis is None:

            continue

        prepared.append(
            {
                "component_id":
                    cylinder.get(
                        "component_id"
                    ),

                "diameter":
                    diameter,

                "length":
                    safe_float(
                        cylinder.get(
                            "length"
                        )
                    ),

                "origin":
                    origin,

                "axis":
                    axis,

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
                    )
            }
        )

    return prepared


# ============================================================
# STRICT STAGE 1 MATCHING
# ============================================================

def strict_stage1_match(
    feature,
    cylinder,
    principal_center,
    principal_axis
):

    feature_diameter = safe_float(
        feature[
            "diameter"
        ]
    )

    cylinder_diameter = safe_float(
        cylinder[
            "diameter"
        ]
    )

    # ========================================================
    # HARD DIAMETER TEST
    # ========================================================

    diameter_difference_ratio = (
        abs(
            feature_diameter -
            cylinder_diameter
        )
        /
        max(
            cylinder_diameter,
            1e-9
        )
    )

    if (
        diameter_difference_ratio
        >
        STAGE1_MAX_DIAMETER_DIFFERENCE_RATIO
    ):

        return None

    # ========================================================
    # PROJECTED CENTER
    # ========================================================

    feature_axis_point = np.asarray(
        feature[
            "physical_axis_point"
        ],
        dtype=float
    )

    cylinder_relative = (
        cylinder[
            "origin"
        ]
        -
        principal_center
    )

    cylinder_axial = float(
        np.dot(
            cylinder_relative,
            principal_axis
        )
    )

    cylinder_axis_point = (
        principal_center
        +
        (
            cylinder_relative
            -
            principal_axis
            *
            cylinder_axial
        )
    )

    projected_center_distance = float(
        np.linalg.norm(
            feature_axis_point -
            cylinder_axis_point
        )
    )

    if (
        projected_center_distance
        >
        STAGE1_MAX_PROJECTED_CENTER_DISTANCE
    ):

        return None

    # ========================================================
    # AXIAL POSITION
    # ========================================================

    feature_center = np.asarray(
        feature[
            "physical_axis_point"
        ],
        dtype=float
    )

    feature_relative = (
        feature_center -
        principal_center
    )

    feature_axial = float(
        np.dot(
            feature_relative,
            principal_axis
        )
    )

    axial_difference = abs(
        feature_axial -
        cylinder_axial
    )

    if (
        axial_difference
        >
        STAGE1_MAX_AXIAL_DIFFERENCE
    ):

        return None

    # ========================================================
    # RADIAL POSITION
    # ========================================================

    cylinder_radial_distance = float(
        np.linalg.norm(
            cylinder_axis_point -
            principal_center
        )
    )

    radial_difference = abs(
        feature[
            "radial_distance"
        ]
        -
        cylinder_radial_distance
    )

    if (
        radial_difference
        >
        STAGE1_MAX_RADIAL_DIFFERENCE
    ):

        return None

    # ========================================================
    # ANGLE
    # ========================================================

    angular_difference = None

    feature_angle = feature.get(
        "angular_position_deg"
    )

    if (
        feature_angle is not None
        and
        cylinder_radial_distance > 5.0
    ):

        u, v = build_basis(
            principal_axis
        )

        cylinder_radial_vector = (
            cylinder_axis_point -
            principal_center
        )

        cylinder_u = float(
            np.dot(
                cylinder_radial_vector,
                u
            )
        )

        cylinder_v = float(
            np.dot(
                cylinder_radial_vector,
                v
            )
        )

        cylinder_angle = math.degrees(
            math.atan2(
                cylinder_v,
                cylinder_u
            )
        )

        if cylinder_angle < 0:

            cylinder_angle += 360.0

        angular_difference = (
            angle_difference_deg(
                float(feature_angle),
                float(cylinder_angle)
            )
        )

        if (
            angular_difference
            >
            STAGE1_MAX_ANGULAR_DIFFERENCE_DEG
        ):

            return None

    # ========================================================
    # MATCH SCORE
    #
    # Only calculated after every hard test passes.
    # ========================================================

    diameter_score = max(
        0.0,
        1.0 -
        (
            diameter_difference_ratio
            /
            STAGE1_MAX_DIAMETER_DIFFERENCE_RATIO
        )
    )

    projected_score = max(
        0.0,
        1.0 -
        (
            projected_center_distance
            /
            STAGE1_MAX_PROJECTED_CENTER_DISTANCE
        )
    )

    axial_score = max(
        0.0,
        1.0 -
        (
            axial_difference
            /
            STAGE1_MAX_AXIAL_DIFFERENCE
        )
    )

    radial_score = max(
        0.0,
        1.0 -
        (
            radial_difference
            /
            STAGE1_MAX_RADIAL_DIFFERENCE
        )
    )

    if angular_difference is None:

        angular_score = 1.0

    else:

        angular_score = max(
            0.0,
            1.0 -
            (
                angular_difference
                /
                STAGE1_MAX_ANGULAR_DIFFERENCE_DEG
            )
        )

    score = (
        diameter_score * 0.45
        +
        projected_score * 0.25
        +
        axial_score * 0.10
        +
        radial_score * 0.10
        +
        angular_score * 0.10
    )

    return {

        "score":
            score,

        "diameter_difference_ratio":
            diameter_difference_ratio,

        "projected_center_distance":
            projected_center_distance,

        "axial_difference":
            axial_difference,

        "radial_difference":
            radial_difference,

        "angular_difference":
            angular_difference
    }


# ============================================================
# ATTACH STAGE 1
# ============================================================

def attach_stage1(
    features,
    stage1_cylinders,
    principal_center,
    principal_axis
):

    print(
        "\n[5] Matching physical axes with Stage 1 cylinders..."
    )

    if not stage1_cylinders:

        for feature in features:

            feature[
                "stage1_match"
            ] = None

        print(
            "[INFO] No Stage 1 cylinders available."
        )

        return features

    matches = 0

    for feature in features:

        valid = []

        for cylinder in stage1_cylinders:

            result = strict_stage1_match(
                feature,
                cylinder,
                principal_center,
                principal_axis
            )

            if result is not None:

                valid.append(
                    (
                        result[
                            "score"
                        ],
                        cylinder,
                        result
                    )
                )

        if not valid:

            feature[
                "stage1_match"
            ] = None

            feature[
                "stage1_match_status"
            ] = "NO_VALID_MATCH"

            continue

        valid.sort(
            key=lambda x: x[0],
            reverse=True
        )

        (
            score,
            cylinder,
            result
        ) = valid[0]

        feature[
            "stage1_match"
        ] = {

            "component_id":
                cylinder[
                    "component_id"
                ],

            "diameter":
                cylinder[
                    "diameter"
                ],

            "length":
                cylinder[
                    "length"
                ],

            "physical_score":
                cylinder[
                    "physical_score"
                ],

            "internal_ratio":
                cylinder[
                    "internal_ratio"
                ],

            "external_ratio":
                cylinder[
                    "external_ratio"
                ],

            "match_score":
                score,

            "diameter_difference_ratio":
                result[
                    "diameter_difference_ratio"
                ],

            "projected_center_distance":
                result[
                    "projected_center_distance"
                ],

            "axial_difference":
                result[
                    "axial_difference"
                ],

            "radial_difference":
                result[
                    "radial_difference"
                ],

            "angular_difference":
                result[
                    "angular_difference"
                ]
        }

        feature[
            "stage1_match_status"
        ] = "STRICT_MATCH"

        matches += 1

    print(
        f"[OK] Strict Stage 1 matches: "
        f"{matches}/{len(features)}"
    )

    return features


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    features
):

    print()
    print("=" * 78)

    print(
        "ALPHA - STAGE 3B RESULT"
    )

    print(
        "PHYSICAL AXIS / LOCATION CONSOLIDATION"
    )

    print("=" * 78)

    print()

    print(
        f"Physical circular features: "
        f"{len(features)}"
    )

    for feature in features:

        print()
        print(
            feature[
                "feature_id"
            ]
        )

        print(
            f"  Position          : "
            f"{feature['position']}"
        )

        print(
            f"  Axis point        : "
            f"{feature['physical_axis_point']}"
        )

        print(
            f"  Radial distance   : "
            f"{feature['radial_distance']:.4f}"
        )

        angle = feature[
            "angular_position_deg"
        ]

        if angle is None:

            print(
                "  Angular position  : CENTRAL"
            )

        else:

            print(
                f"  Angular position  : "
                f"{angle:.2f} deg"
            )

        print(
            f"  Diameter range    : "
            f"{feature['diameter_min']:.4f}"
            f" - "
            f"{feature['diameter_max']:.4f}"
        )

        print(
            f"  Mean diameter     : "
            f"{feature['diameter']:.4f}"
        )

        print(
            f"  Diameter behavior : "
            f"{feature['diameter_behavior']}"
        )

        print(
            f"  Sections          : "
            f"{feature['section_count']}"
        )

        print(
            f"  Section IDs       : "
            f"{feature['section_ids']}"
        )

        print(
            f"  Axial range       : "
            f"{feature['axial_range']['min']:.4f}"
            f" - "
            f"{feature['axial_range']['max']:.4f}"
        )

        print(
            f"  Axial length      : "
            f"{feature['axial_range']['length']:.4f}"
        )

        print(
            f"  Continuity        : "
            f"{feature['continuity']}"
        )

        match = feature.get(
            "stage1_match"
        )

        if match:

            print(
                "  Stage 1 match     : YES"
            )

            print(
                f"    Cylinder Ø      : "
                f"{match['diameter']:.4f}"
            )

            print(
                f"    Diameter diff   : "
                f"{match['diameter_difference_ratio'] * 100:.2f}%"
            )

            print(
                f"    Projected dist  : "
                f"{match['projected_center_distance']:.4f}"
            )

            print(
                f"    Match score     : "
                f"{match['match_score']:.3f}"
            )

        else:

            print(
                "  Stage 1 match     : NO"
            )

            print(
                f"    Status          : "
                f"{feature.get('stage1_match_status')}"
            )

    print()
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)

    print(
        "ALPHA - STAGE 3B"
    )

    print(
        "PHYSICAL AXIS / LOCATION CONSOLIDATION"
    )

    print("=" * 78)

    try:

        # ----------------------------------------------------
        # 1. Load Stage 3
        # ----------------------------------------------------

        stage3 = load_stage3()

        (
            principal_center,
            principal_axis,
            extents
        ) = get_principal_geometry(
            stage3
        )

        print(
            "\n[INFO] Principal axis:"
        )

        print(
            f"       {vector_to_list(principal_axis)}"
        )

        # ----------------------------------------------------
        # 2. Extract raw circles
        # ----------------------------------------------------

        observations = (
            extract_observations(
                stage3,
                principal_center,
                principal_axis
            )
        )

        if not observations:

            print(
                "\n[WARNING] No circular observations found."
            )

            return

        # ----------------------------------------------------
        # 3. Consolidate by physical axis
        # ----------------------------------------------------

        groups = (
            consolidate_observations(
                observations
            )
        )

        # ----------------------------------------------------
        # 4. Finalize
        # ----------------------------------------------------

        features = finalize_features(
            groups,
            extents
        )

        # ----------------------------------------------------
        # 5. Load Stage 1
        # ----------------------------------------------------

        stage1 = load_stage1()

        stage1_cylinders = (
            prepare_stage1_cylinders(
                stage1
            )
        )

        # ----------------------------------------------------
        # 6. Strict Stage 1 matching
        # ----------------------------------------------------

        features = attach_stage1(
            features,
            stage1_cylinders,
            principal_center,
            principal_axis
        )

        # ----------------------------------------------------
        # 7. Build output
        # ----------------------------------------------------

        output = {

            "system":
                "ALPHA",

            "stage":
                "3B",

            "stage_name":
                "PHYSICAL AXIS / LOCATION CONSOLIDATION",

            "source":
                str(STAGE3_FILE),

            "principle":
                "Section circles belonging to the same "
                "physical axis are consolidated primarily "
                "using projected center location. "
                "Axial Z variation is retained as feature "
                "extent rather than used as feature identity.",

            "settings":
                {

                    "projected_center_tolerance":
                        PROJECTED_CENTER_TOLERANCE,

                    "radial_distance_tolerance":
                        RADIAL_DISTANCE_TOLERANCE,

                    "angular_tolerance_deg":
                        ANGULAR_TOLERANCE_DEG,

                    "max_diameter_variation_ratio":
                        MAX_DIAMETER_VARIATION_RATIO
                },

            "principal_geometry":
                {

                    "center":
                        vector_to_list(
                            principal_center
                        ),

                    "axis":
                        vector_to_list(
                            principal_axis
                        ),

                    "extents":
                        [
                            float(x)
                            for x in extents
                        ]
                },

            "raw_observation_count":
                len(observations),

            "physical_feature_count":
                len(features),

            "features":
                features
        }

        # ----------------------------------------------------
        # 8. Save
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

        print(
            f"\n[OK] Saved:\n"
            f"{OUTPUT_FILE}"
        )

        # ----------------------------------------------------
        # 9. Print
        # ----------------------------------------------------

        print_results(
            features
        )

        print()
        print(
            "[SUCCESS] Stage 3B completed."
        )

    except KeyboardInterrupt:

        print(
            "\n[STOPPED] User interrupted."
        )

    except Exception as exc:

        print()
        print("=" * 78)

        print(
            "STAGE 3B ERROR"
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