from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

GEOMETRY_FILE = OUTPUT_DIR / "consolidated_feature_geometry.json"
LOCAL_FILE = OUTPUT_DIR / "local_hole_analysis.json"
TOPOLOGY_FILE = OUTPUT_DIR / "topology_feature_analysis.json"
OUTPUT_FILE = OUTPUT_DIR / "final_feature_map.json"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required analysis file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON object: {path}")
    return data


def index_features(data):
    features = data.get("features", [])
    if not isinstance(features, list):
        return {}
    return {
        str(item.get("feature_id")): item
        for item in features
        if isinstance(item, dict) and item.get("feature_id")
    }


def geometry_quality(feature):
    continuity = str(feature.get("continuity", "")).upper()
    variation = safe_float(feature.get("diameter_variation_ratio"), 1.0)

    if continuity == "STRONG":
        score = 0.95
    elif continuity == "MODERATE":
        score = 0.80
    else:
        score = 0.60

    if variation <= 0.05:
        score += 0.04
    elif variation > 0.20:
        score -= 0.12

    return max(0.0, min(0.99, score))


def fuse_feature(geometry, local, topology):
    feature_id = geometry.get("feature_id") or topology.get("feature_id")

    topology_class = str(topology.get("classification", "UNKNOWN"))
    topology_subtype = str(topology.get("subtype", "UNKNOWN"))
    topology_conf = safe_float(topology.get("confidence"), 0.0)

    local_class = str(local.get("classification", "UNKNOWN"))
    local_subtype = str(local.get("subtype", "UNKNOWN"))
    local_conf = safe_float(local.get("confidence"), 0.0)

    geom_conf = geometry_quality(geometry)

    # Stage 5C is the topology authority. Stage 5B is supporting
    # local termination evidence and is never allowed to upgrade an
    # ambiguous topology result by itself.
    if (
        topology_class == "CYLINDRICAL_SURFACE"
        and topology_subtype == "OUTER_CYLINDRICAL_SURFACE"
    ):
        final_class = "CYLINDRICAL_SURFACE"
        final_subtype = "OUTER_CYLINDRICAL_SURFACE"
        reason = (
            "Topology validation identifies the feature as an external "
            "cylindrical surface; local hole evidence cannot override it."
        )

    elif topology_class == "HOLE":
        final_class = "HOLE"
        final_subtype = topology_subtype

        if topology_subtype == "BLIND_HOLE":
            if local_subtype == "BLIND_HOLE_CANDIDATE":
                reason = (
                    "Topology validation identifies a blind hole and local "
                    "section analysis independently supports internal termination."
                )
            else:
                reason = (
                    "Topology validation identifies a blind hole; local "
                    "termination evidence is supportive or inconclusive."
                )
        elif topology_subtype == "THROUGH_HOLE":
            reason = (
                "Topology validation identifies a through-hole."
            )
        elif topology_subtype == "INTERNAL_HOLE":
            reason = (
                "Topology validation identifies an internal hole without "
                "sufficient evidence to upgrade it to a through-hole."
            )
        else:
            reason = "Topology validation identifies a hole feature."

    elif topology_class == "MIXED_TOPOLOGY":
        final_class = "MIXED_TOPOLOGY"
        final_subtype = "REQUIRES_SECTION_ANALYSIS"
        reason = "Topology evidence is mixed; no stronger deterministic classification was established."

    else:
        final_class = topology_class or "UNKNOWN"
        final_subtype = topology_subtype or "UNKNOWN"
        reason = "Topology evidence did not establish a stronger feature classification."

    # Conservative fusion confidence. The topology result carries the
    # highest weight because it is the authoritative discriminator.
    confidence = (
        0.55 * topology_conf
        + 0.25 * local_conf
        + 0.20 * geom_conf
    )

    # Agreement bonus for hole + local blind candidate.
    if (
        final_class == "HOLE"
        and final_subtype == "BLIND_HOLE"
        and local_subtype == "BLIND_HOLE_CANDIDATE"
    ):
        confidence += 0.02

    # Keep confidence bounded and conservative.
    confidence = max(0.0, min(0.99, confidence))

    axial_range = geometry.get("axial_range", {})
    local_range = local.get("detected_axial_range", {})

    return {
        "feature_id": feature_id,
        "classification": final_class,
        "subtype": final_subtype,
        "confidence": round(confidence, 4),
        "mechanical_position": geometry.get("position"),
        "diameter_behavior": geometry.get("diameter_behavior"),
        "continuity": geometry.get("continuity"),
        "diameter": safe_float(geometry.get("diameter")),
        "diameter_min": safe_float(geometry.get("diameter_min")),
        "diameter_max": safe_float(geometry.get("diameter_max")),
        "diameter_variation_ratio": safe_float(geometry.get("diameter_variation_ratio")),
        "physical_axis_point": geometry.get("physical_axis_point"),
        "radial_distance": safe_float(geometry.get("radial_distance")),
        "angular_position_deg": safe_float(geometry.get("angular_position_deg")),
        "axial_range": axial_range,
        "axial_length": safe_float(axial_range.get("length")),
        "local_detected_axial_range": local_range,
        "local_detected_length": safe_float(local_range.get("length")),
        "opening": {
            "lower_side_open": bool(topology.get("lower_side_open", False)),
            "upper_side_open": bool(topology.get("upper_side_open", False)),
            "lower_side_material": bool(topology.get("lower_side_material", False)),
            "upper_side_material": bool(topology.get("upper_side_material", False)),
        },
        "evidence": {
            "geometry": {
                "continuity": geometry.get("continuity"),
                "section_count": geometry.get("section_count"),
                "diameter_variation_ratio": safe_float(geometry.get("diameter_variation_ratio")),
                "quality_score": round(geom_conf, 4),
            },
            "local_analysis": {
                "classification": local_class,
                "subtype": local_subtype,
                "confidence": local_conf,
                "detected_sections": local.get("detected_sections"),
                "continuity": local.get("continuity"),
                "diameter_profile": local.get("diameter_profile"),
            },
            "topology": {
                "classification": topology_class,
                "subtype": topology_subtype,
                "confidence": topology_conf,
                "valid_sections": topology.get("valid_sections"),
                "loop_sections": topology.get("loop_sections"),
                "internal_sections": topology.get("internal_sections"),
                "outer_sections": topology.get("outer_sections"),
                "internal_ratio": safe_float(topology.get("internal_ratio")),
                "outer_ratio": safe_float(topology.get("outer_ratio")),
            },
        },
        "decision_reason": reason,
    }


def build_final_map():
    geometry_data = load_json(GEOMETRY_FILE)
    local_data = load_json(LOCAL_FILE)
    topology_data = load_json(TOPOLOGY_FILE)

    geometry = index_features(geometry_data)
    local = index_features(local_data)
    topology = index_features(topology_data)

    feature_ids = list(geometry.keys())
    if not feature_ids:
        feature_ids = list(topology.keys())

    features = []
    for feature_id in feature_ids:
        g = geometry.get(feature_id, {})
        l = local.get(feature_id, {})
        t = topology.get(feature_id, {})
        features.append(fuse_feature(g, l, t))

    features.sort(key=lambda x: str(x.get("feature_id", "")))

    counts = {
        "total": len(features),
        "holes": sum(x["classification"] == "HOLE" for x in features),
        "through_holes": sum(
            x["classification"] == "HOLE" and x["subtype"] == "THROUGH_HOLE"
            for x in features
        ),
        "blind_holes": sum(
            x["classification"] == "HOLE" and x["subtype"] == "BLIND_HOLE"
            for x in features
        ),
        "internal_holes": sum(
            x["classification"] == "HOLE" and x["subtype"] == "INTERNAL_HOLE"
            for x in features
        ),
        "outer_cylindrical_surfaces": sum(
            x["classification"] == "CYLINDRICAL_SURFACE"
            and x["subtype"] == "OUTER_CYLINDRICAL_SURFACE"
            for x in features
        ),
        "mixed_topology": sum(x["classification"] == "MIXED_TOPOLOGY" for x in features),
        "unknown": sum(x["classification"] == "UNKNOWN" for x in features),
    }

    confidences = [safe_float(x.get("confidence")) for x in features]
    counts["mean_confidence"] = (
        round(sum(confidences) / len(confidences), 4)
        if confidences else 0.0
    )

    principal = geometry_data.get("principal_geometry", {})

    return {
        "system": "ALPHA",
        "stage": "6",
        "stage_name": "FINAL FEATURE FUSION",
        "method": "Deterministic fusion of physical geometry, local section evidence and topology validation.",
        "topology_authority": "Stage 5C topology validation",
        "ai_required": False,
        "principal_geometry": principal,
        "summary": counts,
        "features": features,
        "source_evidence": {
            "stage_3b": GEOMETRY_FILE.name,
            "stage_5b": LOCAL_FILE.name,
            "stage_5c": TOPOLOGY_FILE.name,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_map = build_final_map()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_map, f, indent=2, ensure_ascii=False)

    summary = final_map["summary"]
    print("=" * 78)
    print("ALPHA - STAGE 6 FINAL FEATURE FUSION")
    print("=" * 78)
    print(f"Unified features : {summary['total']}")
    for feature in final_map["features"]:
        print(
            f"{feature['feature_id']:>24} | "
            f"{feature['classification']:<22} | "
            f"{feature['subtype']:<32} | "
            f"{feature['confidence']:.3f}"
        )
    print("-" * 78)
    print(f"Output written to: {OUTPUT_FILE}")
    print("=" * 78)


if __name__ == "__main__":
    main()
