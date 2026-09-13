import contextlib
import hashlib
import io
import json
import threading
from pathlib import Path

import streamlit as st


# ============================================================
# ALPHA — STL FEATURE INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="ALPHA | CAD Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .alpha-title {
        font-size: 2.7rem;
        font-weight: 850;
        letter-spacing: 0.10em;
        margin-bottom: 0.1rem;
    }

    .alpha-subtitle {
        font-size: 1rem;
        opacity: 0.68;
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 1.22rem;
        font-weight: 750;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
    }

    .result-card {
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 12px;
        padding: 0.8rem 0.95rem;
        min-height: 92px;
    }

    .small-muted {
        font-size: 0.80rem;
        opacity: 0.62;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 10px;
        padding: 0.65rem;
    }

    .stDownloadButton button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "analysis_key": None,
    "analysis_result": None,
    "analysis_logs": "",
    "analysis_error": None,
    "uploaded_name": None,
    "uploaded_bytes": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# A single shared working directory is used because the existing
# deterministic stage files were designed around input/test_part.stl.
# The lock prevents two simultaneous users from overwriting the same
# working analysis files.
ANALYSIS_LOCK = threading.Lock()
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_FILE = INPUT_DIR / "test_part.stl"

OUTPUT_FILES = [
    "mechanical_feature_evidence.json",
    "ai_feature_reasoning.json",
    "feature_geometry_evidence.json",
    "consolidated_feature_geometry.json",
    "local_hole_analysis.json",
    "topology_feature_analysis.json",
    "hole_termination_analysis.json",
    "final_feature_map.json",
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="alpha-title">⚙️ ALPHA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="alpha-subtitle">'
    "CAD Intelligence • Geometry • Topology • Feature Recognition"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("ALPHA")

    st.markdown(
        "**STL Feature Intelligence**"
    )

    st.caption(
        "Deterministic geometry and topology analysis."
    )

    st.divider()

    st.subheader("Analysis stages")
    st.write("✓ Surface-region analysis")
    st.write("✓ Cylindrical evidence")
    st.write("✓ Circular section geometry")
    st.write("✓ Physical feature consolidation")
    st.write("✓ Local opening / termination analysis")
    st.write("✓ Topology validation")
    st.write("✓ Final feature fusion")

    st.divider()

    st.subheader("Input")
    st.write("STL mesh")

    st.divider()

    st.caption("ALPHA engineering prototype")


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload STL Model</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Drop an STL file here",
    type=["stl"],
    accept_multiple_files=False,
    help="Upload one STL mechanical part for feature analysis.",
)


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_key = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.uploaded_name != uploaded_file.name:
        st.session_state.analysis_result = None
        st.session_state.analysis_logs = ""
        st.session_state.analysis_error = None

    st.session_state.uploaded_name = uploaded_file.name
    st.session_state.uploaded_bytes = file_bytes

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("File", uploaded_file.name)

    with c2:
        st.metric("Format", "STL")

    with c3:
        st.metric("Size", f"{uploaded_file.size / (1024 * 1024):.2f} MB")

    st.caption(f"File ID: {file_key[:12]}…")


# ============================================================
# ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Analyze</div>',
    unsafe_allow_html=True,
)

analyze_clicked = st.button(
    "🔍 Analyze STL",
    type="primary",
    use_container_width=True,
    disabled=uploaded_file is None,
)


if analyze_clicked and uploaded_file is not None:
    file_bytes = st.session_state.uploaded_bytes
    file_key = hashlib.sha256(file_bytes).hexdigest()

    if (
        st.session_state.analysis_result is not None
        and st.session_state.analysis_key == file_key
    ):
        st.info("This STL has already been analyzed in this session.")
    else:
        # Heavy modules are imported only after the user starts analysis.
        # This keeps the initial upload page responsive.
        with st.status("Running ALPHA feature analysis…", expanded=True) as status:
            try:
                import importlib

                import src.mechanical_feature_engine as stage1
                import src.feature_geometry_extractor as stage3
                import src.cylinder_feature_consolidator as stage3b
                import src.local_hole_analysis as stage5b
                import src.topology_feature_validator as stage5c
                import src.final_feature_fusion as stage6

                INPUT_DIR.mkdir(parents=True, exist_ok=True)
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                # Make sure an older user's results cannot contaminate this run.
                for filename in OUTPUT_FILES:
                    target = OUTPUT_DIR / filename
                    if target.exists():
                        target.unlink()

                INPUT_FILE.write_bytes(file_bytes)

                # The existing stage implementations use a fixed project input
                # path. We redirect their input selectors to the uploaded file.
                stage1.find_stl = lambda: INPUT_FILE
                stage3.find_stl = lambda: INPUT_FILE
                stage5b.STL_FILE = INPUT_FILE
                stage5c.STL_PATH = str(INPUT_FILE)

                logs = io.StringIO()

                def run_stage(label, module, output_file):
                    status.write(f"**{label}**")
                    with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
                        module.main()
                    if not output_file.exists():
                        raise RuntimeError(
                            f"{label} did not produce {output_file.name}."
                        )
                    status.write(f"✓ {label} complete")

                with ANALYSIS_LOCK:
                    run_stage(
                        "Stage 1 — surface and cylindrical evidence",
                        stage1,
                        OUTPUT_DIR / "mechanical_feature_evidence.json",
                    )
                    run_stage(
                        "Stage 3 — circular section geometry",
                        stage3,
                        OUTPUT_DIR / "feature_geometry_evidence.json",
                    )
                    run_stage(
                        "Stage 3B — physical feature consolidation",
                        stage3b,
                        OUTPUT_DIR / "consolidated_feature_geometry.json",
                    )
                    run_stage(
                        "Stage 5B — local opening / termination analysis",
                        stage5b,
                        OUTPUT_DIR / "local_hole_analysis.json",
                    )
                    run_stage(
                        "Stage 5C — topology validation",
                        stage5c,
                        OUTPUT_DIR / "topology_feature_analysis.json",
                    )
                    run_stage(
                        "Stage 6 — final feature fusion",
                        stage6,
                        OUTPUT_DIR / "final_feature_map.json",
                    )

                with open(
                    OUTPUT_DIR / "final_feature_map.json",
                    "r",
                    encoding="utf-8",
                ) as f:
                    final_result = json.load(f)

                st.session_state.analysis_key = file_key
                st.session_state.analysis_result = final_result
                st.session_state.analysis_logs = logs.getvalue()
                st.session_state.analysis_error = None

                status.update(
                    label="ALPHA analysis complete",
                    state="complete",
                    expanded=False,
                )

            except Exception as exc:
                st.session_state.analysis_result = None
                st.session_state.analysis_key = None
                st.session_state.analysis_logs = logs.getvalue() if "logs" in locals() else ""
                st.session_state.analysis_error = str(exc)
                status.update(
                    label="ALPHA analysis failed",
                    state="error",
                    expanded=True,
                )


# ============================================================
# ERROR DISPLAY
# ============================================================

if st.session_state.analysis_error:
    st.error(
        f"Analysis error: {st.session_state.analysis_error}"
    )

    if st.session_state.analysis_logs:
        with st.expander("Analysis log"):
            st.code(st.session_state.analysis_logs)


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.analysis_result

if result is not None:
    st.divider()

    summary = result.get("summary", {})
    principal = result.get("principal_geometry", {})

    st.markdown(
        '<div class="section-title">Analysis Results</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Final feature summary
    # --------------------------------------------------------

    cols = st.columns(7)

    metrics = [
        ("Features", summary.get("total", 0)),
        ("Holes", summary.get("holes", 0)),
        ("Through", summary.get("through_holes", 0)),
        ("Blind", summary.get("blind_holes", 0)),
        ("Internal", summary.get("internal_holes", 0)),
        ("Outer Cyl.", summary.get("outer_cylindrical_surfaces", 0)),
        ("Confidence", f"{summary.get('mean_confidence', 0.0):.3f}"),
    ]

    for column, (label, value) in zip(cols, metrics):
        with column:
            st.metric(label, value)

    # --------------------------------------------------------
    # Geometry summary
    # --------------------------------------------------------

    st.subheader("Part Geometry")

    extents = principal.get("extents", [])
    axis = principal.get("axis", [])
    center = principal.get("center", [])

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.metric(
            "X extent",
            f"{extents[0]:.3f}" if len(extents) > 0 else "—",
        )

    with g2:
        st.metric(
            "Y extent",
            f"{extents[1]:.3f}" if len(extents) > 1 else "—",
        )

    with g3:
        st.metric(
            "Z extent",
            f"{extents[2]:.3f}" if len(extents) > 2 else "—",
        )

    with g4:
        st.metric(
            "Principal axis",
            str([round(float(x), 4) for x in axis]) if axis else "—",
        )

    if center:
        st.caption(
            "Principal center: "
            + str([round(float(x), 4) for x in center])
        )

    # --------------------------------------------------------
    # Load the analyzed STL for statistics and visualization.
    # --------------------------------------------------------

    import trimesh

    mesh = trimesh.load_mesh(
        io.BytesIO(st.session_state.uploaded_bytes),
        file_type="stl",
        process=False,
    )

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            list(mesh.geometry.values())
        )

    if isinstance(mesh, trimesh.Trimesh):
        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric("STL triangles", f"{len(mesh.faces):,}")

        with p2:
            st.metric("STL vertices", f"{len(mesh.vertices):,}")

        with p3:
            st.metric("Watertight", "Yes" if mesh.is_watertight else "No")

        with p4:
            st.metric("Surface area", f"{mesh.area:.3f}")

    # --------------------------------------------------------
    # 3D viewer
    # --------------------------------------------------------

    st.subheader("3D Model")

    try:
        import plotly.graph_objects as go

        max_faces = 18000
        faces = mesh.faces

        if len(faces) > max_faces:
            step = max(1, len(faces) // max_faces)
            faces = faces[::step][:max_faces]

        used_vertices = sorted(set(faces.reshape(-1).tolist()))
        vertex_map = {old: new for new, old in enumerate(used_vertices)}
        viewer_vertices = mesh.vertices[used_vertices]

        i = [vertex_map[int(x)] for x in faces[:, 0]]
        j = [vertex_map[int(x)] for x in faces[:, 1]]
        k = [vertex_map[int(x)] for x in faces[:, 2]]

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=viewer_vertices[:, 0],
                    y=viewer_vertices[:, 1],
                    z=viewer_vertices[:, 2],
                    i=i,
                    j=j,
                    k=k,
                    flatshading=False,
                    hoverinfo="skip",
                )
            ]
        )

        fig.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "responsive": True,
            },
        )

        if len(mesh.faces) > max_faces:
            st.caption(
                f"Viewer display is reduced to approximately {len(faces):,} triangles; "
                "analysis uses the uploaded STL."
            )

    except Exception as exc:
        st.warning(f"3D viewer could not be rendered: {exc}")

    # --------------------------------------------------------
    # Final feature table
    # --------------------------------------------------------

    st.subheader("Final Feature Map")

    rows = []
    for feature in result.get("features", []):
        rows.append(
            {
                "Feature ID": feature.get("feature_id"),
                "Position": feature.get("mechanical_position"),
                "Classification": feature.get("classification"),
                "Subtype": feature.get("subtype"),
                "Diameter": round(float(feature.get("diameter", 0.0)), 3),
                "Axial Length": round(float(feature.get("axial_length", 0.0)), 3),
                "Confidence": round(float(feature.get("confidence", 0.0)), 3),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Detailed evidence
    # --------------------------------------------------------

    st.subheader("Feature Evidence")

    for feature in result.get("features", []):
        feature_id = feature.get("feature_id", "Feature")
        title = (
            f"{feature_id} — {feature.get('classification', 'UNKNOWN')} / "
            f"{feature.get('subtype', 'UNKNOWN')}"
        )

        with st.expander(title):
            a, b, c = st.columns(3)

            with a:
                st.write("**Geometry**")
                st.json(feature.get("evidence", {}).get("geometry", {}))

            with b:
                st.write("**Local analysis**")
                st.json(feature.get("evidence", {}).get("local_analysis", {}))

            with c:
                st.write("**Topology**")
                st.json(feature.get("evidence", {}).get("topology", {}))

            st.info(feature.get("decision_reason", "No decision reason recorded."))

    # --------------------------------------------------------
    # JSON downloads
    # --------------------------------------------------------

    st.subheader("Machine-Readable Output")

    final_json = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "⬇️ Download final_feature_map.json",
        data=final_json,
        file_name="final_feature_map.json",
        mime="application/json",
    )

    if st.session_state.analysis_logs:
        with st.expander("Technical analysis log"):
            st.code(st.session_state.analysis_logs)

else:
    st.info(
        "Upload an STL mechanical part and click **Analyze STL** to generate the feature map."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ALPHA | Deterministic STL geometry and topology feature analysis"
)
