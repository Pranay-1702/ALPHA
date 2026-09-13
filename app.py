
import json
from pathlib import Path

import streamlit as st


# ============================================================
# ALPHA — CAD INTELLIGENCE DEMO UI
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
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .alpha-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 0.1rem;
    }

    .alpha-subtitle {
        font-size: 1rem;
        opacity: 0.72;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
    }

    .status-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
    }

    .small-muted {
        font-size: 0.82rem;
        opacity: 0.65;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 10px;
        padding: 0.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "CAD Analysis"

if "uploaded_file_bytes" not in st.session_state:
    st.session_state.uploaded_file_bytes = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="alpha-title">⚙️ ALPHA</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="alpha-subtitle">'
    "CAD Intelligence • B-Rep • Feature Recognition • Assembly Analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Analysis")

    analysis_mode = st.radio(
        "Mode",
        [
            "CAD Analysis",
            "AI Interpretation",
        ],
        index=0,
    )

    st.session_state.analysis_mode = analysis_mode

    st.divider()

    st.subheader("AI Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Enter API key for AI features",
        help=(
            "Optional. Core CAD analysis does not require a Gemini "
            "API key. The key is only used for AI interpretation "
            "when that backend is connected."
        ),
    )

    enable_ai = st.checkbox(
        "Enable AI interpretation",
        value=False,
        disabled=not bool(api_key),
    )

    if enable_ai and api_key:
        st.success("AI interpretation enabled")

    elif not api_key:
        st.info(
            "API key is optional for the CAD-analysis layer."
        )

    st.divider()

    st.subheader("Supported formats")

    st.write("• STEP / STP")
    st.write("• STL")

    st.divider()

    st.caption(
        "ALPHA — engineering CAD intelligence prototype"
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">1. Upload CAD Model</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload an existing CAD model",
    type=["step", "stp", "stl"],
    help="Upload a STEP/STP or STL file for analysis.",
)


if uploaded_file is not None:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.uploaded_file_bytes = uploaded_file.getvalue()

    suffix = Path(uploaded_file.name).suffix.lower()
    file_size_mb = uploaded_file.size / (1024 * 1024)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "File",
            uploaded_file.name,
        )

    with col2:
        st.metric(
            "Format",
            suffix.replace(".", "").upper(),
        )

    with col3:
        st.metric(
            "Size",
            f"{file_size_mb:.2f} MB",
        )

    st.success(
        f"Model ready: **{uploaded_file.name}**"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.markdown(
    '<div class="section-title">2. Analyze Model</div>',
    unsafe_allow_html=True,
)

analyze_col, status_col = st.columns(
    [1, 3],
    vertical_alignment="center",
)

with analyze_col:
    analyze_clicked = st.button(
        "🔍 Analyze CAD",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

with status_col:
    if uploaded_file is None:
        st.caption("Upload a STEP/STP or STL file to begin.")

    else:
        st.caption(
            "UI prototype ready. The OpenCascade analysis "
            "backend will be connected in the next phase."
        )


if analyze_clicked and uploaded_file is not None:
    st.session_state.analysis_started = True

    st.success(
        "Analysis request accepted. CAD backend integration "
        "will populate the results below."
    )


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">3. CAD Model Intelligence</div>',
    unsafe_allow_html=True,
)


if not st.session_state.analysis_started:

    st.info(
        "Upload a CAD model and select **Analyze CAD**. "
        "The results panel will display geometry, topology, "
        "features, assembly structure and relationships."
    )

else:

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    st.subheader("Model Summary")

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.metric("Components", "—")

    with m2:
        st.metric("Solids", "—")

    with m3:
        st.metric("Faces", "—")

    with m4:
        st.metric("Edges", "—")

    with m5:
        st.metric("Vertices", "—")

    st.caption(
        "These values will be populated by the OpenCascade "
        "B-Rep reader."
    )

    # --------------------------------------------------------
    # Main analysis columns
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 2],
        gap="large",
    )

    with left:

        st.subheader("Assembly / Feature Tree")

        tree_data = {
            "Assembly": {
                "Components": [
                    "Component 001",
                    "Component 002",
                    "Component 003",
                ],
                "Features": [
                    "Planes",
                    "Cylinders",
                    "Holes",
                    "Interfaces",
                ],
            }
        }

        st.json(tree_data)

        st.subheader("Detected Features")

        st.write("• Cylindrical surfaces")
        st.write("• Planar surfaces")
        st.write("• Hole candidates")
        st.write("• Boss candidates")
        st.write("• Interface candidates")

    with right:

        st.subheader("3D CAD Viewer")

        viewer = st.container(
            height=430,
            border=True,
        )

        with viewer:
            st.markdown(
                """
                <div style="
                    height:390px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    opacity:0.55;
                    font-size:1.1rem;
                ">
                    <div>
                        <div style="font-size:3rem;">◈</div>
                        <div><b>3D CAD Viewer</b></div>
                        <div class="small-muted">
                            STEP/B-Rep visualization will be connected here.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Feature table
    # --------------------------------------------------------

    st.subheader("Feature Recognition")

    feature_rows = [
        {
            "Feature ID": "—",
            "Type": "Awaiting CAD backend",
            "Geometry": "—",
            "Confidence": "—",
        }
    ]

    st.dataframe(
        feature_rows,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Relationships
    # --------------------------------------------------------

    st.subheader("Component Relationships")

    relationship_rows = [
        {
            "Component A": "—",
            "Relationship": "Awaiting B-Rep analysis",
            "Component B": "—",
            "Confidence": "—",
        }
    ]

    st.dataframe(
        relationship_rows,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Machine-readable output
    # --------------------------------------------------------

    st.subheader("Machine-Readable CAD Data")

    demo_json = {
        "status": "UI_READY_BACKEND_PENDING",
        "model": (
            st.session_state.uploaded_file_name
            or "unknown"
        ),
        "assembly": {
            "components": [],
            "relationships": [],
        },
        "features": [],
        "geometry": {},
    }

    json_text = json.dumps(
        demo_json,
        indent=2,
    )

    st.code(
        json_text,
        language="json",
    )

    st.download_button(
        "⬇️ Download JSON",
        data=json_text,
        file_name="alpha_cad_analysis.json",
        mime="application/json",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:
    st.caption(
        "ALPHA | CAD Intelligence Prototype"
    )

with footer_right:
    st.caption(
        "STEP/B-Rep → Features → Relationships → Robotics-ready data"
    )
