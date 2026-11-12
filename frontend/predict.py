"""Predict page: upload a cell image, see the result, confirm or correct it."""

from pathlib import Path

import streamlit as st

import api
from uploads import upload_error

LOW_CONFIDENCE = 0.70  # below this the result is flagged for a manual re-check
SAMPLES = Path(__file__).parent / "samples"
LABEL_TEXT = {"parasitized": "Parasite detected", "uninfected": "No parasite detected"}
OTHER = {"parasitized": "uninfected", "uninfected": "parasitized"}


def sidebar():
    with st.sidebar:
        st.subheader("Service status")
        if not api.ready():
            st.error("Analysis service unavailable")
            return
        try:
            info = api.model_info()
        except api.BackendError as e:
            st.warning(str(e))
            return
        st.success("Ready")
        st.caption(
            f"Model version {info['version']} · test accuracy "
            f"{info['test_accuracy']:.1%} · catches "
            f"{info['test_recall_parasitized']:.1%} of infected cells"
        )


def pick_image():
    uploaded = st.file_uploader(
        "Upload a single-cell image (PNG or JPEG, up to 5 MB)",
        type=["png", "jpg", "jpeg"],
    )
    samples = sorted(SAMPLES.glob("*.png"))
    if samples and not uploaded:
        choice = st.selectbox(
            "…or try a sample image",
            ["(none)"] + [p.name for p in samples],
            help="Cells from the public NIH test set",
        )
        if choice != "(none)":
            path = SAMPLES / choice
            return path.name, path.read_bytes(), "image/png"
    if uploaded:
        return uploaded.name, uploaded.getvalue(), uploaded.type
    return None


def show_result(result):
    label = result["label"]
    if label == "parasitized":
        st.error(f"### 🦠 {LABEL_TEXT[label]}")
    else:
        st.success(f"### ✅ {LABEL_TEXT[label]}")
    conf = result["confidence"]
    st.progress(min(max(conf, 0.0), 1.0), text=f"Confidence {conf:.0%}")
    if conf < LOW_CONFIDENCE:
        st.warning("Low confidence: please have this cell checked by a microscopist.")
    with st.expander("Details"):
        st.write(
            f"Probability of parasite: **{result['p_parasitized']:.3f}** "
            f"(flagged as parasitized at ≥ {result['threshold']:.3f})"
        )
        st.caption(
            f"Model version {result['model_version']} · "
            f"answered in {result['latency_ms']:.0f} ms · "
            f"request {result['request_id']}"
        )


def feedback(image, result):
    st.markdown("**Is this result correct?**")
    sent = st.session_state.setdefault("feedback_sent", set())
    if result["request_id"] in sent:
        st.info(
            "Thanks, your feedback was saved and will be used to improve the model."
        )
        return
    yes, no = st.columns(2)
    choice = None
    if yes.button("👍 Yes, correct", width="stretch"):
        choice = result["label"]
    if no.button(
        f"👎 No, it is {OTHER[result['label']]}",
        width="stretch",
    ):
        choice = OTHER[result["label"]]
    if choice:
        try:
            api.send_feedback(*image, result, choice)
        except api.BackendError as e:
            st.error(f"Feedback could not be saved: {e}")
            return
        sent.add(result["request_id"])
        st.rerun()


sidebar()
st.title("🔬 MalariaScope")
st.markdown(
    "Screening aid for **malaria parasites in thin blood smear images**. "
    "Upload a picture of **one red blood cell** and press **Analyse**."
)
st.caption(
    "This tool supports, and does not replace, examination by a trained microscopist."
)

left, right = st.columns([1, 1], gap="large")
with left:
    image = pick_image()
    if image:
        name, data, _ = image
        error = upload_error(data)
        if error:
            st.error(error)
            image = None
        else:
            st.image(data, caption=name, width=260)
    analyse = st.button(
        "Analyse", type="primary", disabled=image is None, width="stretch"
    )

with right:
    if analyse and image:
        with st.spinner("Analysing the cell…"):
            try:
                st.session_state["result"] = api.predict(*image)
                st.session_state["image"] = image
            except api.BackendError as e:
                st.session_state.pop("result", None)
                st.error(str(e))
    result = st.session_state.get("result")
    # only show a result for the image currently selected
    if result and image and st.session_state.get("image", (None,))[0] == image[0]:
        show_result(result)
        st.divider()
        feedback(st.session_state["image"], result)
    elif not analyse:
        st.info("The result will appear here.")
