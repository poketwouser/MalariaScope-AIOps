"""UI upload checks: a bad file must give a message, never reach st.image (UT-18)."""

import io

import pytest
from PIL import Image
from uploads import MAX_MB, upload_error


def png():
    out = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(out, "PNG")
    return out.getvalue()


def test_good_image_is_accepted():
    assert upload_error(png()) is None


@pytest.mark.parametrize(
    "data",
    [b"", b"this is not an image", png()[:40]],
    ids=["empty", "text_named_png", "truncated_png"],
)
def test_unreadable_file_gets_a_message(data):
    assert "not a readable image" in upload_error(data)


def test_large_file_gets_a_message():
    assert "larger than" in upload_error(b"0" * (MAX_MB * 1024 * 1024 + 1))
