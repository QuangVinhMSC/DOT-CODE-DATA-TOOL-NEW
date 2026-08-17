import numpy as np
import pytest

from dotgen.core.imageops import load_image, resize_set, resize_to, save_image


def test_resize_set_makes_every_image_the_same_size(backgrounds):
    out = resize_set(backgrounds, (320, 240))
    assert {img.shape[:2] for img in out} == {(240, 320)}


def test_wider_image_is_cropped_on_x_only(backgrounds):
    """900x300 into 320x240: cover scale is set by height, so x is cropped."""
    wide = backgrounds[2]
    out = resize_to(wide, (320, 240))

    assert out.shape[:2] == (240, 320)

    # Scale-to-cover by height: 300 -> 240 is 0.8, so the width becomes 720 and
    # only that dimension loses pixels.
    assert 900 * (240 / 300) > 320


def test_taller_image_is_cropped_on_y_only():
    tall = np.zeros((900, 300, 3), np.uint8)
    out = resize_to(tall, (300, 300))
    assert out.shape[:2] == (300, 300)


def test_matching_aspect_ratio_is_not_cropped():
    src = np.zeros((480, 640, 3), np.uint8)
    src[:, :] = 77
    out = resize_to(src, (320, 240))
    assert out.shape[:2] == (240, 320)
    assert int(out.mean()) == 77


def test_crop_takes_the_centre():
    """A marker in the middle survives; the outer thirds are what get cut."""
    src = np.zeros((300, 900, 3), np.uint8)
    src[140:160, 440:460] = 255

    out = resize_to(src, (300, 300))
    h, w = out.shape[:2]

    assert out[h // 2, w // 2].max() == 255
    assert out[5, 5].max() == 0


def test_aspect_ratio_is_preserved_no_stretching():
    """A circle must stay a circle: equal scale on both axes."""
    import cv2

    src = np.zeros((300, 900, 3), np.uint8)
    cv2.circle(src, (450, 150), 60, (255, 255, 255), -1)

    out = resize_to(src, (300, 300))
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(gray > 128)

    width = xs.max() - xs.min()
    height = ys.max() - ys.min()

    assert abs(width - height) <= 2


def test_bad_target_size_is_rejected():
    with pytest.raises(ValueError):
        resize_to(np.zeros((10, 10, 3), np.uint8), (0, 10))


def test_load_image_reports_the_path(tmp_path):
    missing = tmp_path / "nope.png"

    with pytest.raises(IOError) as exc:
        load_image(str(missing))

    assert "nope.png" in str(exc.value)


def test_save_then_load_roundtrip(tmp_path):
    img = np.full((20, 30, 3), 123, np.uint8)
    path = str(tmp_path / "sub" / "img.png")

    save_image(path, img)
    back = load_image(path)

    assert back.shape == img.shape
    assert int(back.mean()) == 123


# ----------------------------------------------------------------------
# Phase 10.4 -- the file dialogs catch ``IOError`` and nothing else, so a
# failure that escapes as any other type reaches the user as a traceback.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, data",
    [
        ("empty.png", b""),
        ("text.png", b"this is not an image, it is a note"),
        ("truncated.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
        ("zipbomb.jpg", b"PK\x03\x04" + b"\xff" * 512),
    ],
)
def test_unreadable_files_all_raise_ioerror(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)

    with pytest.raises(IOError):
        load_image(str(path))


def test_directory_raises_ioerror(tmp_path):
    with pytest.raises(IOError):
        load_image(str(tmp_path))


def test_save_to_an_unwritable_path_raises_ioerror(tmp_path):
    # A file where a directory has to be: makedirs cannot create the parent.
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")

    with pytest.raises(IOError):
        save_image(str(blocker / "sub" / "img.png"), np.zeros((4, 4, 3), np.uint8))


def test_save_of_an_unencodable_array_raises_ioerror(tmp_path):
    with pytest.raises(IOError):
        save_image(str(tmp_path / "img.png"), np.zeros((4, 4, 7), np.float64))
