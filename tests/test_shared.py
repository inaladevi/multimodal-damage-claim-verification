import pandas as pd
from PIL import Image

from shared import ISSUE_TYPES, OBJECT_PARTS, build_prompt, load_images

HISTORY = pd.DataFrame({"user_id": ["u1"], "past_claim_count": [3], "history_flags": ["repeat_claimant"]})
REQUIREMENTS = pd.DataFrame({
    "claim_object": ["car", "laptop", "all"],
    "minimum_image_evidence": ["car needs a wide shot", "laptop needs a screen shot", "images must be clear"],
})


def claim(**overrides):
    row = {"user_id": "u1", "user_claim": "my bumper is dented", "claim_object": "car",
           "image_paths": "images/a.jpg"}
    row.update(overrides)
    return row


def test_prompt_lists_every_allowed_issue_type():
    prompt = build_prompt(claim(), HISTORY, REQUIREMENTS)
    for issue_type in ISSUE_TYPES:
        assert issue_type in prompt


def test_prompt_only_lists_parts_for_the_claimed_object():
    prompt = build_prompt(claim(claim_object="car"), HISTORY, REQUIREMENTS)
    assert "front_bumper" in prompt
    assert "trackpad" not in prompt

    prompt = build_prompt(claim(claim_object="laptop"), HISTORY, REQUIREMENTS)
    assert "trackpad" in prompt
    assert "front_bumper" not in prompt


def test_prompt_includes_history_and_matching_requirements_only():
    prompt = build_prompt(claim(), HISTORY, REQUIREMENTS)
    assert "User Past Claims Count: 3" in prompt
    assert "repeat_claimant" in prompt
    assert "car needs a wide shot" in prompt
    assert "images must be clear" in prompt
    assert "laptop needs a screen shot" not in prompt


def test_unknown_user_gets_empty_history():
    prompt = build_prompt(claim(user_id="nobody"), HISTORY, REQUIREMENTS)
    assert "User Past Claims Count: 0" in prompt
    assert "User History Flags: none" in prompt


def test_every_object_has_an_unknown_part():
    for parts in OBJECT_PARTS.values():
        assert "unknown" in parts


def test_each_image_is_preceded_by_its_id(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "img_1.jpg")
    Image.new("RGB", (10, 10)).save(tmp_path / "img_2.jpg")

    contents = load_images(claim(image_paths="img_1.jpg;img_2.jpg"), str(tmp_path))

    assert contents[0] == "Image ID: img_1"
    assert contents[2] == "Image ID: img_2"
    assert len(contents) == 4


def test_missing_images_are_skipped(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "img_1.jpg")

    contents = load_images(claim(image_paths="img_1.jpg;gone.jpg"), str(tmp_path))

    assert contents[0] == "Image ID: img_1"
    assert len(contents) == 2


def test_large_images_are_downscaled(tmp_path):
    Image.new("RGB", (3000, 1500)).save(tmp_path / "big.jpg")

    contents = load_images(claim(image_paths="big.jpg"), str(tmp_path))

    assert max(contents[1].size) <= 1024