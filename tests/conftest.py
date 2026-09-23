"""Skip tests that need the training datasets when they are not present (they are never part of the public repository)."""
from pathlib import Path
import pytest

DATASET_TESTS = {"test_examples_are_listed_with_existing_images", "test_example_images_are_served",
                 'test_absent_object_questions_are_actually_built', 'test_aligned_state_carries_only_the_coarse_bucket_never_a_caption', 'test_contradicting_states_are_pressed_harder_than_the_default', 'test_converter_is_deterministic', 'test_converter_shape_and_mix', 'test_no_caption_ever_reaches_a_record', 'test_no_source_group_spans_test_and_fit', 'test_two_image_pairs_stay_inside_one_partition', 'test_two_image_records_pair_inside_one_partition'}


def pytest_collection_modifyitems(config, items):
    if Path("data/decision-v1").is_dir() and Path("data/decision-v2/licenses").is_dir(): return
    skip = pytest.mark.skip(reason="needs the training datasets or their licence evidence, which are not distributed")
    for item in items:
        if item.name in DATASET_TESTS or getattr(item, "originalname", item.name) in DATASET_TESTS: item.add_marker(skip)
