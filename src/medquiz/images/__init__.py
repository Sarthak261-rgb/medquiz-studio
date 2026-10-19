"""Image track: OpenStax figures, Stable Diffusion 1.5 with and without a LoRA,
automatic checks and the image review taxonomy."""

IMAGE_FAILURES = {
    "wrong_anatomy": "Anatomy is wrong (shape, position, connections)",
    "missing_or_extra": "A structure is missing or duplicated",
    "garbled_text": "Garbled or fake text and labels",
    "not_subject": "Does not show the requested subject",
    "not_teaching_style": "Not usable as a teaching illustration (style, clutter)",
    "artifact": "Visual artifacts (melted shapes, noise, broken lines)",
}
