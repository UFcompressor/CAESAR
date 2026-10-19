"""Render the authoritative UFL catalog as a public model table."""

README_START = "<!-- BEGIN GENERATED MODEL LIST -->"
README_END = "<!-- END GENERATED MODEL LIST -->"


def model_list(catalog):
    """Render the public list from the authoritative catalog."""
    import html

    def cell(value):
        return (
            html.escape(str(value), quote=False)
            .replace("|", "&#124;")
            .replace("\n", " ")
            .replace("\r", " ")
        )

    lines = [
        README_START,
        "",
        "| Registration | Model | Owner | Status | Checkpoint | Dimensions | Description |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for status, models in (
        ("Active", catalog["models"]),
        ("Retired", catalog.get("retired_models", [])),
    ):
        for model in sorted(models, key=lambda m: m["registration_id"]):
            state = status
            if status == "Active" and model["name"] == catalog["default_model"]:
                state = "Active (default)"
            values = [
                model["registration_id"],
                f"{model['display_name']} ({model['name']})",
                model.get("owner", "Unassigned"),
                state,
                model["filename"],
                f"{model['min_dims']}D–{model['max_dims']}D",
                model["description"],
            ]
            lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    return "\n".join([*lines, "", README_END])


def render_readme(catalog, text):
    block = model_list(catalog)
    if README_START not in text and README_END not in text:
        return text.rstrip() + "\n\n" + block + "\n"
    if text.count(README_START) != 1 or text.count(README_END) != 1:
        raise ValueError("README must contain exactly one model list marker pair")
    start, end = text.index(README_START), text.index(README_END)
    if start >= end:
        raise ValueError("README model list markers are reversed")
    return text[:start] + block + text[end + len(README_END) :]
