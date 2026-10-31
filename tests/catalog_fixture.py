"""Small synthetic catalog for offline registry tests."""


def make_catalog():
    models = []
    for number, name, filename in (
        (1, "caesar_v1", "caesar_v.pt"),
        (2, "caesar_v2", "model_bs64_ep100k.pt"),
    ):
        digest = str(number) * 64
        models.append(
            dict(
                name=name,
                display_name=name,
                filename=filename,
                registration_id=number,
                sha256=digest,
                id=f"ufl:{number}@sha256:{digest}",
                description="Synthetic test model",
                architecture="caesar-bcrn-v1",
                min_dims=3,
                max_dims=5,
                url=f"https://example.com/{filename}",
                owner="E53Klasky",
                owner_id=173562962,
            )
        )
    return dict(schema_version=1, default_model="caesar_v2", models=models)
