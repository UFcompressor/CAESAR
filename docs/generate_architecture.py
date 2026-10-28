"""Generate a compact CAESAR architecture figure (Matplotlib required).

Run: python3 docs/generate_architecture.py
Outputs: vector PDF/SVG and a 300 dpi PNG beside this script.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/caesar-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.015, top=0.985)
    ax.set(xlim=(0, 16.6), ylim=(0, 10.6))
    ax.axis("off")
    # Orange and blue match README.md. Gator Green (#22884C) marks rANS.
    blue, orange, green = "#1B4FA8", "#E8500A", "#22884C"
    ink, white = "#111111", "#ffffff"
    texts = []
    font_scale = 1.25  # Increase every label without enlarging the canvas.

    def panel(x, y, w, h, fill, edge):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.025,rounding_size=.12",
                facecolor=fill,
                edgecolor=edge,
                linewidth=2,
                zorder=0,
            )
        )

    def box(x, y, w, h, title, detail, color=white, title_fs=16, detail_fs=14):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=.02,rounding_size=.08",
                facecolor=color,
                edgecolor=ink,
                linewidth=1.6,
                zorder=3,
            )
        )
        fg = white if color == blue else ink
        title_artist = ax.text(
            x + w / 2,
            y + h - 0.08,
            title,
            ha="center",
            va="top",
            fontsize=font_scale * title_fs,
            weight="bold",
            color=fg,
            zorder=4,
        )
        detail_artist = ax.text(
            x + w / 2,
            y + 0.08,
            detail,
            ha="center",
            va="bottom",
            fontsize=font_scale * detail_fs,
            weight="bold",
            linespacing=1.12,
            color=fg,
            zorder=4,
        )
        texts.append((title_artist, detail_artist, (x, y, w, h)))

    def arrow(points, color=ink, dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                linewidth=1.8,
                linestyle="--" if dashed else "-",
                zorder=2,
            )
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.8,
                linestyle="--" if dashed else "-",
                color=color,
                zorder=2,
            )
        )

    def label(x, y, text, color=ink):
        ax.text(
            x,
            y,
            text,
            fontsize=font_scale * 14,
            weight="bold",
            color=color,
            ha="center",
            va="center",
            zorder=5,
            bbox={"facecolor": white, "edgecolor": "none", "pad": 2},
        )

    ax.text(
        0.2,
        10.28,
        "CAESAR software architecture",
        fontsize=font_scale * 25,
        weight="bold",
    )
    panel(0.15, 8.8, 16.25, 1.25, "#f3f3f3", ink)
    box(
        0.3,
        8.86,
        5.05,
        1.1,
        "Model registration",
        "UFL_MODELS → CAESAR catalog\nRegistration + checkpoint SHA-256",
        title_fs=16,
        detail_fs=13,
    )
    box(
        5.9,
        8.86,
        4.9,
        1.1,
        "Export and installation",
        "3 packages + 6 CDF tables\nManifest: identity, ranks, device",
        detail_fs=13,
    )
    box(
        11.35,
        8.86,
        4.9,
        1.1,
        "Native model loading",
        "Validate installation manifest\nThread-local runners / tables",
        detail_fs=13,
    )
    arrow([(5.35, 9.4), (5.9, 9.4)])
    arrow([(10.8, 9.4), (11.35, 9.4)])

    panel(0.15, 4.05, 16.25, 4.45, "#f0f5ff", blue)
    ax.text(0.35, 8.15, "COMPRESSION →", fontsize=font_scale * 18, color=blue)
    box(0.35, 6.65, 3.25, 1.15, "Input tensor", "Values and shape\nError bound")
    box(
        4.15,
        6.65,
        3.4,
        1.15,
        "Preprocessing",
        "Normalize and block\nRank; padding / filtering",
    )
    box(
        8.1,
        6.65,
        3.55,
        1.15,
        "Neural encoding",
        "Analysis + hyperprior\nQuantized latents",
        blue,
    )
    box(
        12.2,
        6.65,
        3.85,
        1.15,
        "rANS encoding",
        "Latents / hyperlatents\nInstalled CDF tables",
        green,
    )
    for a, b in [(3.6, 4.15), (7.55, 8.1), (11.65, 12.2)]:
        arrow([(a, 7.23), (b, 7.23)])

    box(
        0.35,
        4.3,
        6.3,
        1.35,
        "Reconstruction metadata",
        "Model ID; shape and rank\nNormalization; blocks, padding, filtering\nCorrection method + metadata",
        detail_fs=13.5,
    )
    box(
        7.2,
        4.3,
        4.2,
        1.35,
        "Local reconstruction",
        "Hyperprior + synthesis\nRestore blocks and scale",
        blue,
    )
    box(
        12,
        4.3,
        4.05,
        1.35,
        "Residual correction",
        "Input vs. reconstruction\nGAE / LBRC / NGLR",
        orange,
        detail_fs=13,
    )
    arrow([(9.85, 6.65), (9.85, 5.65)], blue)
    label(9.85, 6.16, "Quantized latents", blue)
    arrow([(11.4, 4.98), (12, 4.98)], orange)
    arrow([(2, 6.65), (2, 5.95), (14, 5.95), (14, 5.65)])
    label(5.3, 5.95, "Original tensor reference")
    arrow([(5.85, 6.65), (5.85, 6.3), (3.4, 6.3), (3.4, 5.65)])

    panel(0.15, 2.68, 16.25, 1.17, "#f3f3f3", ink)
    box(
        0.35,
        2.83,
        15.7,
        0.87,
        "Serialized compressed representation",
        "Model identity + metadata + latent streams + correction payload (NGLR weights when used)",
        title_fs=17,
        detail_fs=13.5,
    )
    arrow([(3.4, 4.3), (3.4, 3.7)])
    arrow([(14, 4.3), (14, 3.7)], orange)
    arrow([(16.05, 7.23), (16.25, 7.23), (16.25, 3.24), (16.05, 3.24)], blue)

    panel(0.15, 0.5, 16.25, 1.98, "#fff3ec", orange)
    ax.text(0.35, 2.12, "← DECOMPRESSION", fontsize=font_scale * 18, color=ink)
    box(
        12.2,
        0.7,
        3.85,
        1.15,
        "Parse and validate",
        "Streams and metadata\nCheck installed model ID",
        detail_fs=13,
    )
    box(
        8.1,
        0.7,
        3.55,
        1.15,
        "rANS decoding",
        "Recover latent symbols\nInstalled CDF tables",
        green,
    )
    box(
        4.15,
        0.7,
        3.4,
        1.15,
        "Reconstruction",
        "Hyperprior + synthesis\nRestore blocks and scale",
        blue,
        detail_fs=13,
    )
    box(
        0.35,
        0.7,
        3.25,
        1.15,
        "Apply correction",
        "GAE / LBRC / NGLR\nReconstructed tensor",
        orange,
        detail_fs=13,
    )
    arrow([(14.1, 2.83), (14.1, 1.85)])
    for a, b in [(12.2, 11.65), (8.1, 7.55), (4.15, 3.6)]:
        arrow([(a, 1.27), (b, 1.27)])

    # One dependency bus feeds the two phases without wrapping the figure.
    arrow([(13.8, 8.95), (13.8, 8.66), (9.85, 8.66), (9.85, 7.8)], dashed=True)
    arrow([(13.8, 8.66), (14.1, 8.66), (14.1, 7.8)], dashed=True)
    # The caption explains shared installation reuse on decompression as well.
    ax.text(
        0.25,
        0.14,
        "Blue: neural     Orange: correction     Green: rANS     White: data / metadata",
        fontsize=font_scale * 15,
        weight="bold",
    )

    # Check that large labels remain inside their boxes rather than silently
    # shrinking fonts to accommodate text.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for title, detail, (x, y, w, h) in texts:
        bounds = ax.transData.transform([(x, y), (x + w, y + h)])
        for artist in (title, detail):
            extent = artist.get_window_extent(renderer)
            if extent.x0 < bounds[0][0] or extent.x1 > bounds[1][0]:
                raise ValueError(f"Text exceeds box width: {artist.get_text()}")
        if title.get_window_extent(renderer).y0 < detail.get_window_extent(renderer).y1:
            raise ValueError(f"Title overlaps detail: {title.get_text()}")
    output = Path(__file__).resolve().parent / "caesar_architecture"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300, facecolor=white)
    plt.close(fig)


if __name__ == "__main__":
    main()
