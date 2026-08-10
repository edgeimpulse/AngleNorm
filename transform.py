"""AngleForge — Edge Impulse custom synthetic data block.

Normalises the camera angle of your existing project images so the whole
dataset is visually consistent (e.g. everything shot top-down). For each source
image it:

1. Lists your existing samples via the Studio API.
2. Downloads the image.
3. Sends it to the public **AngleForge** Hugging Face Space, which runs the real
   Qwen-Image-Edit multi-angle model on ZeroGPU, requesting the selected angle.
4. Uploads the re-rendered image back to the project via the Ingestion API,
   tagged with the ``x-synthetic-data-job-id`` header so it previews in the
   Synthetic data tab.

This block operates in ``standalone`` mode (no input directory); all data flows
through the Studio + Ingestion APIs. See:
https://docs.edgeimpulse.com/studio/organizations/custom-blocks/custom-synthetic-data-blocks
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

import requests

# --------------------------------------------------------------------------- #
# Environment (always provided by Edge Impulse to synthetic data blocks)
# --------------------------------------------------------------------------- #
API_ENDPOINT = os.environ.get("EI_API_ENDPOINT", "https://studio.edgeimpulse.com/v1").rstrip("/")
PROJECT_ID = os.environ.get("EI_PROJECT_ID", "")
# Studio (read) key: the project API key. The /api/{projectId}/raw-data endpoints
# reject organization keys (403 "This endpoint requires a project API Key"), so
# prefer EI_PROJECT_API_KEY and only fall back to EI_API_KEY.
STUDIO_API_KEY = os.environ.get("EI_PROJECT_API_KEY") or os.environ.get("EI_API_KEY", "")
# Ingestion (write) key: the project API key.
INGESTION_API_KEY = os.environ.get("EI_PROJECT_API_KEY") or os.environ.get("EI_API_KEY", "")
INGESTION_HOST = os.environ.get("EI_INGESTION_HOST", "edgeimpulse.com")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

INGESTION_URL = "https://ingestion." + INGESTION_HOST
if INGESTION_HOST.endswith(".test.edgeimpulse.com"):
    INGESTION_URL = "http://ingestion." + INGESTION_HOST
elif INGESTION_HOST == "host.docker.internal":
    INGESTION_URL = "http://" + INGESTION_HOST + ":4810"


def _require(value: str, name: str) -> str:
    if not value:
        print(f"Missing required environment variable: {name}")
        sys.exit(1)
    return value


# --------------------------------------------------------------------------- #
# Arguments (defined in parameters.json + auto-passed synthetic-data args)
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Normalise existing Edge Impulse images to a consistent camera angle via AngleForge."
)
parser.add_argument("--angle", type=str, default="top_down", help="Angle preset key")
parser.add_argument("--angleforge-space", type=str, default="eoinedge/angleforge",
                    help="AngleForge Hugging Face Space id (owner/name)")
parser.add_argument("--image-size", type=int, default=512, help="Longest side of output image")
parser.add_argument("--source-category", type=str, default="training",
                    help="training | testing | all")
parser.add_argument("--upload-category", type=str, default="training",
                    help="same | split | training | testing")
parser.add_argument("--labels", type=str, default="", help="Comma-separated label filter")
parser.add_argument("--max-samples", type=int, default=0, help="Max images to process (0 = all)")
parser.add_argument("--seed", type=int, default=1234, help="Generation seed")
parser.add_argument("--synthetic-data-job-id", type=int, required=False, default=None,
                    help="Synthetic data job id (passed automatically by Edge Impulse)")
parser.add_argument("--skip-upload", action="store_true", help="Do not upload results back to EI")
args, _unknown = parser.parse_known_args()

_require(PROJECT_ID, "EI_PROJECT_ID")
_require(STUDIO_API_KEY, "EI_API_KEY / EI_PROJECT_API_KEY")

label_filter = {s.strip() for s in args.labels.split(",") if s.strip()}


# --------------------------------------------------------------------------- #
# Studio API helpers
# --------------------------------------------------------------------------- #
def list_image_samples(category: str):
    """Yield image samples for the given category, paging through the API."""
    headers = {"x-api-key": STUDIO_API_KEY, "Accept": "application/json"}
    offset = 0
    page = 1000
    while True:
        params = {"category": category, "dataType": "image", "limit": page, "offset": offset}
        resp = requests.get(
            f"{API_ENDPOINT}/api/{PROJECT_ID}/raw-data",
            headers=headers, params=params, timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"listSamples failed ({resp.status_code}): {resp.text}")
        body = resp.json()
        if not body.get("success", False):
            raise RuntimeError(f"listSamples error: {body.get('error')}")
        samples = body.get("samples", [])
        for s in samples:
            yield s
        total = body.get("totalCount", 0)
        offset += len(samples)
        if not samples or offset >= total:
            break


def download_image(sample_id: int) -> bytes:
    headers = {"x-api-key": STUDIO_API_KEY}
    resp = requests.get(
        f"{API_ENDPOINT}/api/{PROJECT_ID}/raw-data/{sample_id}/image",
        headers=headers, timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"getSampleAsImage failed ({resp.status_code}): {resp.text[:200]}")
    return resp.content


def upload_image(png_bytes: bytes, filename: str, label: str, category: str, metadata: dict):
    if category == "same":
        category = "split"  # ingestion has no "same"; caller resolves per-sample
    headers = {
        "x-label": label,
        "x-api-key": INGESTION_API_KEY,
        "x-metadata": json.dumps(metadata),
    }
    if args.synthetic_data_job_id is not None:
        headers["x-synthetic-data-job-id"] = str(args.synthetic_data_job_id)
    resp = requests.post(
        f"{INGESTION_URL}/api/{category}/files",
        headers=headers,
        files={"data": (filename, png_bytes, "image/png")},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Ingestion failed ({resp.status_code}): {resp.text[:300]}")
    body = resp.json()
    if not body.get("success", False):
        raise RuntimeError(f"Ingestion error: {body.get('error')}")
    files = body.get("files") or [{}]
    if not files[0].get("success", True):
        raise RuntimeError(f"Ingestion file error: {files[0].get('error')}")


# --------------------------------------------------------------------------- #
# AngleForge (Hugging Face Space) client
# --------------------------------------------------------------------------- #
def _result_to_png_bytes(item) -> bytes:
    """Turn one AngleForge gallery item into real PNG bytes.

    The ``/grab_viewpoints_ui`` endpoint returns gallery items as
    ``{"image": <local filepath or URL>, "caption": ...}``; older/other shapes
    may be a PIL Image, a bare path, or a URL string. Everything is re-encoded
    to PNG via PIL so the uploaded bytes always match the ``.png`` filename.
    """
    import io

    from PIL import Image as _PILImage  # provided by the block's requirements

    def _png(img) -> bytes:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    if isinstance(item, _PILImage.Image):
        return _png(item)

    path = None
    url = None
    if isinstance(item, str):
        if item.startswith(("http://", "https://")):
            url = item
        else:
            path = item
    elif isinstance(item, dict):
        for key in ("image", "path", "name"):
            val = item.get(key)
            if isinstance(val, str) and val:
                if val.startswith(("http://", "https://")):
                    url = val
                else:
                    path = val
                break
        if not path and not url:
            maybe_url = item.get("url")
            if isinstance(maybe_url, str) and maybe_url:
                url = maybe_url

    if path and os.path.exists(path):
        return _png(_PILImage.open(path))
    if url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return _png(_PILImage.open(io.BytesIO(r.content)))

    raise RuntimeError(f"Unexpected AngleForge output item: {item!r}")


def make_client(space: str):
    from gradio_client import Client

    # AngleForge is public, so no token is required. Only pass one for a private
    # Space, and do it defensively since the keyword differs across
    # gradio_client versions (hf_token vs. headers-based auth).
    if not HF_TOKEN:
        return Client(space)
    try:
        return Client(space, hf_token=HF_TOKEN)
    except TypeError:
        return Client(space, headers={"Authorization": f"Bearer {HF_TOKEN}"})


def render_angle(client, image_path: str, angle: str, size: int, seed: int) -> bytes:
    from gradio_client import handle_file

    # Use the UI endpoint: it returns a proper Gallery (real downloadable image
    # files) plus a status string. The programmatic `/grab_viewpoints` gr.api
    # endpoint serialises PIL images as an untyped `Any`, which comes back as
    # the useless `str(image)` repr on some gradio_client versions.
    result = client.predict(
        handle_file(image_path),
        [angle],
        seed,
        size,
        "",  # hf_token for the Space's own serverless backend (unused: ZeroGPU)
        api_name="/grab_viewpoints_ui",
    )

    if isinstance(result, (list, tuple)) and len(result) >= 2:
        gallery, status = result[0], result[1]
    else:
        gallery, status = result, ""

    if not gallery:
        raise RuntimeError(f"AngleForge returned no image. {status}".strip())
    return _result_to_png_bytes(gallery[0])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(f"AngleForge synthetic-data block")
    print(f"  Space:            {args.angleforge_space}")
    print(f"  Angle:            {args.angle}")
    print(f"  Source category:  {args.source_category}")
    print(f"  Upload category:  {args.upload_category}")
    print(f"  Image size:       {args.image_size}")
    print(f"  Label filter:     {sorted(label_filter) or 'all'}")
    print(f"  Max images:       {args.max_samples or 'no limit'}")
    print("")

    try:
        client = make_client(args.angleforge_space)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to connect to AngleForge Space '{args.angleforge_space}': {exc}")
        print(traceback.format_exc())
        sys.exit(1)

    processed = 0
    failed = 0
    for sample in list_image_samples(args.source_category):
        if args.max_samples and processed >= args.max_samples:
            break
        label = str(sample.get("label", "") or "unknown")
        if label_filter and label not in label_filter:
            continue
        sample_id = sample.get("id")
        src_category = str(sample.get("category", "training") or "training")

        print(f"[{processed + 1}] sample {sample_id} ({label})...", end="", flush=True)
        try:
            img_bytes = download_image(sample_id)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            png = render_angle(client, tmp_path, args.angle, args.image_size, args.seed)

            if not args.skip_upload:
                upload_category = args.upload_category
                if upload_category == "same":
                    upload_category = src_category if src_category in ("training", "testing") else "split"
                filename = f"{label}.angle_{args.angle}.{sample_id}.png"
                upload_image(
                    png, filename, label, upload_category,
                    metadata={
                        "generated_by": "angleforge",
                        "angle": args.angle,
                        "source_sample_id": str(sample_id),
                        "angleforge_space": args.angleforge_space,
                    },
                )
            processed += 1
            print(" OK")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f" FAILED: {exc}")

    print("")
    print(f"Done. Normalised {processed} image(s) to '{args.angle}', {failed} failed.")
    if processed == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
