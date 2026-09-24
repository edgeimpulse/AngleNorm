# AngleNorm — Edge Impulse custom synthetic data block

Normalise the camera angle of the images already in your Edge Impulse project so
the whole dataset is visually consistent (e.g. **everything top-down**). It runs
the real **Qwen-Image-Edit multi-angle** model via the public
[AngleNorm](https://huggingface.co/spaces/eoinedge/AngleNorm) Hugging Face
Space (ZeroGPU), so the block itself needs no GPU.

> **Enterprise only.** Custom synthetic data blocks require an Edge Impulse
> Enterprise plan.

## How it works

For every existing image sample it:

1. Lists your samples via the Studio API (`GET /api/{projectId}/raw-data`).
2. Downloads the image (`GET /api/{projectId}/raw-data/{id}/image`).
3. Sends it to the AngleNorm Space `grab_viewpoints` API for the selected angle.
4. Uploads the re-rendered image back through the Ingestion API with the
   `x-synthetic-data-job-id` header (so it previews under **Data acquisition →
   Synthetic data**).

Original samples are left untouched; normalised copies are added with a
`generated_by=anglenorm` metadata tag and filename `label.angle_<angle>.<id>.png`.

## Parameters

| Param | Description |
| ----- | ----------- |
| `angle` | Target camera angle preset (top_down, birds_eye, rotate_left_45, …). |
| `anglenorm-space` | HF Space id running the model (default `eoinedge/AngleNorm`). |
| `HF_TOKEN` | Secret. Only needed if the Space is private. |
| `image-size` | Longest side of the output image (px). |
| `source-category` | Which images to normalise: training / testing / all. |
| `upload-category` | Where to put results: same / split / training / testing. |
| `labels` | Optional comma-separated label filter. |
| `max-samples` | Cap the number of images processed (0 = all). |
| `seed` | Reproducible generation seed. |

## Test locally

Synthetic data blocks are **not** supported by `edge-impulse-blocks runner`, so
build and run the container directly:

```bash
docker build -t anglenorm-synthetic .
docker run --rm \
  -e EI_PROJECT_ID='<project-id>' \
  -e EI_API_KEY='ei_...'             `# org or project API key (Studio reads)` \
  -e EI_PROJECT_API_KEY='ei_...'     `# project API key (ingestion writes)` \
  -e EI_INGESTION_HOST='edgeimpulse.com' \
  -e EI_API_ENDPOINT='https://studio.edgeimpulse.com/v1' \
  anglenorm-synthetic \
  --synthetic-data-job-id 0 \
  --angle top_down --source-category training --upload-category training \
  --max-samples 3
```

## Push to Edge Impulse

```bash
edge-impulse-blocks init      # choose: Synthetic data block
edge-impulse-blocks push
```

Then use it from **Data acquisition → Synthetic data** in any project in your
organization.
