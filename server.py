"""
ViMax API Server for Cloud Run deployment.

Provides HTTP endpoints to submit video generation jobs,
check their status, and download the results.

Cloud Run requires a web server to handle requests — this wraps
the ViMax pipeline behind a REST API.
"""

import os
import json
import asyncio
import uuid
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vimax-server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
jobs: dict = {}  # job_id -> job info dict
pipeline = None  # Lazy-loaded pipeline instance

# Config path — can be overridden via env var
CONFIG_PATH = os.environ.get("VIMAX_CONFIG", "configs/idea2video.yaml")


# ---------------------------------------------------------------------------
# Lifespan: initialize pipeline on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the ViMax pipeline on startup."""
    global pipeline
    logger.info("ViMax server starting up...")
    logger.info(f"Config path: {CONFIG_PATH}")

    # Check if credentials are configured
    try:
        import yaml
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        has_vertex = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        has_api_key = bool(os.environ.get("GOOGLE_API_KEY"))

        if not has_vertex and not has_api_key:
            logger.warning(
                "⚠️  No credentials configured. Set GOOGLE_CLOUD_PROJECT + "
                "GOOGLE_CLOUD_LOCATION for Vertex AI, or GOOGLE_API_KEY for "
                "AI Studio."
            )
        elif has_vertex:
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
            logger.info(f"✅ Vertex AI configured: project={project}, location={location}")
        else:
            logger.info("✅ Google AI Studio API key configured")

    except Exception as e:
        logger.warning(f"Could not read config at startup: {e}")

    yield
    logger.info("ViMax server shutting down...")


app = FastAPI(
    title="Visualiser API",
    description="Agentic Video Generation API — powered by Visualiser multi-agent pipeline",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class Idea2VideoRequest(BaseModel):
    """Submit an idea to generate a video."""
    idea: str
    user_requirement: str = "Do not exceed 3 scenes. Each scene should be no more than 5 shots."
    style: str = "Realistic"
    config: Optional[str] = None  # Optional config path override


class Script2VideoRequest(BaseModel):
    """Submit a script to generate a video."""
    script: str
    user_requirement: str = "Fast-paced with no more than 15 shots."
    style: str = "Anime Style"
    config: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Status of a video generation job."""
    job_id: str
    status: str  # pending, running, completed, failed
    message: str = ""
    video_path: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper: get or create pipeline
# ---------------------------------------------------------------------------
def get_pipeline(config_path: str = None):
    """Lazy-load the pipeline to avoid startup delays."""
    global pipeline
    if pipeline is None:
        from pipelines.idea2video_pipeline import Idea2VideoPipeline
        cfg = config_path or CONFIG_PATH
        pipeline = Idea2VideoPipeline.init_from_config(config_path=cfg)
    return pipeline


def inject_env_api_keys():
    """Inject API credentials from environment variables into the config.

    Supports two authentication modes:

    1. **Vertex AI** (recommended for Cloud Run): Set ``GOOGLE_CLOUD_PROJECT``
       and ``GOOGLE_CLOUD_LOCATION``. The Cloud Run service account handles
       auth automatically — no API key needed.

    2. **Google AI Studio**: Set ``GOOGLE_API_KEY`` as a fallback.
       Legacy vars ``VIMAX_CHAT_API_KEY`` / ``OPENROUTER_API_KEY`` are also
       supported for backward compatibility.
    """
    import yaml

    # Read the config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    changed = False

    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    gcp_location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    # --- Vertex AI credentials ---
    if gcp_project:
        # Chat model (google_vertexai provider)
        chat_args = config.get("chat_model", {}).get("init_args", {})
        if not chat_args.get("project"):
            chat_args["project"] = gcp_project
            changed = True
        if not chat_args.get("location"):
            chat_args["location"] = gcp_location
            changed = True

        # Image generator
        img_args = config.get("image_generator", {}).get("init_args", {})
        if not img_args.get("project"):
            img_args["project"] = gcp_project
            changed = True
        if not img_args.get("location"):
            img_args["location"] = gcp_location
            changed = True

        # Video generator
        vid_args = config.get("video_generator", {}).get("init_args", {})
        if not vid_args.get("project"):
            vid_args["project"] = gcp_project
            changed = True
        if not vid_args.get("location"):
            vid_args["location"] = gcp_location
            changed = True

    # --- Google AI Studio API key (fallback) ---
    google_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("VIMAX_GOOGLE_API_KEY")
    )
    if google_key and not gcp_project:
        # Only use API key if Vertex AI is not configured
        chat_args = config.get("chat_model", {}).get("init_args", {})
        if not chat_args.get("api_key"):
            chat_args["api_key"] = google_key
            changed = True

        img_args = config.get("image_generator", {}).get("init_args", {})
        if not img_args.get("api_key"):
            img_args["api_key"] = google_key
            changed = True

        vid_args = config.get("video_generator", {}).get("init_args", {})
        if not vid_args.get("api_key"):
            vid_args["api_key"] = google_key
            changed = True

    # --- Legacy OpenRouter support ---
    chat_key = (
        os.environ.get("VIMAX_CHAT_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if chat_key and not gcp_project:
        chat_args = config.get("chat_model", {}).get("init_args", {})
        if not chat_args.get("api_key"):
            chat_args["api_key"] = chat_key
            changed = True

    if changed:
        # Write back so the pipeline picks it up
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info("Injected API credentials from environment variables into config.")


# ---------------------------------------------------------------------------
# Background task: run the pipeline
# ---------------------------------------------------------------------------
async def run_idea2video(job_id: str, idea: str, user_requirement: str, style: str, config_path: str = None):
    """Run the idea2video pipeline in the background."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["message"] = "Pipeline is running..."

        # Inject API keys from env vars
        inject_env_api_keys()

        # Force re-initialization of pipeline with updated config
        global pipeline
        pipeline = None

        p = get_pipeline(config_path)
        video_path = await p(idea=idea, user_requirement=user_requirement, style=style)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Video generated successfully!"
        jobs[job_id]["video_path"] = video_path
        logger.info(f"Job {job_id} completed: {video_path}")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Pipeline failed: {e}"
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)


async def run_script2video(job_id: str, script: str, user_requirement: str, style: str, config_path: str = None):
    """Run the script2video pipeline in the background."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["message"] = "Pipeline is running..."

        inject_env_api_keys()

        from pipelines.script2video_pipeline import Script2VideoPipeline
        global pipeline
        pipeline = None

        cfg = config_path or "configs/script2video.yaml"
        p = Script2VideoPipeline.init_from_config(config_path=cfg)
        video_path = await p(script=script, user_requirement=user_requirement, style=style)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Video generated successfully!"
        jobs[job_id]["video_path"] = video_path
        logger.info(f"Job {job_id} completed: {video_path}")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = f"Pipeline failed: {e}"
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "service": "visualiser-api"}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Visualiser API",
        "version": "0.2.0",
        "provider": "Vertex AI" if os.environ.get("GOOGLE_CLOUD_PROJECT") else "Google AI Studio",
        "endpoints": {
            "POST /idea2video": "Submit an idea to generate a video",
            "POST /script2video": "Submit a script to generate a video",
            "GET /jobs/{job_id}": "Check job status",
            "GET /jobs": "List all jobs",
            "GET /download/{job_id}": "Download the generated video",
            "GET /health": "Health check",
        },
    }


@app.post("/idea2video", response_model=JobStatusResponse)
async def idea2video(request: Idea2VideoRequest, background_tasks: BackgroundTasks):
    """Submit an idea to generate a video.

    The pipeline runs asynchronously. Use the returned job_id to check status.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": "Job queued, waiting to start...",
        "type": "idea2video",
        "video_path": None,
        "error": None,
    }

    background_tasks.add_task(
        run_idea2video,
        job_id=job_id,
        idea=request.idea,
        user_requirement=request.user_requirement,
        style=request.style,
        config_path=request.config,
    )

    return JobStatusResponse(
        job_id=job_id,
        status="pending",
        message="Job queued. Poll /jobs/{job_id} for status.",
    )


@app.post("/script2video", response_model=JobStatusResponse)
async def script2video(request: Script2VideoRequest, background_tasks: BackgroundTasks):
    """Submit a script to generate a video.

    The pipeline runs asynchronously. Use the returned job_id to check status.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": "Job queued, waiting to start...",
        "type": "script2video",
        "video_path": None,
        "error": None,
    }

    background_tasks.add_task(
        run_script2video,
        job_id=job_id,
        script=request.script,
        user_requirement=request.user_requirement,
        style=request.style,
        config_path=request.config,
    )

    return JobStatusResponse(
        job_id=job_id,
        status="pending",
        message="Job queued. Poll /jobs/{job_id} for status.",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a video generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job["message"],
        video_path=job.get("video_path"),
        error=job.get("error"),
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs and their statuses."""
    return {"jobs": list(jobs.values()), "total": len(jobs)}


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the generated video for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job['status']}, not completed")

    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"visualiser_{job_id}.mp4",
    )


# ---------------------------------------------------------------------------
# Run with: uvicorn server:app --host 0.0.0.0 --port 8080
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
