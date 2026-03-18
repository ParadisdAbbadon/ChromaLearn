"""
ChromaLearn Image Classifier
Upload an image → classify it against the DB, or add it if it's new.
"""
import uuid

import gradio as gr
import numpy as np
from PIL import Image

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# ── Config ──────────────────────────────────────────────────────────────────
COLLECTION_NAME   = "Pet_Image_Collection"
NOVELTY_THRESHOLD = 0.75   # cosine distance; above this → "never seen before"

SEED_IMAGES = {
    "corgi":       {"path": "image_set_1/corgi.jpg",       "animal": "dog"},
    "husky":       {"path": "image_set_1/husky.jpg",       "animal": "dog"},
    "labrador":    {"path": "image_set_1/labrador.jpg",    "animal": "dog"},
    "orange_cat":  {"path": "image_set_1/orange_cat.jpg",  "animal": "cat"},
    "siamese_cat": {"path": "image_set_1/siamese_cat.jpg", "animal": "cat"},
}

# ── DB initialisation ────────────────────────────────────────────────────────

# Persistent Client
#UPLOADS_DIR.mkdir(exist_ok=True)
#_client   = chromadb.PersistentClient(path=CHROMA_PATH)

# Ephemeral Client
_client = chromadb.Client()

_embed_fn = OpenCLIPEmbeddingFunction()
_loader   = ImageLoader()

existing_names = [c.name for c in _client.list_collections()]

if COLLECTION_NAME not in existing_names:
    collection = _client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,
        data_loader=_loader,
    )
    collection.add(
        ids=list(SEED_IMAGES.keys()),
        uris=[v["path"] for v in SEED_IMAGES.values()],
        metadatas=[
            {"filename": k + ".jpg", "animal": v["animal"], "description": v["animal"]}
            for k, v in SEED_IMAGES.items()
        ],
    )
    print(f"Seeded '{COLLECTION_NAME}' with {len(SEED_IMAGES)} images.")
else:
    collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,
        data_loader=_loader,
    )
    print(f"Loaded '{COLLECTION_NAME}' ({collection.count()} images).")


# ── Core logic ───────────────────────────────────────────────────────────────
def identify_image(uploaded_pil, state):
    """Query the DB with the uploaded image and decide if it's known or novel."""
    if uploaded_pil is None:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False),
            None,
            state,
        )

    img_array = np.array(uploaded_pil)
    n_results = min(3, collection.count())

    results = collection.query(
        query_images=[img_array],
        n_results=n_results,
        include=["uris", "metadatas", "distances"],
    )

    top_dist = results["distances"][0][0]
    top_meta = results["metadatas"][0][0]
    top_uri  = results["uris"][0][0]

    state = {"img_array": img_array}

    if top_dist <= NOVELTY_THRESHOLD:
        label = top_meta.get("animal") or top_meta.get("description", "unknown")
        result_md = (
            f"### This looks like a **{label}**!\n\n"
            f"| | |\n|---|---|\n"
            f"| Closest match | `{top_meta.get('filename', top_uri.split('/')[-1])}` |\n"
            f"| Similarity distance | `{top_dist:.4f}` *(lower = more similar)* |\n"
            f"| Description | _{top_meta.get('description', label)}_ |"
        )
        matched_img = Image.open(top_uri)
        return (
            gr.update(visible=True),            # result_box
            gr.update(visible=False),           # novel_box
            gr.update(value=result_md, visible=True),
            matched_img,
            state,
        )
    else:
        return (
            gr.update(visible=False),           # result_box
            gr.update(visible=True),            # novel_box
            gr.update(value="", visible=False),
            None,
            state,
        )


def add_to_database(description, state):
    """Add the pending image to the DB with the user's description."""
    if not state:
        return gr.update(), "No image pending — please upload one first.", state

    description = description.strip()
    if not description:
        return gr.update(), "Please enter a description before adding.", state

    new_id = f"user_{uuid.uuid4().hex[:8]}"

    collection.add(
        ids=[new_id],
        images=[state["img_array"]],
        metadatas=[{
            "animal":      description,
            "description": description,
            "user_added":  "true",
        }],
    )

    msg = (
        f"Added **\"{description}\"** to the database!  \n"
        f"ID: `{new_id}` — Total images in DB: **{collection.count()}**"
    )
    return (
        gr.update(visible=False),   # hide novel_box
        msg,
        {},                         # clear pending state
    )


def reset_ui():
    """Clear the UI back to the initial state."""
    return (
        None,                           # uploaded image
        gr.update(visible=False),       # result_box
        gr.update(visible=False),       # novel_box
        gr.update(value="", visible=False),
        None,                           # matched image
        "",                             # description input
        "",                             # status text
        {},                             # state
    )


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="ChromaLearn Image Classifier") as demo:

    state = gr.State({})

    gr.Markdown(
        "# ChromaLearn Image Classifier\n"
        "Upload an image and the app will try to classify it against the database. "
        f"Currently **{collection.count()} images** in the database."
    )

    with gr.Row():
        # ── Left column: upload ──────────────────────────────────────────────
        with gr.Column(scale=1):
            upload = gr.Image(label="Upload an image", type="pil")
            with gr.Row():
                identify_btn = gr.Button("Identify", variant="primary")
                reset_btn    = gr.Button("Reset")

        # ── Right column: results ────────────────────────────────────────────
        with gr.Column(scale=1):
            result_md = gr.Markdown(visible=False)

            # Shown when image is recognised
            with gr.Group(visible=False) as result_box:
                matched_img = gr.Image(label="Closest match in database", interactive=False)

            # Shown when image is novel
            with gr.Group(visible=False) as novel_box:
                gr.Markdown(
                    "### Woah! I've never seen this before. What is it?\n"
                    "Write a short description and I'll add it to the database."
                )
                description_input = gr.Textbox(
                    label="Description",
                    placeholder="e.g. golden retriever, tabby cat, bald eagle…",
                    max_lines=2,
                )
                add_btn = gr.Button("Add to Database", variant="primary")

            status_text = gr.Markdown("")

    # ── Event wiring ─────────────────────────────────────────────────────────
    identify_btn.click(
        fn=identify_image,
        inputs=[upload, state],
        outputs=[result_box, novel_box, result_md, matched_img, state],
    )

    add_btn.click(
        fn=add_to_database,
        inputs=[description_input, state],
        outputs=[novel_box, status_text, state],
    )

    reset_btn.click(
        fn=reset_ui,
        inputs=[],
        outputs=[upload, result_box, novel_box, result_md, matched_img,
                 description_input, status_text, state],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
