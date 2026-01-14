import os

def download_model_from_hub(model_path_or_id):
    """
    Check if the model is a local path or a HuggingFace Hub ID.
    If it's a local path and exists, return it.
    If it's a Hub ID, return it (transformers will handle the download).
    """
    if os.path.exists(model_path_or_id):
        return model_path_or_id
    # If it's not a local path, assume it's a Hub ID.
    return model_path_or_id
