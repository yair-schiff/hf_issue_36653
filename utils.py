from transformers import PreTrainedModel


def push_my_model_to_hub(
    model: PreTrainedModel,
    repo_id: str = "yairschiff/hf-issue-36653",
    private: bool = False,
    commit_message: str = "Initial commit",
    safe_serialization: bool = False
) -> None:
    model.__class__.register_for_auto_class()
    model.__class__.register_for_auto_class("AutoModel")
    model.config.auto_map = model.config.auto_map

    # Pushes model and `modeling.py` to HF hub
    model.push_to_hub(
      repo_id=repo_id,
      private=private,
      commit_message=commit_message,
      safe_serialization=safe_serialization,
    )
