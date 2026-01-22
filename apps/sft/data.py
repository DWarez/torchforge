import os

from forge.data.collate import collate_padded
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.distributed import ParallelDims


def setup_dataloader(
    dataset_configs: list[dict],
    hf_assets_path: str,
    batch_size: int,
    parallel_dims: ParallelDims | None = None,
) -> StatefulDataLoader:
    if len(dataset_configs) > 1:
        raise ValueError(
            f"Multiple training datasets not supported yet. Got {len(dataset_configs)} datasets."
        )

    dataset_config = dataset_configs[0]

    tokenizer = HuggingFaceModelTokenizer(
        tokenizer_json_path=os.path.join(hf_assets_path, "tokenizer.json"),
        tokenizer_config_json_path=os.path.join(hf_assets_path, "tokenizer_config.json"),
        generation_config_path=os.path.join(hf_assets_path, "generation_config.json"),
        chat_template_path=(
            path
            if os.path.exists(path := os.path.join(hf_assets_path, "chat_template.jinja"))
            else None
        ),
    )

    dp_mesh = None
    if parallel_dims is not None and parallel_dims.dp_enabled:
        dp_mesh = parallel_dims.world_mesh.get_group("dp")

    dataset = sft_iterable_dataset(
        model_transform=tokenizer,
        # Todo: change this to some sort of json dataset name -> mapping -> transform
        message_transform=AlpacaToMessages(),
        dp_mesh=dp_mesh,
        **dataset_config,
    )

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_padded,
    )

    return dataloader