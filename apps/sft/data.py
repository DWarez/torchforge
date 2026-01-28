import os
from typing import Any

from forge.data.collate import collate_padded
from forge.data.datasets.sft_dataset import AlpacaToMessages, sft_iterable_dataset
from forge.data.tokenizer import HuggingFaceModelTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.distributed import ParallelDims

from forge.data.utils import TuneMessage, mask_messages

class GenericSFTToMessages:
    ROLE_MAP = {
        "human": "user",
        "gpt": "assistant", 
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }

    def __init__(
        self,
        masking_strategy: str = "train_on_assistant",
        messages_column: str | None = None,
        user_column: str | None = None,
        assistant_column: str | None = None,
        system_prompt: str | None = None,
    ):
        self.masking_strategy = masking_strategy
        self.messages_column = messages_column
        self.user_column = user_column
        self.assistant_column = assistant_column
        self.system_prompt = system_prompt

    def _normalize_role(self, role: str) -> str:
        return self.ROLE_MAP.get(role.lower(), role.lower())

    def _from_messages_list(self, messages_data: list[dict]) -> list[TuneMessage]:
        messages = []
        for msg in messages_data:
            role = self._normalize_role(msg.get("role", msg.get("from", "")))
            content = msg.get("content", msg.get("value", msg.get("text", "")))
            messages.append(TuneMessage(role=role, content=content, eot=True))
        return messages

    def _from_prompt_completion(
        self, user_content: str, assistant_content: str
    ) -> list[TuneMessage]:
        messages = []
        if self.system_prompt:
            messages.append(
                TuneMessage(role="system", content=self.system_prompt, eot=True)
            )
        messages.extend([
            TuneMessage(role="user", content=user_content, eot=True),
            TuneMessage(role="assistant", content=assistant_content, eot=True),
        ])
        return messages

    def _detect_and_parse(self, sample: dict[str, Any]) -> list[TuneMessage]:
        if self.messages_column and self.messages_column in sample:
            return self._from_messages_list(sample[self.messages_column])

        if self.user_column and self.assistant_column:
            return self._from_prompt_completion(
                sample[self.user_column], sample[self.assistant_column]
            )

        for col in ("messages", "conversations", "conversation"):
            if col in sample and isinstance(sample[col], list):
                return self._from_messages_list(sample[col])

        prompt_keys = ("prompt", "instruction", "question", "input", "human")
        completion_keys = ("completion", "response", "output", "answer", "assistant", "gpt")

        user_content = None
        assistant_content = None

        for key in prompt_keys:
            if key in sample and sample[key]:
                user_content = sample[key]
                break

        for key in completion_keys:
            if key in sample and sample[key]:
                assistant_content = sample[key]
                break

        if user_content and assistant_content:
            return self._from_prompt_completion(user_content, assistant_content)

        raise ValueError(
            f"Could not detect format. Available columns: {list(sample.keys())}"
        )

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        messages = self._detect_and_parse(sample)
        mask_messages(messages, self.masking_strategy)
        return {"messages": messages}

def setup_dataloader(
    dataset_configs: list[dict],
    max_seq_len: int,
    hf_assets_path: str,
    batch_size: int,
    parallel_dims: ParallelDims | None = None,
) -> StatefulDataLoader:
    # Todo: multidataset
    if len(dataset_configs) > 1:
        raise ValueError(
            f"Multiple training datasets not supported yet. Got {len(dataset_configs)} datasets."
        )

    dataset_config = dataset_configs[0]

    tokenizer = HuggingFaceModelTokenizer(
        tokenizer_json_path=os.path.join(hf_assets_path, "tokenizer.json"),
        tokenizer_config_json_path=os.path.join(
            hf_assets_path, "tokenizer_config.json"
        ),
        generation_config_path=os.path.join(hf_assets_path, "generation_config.json"),
        chat_template_path=(
            path
            if os.path.exists(
                path := os.path.join(hf_assets_path, "chat_template.jinja")
            )
            else None
        ),
        max_seq_len=max_seq_len,
    )

    dp_mesh = None
    if parallel_dims is not None:
        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.world_mesh.get_group("dp")
        elif parallel_dims.dp_replicate_enabled:
            dp_mesh = parallel_dims.world_mesh.get_group("dp_replicate")
        elif parallel_dims.dp_shard_enabled:
            dp_mesh = parallel_dims.world_mesh.get_group("dp_shard")
        else:
            import torch.distributed as dist

            dp_mesh = dist.new_group([dist.get_rank()])

    dataset = sft_iterable_dataset(
        model_transform=tokenizer,
        # Todo: change this to some sort of json dataset name -> mapping -> transform
        # message_transform=AlpacaToMessages(),
        message_transform=GenericSFTToMessages(),
        dp_mesh=dp_mesh,
        **dataset_config,
    )

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_padded,
    )

    return dataloader
