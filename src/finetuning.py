from safetensors.torch import load_file, save_file
from model import Vocos


def convert_vocos_keys(state_dict: dict) -> dict:
    """Convert HuggingFace Vocos keys to Vocos-v2 keys."""

    converted = {}

    for k, v in state_dict.items():
        nk = k

        nk = nk.replace("embed.", "in_conv.")
        nk = nk.replace("decoder.out.", "out_conv.")
        nk = nk.replace("final_layer_norm.", "norm_last.")

        nk = nk.replace(".dwconv.", ".dw_conv.")
        nk = nk.replace(".pwconv1.", ".mlp.0.")
        nk = nk.replace(".pwconv2.", ".mlp.2.")
        nk = nk.replace(".layer_scale_parameter", ".layer_scale")

        converted[nk] = v

    return converted


def load_matching_weights(model, state_dict):
    """
    Copy matching weights into model.

    Supports:
      - identical tensors
      - Linear -> Conv1d(kernel_size=1)
    """

    model_state = model.state_dict()

    loaded = []
    skipped = []

    for k, src in state_dict.items():

        if k not in model_state:
            skipped.append((k, "missing"))
            continue

        dst = model_state[k]

        # Same shape
        if dst.shape == src.shape:
            dst.copy_(src)
            loaded.append(k)
            continue

        # Linear -> Conv1d(kernel=1)
        if (
            src.ndim + 1 == dst.ndim
            and dst.shape[-1] == 1
            and src.shape == dst.shape[:-1]
        ):
            dst.copy_(src.unsqueeze(-1))
            loaded.append(k)
            continue

        skipped.append((k, dst.shape, src.shape))

    model.load_state_dict(model_state)

    return loaded, skipped


def load_pretrained_vocos(model_path: str = "checkpoints/model.safetensors") -> Vocos:
    """
    Load pretrained Vocos model from HuggingFace checkpoint.
    """
    # Create model
    model = Vocos()

    # HuggingFace checkpoint:
    # https://huggingface.co/hf-audio/vocos-mel-24khz
    state_dict = load_file(model_path)

    # Convert key names
    state_dict = convert_vocos_keys(state_dict)

    # Copy weights
    loaded, skipped = load_matching_weights(model, state_dict)

    print(f"Loaded : {len(loaded)} tensors")
    print(f"Skipped: {len(skipped)} tensors")

    if skipped:
        print("\nSkipped keys:")
        for x in skipped:
            print(x)

    return model


if __name__ == "__main__":
    model = load_pretrained_vocos("checkpoints/model.safetensors")

    # Save converted checkpoint
    save_file(model.state_dict(), "vocos-v2.safetensors")
    print("Saved -> vocos-v2.safetensors")
