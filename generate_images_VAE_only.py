import argparse
import torch
import random
import numpy as np
import os
from monai.utils import set_determinism
from scripts.utils import define_instance
import json

def set_seed(seed):
    set_determinism(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_configs(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    args = argparse.Namespace()

    # Set model config
    for k, v in config["model_config"].items():
        setattr(args, k, v)

    # Set paths
    for k, v in config["paths"].items():
        setattr(args, k, v)

    # Get dimensions from inference config
    args.dim = config["inference"]["diffusion_unet_inference"]["dim"]

    # Calculate latent dimensions
    args.latent_shape = [
        args.latent_channels,
        args.dim[0] // 4,
        args.dim[1] // 4
    ]

    return args

def load_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.spatial_dims = 2  # Keep 2D spatial dimensions
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint = torch.load(args.trained_autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    return autoencoder

def print_model_structure(autoencoder):
    """Print detailed information about the autoencoder's structure"""
    print("\nAutoencoder Structure:")
    print("=====================")

    print("\nDecoder Blocks:")
    for i, block in enumerate(autoencoder.decoder.blocks):
        print(f"\nBlock {i}:")
        print(f"Type: {type(block).__name__}")
        print("Attributes:", [attr for attr in dir(block) if not attr.startswith('_')])

        if hasattr(block, 'in_channels'):
            print(f"in_channels: {block.in_channels}")
        if hasattr(block, 'out_channels'):
            print(f"out_channels: {block.out_channels}")

    if hasattr(autoencoder.decoder, 'final_conv'):
        print("\nFinal Conv Layer:")
        final_conv = autoencoder.decoder.final_conv
        print(f"Type: {type(final_conv).__name__}")
        if hasattr(final_conv, 'in_channels'):
            print(f"in_channels: {final_conv.in_channels}")
        if hasattr(final_conv, 'out_channels'):
            print(f"out_channels: {final_conv.out_channels}")

def generate_vae_samples(autoencoder, num_samples, latent_channels, device):
    with torch.no_grad():
        # Generate latents in float32
        latents = torch.randn(num_samples, latent_channels, 64, 64).to(device).float()
        print(f"Latent shape: {latents.shape}, dtype: {latents.dtype}")

        # Process through decoder
        h = autoencoder.post_quant_conv(latents)
        print(f"After post_quant_conv shape: {h.shape}, dtype: {h.dtype}")

        # Decoder processing
        for i, block in enumerate(autoencoder.decoder.blocks):
            if isinstance(block, torch.nn.Module):
                try:
                    # Handle different MAISI block types
                    block_type = type(block).__name__
                    print(f"\nProcessing {block_type}")

                    # Handle each block type
                    if block_type == 'MaisiResBlock':
                        # Convert input to float32 for ResBlock operations
                        h = h.float()

                        # Process through ResBlock components
                        identity = h

                        # Normalization and activation
                        h = block.norm1(h)
                        h = torch.nn.functional.silu(h)

                        # First convolution
                        if hasattr(block.conv1, 'forward_no_split'):
                            h = block.conv1.forward_no_split(h)
                        else:
                            h = block.conv1.conv(h)  # Use conv directly to avoid splitting

                        # Second normalization and activation
                        h = block.norm2(h)
                        h = torch.nn.functional.silu(h)

                        # Second convolution
                        if hasattr(block.conv2, 'forward_no_split'):
                            h = block.conv2.forward_no_split(h)
                        else:
                            h = block.conv2.conv(h)  # Use conv directly to avoid splitting

                        # Handle shortcut
                        if block.nin_shortcut is not None and not isinstance(block.nin_shortcut, torch.nn.Identity):
                            if hasattr(block.nin_shortcut, 'forward_no_split'):
                                identity = block.nin_shortcut.forward_no_split(identity)
                            else:
                                identity = block.nin_shortcut.conv(identity)  # Use conv directly

                        h = identity + h

                    elif block_type == 'MaisiUpsample':
                        h = h.float()  # Ensure float32
                        if block.use_convtranspose:
                            h = block.conv.conv(h)  # Use conv directly
                        else:
                            h = torch.nn.functional.interpolate(h, scale_factor=2.0, mode='nearest')
                            h = block.conv.conv(h)  # Use conv directly

                    elif block_type == 'MaisiGroupNorm3D':
                        h = h.float()  # Ensure float32
                        num_groups = block.num_groups if hasattr(block, 'num_groups') else 32
                        h = torch.nn.functional.group_norm(
                            h,
                            num_groups=num_groups,
                            weight=block.weight.float() if hasattr(block, 'weight') else None,
                            bias=block.bias.float() if hasattr(block, 'bias') else None,
                            eps=block.eps if hasattr(block, 'eps') else 1e-6
                        )

                    elif block_type == 'MaisiConvolution':
                        h = h.float()  # Ensure float32
                        h = block.conv(h)  # Use the standard forward pass

                    else:
                        # For any other block types
                        h = h.float()  # Ensure float32
                        h = block(h)

                    print(f"After block {i} ({block_type}) shape: {h.shape}, dtype: {h.dtype}")

                except Exception as e:
                    print(f"Error in block {i} ({block_type}): {str(e)}")
                    print(f"Current tensor shape: {h.shape}")
                    print(f"Current tensor dtype: {h.dtype}")
                    if hasattr(block, 'conv1'):
                        print(f"Conv1 module type: {type(block.conv1)}")
                    raise

        # Final processing if needed
        if hasattr(autoencoder.decoder, 'final_conv'):
            h = h.float()  # Ensure float32
            if hasattr(autoencoder.decoder.final_conv, 'conv'):
                h = autoencoder.decoder.final_conv.conv(h)  # Use conv directly
            else:
                h = autoencoder.decoder.final_conv(h)
            print(f"After final_conv shape: {h.shape}, dtype: {h.dtype}")

        return h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    config = load_configs(args.config_path)

    autoencoder = load_autoencoder(config)

    def convert_model_to_float32(model):
        for param in model.parameters():
            param.data = param.data.float()
        return model

    # In your main function, before generating:
    autoencoder = convert_model_to_float32(autoencoder)

    device = next(autoencoder.parameters()).device

    print(f"Generating {args.num_images} images...")
    print_model_structure(autoencoder)  # Add this line before generate_vae_samples
    images = generate_vae_samples(autoencoder, args.num_images, config.latent_channels, device)

    for i, img in enumerate(images):
        save_path = f"{args.output_dir}/vae_sample_{i}.nii.gz"
        torch.save(img.cpu(), save_path)
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()