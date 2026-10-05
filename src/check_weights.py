import torch
import sys

def check_checkpoint(ckpt_path):
    print(f"Loading {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    state_dict = checkpoint.get("state_dict", checkpoint)
    
    with open("checkpoint_keys.txt", "w") as f:
        f.write("--- All keys and shapes in checkpoint ---\n")
        keys = list(state_dict.keys())
        f.write(f"Total keys: {len(keys)}\n\n")
        
        for k in keys:
            f.write(f"{k}: {state_dict[k].shape}\n")
            
    print("Successfully wrote all keys and shapes to 'checkpoint_keys.txt'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_checkpoint(sys.argv[1])
    else:
        check_checkpoint("/home/mav24/projects/MaRaI/weights/3d_et_rt/checkpoints/epoch_latest.pt")
