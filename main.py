import argparse
import os
import sys
import shutil
from dtcore import trainer



def generate_c_code_from_model(model_dir: str, out_dir: str) -> None:
    """Generate C code from saved params.pkl using weights_to_firmware submodule."""
    submodule_path = os.path.join(os.path.dirname(__file__), 'weights_to_firmware')
    if submodule_path not in sys.path:
        sys.path.append(submodule_path)
    try:
        import quad_gen.get_models as wt_get_models
    except Exception as e:
        raise RuntimeError(f"Failed to import weights_to_firmware (looked in: {submodule_path}). Error: {e}")

    # absolute_path=True to use out_dir exactly as given
    wt_get_models.save_result(model_dir=model_dir, out_dir=out_dir, osi=False, absolute_path=True)


def move_model_to_weights_to_firmware(model_dir: str) -> str:
    """Move the pkl file from model_dir to weights_to_firmware's input_model folder."""
    # Find the pkl file in the model directory
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not pkl_files:
        raise RuntimeError(f"No .pkl files found in {model_dir}")
    
    pkl_file = pkl_files[0]  # Take the first pkl file found
    source_path = os.path.join(model_dir, pkl_file)
    
    # Create input_model directory if it doesn't exist
    input_model_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware', 'input_model')
    os.makedirs(input_model_dir, exist_ok=True)
    
    # Move the pkl file
    dest_path = os.path.join(input_model_dir, pkl_file)
    shutil.copy2(source_path, dest_path)
    print(f"Moved {pkl_file} to {dest_path}")
    
    return input_model_dir


def move_network_evaluate_to_firmware(weights_to_firmware_dir: str) -> None:
    """Move network_evaluate.c from weights_to_firmware output to firmware/network_evaluate.h."""
    # Look for network_evaluate.c in the output directory
    output_dir = os.path.join(weights_to_firmware_dir, 'output_model')
    network_evaluate_c = os.path.join(output_dir, 'network_evaluate.c')
    
    if not os.path.exists(network_evaluate_c):
        raise RuntimeError(f"network_evaluate.c not found at {network_evaluate_c}")
    
    # Create firmware directory if it doesn't exist
    firmware_dir = os.path.join(os.path.dirname(__file__), 'firmware')
    os.makedirs(firmware_dir, exist_ok=True)
    
    # Move and rename to network_evaluate.h
    dest_path = os.path.join(firmware_dir, 'network_evaluate.h')
    shutil.copy2(network_evaluate_c, dest_path)
    print(f"Moved network_evaluate.c to {dest_path}")


def main():
    parser = argparse.ArgumentParser(description='DroneTrain entrypoint')
    parser.add_argument('--env', default='simple', help='Environment name registered in Brax')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training or with loaded params')
    parser.add_argument('--model_dir', default='models/mjx_brax_policy', help='Directory to save/load model params')
    parser.add_argument('--video', default='gifs/simple_train.mp4', help='Video path for evaluation (set empty to skip)')
    parser.add_argument('--steps', type=int, default=200, help='Evaluation steps')
    parser.add_argument('--model_xml', default=None, help='Override MuJoCo XML path for the environment')
    args = parser.parse_args()

    env_kwargs = {'model_xml_path': args.model_xml} if args.model_xml else None

    make_inference_fn = None
    params = None

    if args.train:
        make_inference_fn, params, _ = trainer.train(env_name=args.env, model_dir=args.model_dir, env_kwargs=env_kwargs)
        
        # After training, perform the complete workflow:
        # 1. Move pkl file to weights_to_firmware input_model folder
        print("Moving model file to weights_to_firmware input_model folder...")
        input_model_dir = move_model_to_weights_to_firmware(args.model_dir)
        
        # 2. Generate C code using the moved params.pkl
        print("Generating C code from moved model file...")
        out_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware', 'output_model')
        os.makedirs(out_dir, exist_ok=True)
        generate_c_code_from_model(input_model_dir, out_dir)
        print("C code generation completed.")
        
        # 3. Move network_evaluate.c to firmware/network_evaluate.h
        print("Moving network_evaluate.c to firmware/network_evaluate.h...")
        weights_to_firmware_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware')
        move_network_evaluate_to_firmware(weights_to_firmware_dir)
        print("Complete workflow finished successfully!")
    
    if args.eval:
        if make_inference_fn is None or params is None:
            # load and build inference
            from brax.io import model as brax_model_io
            make_inference_fn, _, _ = trainer.train(env_name=args.env, config_overrides={'num_timesteps': 0}, env_kwargs=env_kwargs)
            params = brax_model_io.load_params(args.model_dir)
        trainer.evaluate(env_name=args.env, make_inference_fn=make_inference_fn, params=params, n_steps=args.steps, video_path=args.video or None, env_kwargs=env_kwargs)


if __name__ == "__main__":
    main()
