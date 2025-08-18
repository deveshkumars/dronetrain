import argparse
import os
import sys
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
        # After training, generate C code using the saved params.pkl
        out_dir = os.path.join(os.path.dirname(__file__), 'weights_to_firmware', 'output_model')
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating C code from model_dir={args.model_dir} into out_dir={out_dir} ...")
        generate_c_code_from_model(model_dir=args.model_dir, out_dir=out_dir)
        print("C code generation completed. See output files in:", out_dir)
    
    if args.eval:
        if make_inference_fn is None or params is None:
            # load and build inference
            from brax.io import model as brax_model_io
            make_inference_fn, _, _ = trainer.train(env_name=args.env, config_overrides={'num_timesteps': 0}, env_kwargs=env_kwargs)
            params = brax_model_io.load_params(args.model_dir)
        trainer.evaluate(env_name=args.env, make_inference_fn=make_inference_fn, params=params, n_steps=args.steps, video_path=args.video or None, env_kwargs=env_kwargs)


if __name__ == "__main__":
    main()
