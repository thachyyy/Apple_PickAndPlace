"""RGB-guided Apple Pick-and-Place with robosuite (multi-apple, LLM instruction parsing).

instruction --(keyword / local LLM)--> (color, count)
camera RGB --(color mask, blobs)--> pixels --(camera ray ∩ table plane)--> target XYZ
target XYZ --(scripted OSC waypoints)--> apple in the box


"""

import argparse
import sys

from src.language import HF_MODEL, InstructionError, parse_instruction


def main():
    parser = argparse.ArgumentParser(description="RGB-guided apple pick-and-place")
    parser.add_argument(
        "--instruction",
        default="Put the red apple into the box.",
        help="Examples: 'Put the red apple into the box.', 'Đặt toàn bộ táo xanh vào hộp.'",
    )
    parser.add_argument("--no-render", action="store_true", help="Disable the on-screen viewer")
    parser.add_argument(
        "--no-llm", action="store_true", help="Use the keyword parser instead of the LLM"
    )
    parser.add_argument("--llm-model", default=HF_MODEL, help="Hugging Face model id for the parser")
    parser.add_argument(
        "--save-steps",
        metavar="DIR",
        help="Save an image for every pipeline step (detection, each motion phase) into DIR",
    )
    args = parser.parse_args()
    render = not args.no_render

    # Parse first: a bad instruction should fail before the simulator starts.
    try:
        color, count = parse_instruction(
            args.instruction, use_llm=not args.no_llm, model_name=args.llm_model
        )
    except InstructionError as exc:
        sys.exit(f"Cannot run instruction {args.instruction!r}: {exc}")

    recorder = None
    if args.save_steps:
        from src.recorder import StepRecorder

        recorder = StepRecorder(args.save_steps)
        recorder.save_json(
            "instruction",
            {"instruction": args.instruction, "color": color, "count": count or "all"},
        )

    # Imported here so robosuite only loads once the instruction is valid.
    from src.control import run_pick_place
    from src.env import make_env

    env = make_env(render=render, record=recorder is not None)
    try:
        obs = env.reset()
        print("RGB vision guides target selection; this is not a trained VLA.")
        run_pick_place(env, obs, color=color, count=count, render=render, recorder=recorder)
        if render:
            print("Close the simulation window to exit.")
            while True:
                env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
