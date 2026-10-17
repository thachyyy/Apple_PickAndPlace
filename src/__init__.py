"""RGB-guided apple pick-and-place with robosuite, split by pipeline stage.

config    scene geometry, camera and motion constants (no robosuite)
language  instruction -> (color, count), keyword or local Hugging Face LLM (no robosuite)
sim_setup robosuite runtime tweaks; import before any other robosuite module
env       the robosuite scene: Panda, table, open box, red/green apples
vision    RGB color segmentation and pixel -> table-plane projection
control   scripted waypoint motion and the pick-and-place task loop
recorder  optional per-step image export (--save-steps)
"""
