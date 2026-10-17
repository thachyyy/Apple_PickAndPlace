"""robosuite runtime tweaks. Import this before any other robosuite module."""

import os
import platform

import robosuite.macros as macros

# Use the normal top-left image convention so pixel coordinates match camera math.
macros.IMAGE_CONVENTION = "opencv"

# robosuite forces MUJOCO_GL=egl on every non-macOS system when GPU rendering is
# enabled, but EGL is Linux-only. On Windows, fall back to WGL (GLFW context).
if platform.system() == "Windows":
    import robosuite.utils.binding_utils as binding_utils

    os.environ["MUJOCO_GL"] = "wgl"
    binding_utils._MUJOCO_GL = "wgl"
