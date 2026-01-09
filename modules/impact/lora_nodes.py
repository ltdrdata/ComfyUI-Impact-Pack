# modules/impact/lora_nodes.py

# ------------------------------------------------------------
# Internal helpers (shared by Simple and Advanced nodes)
# ------------------------------------------------------------


def _map_schedule_range(schedule_range):
    if schedule_range == "early":
        return 0.0, 0.4
    if schedule_range == "mid":
        return 0.2, 0.7
    if schedule_range == "late":
        return 0.5, 1.0
    return 0.0, 1.0  # full


def _map_schedule_quality(schedule_quality):
    if schedule_quality == "low":
        return 3
    if schedule_quality == "high":
        return 7
    return 5  # medium


def _map_schedule_mode(schedule_mode, strength):
    if schedule_mode == "fade_in":
        return 0.0, strength
    if schedule_mode == "fade_out":
        return strength, 0.0
    if schedule_mode == "fade_in_out":
        return 0.0, strength
    if schedule_mode == "late_boost":
        return 0.0, strength
    return strength, strength


class ScheduledLoRALoader:
    """
    Scheduled LoRA Loader.

    Loads a single LoRA and applies an explicit scheduling curve to it.
    The scheduling curve (strength_start -> strength_end) is the single
    source of truth for LoRA influence.

    Also outputs a visual curve preview as an IMAGE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        lora_list = folder_paths.get_filename_list("loras")
        lora_list.insert(0, "None")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (lora_list,),
                "strength_start": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05},
                ),
                "strength_end": (
                    "FLOAT",
                    {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.05},
                ),
                "interpolation": (
                    ["linear", "ease_in", "ease_out", "ease_in_out"],
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "keyframes_count": (
                    "INT",
                    {"default": 5, "min": 2, "max": 12, "step": 1},
                ),
            },
            "optional": {
                "apply_to_conds": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "IMAGE")
    RETURN_NAMES = ("model", "clip", "curve_preview")

    FUNCTION = "load"
    CATEGORY = "ImpactPack/LoRA"

    def load(
        self,
        model,
        clip,
        lora_name,
        strength_start,
        strength_end,
        interpolation,
        start_percent,
        end_percent,
        keyframes_count,
        apply_to_conds=True,
    ):
        # ----------------------------------------------------
        # Generate curve preview (always safe)
        # ----------------------------------------------------
        curve_preview = self._generate_curve_preview(
            lora_name,
            strength_start,
            strength_end,
            interpolation,
            start_percent,
            end_percent,
            keyframes_count,
        )

        # ----------------------------------------------------
        # Fast path: no LoRA or constant strength
        # ----------------------------------------------------
        if lora_name == "None" or strength_start == strength_end:
            if lora_name == "None":
                return model, clip, curve_preview

            import comfy.sd
            import comfy.utils
            import folder_paths

            lora_path = folder_paths.get_full_path("loras", lora_name)
            if lora_path is None:
                raise FileNotFoundError(f"LoRA not found: {lora_name}")

            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

            model, clip = comfy.sd.load_lora_for_models(
                model,
                clip,
                lora,
                strength_end,
                0.0,
            )
            return model, clip, curve_preview

        # ----------------------------------------------------
        # Scheduling path
        # ----------------------------------------------------
        from comfy_extras.nodes_hooks import (CreateHookKeyframesInterpolated,
                                              CreateHookLora, SetClipHooks,
                                              SetHookKeyframes)

        # 1. Create hook for this LoRA
        (hooks,) = CreateHookLora().create_hook(
            lora_name=lora_name,
            strength_model=strength_end,
            strength_clip=0.0,
            prev_hooks=None,
        )

        # 2. Create keyframes
        (hook_kf,) = CreateHookKeyframesInterpolated().create_hook_keyframes(
            strength_start=strength_start,
            strength_end=strength_end,
            interpolation=interpolation,
            start_percent=start_percent,
            end_percent=end_percent,
            keyframes_count=keyframes_count,
            print_keyframes=False,
            prev_hook_kf=None,
        )

        # 3. Attach keyframes
        (hooks,) = SetHookKeyframes().set_hook_keyframes(
            hooks=hooks,
            hook_kf=hook_kf,
        )

        # 4. Apply hooks to CLIP
        (clip,) = SetClipHooks().apply_hooks(
            clip=clip,
            hooks=hooks,
            apply_to_conds=apply_to_conds,
            schedule_clip=False,
        )

        return model, clip, curve_preview

    # --------------------------------------------------------
    # Curve visualization helper
    # --------------------------------------------------------
    def _generate_curve_preview(
        self,
        lora_name,
        strength_start,
        strength_end,
        interpolation,
        start_percent,
        end_percent,
        keyframes_count=None,
        width=384,  # CAMBIO: Ajustado a 16:9 (más angosto)
        height=216,  # CAMBIO: Ajustado a 16:9
    ):
        import os

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        import torch
        from matplotlib.backends.backend_agg import \
            FigureCanvasAgg as FigureCanvas

        # ----------------------------------------------------
        # Build discrete keyframe-based curve
        # ----------------------------------------------------
        k_count = max(2, int(keyframes_count))
        start_t = max(0.0, min(start_percent, 1.0))
        end_t = max(start_t, min(end_percent, 1.0))

        x_list, y_list = [], []

        if start_t > 0.0:
            x_list.extend([0.0, start_t])
            y_list.extend([0.0, 0.0])

        x_active = np.linspace(start_t, end_t, k_count)
        t_active = (x_active - start_t) / max(end_t - start_t, 1e-6)

        if interpolation == "linear":
            curve_vals = t_active
        elif interpolation == "ease_in":
            curve_vals = t_active**2
        elif interpolation == "ease_out":
            curve_vals = 1 - (1 - t_active) ** 2
        elif interpolation == "ease_in_out":
            curve_vals = np.where(
                t_active < 0.5,
                2 * t_active**2,
                1 - (-2 * t_active + 2) ** 2 / 2,
            )
        else:
            curve_vals = t_active

        y_active = strength_start + curve_vals * (strength_end - strength_start)

        x_list.extend(x_active)
        y_list.extend(y_active)

        if end_t < 1.0:
            x_list.append(1.0)
            y_list.append(y_active[-1])

        x_plot = np.array(x_list)
        y_plot = np.array(y_list)

        # ----------------------------------------------------
        # Figure setup
        # ----------------------------------------------------
        chart_title = "LoRA Scheduling Curve"
        text_color = "#e6e6e6"
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor((0.18, 0.18, 0.20, 0.60))
        
        fig.text(
            0.5,
            0.92,
            chart_title,
            ha="center",
            va="center",
            color=text_color,
            fontsize=9,
            alpha=0.9,
        )

        # Chart area
        ax = fig.add_axes([0.10, 0.12, 0.85, 0.73])

        ax.set_facecolor("none")
        ax.grid(True, color="#8a8a8a", linewidth=0.7, alpha=0.22)

        s_min = min(0.0, strength_start, strength_end)
        s_max = max(0.0, strength_start, strength_end)
        span = max(0.1, s_max - s_min)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(s_min - span * 0.05, s_max + span * 0.05)

        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        ax.tick_params(colors=text_color, labelsize=8)

        for spine in ax.spines.values():
            spine.set_edgecolor("#777777")
            spine.set_linewidth(1.0)

        # ----------------------------------------------------
        # Plot curve
        # ----------------------------------------------------
        ax.axhline(
            0, color="#ffffff", linewidth=1.0, linestyle="--", alpha=0.35
        )

        ax.fill_between(
            x_plot,
            y_plot,
            0,
            where=(y_plot >= 0),
            color="#5b8cff",
            alpha=0.55,
            interpolate=True,
        )

        ax.fill_between(
            x_plot,
            y_plot,
            0,
            where=(y_plot < 0),
            color="#e06c3f",
            alpha=0.55,
            interpolate=True,
        )

        ax.plot(x_plot, y_plot, color="#ffffff", linewidth=2.0, alpha=0.9)
        ax.scatter(x_active, y_active, color="#ffffff", s=14, zorder=5)

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------
        canvas = FigureCanvas(fig)
        canvas.draw()

        buf = np.asarray(canvas.buffer_rgba())
        plt.close(fig)

        img = torch.from_numpy(buf).float() / 255.0
        img = img.unsqueeze(0)

        return img
