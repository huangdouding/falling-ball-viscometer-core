"""
批量运行新管线处理第一周所有实验视频。

用法:
    python batch_run.py

输出目录: C:/Users/25570/Desktop/物理竞赛1/实验/第一周/新管线结果/
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace")

from src.utils import load_config, normalize_config_keys, ensure_output_dir
from src.pipeline import run_pipeline

BASE_VIDEO = "C:/Users/25570/Desktop/物理竞赛1/实验/第一周"
OUTPUT_ROOT = os.path.join(BASE_VIDEO, "新管线结果")

# ── 以下参数仅在 config.yaml 缺少对应值时作为回退 ──
BALL_DENSITY = 7850.0      # kg/m3
LIQUID_DENSITY = 950.0     # kg/m3
LIQUID_HEIGHT_MM = 335.0   # 33.5 cm
_SCALE_FALLBACK = 0.286902  # 仅在 config.yaml 无 scale_mm_per_px 时使用

# 球径 → 半径映射
DIAM_TO_RADIUS = {1.5: 0.75, 2.0: 1.0, 2.5: 1.25}

# 参考黏度
REF_VISC = {30: 0.451, 40: 0.231}


def run_single(video_path: str, ball_size: float, temp: int, output_dir: str) -> dict:
    """Run pipeline on single video, return result summary."""
    config = load_config("config.yaml")
    config = normalize_config_keys(config)

    config["video_path"] = video_path
    config["ball_radius_mm"] = DIAM_TO_RADIUS[ball_size]
    config["ball_density_kg_m3"] = BALL_DENSITY
    config["liquid_density_kg_m3"] = LIQUID_DENSITY
    config["liquid_height_mm"] = LIQUID_HEIGHT_MM
    config["cylinder_radius_mm"] = 20.0  # unchanged
    # ★ 优先使用 config.yaml 的比例尺，仅回退到脚本常量
    if config.get("scale_mm_per_px") is None:
        config["scale_mm_per_px"] = _SCALE_FALLBACK
    config["temperature_c"] = float(temp)
    config["reference_viscosity_pa_s"] = REF_VISC[temp]
    config["output_dir"] = output_dir
    config["manual_fps"] = 240.0
    config["save_plots"] = True
    config["save_marked_video"] = False
    config["gravity_m_s2"] = 9.98

    ensure_output_dir(output_dir)

    start_t = time.time()
    result = run_pipeline(
        video_path, config, output_dir,
        debug_callback=lambda m: None,
        progress_callback=lambda c, t: None,
    )
    elapsed = time.time() - start_t

    # Build summary
    tr = result.get("terminal_region", {})
    vr = result.get("viscosity_result")
    ti = result.get("track_interval")

    summary = {
        "success": result["success"],
        "elapsed_s": round(elapsed, 1),
        "terminal_found": tr.get("found", False),
        "vt_m_s": tr.get("terminal_velocity_m_s"),
        "r2": tr.get("r2"),
        "cv": tr.get("cv"),
        "region": f"{tr.get('start_time_s','?'):.3f}-{tr.get('end_time_s','?'):.3f}s" if tr.get("found") else "N/A",
        "frames_raw": int((result["traj_df"]["point_type"] == "raw").sum()),
        "frames_total": len(result["traj_df"]),
        "y_disp_m": round(result["traj_df"]["y_m"].max() - result["traj_df"]["y_m"].min(), 6),
    }
    if vr:
        summary["eta_basic_Pa_s"] = round(float(vr["eta_basic_pa_s"]), 6)
        summary["eta_final_Pa_s"] = round(float(vr["eta_final_pa_s"]), 6)
        summary["wall_factor"] = round(float(vr.get("wall_correction_factor", 1.0)), 4)
        summary["reynolds_factor"] = round(float(vr["reynolds_factor"]), 4)
        summary["Re"] = round(float(vr["reynolds_number"]), 4)
        summary["ref_Pa_s"] = REF_VISC[temp]
        if summary.get("eta_final_Pa_s") and summary["ref_Pa_s"]:
            summary["error_pct"] = round(
                (summary["eta_final_Pa_s"] - summary["ref_Pa_s"]) / summary["ref_Pa_s"] * 100, 1
            )
    if hasattr(ti, "start_frame"):
        summary["interval_frames"] = f"F{ti.start_frame}-F{ti.end_frame}"
        summary["interval_s"] = round(ti.duration_s, 3)

    return summary


def main():
    print("=" * 60)
    print("  批量处理: 第一周所有实验视频")
    print(f"  输出: {OUTPUT_ROOT}")
    print("=" * 60)

    all_summaries = []
    total_videos = 0
    failed_videos = 0

    # Walk video directory structure
    for ball_str in sorted(os.listdir(BASE_VIDEO)):
        ball_dir = os.path.join(BASE_VIDEO, ball_str)
        if not os.path.isdir(ball_dir) or ball_str == "新管线结果":
            continue
        try:
            ball_size = float(ball_str)
        except ValueError:
            continue

        for temp_str in sorted(os.listdir(ball_dir)):
            temp_dir = os.path.join(ball_dir, temp_str)
            if not os.path.isdir(temp_dir):
                continue
            try:
                temp = int(temp_str)
            except ValueError:
                continue

            for video_file in sorted(os.listdir(temp_dir)):
                if not video_file.endswith(".mp4"):
                    continue

                video_path = os.path.join(temp_dir, video_file)
                name_noext = os.path.splitext(video_file)[0]

                # Output dir: 新管线结果/ball/temp/video_name/
                out_dir = os.path.join(OUTPUT_ROOT, ball_str, temp_str, name_noext)
                total_videos += 1

                print(f"\n[{total_videos}] {video_file}  ({ball_str}mm, {temp}C) ...", end=" ", flush=True)

                try:
                    summary = run_single(video_path, ball_size, temp, out_dir)
                    summary["video"] = video_file
                    summary["ball_mm"] = ball_size
                    summary["temp_C"] = temp
                    all_summaries.append(summary)

                    if summary["terminal_found"]:
                        eta = summary.get("eta_final_Pa_s", "?")
                        err = summary.get("error_pct", "?")
                        print(f"OK  vt={summary['vt_m_s']:.4f}  eta={eta}  err={err}%  ({summary['elapsed_s']}s)")
                    else:
                        print(f"OK  no terminal velocity  ({summary['elapsed_s']}s)")
                except Exception as e:
                    failed_videos += 1
                    print(f"FAILED: {e}")
                    all_summaries.append({
                        "video": video_file,
                        "ball_mm": ball_size,
                        "temp_C": temp,
                        "success": False,
                        "error": str(e),
                    })

    # ── Print summary table ──
    print("\n\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(f"{'视频':22s} {'球径':6s} {'温度':6s} {'检测率':16s} {'Y位移':10s} {'v_t':10s} {'R²':8s} {'η_final':10s} {'参考':8s} {'误差':8s}")
    print("-" * 70)

    for s in all_summaries:
        if s.get("success"):
            det = f"{s.get('frames_raw','?')}/{s.get('frames_total','?')}"
            vt = f"{s.get('vt_m_s','?'):.4f}" if s.get("vt_m_s") else "N/A"
            r2 = f"{s.get('r2','?'):.4f}" if s.get("r2") else "N/A"
            eta = f"{s.get('eta_final_Pa_s','?'):.4f}" if s.get("eta_final_Pa_s") else "N/A"
            ref = f"{s.get('ref_Pa_s','?'):.4f}" if s.get("ref_Pa_s") else "N/A"
            err = f"{s.get('error_pct','?'):+.1f}%" if s.get("error_pct") is not None else "N/A"
            ydisp = f"{s.get('y_disp_m',0):.4f}"
            print(f"{s['video']:22s} {s['ball_mm']}mm   {s['temp_C']}C    {det:14s} {ydisp:8s} {vt:8s} {r2:6s} {eta:8s} {ref:6s} {err:8s}")
        else:
            print(f"{s.get('video','?'):22s} {'?':6s} {'?':6s} {'FAILED':16s}")

    print("-" * 70)
    print(f"总计: {total_videos} 个视频, 失败: {failed_videos}")

    # Save summary CSV
    import csv
    csv_path = os.path.join(OUTPUT_ROOT, "summary.csv")
    fieldnames = [
        "video", "ball_mm", "temp_C", "success", "elapsed_s",
        "terminal_found", "vt_m_s", "r2", "cv", "region",
        "frames_raw", "frames_total", "y_disp_m",
        "eta_basic_Pa_s", "eta_final_Pa_s", "wall_factor", "reynolds_factor", "Re",
        "ref_Pa_s", "error_pct", "interval_frames", "interval_s", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_summaries)
    print(f"\n详细结果已保存: {csv_path}")
    print(f"结果目录: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
