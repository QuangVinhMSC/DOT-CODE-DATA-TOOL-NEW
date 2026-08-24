"""Re-run the YOLO export from the saved job archive.

The exporter has no resume, but every image is a pure function of
(job.id, export seed, image index) and every split assignment is a pure
function of the index -- so re-running with the *saved* job (which keeps its
id, unlike a freshly snapshotted one) rewrites the images already on disk
byte-identically and fills in whatever is missing.
"""
import argparse, sys, time

from dotgen.core import io_config
from dotgen.core.exporter import preflight, run_export, report_text
from dotgen.core.state import AppState


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="cofig/orig_match.dotjobs")
    ap.add_argument("--config", default="cofig/orig_match.dotcfg")
    ap.add_argument("--images", type=int, default=0, help="0 = use the saved value")
    ap.add_argument("--out", default="", help="empty = use the saved value")
    args = ap.parse_args()

    state = AppState()
    io_config.load_config(state, args.config)
    spec = state.export
    if args.images:
        spec.images_per_job = args.images
    if args.out:
        spec.out_dir = args.out

    jobs = io_config.load_jobs(args.jobs)
    print("job        :", jobs[0].name, jobs[0].id,
          "| dot model", jobs[0].dot_model.n_samples, "samples")
    print("classes    :", [c.name for c in jobs[0].classes if c.enabled])
    print("export     :", spec.out_dir, spec.images_per_job, "images, seed", spec.seed)

    errors = preflight(jobs, spec)
    print("preflight  :", errors or "clean")
    if errors:
        return 1

    t0 = time.time()
    last = [0]

    def progress(done: int, total: int) -> bool:
        if done - last[0] >= 500 or done == total:
            last[0] = done
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"  {done}/{total}  {el:6.1f}s  {rate:5.1f} img/s  "
                  f"eta {(total - done) / rate / 60 if rate else 0:5.1f} min", flush=True)
        return True

    report = run_export(jobs, spec, progress)
    print(report_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
