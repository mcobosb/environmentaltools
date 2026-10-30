"""Command line of the SWAN workflow: ``python -m environmentaltools.models.swan <step> swan.yaml``.

Steps:
    prepare       create or update the cases (sea states × variants) and the overview to review them
    run           run SWAN in the cases without result (only the active variants; --variant NAME, --case NAME,
                  --filter TEXT for some; --again to repeat)
    postprocess   tables of all the cases (points, convergence) and maps of the chosen cases (--again for all)
    all           the three steps in a row
"""

import argparse
import sys

from . import load_config, postprocess, prepare, run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m environmentaltools.models.swan", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "run", "postprocess", "all"])
    parser.add_argument("project", help="project file (swan.yaml)")
    parser.add_argument("--case", action="append", help="only this case (can be repeated)")
    parser.add_argument("--variant", action="append", help="only the cases of this variant (can be repeated)")
    parser.add_argument("--filter", help="only the cases whose name contains this text")
    parser.add_argument("--threads", type=int, help="OpenMP threads of SWAN")
    parser.add_argument("--again", action="store_true", help="repeat finished cases (run) or everything (postprocess)")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    cfg = load_config(args.project)
    if args.step in ("prepare", "all"):
        prepare(cfg)
    if args.step in ("run", "all"):
        try:
            log = run(cfg, filter=args.filter, cases=args.case, threads=args.threads, force=args.again,
                      variants=args.variant)
        except FileNotFoundError as exc:
            sys.exit(str(exc))
        if len(log) and "ok" in log and not log["ok"].all():
            sys.exit("Some case failed: see screen_*.log and PRINT_* of the case")
    if args.step in ("postprocess", "all"):
        postprocess(cfg, force=args.again)


if __name__ == "__main__":
    main()
