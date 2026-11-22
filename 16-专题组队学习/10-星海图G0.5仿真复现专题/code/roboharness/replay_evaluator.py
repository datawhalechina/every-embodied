"""Run the stock evaluator in a fresh, privately leased simulator cache."""
import os

from private_storage import private_appdata, run_private_preflight


def main():
    from behavior_interface import runtime_storage
    from behavior_interface_eval_test import official_evaluator_entrypoint

    base = os.environ["OMNIGIBSON_APPDATA_PATH"]
    appdata, lease, created_before = private_appdata(base)
    os.environ["OMNIGIBSON_APPDATA_PATH"] = str(appdata)
    original = runtime_storage.preflight_storage

    def checked(*args, **kwargs):
        return run_private_preflight(runtime_storage, original, appdata, lease,
                                     created_before, *args, **kwargs)

    runtime_storage.preflight_storage = checked
    try:
        official_evaluator_entrypoint.main()
    finally:
        runtime_storage.preflight_storage = original
        lease.close()


if __name__ == "__main__":
    main()
