import time
import torch
import sys

def monitor_gpu(interval=5, stop_event=None, usage_list=None):
    while True:
        if stop_event is not None and stop_event.is_set():
            break

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # in MB
            cached = torch.cuda.memory_reserved() / 1024**2  # in MB

            print(f"[GPU MONITOR] {time.strftime('%H:%M:%S')}", file=sys.stderr)
            print(f"Allocated: {allocated:.2f} MB", file=sys.stderr)
            print(f"Cached:    {cached:.2f} MB", file=sys.stderr)

            if usage_list is not None:
                usage_list.append(allocated)

        else:
            print("[GPU MONITOR] CUDA is not available.", file=sys.stderr)

        time.sleep(interval)
