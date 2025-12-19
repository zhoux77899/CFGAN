import psutil
import pynvml


def monitor_disk_usage() -> dict[str, float]:
    disk_usage = {}

    disk_partitions = psutil.disk_partitions()
    for disk in disk_partitions:
        usage = psutil.disk_usage(disk.device)
        disk_usage[disk.device[0]] = usage.percent

    return disk_usage


def monitor_gpu_memory(device_id: int) -> float:
    pynvml.nvmlInit()
    handler = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handler)
    return memory_info.free / 1024 ** 2
