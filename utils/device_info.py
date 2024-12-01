import os
import platform

def get_cpu_info():
    """
    Get CPU model information based on the operating system.
    """
    try:
        if platform.system() == "Windows":
            # Use WMIC command for Windows
            # print("WWWWWWWWWWWWWWWWWWWindoows")
            result = os.popen("wmic cpu get name").read()
            # print(f"Raw result: {result}")
            lines = [line.strip() for line in result.split("\n") if line.strip()]
            if len(lines) > 1:  # 第一个是标题 'Name'，第二个是实际内容
                return lines[1]
            else:
                return "Unknown CPU"
        elif platform.system() == "Linux":
            # Read from /proc/cpuinfo for Linux
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        elif platform.system() == "Darwin":  # macOS
            # Use sysctl command for macOS
            return os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
        else:
            return "Unsupported OS"
    except Exception as e:
        return f"Error retrieving CPU information: {e}"

def get_gpu_info_nvidia_smi():
    """
    Get GPU information using the nvidia-smi command.
    """
    try:
        gpu_info = os.popen("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv").read()
        return gpu_info.strip()
    except Exception as e:
        return f"Error retrieving GPU information: {e}"