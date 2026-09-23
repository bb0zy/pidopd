import torch, time, sys

print(f'torch: {torch.__version__}')
print(f'cuda: {torch.version.cuda}')
print(f'可用 GPU: {torch.cuda.device_count()}')
print(f'名字: {torch.cuda.get_device_name(0)}')
print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
print()
print("开始烧 GPU，跑 60 秒...")
print("-" * 60)

a = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()

start = time.time()
iter_count = 0
flops = 0.0
next_report = time.time() + 5

while time.time() - start < 60:
    for _ in range(50):
        a = a @ a / 4096.0
    torch.cuda.synchronize()
    iter_count += 50
    flops += 50 * 2 * 4096**3

    if time.time() >= next_report:
        elapsed = time.time() - start
        print(f"  {elapsed:6.1f}s | iter {iter_count:6d} | "
              f"{flops/elapsed/1e12:6.2f} TFLOPS | "
              f"GSM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
        next_report += 5

torch.cuda.synchronize()
print("-" * 60)
print(f"总时长: {time.time()-start:.1f}s")
print(f"均速: {flops/(time.time()-start)/1e12:.2f} TFLOPS")
print("GPU 正常工作 ✅")
