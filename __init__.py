import torch as th

print(th.version.cuda)
print(th.cuda.is_available())
print(th.cuda.device_count())