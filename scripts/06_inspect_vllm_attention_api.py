import inspect
import vllm

print("vLLM:", vllm.__version__)

try:
    import vllm.v1.attention.layer as m

    print("module:", inspect.getsourcefile(m))
    for name in dir(m):
        if "Attention" in name:
            obj = getattr(m, name)
            print("\n", name)
            try:
                print(inspect.signature(obj))
            except Exception:
                pass
            try:
                print(inspect.getsourcefile(obj))
            except Exception:
                pass
except Exception as e:
    print("inspect failed:", e)
