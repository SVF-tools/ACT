# act/pipeline/tmp_debug_net.py

import logging
from act.pipeline.verification.model_factory import ModelFactory

def dump_net_var_ids(net_name: str = "mnist_robust_easy") -> None:
    """
    打印指定 ACT Net 中每一层的 in_vars / out_vars 范围，
    用来对照 box 约束里的 var_ids。
    """
    factory = ModelFactory()
    net = factory.get_act_net(net_name)

    print("=" * 80)
    print(f"Net name: {net_name}")
    print("=" * 80)

    for L in net.layers:
        in_ids  = list(L.in_vars)
        out_ids = list(L.out_vars)

        in_min  = min(in_ids)  if in_ids  else None
        in_max  = max(in_ids)  if in_ids  else None
        out_min = min(out_ids) if out_ids else None
        out_max = max(out_ids) if out_ids else None

        print(
            f"Layer {L.id:2d}  kind={L.kind:12s}  "
            f"in_n={len(in_ids):4d}  in_min={str(in_min):>4}  in_max={str(in_max):>4}  |  "
            f"out_n={len(out_ids):4d} out_min={str(out_min):>4} out_max={str(out_max):>4}"
        )

def main():
    logging.basicConfig(level=logging.INFO)
    # 你可以改成 "cifar_margin_strict" 等等
    dump_net_var_ids("mnist_robust_easy")

if __name__ == "__main__":
    main()
