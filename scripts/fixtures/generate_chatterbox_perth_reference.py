#!/usr/bin/env python3
"""Generate the PerTh watermarker fixture with upstream's PyTorch code.

The reference is resemble-ai/Perth at ff1c8ac5 (MIT), the implicit
watermarker Chatterbox applies to every utterance, with its checkpoint
`perth_net_250000.pth.tar` (sha256 a15bce45…), torch 2.6.0, F32 on the CPU.
The state dict is read with `torch.load(weights_only=True)`, which loads
tensors and plain containers only.

The input is the synthetic signal of `generate_chatterbox_clone_reference.py`
(bit-reproducible in Rust), taken as 32 kHz audio so that upstream's
`apply_watermark` runs without resampling; resampling is izwi's own
(torchaudio's sinc resampler) and is not under test here. The fixture
records, as strided sample points:
  * the normalised dB magnitude and the phase of the STFT;
  * the frame mask and the encoder's residual on the lowest 128 bins;
  * the watermarked signal;
  * upstream's detector on the original and the watermarked signal.

Usage:

    python3 -I scripts/fixtures/generate_chatterbox_perth_reference.py \\
        path/to/perth_net_250000.pth.tar crates/izwi-core/tests/fixtures/chatterbox
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

SOURCE = "resemble-ai/Perth@ff1c8ac55a976971245cdd53c18d6131ca00d993"
HERE = Path(__file__).resolve().parent


def clone_script():
    spec = importlib.util.spec_from_file_location(
        "clone_reference", HERE / "generate_chatterbox_clone_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    from perth.perth_net.perth_net_implicit.config import default_hp
    from perth.perth_net.perth_net_implicit.model.perth_net import PerthNet
    from perth.perth_net.perth_net_implicit.model import magmask
    from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker

    clone = clone_script()
    torch.set_grad_enabled(False)
    net = PerthNet(default_hp)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    net.load_state_dict(state)
    net.eval()
    watermarker = PerthImplicitWatermarker(run_name=None, perth_net=net)

    signal = clone.reference_signal()
    magspec, phase = net.ap.signal_to_magphase(torch.from_numpy(signal))
    mask = magmask(magspec[None])[0]
    residual = net.encoder.layers(magspec[None, : net.encoder.subband])[0] * mask
    watermarked = watermarker.apply_watermark(signal, sample_rate=default_hp.sample_rate)
    detect = lambda x: float(watermarker.get_watermark(x, default_hp.sample_rate, round=False))

    fixture = {
        "source": SOURCE,
        "sample_rate": default_hp.sample_rate,
        "signal": clone.sample(torch.from_numpy(signal)),
        "magspec": clone.sample(magspec),
        "phase": clone.sample(phase),
        "mask": [int(v) for v in mask.tolist()],
        "residual": clone.sample(residual),
        "watermarked": clone.sample(torch.from_numpy(watermarked)),
        "detector": {"original": detect(signal), "watermarked": detect(watermarked)},
    }
    out = args.out_dir / "perth_reference.json"
    out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {out}: {magspec.shape[1]} frames, detector {fixture['detector']}")


if __name__ == "__main__":
    main()
