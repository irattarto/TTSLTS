from ttslts import gpu_memory
from ttslts.controller import Controller
from ttslts.gpu_memory import GpuMemoryGate
from ttslts.settings import Settings


def test_parse_nvidia_smi_output():
    assert gpu_memory.parse_free_mb("6012\n") == 6012
    assert gpu_memory.parse_free_mb("[N/A]\n4000, 8188\n") == 4000
    assert gpu_memory.parse_free_mb("") is None


def gate(values):
    clock = [0.0]
    seq = iter(values)
    g = GpuMemoryGate(read=lambda: next(seq), clock=lambda: clock[0])

    def at(seconds):
        clock[0] = seconds
        return g.allowed()

    return at


def test_hysteresis_keeps_models_from_bouncing():
    at = gate([4000, 1000, 400, 1000, 2000])
    assert at(0)  # plenty free
    assert at(6)  # 1000 MB free: below the start limit, but the models are already on the GPU
    assert not at(12)  # under 512 MB: move off the GPU
    assert not at(18)  # 1000 MB is not enough to go back
    assert at(24)  # 2000 MB is


def test_the_card_is_only_read_every_few_seconds():
    reads = []
    clock = [0.0]
    g = GpuMemoryGate(read=lambda: reads.append(1) or 4000, clock=lambda: clock[0])
    for t in (0, 1, 2, 4):
        clock[0] = t
        g.allowed()
    assert len(reads) == 1
    clock[0] = 6
    g.allowed()
    assert len(reads) == 2


def test_unreadable_memory_counts_as_allowed():
    assert gate([None])(0)


def test_controller_keeps_models_off_a_full_gpu():
    controller = Controller(Settings(use_gpu=True), save=lambda s: None)
    controller._gpu_memory = GpuMemoryGate(read=lambda: 100, clock=lambda: 0.0)
    assert not controller.gpu_allowed()
    controller._gpu_memory = GpuMemoryGate(read=lambda: 4000, clock=lambda: 0.0)
    assert controller.gpu_allowed()
