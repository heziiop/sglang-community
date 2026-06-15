import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import MINIMAX_M3_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=600,
    suite="nightly-8-npu-a3",
    nightly=True,
)


class TestMiniMaxM3W8A8(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the Eco-Tech/MiniMax-M3-w8a8
    model on the GSM8K dataset is no less than 0.6.

    [Test Category] Model
    [Test Target] Eco-Tech/MiniMax-M3-w8a8
    """

    model = MINIMAX_M3_W8A8_WEIGHTS_PATH
    accuracy = 0.6
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.88",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--max-running-requests",
        "64",
        "--chunked-prefill-size",
        "-1",
        "--quantization",
        "modelslim",
    ]
    timeout_for_server_launch = 1800


if __name__ == "__main__":
    unittest.main()
