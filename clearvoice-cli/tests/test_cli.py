from pathlib import Path
import unittest

from clearvoice_cli.cli import build_parser, resolve_output_path, should_use_online_write, validate_backend


class CliParserTests(unittest.TestCase):
    def test_enhance_parser_default_model(self) -> None:
        args = build_parser().parse_args(["enhance", "input.wav"])

        self.assertEqual(args.command, "enhance")
        self.assertEqual(args.model, "MossFormer2_SE_48K")
        self.assertEqual(args.backend, "torch")
        self.assertEqual(args.input, Path("input.wav"))

    def test_separate_parser_default_model(self) -> None:
        args = build_parser().parse_args(["separate", "input.wav", "--backend", "mlx"])

        self.assertEqual(args.command, "separate")
        self.assertEqual(args.model, "MossFormer2_SS_16K")
        self.assertEqual(args.backend, "mlx")

    def test_mlx_backend_is_allowed_for_separation(self) -> None:
        validate_backend(command="separate", model_name="MossFormer2_SS_16K", backend="mlx")

    def test_mlx_backend_is_allowed_for_48k_enhancement(self) -> None:
        validate_backend(command="enhance", model_name="MossFormer2_SE_48K", backend="mlx")

    def test_mlx_backend_is_rejected_for_unported_16k_enhancement(self) -> None:
        with self.assertRaises(SystemExit):
            validate_backend(command="enhance", model_name="FRCRN_SE_16K", backend="mlx")

    def test_scp_uses_online_write(self) -> None:
        self.assertTrue(should_use_online_write(Path("files.scp")))

    def test_default_single_file_output_path(self) -> None:
        output_path = resolve_output_path(
            command="enhance",
            input_path=Path("/tmp/input.wav"),
            output_path=None,
            model_name="MossFormer2_SE_48K",
        )

        self.assertEqual(output_path.name, "input_enhanced_MossFormer2_SE_48K.wav")


if __name__ == "__main__":
    unittest.main()
