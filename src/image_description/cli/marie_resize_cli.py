"""Resize Marie-Parie photographs without invoking AI descriptions."""
from image_description.cli.marie_prepare_cli import main as prepare_main


def main(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--no-describe" not in arguments:
        arguments.append("--no-describe")
    prepare_main(arguments)


if __name__ == "__main__":
    main()
